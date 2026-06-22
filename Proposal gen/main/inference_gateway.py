"""Bounded, observable access to proposal-related Ollama inference."""

import threading
import time
from numbers import Real
from typing import Any, Dict, Optional

from ollama import Client

from .config import (
    LLM_MAX_CONCURRENCY,
    LLM_MAX_RETRIES,
    LLM_OVERLOAD_COOLDOWN_SECONDS,
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_RETRY_BACKOFF_SECONDS,
    OLLAMA_HOST,
)


_NATIVE_METRIC_FIELDS = (
    "prompt_eval_count",
    "eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)


class InferenceGateway:
    """Preserve the Ollama chat contract while bounding process load."""

    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        host: str = OLLAMA_HOST,
        max_concurrency: int = LLM_MAX_CONCURRENCY,
        request_timeout_seconds: int = LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = LLM_MAX_RETRIES,
        retry_backoff_seconds: float = LLM_RETRY_BACKOFF_SECONDS,
        overload_cooldown_seconds: float = LLM_OVERLOAD_COOLDOWN_SECONDS,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self._client = client if client is not None else Client(host=host, timeout=request_timeout_seconds)
        self._semaphore = threading.BoundedSemaphore(max(1, int(max_concurrency)))
        self._max_retries = min(1, max(0, int(max_retries)))
        self._retry_backoff_seconds = min(5.0, max(0.0, float(retry_backoff_seconds)))
        self._overload_cooldown_seconds = min(60.0, max(0.0, float(overload_cooldown_seconds)))
        self._clock = clock
        self._sleep = sleep
        self._cooldown_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cooldown_until = 0.0
        self._metrics: Dict[str, Real] = {
            "calls": 0,
            "failures": 0,
            "retries": 0,
            "overloads": 0,
            "wait_seconds": 0.0,
            "wall_seconds": 0.0,
            **{field: 0 for field in _NATIVE_METRIC_FIELDS},
        }

    def chat(self, *args, **kwargs):
        """Call ``Client.chat`` with unchanged arguments and response shape."""
        call_started = self._clock()
        self._increment_metric("calls")
        retries = 0
        try:
            while True:
                try:
                    response = self._invoke(*args, **kwargs)
                    self._record_native_metrics(response)
                    return response
                except Exception as exc:
                    status_code = self._status_code(exc)
                    if status_code not in {429, 503}:
                        raise
                    if retries >= self._max_retries:
                        raise
                    retries += 1
                    self._increment_metric("retries")
                    if self._retry_backoff_seconds:
                        self._sleep(self._retry_backoff_seconds)
        except Exception:
            self._increment_metric("failures")
            raise
        finally:
            self._increment_metric("wall_seconds", max(0.0, self._clock() - call_started))

    def metrics_snapshot(self) -> Dict[str, Real]:
        """Return a thread-safe copy of aggregate operational metrics."""
        with self._state_lock:
            return dict(self._metrics)

    def _invoke(self, *args, **kwargs):
        wait_started = self._clock()
        with self._semaphore:
            capacity_acquired = self._clock()
            with self._state_lock:
                cooldown_active = capacity_acquired < self._cooldown_until
            if cooldown_active:
                with self._cooldown_lock:
                    return self._call_client(wait_started, self._clock(), args, kwargs)
            return self._call_client(wait_started, capacity_acquired, args, kwargs)

    def _call_client(self, wait_started: float, wait_finished: float, args, kwargs):
        self._increment_metric("wait_seconds", max(0.0, wait_finished - wait_started))
        try:
            return self._client.chat(*args, **kwargs)
        except Exception as exc:
            if self._status_code(exc) in {429, 503}:
                # Record cooldown before releasing semaphore capacity so callers
                # already queued behind this attempt cannot bypass it.
                self._mark_overload()
            raise

    def _mark_overload(self) -> None:
        now = self._clock()
        with self._state_lock:
            self._metrics["overloads"] += 1
            self._cooldown_until = max(
                self._cooldown_until,
                now + self._overload_cooldown_seconds,
            )

    def _record_native_metrics(self, response: Any) -> None:
        values: Dict[str, Real] = {}
        for field in _NATIVE_METRIC_FIELDS:
            value = response.get(field) if isinstance(response, dict) else getattr(response, field, None)
            if isinstance(value, Real) and not isinstance(value, bool):
                values[field] = value
        if not values:
            return
        with self._state_lock:
            for field, value in values.items():
                self._metrics[field] += value

    def _increment_metric(self, field: str, value: Real = 1) -> None:
        with self._state_lock:
            self._metrics[field] += value

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        value = getattr(exc, "status_code", None)
        if value is None:
            value = getattr(getattr(exc, "response", None), "status_code", None)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None


_gateway: Optional[InferenceGateway] = None
_gateway_lock = threading.Lock()


def get_inference_gateway() -> InferenceGateway:
    """Return the one inference gateway shared by this Python process."""
    global _gateway
    if _gateway is None:
        with _gateway_lock:
            if _gateway is None:
                _gateway = InferenceGateway()
    return _gateway
