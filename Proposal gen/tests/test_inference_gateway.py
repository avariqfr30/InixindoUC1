import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("ollama") is None:
    ollama_stub = types.ModuleType("ollama")

    class MissingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Tests must inject or patch the Ollama client")

    ollama_stub.Client = MissingClient
    sys.modules["ollama"] = ollama_stub


class StatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"status={status_code}")
        self.status_code = status_code


class ObservedSemaphore:
    def __init__(self, value, expected_waiters):
        self._semaphore = threading.BoundedSemaphore(value)
        self._state_lock = threading.Lock()
        self._waiters = 0
        self._expected_waiters = expected_waiters
        self.waiters_ready = threading.Event()

    def __enter__(self):
        if not self._semaphore.acquire(blocking=False):
            with self._state_lock:
                self._waiters += 1
                if self._waiters == self._expected_waiters:
                    self.waiters_ready.set()
            self._semaphore.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._semaphore.release()


class InferenceGatewayTests(unittest.TestCase):
    def _gateway(self, client, **overrides):
        from main.inference_gateway import InferenceGateway

        options = {
            "client": client,
            "max_concurrency": 2,
            "max_retries": 1,
            "retry_backoff_seconds": 0,
            "overload_cooldown_seconds": 10,
        }
        options.update(overrides)
        return InferenceGateway(**options)

    def test_chat_preserves_arguments_and_response_identity(self):
        response = {"message": {"content": "unchanged"}}

        class Client:
            def chat(inner_self, *args, **kwargs):
                inner_self.args = args
                inner_self.kwargs = kwargs
                return response

        client = Client()
        gateway = self._gateway(client)

        actual = gateway.chat("model-name", messages=[{"role": "user", "content": "hello"}])

        self.assertIs(actual, response)
        self.assertEqual(client.args, ("model-name",))
        self.assertEqual(client.kwargs, {"messages": [{"role": "user", "content": "hello"}]})

    def test_chat_never_exceeds_configured_concurrency(self):
        release = threading.Event()
        two_started = threading.Event()
        state_lock = threading.Lock()
        state = {"active": 0, "maximum": 0}

        class BlockingClient:
            def chat(self, *args, **kwargs):
                with state_lock:
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                    if state["active"] == 2:
                        two_started.set()
                release.wait(timeout=2)
                with state_lock:
                    state["active"] -= 1
                return {"message": {"content": "ok"}}

        gateway = self._gateway(BlockingClient(), overload_cooldown_seconds=0)
        threads = [threading.Thread(target=gateway.chat) for _ in range(4)]
        for thread in threads:
            thread.start()
        self.assertTrue(two_started.wait(timeout=1), "two permitted calls did not start")
        time.sleep(0.05)
        release.set()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(state["maximum"], 2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_retries_once_only_for_overload_statuses(self):
        for status_code in (429, 503):
            with self.subTest(status_code=status_code):
                calls = []

                class OverloadedOnceClient:
                    def chat(self, *args, **kwargs):
                        calls.append(status_code)
                        if len(calls) == 1:
                            raise StatusError(status_code)
                        return {"message": {"content": "recovered"}}

                gateway = self._gateway(OverloadedOnceClient(), max_retries=7)
                result = gateway.chat()

                self.assertEqual(result["message"]["content"], "recovered")
                self.assertEqual(len(calls), 2)
                self.assertEqual(gateway.metrics_snapshot()["retries"], 1)

        calls = []

        class NonOverloadClient:
            def chat(self, *args, **kwargs):
                calls.append("called")
                raise StatusError(500)

        gateway = self._gateway(NonOverloadClient())
        with self.assertRaises(StatusError):
            gateway.chat()
        self.assertEqual(len(calls), 1)
        metrics = gateway.metrics_snapshot()
        self.assertEqual(metrics["retries"], 0)
        self.assertEqual(metrics["failures"], 1)

        calls = []

        class AlwaysOverloadedClient:
            def chat(self, *args, **kwargs):
                calls.append("called")
                raise StatusError(503)

        gateway = self._gateway(AlwaysOverloadedClient(), max_retries=99)
        with self.assertRaises(StatusError):
            gateway.chat()
        self.assertEqual(len(calls), 2)

    def test_overload_cooldown_temporarily_serializes_later_calls(self):
        first_release = threading.Event()
        first_started = threading.Event()
        second_started = threading.Event()
        state_lock = threading.Lock()
        state = {"phase": "overload", "active": 0, "maximum": 0, "successes": 0}

        class CooldownClient:
            def chat(self, *args, **kwargs):
                if state["phase"] == "overload":
                    state["phase"] = "success"
                    raise StatusError(429)
                with state_lock:
                    state["active"] += 1
                    state["successes"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                    if state["successes"] == 1:
                        first_started.set()
                    else:
                        second_started.set()
                first_release.wait(timeout=2)
                with state_lock:
                    state["active"] -= 1
                return {"message": {"content": "ok"}}

        gateway = self._gateway(
            CooldownClient(),
            max_retries=0,
            clock=lambda: 0.0,
        )
        with self.assertRaises(StatusError):
            gateway.chat()

        threads = [threading.Thread(target=gateway.chat) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.assertTrue(first_started.wait(timeout=1))
        self.assertFalse(second_started.wait(timeout=0.1), "cooldown allowed concurrent calls")
        first_release.set()
        self.assertTrue(second_started.wait(timeout=1))
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(state["maximum"], 1)

    def test_queued_callers_recheck_cooldown_after_capacity_becomes_available(self):
        overload_started = threading.Event()
        occupier_started = threading.Event()
        release_overload = threading.Event()
        release_occupier = threading.Event()
        first_success_started = threading.Event()
        second_success_started = threading.Event()
        release_first_success = threading.Event()
        state_lock = threading.Lock()
        state = {"calls": 0}

        class QueuedClient:
            def chat(self, *args, **kwargs):
                with state_lock:
                    state["calls"] += 1
                    call_number = state["calls"]
                if call_number == 1:
                    overload_started.set()
                    release_overload.wait(timeout=2)
                    raise StatusError(429)
                if call_number == 2:
                    occupier_started.set()
                    release_occupier.wait(timeout=2)
                    return {"message": {"content": "occupier"}}
                if call_number == 3:
                    first_success_started.set()
                    release_first_success.wait(timeout=2)
                else:
                    second_success_started.set()
                return {"message": {"content": "ok"}}

        gateway = self._gateway(QueuedClient(), max_retries=0, overload_cooldown_seconds=30)
        observed_semaphore = ObservedSemaphore(value=2, expected_waiters=2)
        gateway._semaphore = observed_semaphore
        errors = []

        def call_gateway():
            try:
                gateway.chat()
            except Exception as exc:
                errors.append(exc)

        overload_thread = threading.Thread(target=call_gateway)
        overload_thread.start()
        self.assertTrue(overload_started.wait(timeout=1))
        occupier_thread = threading.Thread(target=call_gateway)
        occupier_thread.start()
        self.assertTrue(occupier_started.wait(timeout=1))

        queued_threads = [threading.Thread(target=call_gateway) for _ in range(2)]
        for thread in queued_threads:
            thread.start()
        self.assertTrue(observed_semaphore.waiters_ready.wait(timeout=1))

        release_overload.set()
        self.assertTrue(first_success_started.wait(timeout=1))
        release_occupier.set()
        self.assertFalse(
            second_success_started.wait(timeout=0.1),
            "a caller queued before overload bypassed active cooldown",
        )
        release_first_success.set()
        self.assertTrue(second_success_started.wait(timeout=1))

        for thread in [overload_thread, occupier_thread, *queued_threads]:
            thread.join(timeout=2)
        self.assertTrue(all(not thread.is_alive() for thread in queued_threads))
        self.assertEqual(len(errors), 1)
        self.assertEqual(getattr(errors[0], "status_code", None), 429)

    def test_metrics_accumulate_native_ollama_usage_fields(self):
        values = iter([0.0, 0.5, 1.0, 3.0])
        response = {
            "prompt_eval_count": 11,
            "eval_count": 7,
            "total_duration": 101,
            "load_duration": 13,
            "prompt_eval_duration": 17,
            "eval_duration": 71,
        }
        gateway = self._gateway(
            types.SimpleNamespace(chat=lambda *args, **kwargs: response),
            clock=lambda: next(values),
        )

        gateway.chat()

        metrics = gateway.metrics_snapshot()
        self.assertEqual(metrics["calls"], 1)
        self.assertEqual(metrics["failures"], 0)
        self.assertEqual(metrics["overloads"], 0)
        self.assertEqual(metrics["wait_seconds"], 0.5)
        self.assertEqual(metrics["wall_seconds"], 3.0)
        for field, expected in response.items():
            self.assertEqual(metrics[field], expected)

    def test_constructed_client_receives_request_timeout(self):
        from main import inference_gateway as module

        captured = {}

        class Client:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        with patch.object(module, "Client", Client):
            module.InferenceGateway(
                host="http://ollama.internal:11434",
                request_timeout_seconds=77,
            )

        self.assertEqual(captured, {"host": "http://ollama.internal:11434", "timeout": 77})

    def test_factory_returns_one_process_wide_gateway(self):
        from main import inference_gateway as module

        class Client:
            def __init__(self, *args, **kwargs):
                pass

        with patch.object(module, "Client", Client), patch.object(module, "_gateway", None):
            first = module.get_inference_gateway()
            second = module.get_inference_gateway()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
