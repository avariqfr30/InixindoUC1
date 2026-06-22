"""Serialized Internal API reads used by proposal generation preflight."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
from typing import Any


logger = logging.getLogger(__name__)
_internal_api_lock = threading.RLock()


@contextmanager
def internal_api_serial_access():
    """Serialize access to the process-wide Internal API client."""

    with _internal_api_lock:
        yield


@dataclass(frozen=True)
class GenerationPreflightSnapshot:
    """Immutable container for values read through the shared Internal API client."""

    expert_bench_context: Any
    framework_context: Any
    firm_data: Any
    firm_profile: Any
    relationship_context: Any


def load_internal_preflight(
    firm_api: Any,
    *,
    project_type: str,
    base_client: str,
    expert_bench_context: Any = None,
    expert_bench_resolved: bool = False,
    framework_context: Any = None,
) -> GenerationPreflightSnapshot:
    """Read Internal API inputs serially so the shared client is never concurrent."""

    with internal_api_serial_access():
        resolved_expert = expert_bench_context
        if not expert_bench_resolved and (
            not isinstance(resolved_expert, dict) or not resolved_expert.get("available")
        ):
            try:
                resolved_expert = firm_api.get_expert_bench_context(limit_products=8)
            except Exception:
                logger.exception("Internal evidence enrichment failed")
                resolved_expert = None

        resolved_framework = framework_context
        if not resolved_framework:
            try:
                resolved_framework = firm_api.get_framework_catalog()
            except Exception:
                resolved_framework = None

        firm_data = firm_api.get_delivery_guidance(project_type)
        firm_profile = firm_api.get_firm_profile()
        relationship_context = firm_api.get_client_relationship(base_client)

    return GenerationPreflightSnapshot(
        expert_bench_context=resolved_expert,
        framework_context=resolved_framework,
        firm_data=firm_data,
        firm_profile=firm_profile,
        relationship_context=relationship_context,
    )
