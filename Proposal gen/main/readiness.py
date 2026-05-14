"""Readiness probe composition for the proposal generator service."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import requests


def build_readiness_payload(
    project_db_path: Path,
    project_csv_path: Path,
    knowledge_base: Any,
    app_state_store: Any,
    ollama_host: str,
) -> Tuple[Dict[str, Any], int]:
    checks: Dict[str, Dict[str, Any]] = {}

    checks["project_db"] = {
        "ok": project_db_path.exists(),
        "path": str(project_db_path),
    }
    checks["project_csv"] = {
        "ok": project_csv_path.exists(),
        "path": str(project_csv_path),
    }
    checks["knowledge_base"] = {
        "ok": bool(
            getattr(knowledge_base, "df", None) is not None
            and not getattr(knowledge_base.df, "empty", True)
            and "entity" in getattr(knowledge_base.df, "columns", [])
            and "topic" in getattr(knowledge_base.df, "columns", [])
            and getattr(knowledge_base, "vector_ready", False)
            and not getattr(knowledge_base, "last_refresh_error", "")
        ),
        "error": getattr(knowledge_base, "last_refresh_error", ""),
        "sync_in_progress": getattr(knowledge_base, "sync_in_progress", False),
        "vector_store_dir": str(getattr(knowledge_base, "vector_store_dir", "")),
    }

    app_state_ok = False
    app_state_error = ""
    try:
        app_state_store.get_settings()
        app_state_ok = (
            app_state_store.db_path.exists()
            and app_state_store.generated_dir.exists()
            and app_state_store.templates_dir.exists()
            and app_state_store.supporting_docs_dir.exists()
        )
    except Exception as exc:
        app_state_error = str(exc)
    checks["app_state"] = {
        "ok": app_state_ok,
        "db_path": str(app_state_store.db_path),
        "generated_dir": str(app_state_store.generated_dir),
        "templates_dir": str(app_state_store.templates_dir),
        "supporting_docs_dir": str(app_state_store.supporting_docs_dir),
        "error": app_state_error,
    }

    ollama_ok = False
    ollama_error = ""
    try:
        resp = requests.get(f"{ollama_host.rstrip('/')}/api/version", timeout=4)
        ollama_ok = bool(resp.ok)
        if not resp.ok:
            ollama_error = f"HTTP {resp.status_code}"
    except Exception as exc:
        ollama_error = str(exc)
    checks["ollama"] = {
        "ok": ollama_ok,
        "host": ollama_host,
        "error": ollama_error,
    }

    ready_ok = all(item.get("ok") for item in checks.values())
    return {
        "status": "ready" if ready_ok else "degraded",
        "checks": checks,
        "timestamp": time.time(),
    }, 200 if ready_ok else 503
