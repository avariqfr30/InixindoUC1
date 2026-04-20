"""Operational diagnostics for profile and internal data connectivity."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import requests

from .config import (
    APP_PROFILE,
    FIRM_API_CONFIG_FILE,
    INTERNAL_DATA_FALLBACK,
    INTERNAL_DATA_SOURCE,
    OLLAMA_HOST,
    PROJECT_CSV_PATH,
    PROJECT_DB_PATH,
    SERPER_API_KEY,
)
from .runtime_components import InternalDataClient


def _bool_status(value: bool) -> str:
    return "ok" if value else "error"


def build_snapshot(project_type: str, client_name: str) -> Dict[str, Any]:
    client = InternalDataClient()
    snapshot = client.doctor_snapshot(project_type=project_type, client_name=client_name)
    snapshot["environment"] = {
        "app_profile": APP_PROFILE,
        "internal_data_source": INTERNAL_DATA_SOURCE,
        "internal_data_fallback": INTERNAL_DATA_FALLBACK,
        "firm_api_config_file": FIRM_API_CONFIG_FILE,
        "firm_api_config_exists": bool(FIRM_API_CONFIG_FILE and Path(FIRM_API_CONFIG_FILE).exists()),
        "project_db_exists": PROJECT_DB_PATH.exists(),
        "project_csv_exists": PROJECT_CSV_PATH.exists(),
        "serper_configured": bool(str(SERPER_API_KEY or "").strip() and str(SERPER_API_KEY).strip() != "SERPER_API"),
    }
    try:
        resp = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/version", timeout=4)
        snapshot["environment"]["ollama"] = {
            "ok": bool(resp.ok),
            "status_code": resp.status_code,
            "host": OLLAMA_HOST,
        }
    except Exception as exc:
        snapshot["environment"]["ollama"] = {
            "ok": False,
            "host": OLLAMA_HOST,
            "error": str(exc),
        }

    env_ok = (
        snapshot["environment"]["project_db_exists"]
        and snapshot["environment"]["project_csv_exists"]
        and snapshot["environment"]["ollama"].get("ok", False)
    )
    snapshot["ok"] = bool(snapshot.get("ok")) and env_ok
    return snapshot


def render_text(snapshot: Dict[str, Any]) -> str:
    runtime = snapshot.get("runtime", {})
    environment = snapshot.get("environment", {})
    resources = snapshot.get("resources", {})
    lines = [
        f"Overall: {_bool_status(bool(snapshot.get('ok')))}",
        f"Profile: {environment.get('app_profile', '')}",
        f"Internal data: {runtime.get('operator_mode', '')}",
        f"API config file: {environment.get('firm_api_config_file') or '-'} ({_bool_status(bool(environment.get('firm_api_config_exists')) or not environment.get('firm_api_config_file'))})",
        f"Ollama: {_bool_status(bool((environment.get('ollama') or {}).get('ok')))}",
        f"Project DB: {_bool_status(bool(environment.get('project_db_exists')))}",
        f"Project CSV: {_bool_status(bool(environment.get('project_csv_exists')))}",
        f"Serper: {_bool_status(bool(environment.get('serper_configured')))}",
        "",
        "Resources:",
    ]
    for name, payload in resources.items():
        lines.append(f"- {name}: {_bool_status(bool(payload.get('ok')))}")
        error = str(payload.get("error") or "").strip()
        if error:
            lines.append(f"  error: {error}")
            continue
        fields = payload.get("fields") or {}
        if fields:
            field_bits = ", ".join(f"{key}={'ok' if bool(value) else 'missing'}" for key, value in fields.items())
            lines.append(f"  fields: {field_bits}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check app profile and internal data readiness.")
    parser.add_argument("--project-type", default="Implementation")
    parser.add_argument("--client-name", default="PT Contoh Klien")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    snapshot = build_snapshot(project_type=args.project_type, client_name=args.client_name)
    if args.format == "json":
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        print(render_text(snapshot))
    return 0 if snapshot.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
