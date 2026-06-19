"""Runtime payload helpers for operator-facing Internal API controls."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


READINESS_LABELS: Dict[str, tuple[str, str]] = {
    "account_records": ("Daftar klien", "Nama klien dan metadata akun"),
    "project_records": ("Riwayat proyek & tenaga ahli", "Portofolio, produk, peran, dan tenaga ahli"),
    "client_relationship": ("Konteks relasi klien", "Riwayat proyek yang relevan dengan klien"),
    "firm_profile": ("Profil perusahaan penyusun", "Kontak, kredensial, dan portofolio resmi"),
}
READINESS_ORDER = ["account_records", "project_records", "client_relationship", "firm_profile"]


def internal_api_has_runtime_resources(api_client: Any) -> bool:
    resources = getattr(api_client, "resource_config", {}) or {}
    return bool(resources.get("project_records") or resources.get("account_records"))


def internal_api_readiness(validation: Dict[str, Any]) -> Dict[str, Any]:
    resources = (validation or {}).get("resources") if isinstance(validation, dict) else {}
    resources = resources if isinstance(resources, dict) else {}
    ordered_keys = [key for key in READINESS_ORDER if key in resources]
    ordered_keys.extend(sorted(key for key in resources.keys() if key not in READINESS_ORDER))
    readiness_resources = []
    for key in ordered_keys:
        payload = resources.get(key) if isinstance(resources.get(key), dict) else {}
        label, description = READINESS_LABELS.get(key, (str(key).replace("_", " ").title(), "Sumber data internal"))
        missing_count = len(payload.get("missing_required_mapping") or [])
        ready = bool(payload.get("ok"))
        readiness_resources.append(
            {
                "key": key,
                "label": label,
                "description": description,
                "ready": ready,
                "known": key in READINESS_LABELS,
                "missing_required_count": missing_count,
                "status_label": "Siap otomatis" if ready else "Perlu mapping",
            }
        )
    return {
        "ok": bool(readiness_resources) and all(item["ready"] for item in readiness_resources),
        "resources": readiness_resources,
    }


def internal_api_setup_status_payload(api_client: Any, knowledge_base: Any, managed_config_path: Path) -> Dict[str, Any]:
    config_file = api_client.config_file or str(managed_config_path)
    project_data_source = getattr(knowledge_base, "project_data_source", "local")
    config_exists = bool(config_file and Path(config_file).exists())
    api_connection_active = project_data_source == "api" and (
        config_exists or internal_api_has_runtime_resources(api_client)
    )
    validation = api_client.validate_config()
    return {
        "config_file": config_file,
        "config_exists": config_exists,
        "project_data_source": project_data_source,
        "api_connection_active": api_connection_active,
        "can_refresh_dataset": api_connection_active,
        "sync_in_progress": getattr(knowledge_base, "sync_in_progress", False),
        "last_refresh_error": getattr(knowledge_base, "last_refresh_error", ""),
        "connection_label": (
            "Aktif memakai Internal API/APIDog"
            if api_connection_active
            else "Belum aktif untuk sesi berjalan"
        ),
        "runtime": api_client.describe_runtime(),
        "readiness": internal_api_readiness(validation),
        "validation": validation,
    }


def internal_api_activation_payload(
    target_path: Path,
    activated: bool,
    refresh_started: bool,
    knowledge_base: Any,
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "ok",
        "config_file": str(target_path),
        "activated": activated,
        "refresh_started": refresh_started,
        "project_data_source": getattr(knowledge_base, "project_data_source", "local"),
        "api_connection_active": getattr(knowledge_base, "project_data_source", "local") == "api",
        "can_refresh_dataset": getattr(knowledge_base, "project_data_source", "local") == "api",
        "sync_in_progress": getattr(knowledge_base, "sync_in_progress", False),
        "last_refresh_error": getattr(knowledge_base, "last_refresh_error", ""),
        "connection_label": "Aktif memakai Internal API/APIDog" if activated else "Belum aktif untuk sesi berjalan",
        "readiness": internal_api_readiness(validation),
        "validation": validation,
        "notes": [
            "Kredensial tetap dibaca dari environment variable agar password tidak disimpan di UI.",
            "Agar aktif permanen setelah restart, set PROJECT_DATA_SOURCE=api dan FIRM_API_CONFIG_FILE ke path config ini.",
        ],
    }


def internal_api_refresh_payload(config_file: str, refresh_started: bool, knowledge_base: Any) -> Dict[str, Any]:
    return {
        "status": "refreshing" if refresh_started else "current",
        "refresh_started": refresh_started,
        "project_data_source": getattr(knowledge_base, "project_data_source", "api"),
        "api_connection_active": True,
        "can_refresh_dataset": True,
        "config_file": config_file,
        "sync_in_progress": getattr(knowledge_base, "sync_in_progress", False),
        "last_refresh_error": getattr(knowledge_base, "last_refresh_error", ""),
        "connection_label": "Aktif memakai Internal API/APIDog",
        "message": "Dataset internal sedang diambil ulang dari endpoint API.",
    }
