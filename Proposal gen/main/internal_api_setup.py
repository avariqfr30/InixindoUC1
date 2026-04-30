"""Internal API/APIDog setup helpers shared by the web UI and CLI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


RESOURCE_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "firm_profile": {
        "office_address": ["office address", "office_address", "address", "alamat", "alamat kantor", "company_address"],
        "email": ["email", "official email", "official_email", "company_email", "contact_email"],
        "phone": ["phone", "telephone", "telp", "telepon", "phone_number", "contact_number"],
        "whatsapp": ["whatsapp", "wa"],
        "website": ["website", "url", "website_url", "company_website", "site"],
        "legal_name": ["legal_name", "company_legal_name", "company_name", "nama_perusahaan"],
        "operating_hours": ["operating_hours", "business_hours", "jam_operasional"],
        "profile_summary": ["profile_summary", "company_summary", "summary", "description"],
        "credential_highlights": ["credential_highlights", "credentials", "capabilities", "certification"],
        "portfolio_highlights": ["portfolio_highlights", "portfolio", "experience", "case_study"],
    },
    "project_standards": {
        "methodology": ["methodology", "metodologi", "delivery_methodology", "approach", "framework", "metode_kerja"],
        "team": ["team", "team_structure", "delivery_team", "resource_plan", "staffing", "struktur_tim"],
        "commercial": ["commercial", "commercial_terms", "pricing_terms", "payment_terms", "scope_terms"],
    },
    "client_relationship": {
        "summary": ["summary", "relationship_summary", "description", "notes", "history", "engagement_history"],
        "mode": ["mode", "status", "relationship_status", "client_status", "engagement_status", "has_relationship"],
    },
    "project_records": {
        "entity": ["entity", "client_entity", "client", "client_name", "company", "company_name", "customer", "nama_klien"],
        "topic": ["topic", "strategic_initiative", "initiative", "project", "project_name", "program", "inisiatif"],
        "budget": ["budget", "investment_estimation", "investment", "estimated_budget", "anggaran", "nilai_proyek"],
    },
    "account_records": {
        "company_name": ["company_name", "company", "client", "client_name", "nama_perusahaan", "nama_klien"],
        "company_region_name": ["company_region_name", "region", "city", "kota", "wilayah"],
        "company_province_name": ["company_province_name", "province", "provinsi"],
        "company_segment": ["company_segment", "segment", "segmen"],
        "company_sub_segment": ["company_sub_segment", "sub_segment", "subsegmen"],
    },
}


DEFAULT_DATASETS = {
    "firm_profile": "ReferenceAccount",
    "project_standards": "ProjectStandards",
    "client_relationship": "ConsultantProjectExpertHistory",
    "project_records": "ConsultantProjectExpertHistory",
    "account_records": "ReferenceAccount",
}


DEFAULT_RESPONSE_PATHS = {
    "firm_profile": "data.dataset_result.0",
    "project_standards": "data.dataset_result",
    "client_relationship": "data.dataset_result",
    "project_records": "data.dataset_result",
    "account_records": "data.dataset_result",
}


def normalize_api_setup_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    endpoint_url = str((data or {}).get("url") or "").strip()
    if not re.match(r"^https?://", endpoint_url, flags=re.IGNORECASE):
        raise ValueError("Endpoint API harus berupa URL HTTP/HTTPS penuh.")

    method = str((data or {}).get("method") or "POST").strip().upper()
    if method not in {"GET", "POST", "PUT", "PATCH"}:
        method = "POST"

    body_encoding = str((data or {}).get("body_encoding") or "form").strip().lower()
    if body_encoding not in {"form", "json"}:
        body_encoding = "form"

    auth_mode = str((data or {}).get("auth_mode") or "basic").strip().lower()
    if auth_mode not in {"basic", "bearer", "none"}:
        auth_mode = "basic"

    datasets = data.get("datasets") if isinstance(data.get("datasets"), dict) else {}
    response_paths = data.get("response_paths") if isinstance(data.get("response_paths"), dict) else {}

    return {
        "url": endpoint_url,
        "method": method,
        "body_encoding": body_encoding,
        "auth_mode": auth_mode,
        "datasets": datasets,
        "response_paths": response_paths,
    }


def build_internal_api_config(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_api_setup_payload(data)
    datasets = normalized["datasets"]
    paths = normalized["response_paths"]

    return {
        "mode": "generic",
        "auth_mode": normalized["auth_mode"],
        "request_defaults": {
            "url": normalized["url"],
            "method": normalized["method"],
            "body_encoding": normalized["body_encoding"],
            "params": {},
            "headers": {},
        },
        "resources": {
            "firm_profile": {
                "request": {"body": {"dataset": str(datasets.get("firm_profile") or DEFAULT_DATASETS["firm_profile"])}},
                "response_path": str(paths.get("firm_profile") or DEFAULT_RESPONSE_PATHS["firm_profile"]),
                "field_mapping": {
                    "office_address": "company_address",
                    "email": "official_email",
                    "phone": "telephone",
                    "whatsapp": "whatsapp",
                    "website": "website_url",
                    "legal_name": "legal_name",
                    "operating_hours": "operating_hours",
                    "profile_summary": "profile_summary",
                    "credential_highlights": "credential_highlights",
                    "portfolio_highlights": "portfolio_highlights",
                },
            },
            "project_standards": {
                "request": {"body": {"dataset": str(datasets.get("project_standards") or DEFAULT_DATASETS["project_standards"])}},
                "response_path": str(paths.get("project_standards") or DEFAULT_RESPONSE_PATHS["project_standards"]),
                "record_filters": {"project_type": "{project_type}"},
                "field_mapping": {
                    "methodology": "delivery_methodology",
                    "team": "team_composition",
                    "commercial": "commercial_terms",
                },
            },
            "client_relationship": {
                "request": {"body": {"dataset": str(datasets.get("client_relationship") or DEFAULT_DATASETS["client_relationship"])}},
                "response_path": str(paths.get("client_relationship") or DEFAULT_RESPONSE_PATHS["client_relationship"]),
                "record_filters": {"project_name__icontains": "{client_name}"},
                "field_mapping": {
                    "summary": "project_name",
                    "project_name": "project_name",
                    "product_name": "product_name",
                    "expert_name": "expert_name",
                    "position_name": "position_name",
                },
            },
            "project_records": {
                "request": {"body": {"dataset": str(datasets.get("project_records") or DEFAULT_DATASETS["project_records"])}},
                "response_path": str(paths.get("project_records") or DEFAULT_RESPONSE_PATHS["project_records"]),
                "field_mapping": {
                    "entity": "project_name",
                    "topic": "product_name",
                    "project_name": "project_name",
                    "expert_name": "expert_name",
                    "position_name": "position_name",
                },
            },
            "account_records": {
                "request": {"body": {"dataset": str(datasets.get("account_records") or DEFAULT_DATASETS["account_records"])}},
                "response_path": str(paths.get("account_records") or DEFAULT_RESPONSE_PATHS["account_records"]),
                "record_filters": {"company_name__icontains": "{client_name}"},
                "field_mapping": {
                    "company_name": "company_name",
                    "company_region_name": "company_region_name",
                    "company_province_name": "company_province_name",
                    "company_segment": "company_segment",
                    "company_sub_segment": "company_sub_segment",
                    "company_category_name": "company_category_name",
                    "company_category_desc": "company_category_desc",
                },
            },
        },
    }


def write_json_config(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_key(value: str) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def extract_path(payload: Any, path: str) -> Any:
    if not path:
        return payload
    try:
        import jmespath

        return jmespath.search(path, payload)
    except Exception:
        current = payload
        for part in path.split("."):
            if not part:
                continue
            if isinstance(current, list):
                if not part.isdigit():
                    return None
                idx = int(part)
                if idx >= len(current):
                    return None
                current = current[idx]
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current


def flatten_leaf_paths(payload: Any, prefix: str = "") -> List[Tuple[str, str]]:
    paths: List[Tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            child_path = f"{prefix}.{key_text}" if prefix else key_text
            paths.extend(flatten_leaf_paths(value, child_path))
    elif isinstance(payload, list):
        if payload:
            child_path = f"{prefix}.0" if prefix else "0"
            paths.extend(flatten_leaf_paths(payload[0], child_path))
    else:
        paths.append((prefix, str(payload or "")))
    return paths


def infer_mapping(resource: str, sample_node: Any) -> Dict[str, str]:
    alias_map = RESOURCE_ALIASES[resource]
    leaf_paths = flatten_leaf_paths(sample_node)
    mapping: Dict[str, str] = {}
    used_paths = set()
    for target_field, aliases in alias_map.items():
        alias_keys = {normalize_key(target_field), *(normalize_key(alias) for alias in aliases)}
        best_path = ""
        best_score = 0
        for path, _value in leaf_paths:
            if path in used_paths:
                continue
            leaf_key = normalize_key(path.split(".")[-1])
            path_key = normalize_key(path)
            score = 0
            if leaf_key in alias_keys:
                score = 3
            elif any(alias and alias in leaf_key for alias in alias_keys):
                score = 2
            elif any(alias and alias in path_key for alias in alias_keys):
                score = 1
            if score > best_score:
                best_score = score
                best_path = path
        if best_path:
            mapping[target_field] = best_path
            used_paths.add(best_path)
    return mapping
