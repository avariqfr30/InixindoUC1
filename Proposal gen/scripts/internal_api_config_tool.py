#!/usr/bin/env python3
"""Create and inspect internal API config files for production handover."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


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
}


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_starter_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "mode": "generic",
        "auth_mode": args.auth_mode,
        "request_defaults": {
            "url": args.url,
            "method": args.method,
            "body_encoding": args.body_encoding,
            "params": {},
            "headers": {},
        },
        "resources": {
            "firm_profile": {
                "request": {"body": {"dataset": args.firm_dataset}},
                "response_path": args.firm_response_path,
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
                "request": {"body": {"dataset": args.standards_dataset}},
                "response_path": args.standards_response_path,
                "record_filters": {"project_type": "{project_type}"},
                "field_mapping": {
                    "methodology": "delivery_methodology",
                    "team": "team_composition",
                    "commercial": "commercial_terms",
                },
            },
            "client_relationship": {
                "request": {"body": {"dataset": args.relationship_dataset}},
                "response_path": args.relationship_response_path,
                "record_filters": {"client_name": "{client_name}"},
                "field_mapping": {
                    "summary": "relationship_summary",
                    "mode": "relationship_status",
                },
            },
            "project_records": {
                "request": {"body": {"dataset": args.project_records_dataset}},
                "response_path": args.project_records_response_path,
                "field_mapping": {
                    "entity": "client_entity",
                    "topic": "strategic_initiative",
                    "budget": "investment_estimation",
                },
            },
        },
    }


def command_init(args: argparse.Namespace) -> int:
    output = Path(args.out).expanduser().resolve()
    config = build_starter_config(args)
    write_json(output, config)
    print(f"Wrote {output}")
    print("Set these env values for production:")
    print("APP_PROFILE=production")
    print("INTERNAL_DATA_SOURCE=api")
    print("PROJECT_DATA_SOURCE=api")
    print(f"FIRM_API_CONFIG_FILE={output}")
    return 0


def command_infer(args: argparse.Namespace) -> int:
    sample_path = Path(args.sample_json).expanduser().resolve()
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    node = extract_path(payload, args.response_path)
    if isinstance(node, list):
        node = node[0] if node else {}
    if not isinstance(node, dict):
        print("Selected response path does not point to a JSON object/list item.", file=sys.stderr)
        return 2
    mapping = infer_mapping(args.resource, node)
    result = {
        "resource": args.resource,
        "response_path": args.response_path,
        "field_mapping": mapping,
        "unmapped_expected_fields": sorted(set(RESOURCE_ALIASES[args.resource].keys()) - set(mapping.keys())),
    }
    if args.out:
        output = Path(args.out).expanduser().resolve()
        write_json(output, result)
        print(f"Wrote {output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    os.environ["APP_PROFILE"] = "production"
    os.environ["INTERNAL_DATA_SOURCE"] = "api"
    os.environ["PROJECT_DATA_SOURCE"] = "api"
    os.environ["FIRM_API_CONFIG_FILE"] = str(config_path)
    if args.base_url:
        os.environ["FIRM_API_URL"] = args.base_url

    from main.runtime_components import FirmAPIClient  # noqa: E402

    client = FirmAPIClient(force_source="api")
    sample_payload = None
    if args.sample_json:
        sample_payload = json.loads(Path(args.sample_json).expanduser().read_text(encoding="utf-8"))
    report = client.validate_config(sample_payload=sample_payload)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal API config helper for APIDog/JSON handover.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter internal_api_config.json.")
    init_parser.add_argument("--url", required=True, help="Full API endpoint URL, e.g. https://host/api/Resource/dataset")
    init_parser.add_argument("--out", default="internal_api_config.json")
    init_parser.add_argument("--method", default="POST", choices=("GET", "POST", "PUT", "PATCH"))
    init_parser.add_argument("--body-encoding", default="form", choices=("json", "form"))
    init_parser.add_argument("--auth-mode", default="basic", choices=("basic", "bearer", "none"))
    init_parser.add_argument("--firm-dataset", default="ReferenceAccount")
    init_parser.add_argument("--standards-dataset", default="ProjectStandards")
    init_parser.add_argument("--relationship-dataset", default="ClientRelationship")
    init_parser.add_argument("--project-records-dataset", default="Projects")
    init_parser.add_argument("--firm-response-path", default="data.dataset_result.0")
    init_parser.add_argument("--standards-response-path", default="data.dataset_result")
    init_parser.add_argument("--relationship-response-path", default="data.dataset_result")
    init_parser.add_argument("--project-records-response-path", default="data.dataset_result")
    init_parser.set_defaults(func=command_init)

    infer_parser = subparsers.add_parser("infer", help="Infer field_mapping from a sample JSON response.")
    infer_parser.add_argument("--sample-json", required=True)
    infer_parser.add_argument("--resource", required=True, choices=tuple(RESOURCE_ALIASES.keys()))
    infer_parser.add_argument("--response-path", default="")
    infer_parser.add_argument("--out", default="")
    infer_parser.set_defaults(func=command_infer)

    validate_parser = subparsers.add_parser("validate", help="Validate an internal API config file.")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--sample-json", default="")
    validate_parser.add_argument("--base-url", default="")
    validate_parser.set_defaults(func=command_validate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
