#!/usr/bin/env python3
"""Create and inspect internal API config files for production handover."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from main.internal_api_setup import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_RESPONSE_PATHS,
    RESOURCE_ALIASES,
    build_internal_api_config,
    extract_path,
    infer_mapping,
    write_json_config,
)


def command_init(args: argparse.Namespace) -> int:
    output = Path(args.out).expanduser().resolve()
    config = build_internal_api_config(
        {
            "url": args.url,
            "method": args.method,
            "body_encoding": args.body_encoding,
            "auth_mode": args.auth_mode,
            "datasets": {
                "firm_profile": args.firm_dataset,
                "project_standards": args.standards_dataset,
                "client_relationship": args.relationship_dataset,
                "project_records": args.project_records_dataset,
            },
            "response_paths": {
                "firm_profile": args.firm_response_path,
                "project_standards": args.standards_response_path,
                "client_relationship": args.relationship_response_path,
                "project_records": args.project_records_response_path,
            },
        }
    )
    write_json_config(output, config)
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
        write_json_config(output, result)
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

    from main.data_sources import FirmAPIClient  # noqa: E402

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
    init_parser.add_argument("--firm-dataset", default=DEFAULT_DATASETS["firm_profile"])
    init_parser.add_argument("--standards-dataset", default=DEFAULT_DATASETS["project_standards"])
    init_parser.add_argument("--relationship-dataset", default=DEFAULT_DATASETS["client_relationship"])
    init_parser.add_argument("--project-records-dataset", default=DEFAULT_DATASETS["project_records"])
    init_parser.add_argument("--firm-response-path", default=DEFAULT_RESPONSE_PATHS["firm_profile"])
    init_parser.add_argument("--standards-response-path", default=DEFAULT_RESPONSE_PATHS["project_standards"])
    init_parser.add_argument("--relationship-response-path", default=DEFAULT_RESPONSE_PATHS["client_relationship"])
    init_parser.add_argument("--project-records-response-path", default=DEFAULT_RESPONSE_PATHS["project_records"])
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
