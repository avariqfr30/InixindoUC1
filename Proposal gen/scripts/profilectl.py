#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


MANAGED_KEYS = (
    "APP_PROFILE",
    "INTERNAL_DATA_SOURCE",
    "INTERNAL_DATA_FALLBACK",
    "FIRM_API_URL",
    "FIRM_API_AUTH_MODE",
    "FIRM_API_CONFIG_FILE",
    "PROJECT_DATA_SOURCE",
)


def parse_env_lines(lines: List[str]) -> Dict[str, int]:
    positions: Dict[str, int] = {}
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw:
            continue
        key = raw.split("=", 1)[0].strip()
        if key:
            positions[key] = idx
    return positions


def set_env_value(lines: List[str], positions: Dict[str, int], key: str, value: str) -> None:
    rendered = f"{key}={value}\n"
    if key in positions:
        lines[positions[key]] = rendered
    else:
        if lines and lines[-1] and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(rendered)
        positions[key] = len(lines) - 1


def remove_env_value(lines: List[str], positions: Dict[str, int], key: str) -> None:
    idx = positions.get(key)
    if idx is None:
        return
    lines.pop(idx)
    positions.clear()
    positions.update(parse_env_lines(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Switch app profile between demo and production without editing .env by hand."
    )
    parser.add_argument("profile", choices=("demo", "production"))
    parser.add_argument("--env-file", default=".env", help="Path to the env file to update.")
    parser.add_argument("--fallback", choices=("none", "demo"), default="none")
    parser.add_argument("--api-config", default="", help="Path to the internal API config file for production mode.")
    parser.add_argument("--api-url", default="", help="Internal API base URL or endpoint URL for production mode.")
    parser.add_argument("--auth-mode", choices=("basic", "bearer", "none"), default="", help="Internal API auth mode.")
    parser.add_argument(
        "--project-data-source",
        choices=("local", "api"),
        default="",
        help="Source for the project knowledge-base records. Defaults to the existing value in production.",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file).expanduser().resolve()
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = []
    positions = parse_env_lines(lines)

    if args.profile == "demo":
        set_env_value(lines, positions, "APP_PROFILE", "demo")
        set_env_value(lines, positions, "INTERNAL_DATA_SOURCE", "demo")
        set_env_value(lines, positions, "INTERNAL_DATA_FALLBACK", "none")
        set_env_value(lines, positions, "PROJECT_DATA_SOURCE", "local")
        remove_env_value(lines, positions, "FIRM_API_URL")
        remove_env_value(lines, positions, "FIRM_API_AUTH_MODE")
        remove_env_value(lines, positions, "FIRM_API_CONFIG_FILE")
    else:
        set_env_value(lines, positions, "APP_PROFILE", "production")
        set_env_value(lines, positions, "INTERNAL_DATA_SOURCE", "api")
        set_env_value(lines, positions, "INTERNAL_DATA_FALLBACK", args.fallback)
        if args.project_data_source:
            set_env_value(lines, positions, "PROJECT_DATA_SOURCE", args.project_data_source)
        if args.api_url:
            set_env_value(lines, positions, "FIRM_API_URL", args.api_url)
        if args.auth_mode:
            set_env_value(lines, positions, "FIRM_API_AUTH_MODE", args.auth_mode)
        if args.api_config:
            set_env_value(lines, positions, "FIRM_API_CONFIG_FILE", args.api_config)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("".join(lines), encoding="utf-8")

    print(f"Updated {env_path}")
    print(f"APP_PROFILE={args.profile}")
    print(f"INTERNAL_DATA_SOURCE={'demo' if args.profile == 'demo' else 'api'}")
    print(f"INTERNAL_DATA_FALLBACK={'none' if args.profile == 'demo' else args.fallback}")
    if args.profile == "demo":
        print("PROJECT_DATA_SOURCE=local")
    elif args.project_data_source:
        print(f"PROJECT_DATA_SOURCE={args.project_data_source}")
    else:
        print("PROJECT_DATA_SOURCE left unchanged")
    if args.profile == "production":
        if args.api_config:
            print(f"FIRM_API_CONFIG_FILE={args.api_config}")
        else:
            print("FIRM_API_CONFIG_FILE left unchanged")
        if args.api_url:
            print(f"FIRM_API_URL={args.api_url}")
        if args.auth_mode:
            print(f"FIRM_API_AUTH_MODE={args.auth_mode}")
    print("Next steps:")
    print("1. Restart proposal-gen")
    print("2. Run: python -m main.doctor --format text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
