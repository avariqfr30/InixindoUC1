#!/usr/bin/env python3
"""Approve pending Proposal Generator users from the server shell."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.state_store import AppStateStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve a pending Proposal Generator account.")
    parser.add_argument("username", help="Username to approve.")
    parser.add_argument("--approved-by", default="admin-cli", help="Audit label for who approved the account.")
    args = parser.parse_args()

    store = AppStateStore()
    user = store.get_user(args.username)
    if not user:
        print(f"User not found: {args.username}", file=sys.stderr)
        return 1
    if str(user.get("status") or "") == "approved":
        print(f"User is already approved: {user['username']}")
        return 0
    if not store.approve_user(args.username, approved_by=args.approved_by):
        print(f"Unable to approve user: {args.username}", file=sys.stderr)
        return 1
    print(f"Approved user: {user['username']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
