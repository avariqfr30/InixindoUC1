"""Deterministic proposal technique contract helpers."""
from __future__ import annotations

import re
from typing import Any, Dict, List


_ACCOUNT_METADATA_PATTERNS = (
    r"\bkonteks\s+akun\s+internal\b",
    r"\bmenempatkan\b",
    r"\bberlokasi\b|\bdi\s+yogyakarta\b|\bjakarta\b|\bprovinsi\b|\bkota\b",
    r"\bsegmentasi\b|\bsegmen\b|\bklasifikasi\s+swasta\b",
)


def _compact(value: Any, fallback: str = "", max_words: int = 32) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -|,;:"))
    if not text:
        return fallback
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip(" ,;:.")


def _without_account_metadata(value: Any) -> str:
    text = _compact(value, max_words=48)
    if not text:
        return ""
    lowered = text.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _ACCOUNT_METADATA_PATTERNS):
        return ""
    return text


def _unique_items(items: List[str], limit: int = 5) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        cleaned = _compact(item, max_words=18)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _context_fragments(value: Any, keys: tuple[str, ...], limit: int = 3) -> List[str]:
    """Extract compact hidden-planning hints from optional structured context."""

    if not value:
        return []
    records = value if isinstance(value, list) else [value]
    fragments: List[str] = []
    for record in records:
        if isinstance(record, dict):
            parts: List[str] = []
            for key in keys:
                raw = record.get(key)
                if isinstance(raw, list):
                    parts.extend(str(item) for item in raw[:3])
                elif raw:
                    parts.append(str(raw))
            fragment = _compact(" ".join(parts), max_words=20)
        else:
            fragment = _compact(record, max_words=20)
        if fragment:
            fragments.append(fragment)
        if len(fragments) >= limit:
            break
    return fragments


def build_proposal_technique_contract(
    client: str,
    goals: str,
    customer_notes: str = "",
    existing_condition: str = "",
    frameworks: str = "",
    framework_context: Any = None,
    client_use_cases: Any = None,
    osint_facts: Any = None,
    kak_contract: Any = None,
) -> Dict[str, Any]:
    """Build the hidden contract that keeps chapter generation in the intended order."""

    client_name = _compact(client, "klien", max_words=8)
    goal_basis = _compact(goals, f"menajamkan tujuan proyek {client_name}", max_words=36)
    safe_notes = _without_account_metadata(customer_notes)
    safe_existing = _without_account_metadata(existing_condition)
    framework_hints = _context_fragments(
        framework_context,
        ("label", "value", "description", "use_cases"),
        limit=3,
    )
    use_case_hints = _context_fragments(
        client_use_cases,
        ("project_name", "topic", "product_name", "description"),
        limit=3,
    )
    osint_hints = _context_fragments(osint_facts, ("fact", "title", "summary"), limit=2)
    kak_hints = []
    if isinstance(kak_contract, dict):
        kak_hints = _context_fragments(
            [
                item
                for key in ("objectives", "scope", "deliverables", "success_criteria")
                for item in (kak_contract.get(key) or [])
            ],
            (),
            limit=3,
        )

    scope_inputs = _unique_items([
        goal_basis,
        safe_notes,
        safe_existing,
        *use_case_hints,
        *osint_hints,
        *kak_hints,
    ])
    if not scope_inputs:
        scope_inputs = [f"penajaman kebutuhan dan ruang lingkup prioritas {client_name}"]

    framework_basis = _compact(
        " ".join(item for item in [frameworks, *framework_hints, scope_inputs[0] if scope_inputs else ""] if item),
        "kerangka kerja dipilih dari batas ruang lingkup dan kebutuhan proyek",
        max_words=34,
    )
    methodology_basis = _compact(
        " ".join(item for item in [scope_inputs[0], *use_case_hints[:1], *kak_hints[:1], frameworks] if item),
        "metodologi diturunkan dari ruang lingkup dan kerangka kerja yang dipilih",
        max_words=34,
    )

    return {
        "client": client_name,
        "goals": goal_basis,
        "customer_notes": safe_notes,
        "existing_condition": safe_existing,
        "background_basis": goal_basis,
        "objective_basis": goal_basis,
        "scope_basis": "; ".join(scope_inputs),
        "scope_contract_seed": {
            "in_scope": scope_inputs,
            "out_of_scope": [
                "implementasi penuh atau perluasan objek kerja di luar scope hanya dilakukan melalui persetujuan perubahan ruang lingkup"
            ],
        },
        "framework_basis": framework_basis,
        "methodology_basis": methodology_basis,
    }
