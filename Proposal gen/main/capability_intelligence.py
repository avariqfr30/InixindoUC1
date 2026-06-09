"""Dataset-wide capability evidence summarization for internal expert history."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set


Naturalizer = Callable[[Any], str]


def _default_naturalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _dedupe_text(items: Iterable[str], limit: int = 0) -> List[str]:
    values: List[str] = []
    seen: Set[str] = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item or "").strip())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(cleaned)
        if limit and len(values) >= limit:
            break
    return values


def _normalize_term(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _confidence(record_count: int, project_count: int, expert_count: int, role_count: int) -> str:
    if record_count >= 4 and project_count >= 2 and (expert_count >= 2 or role_count >= 2):
        return "high"
    if record_count >= 2 and project_count >= 2 and expert_count >= 2 and role_count >= 2:
        return "high"
    if record_count >= 2 or project_count >= 2 or expert_count >= 2:
        return "medium"
    return "low"


def _sentence_for_card(card: Dict[str, Any]) -> str:
    product = str(card.get("capability") or "kapabilitas terkait")
    record_count = int(card.get("record_count") or 0)
    project_count = int(card.get("project_count") or 0)
    expert_count = int(card.get("expert_count") or 0)
    roles = card.get("role_coverage") or []
    role_text = ", ".join(roles[:3]) if roles else "peran pelaksana yang tercatat"
    parts = [
        f"Riwayat internal menunjukkan pengalaman pada {product}",
        f"berdasarkan {record_count} catatan",
    ]
    if project_count:
        parts.append(f"{project_count} contoh proyek")
    if expert_count:
        parts.append(f"{expert_count} tenaga ahli tercatat")
    parts.append(f"dengan cakupan peran {role_text}")
    return ", ".join(parts) + "."


def build_capability_intelligence(
    records: Iterable[Dict[str, Any]],
    focus_terms: Optional[Iterable[str]] = None,
    naturalize: Naturalizer = _default_naturalize,
    limit_cards: int = 8,
    limit_examples: int = 3,
) -> Dict[str, Any]:
    """Analyze every usable expert-history record into compact evidence cards.

    The function is intentionally pure: it does not fetch data and it does not
    know about proposal wording. It turns internal records into reusable
    capability intelligence that downstream proposal code can consume safely.
    """

    focus_values = _dedupe_text(focus_terms or [])
    normalized_focus = [_normalize_term(term) for term in focus_values if _normalize_term(term)]
    total_records = 0
    usable_records = 0
    grouped: Dict[str, Dict[str, Any]] = {}
    product_order: List[str] = []

    for raw in records or []:
        total_records += 1
        if not isinstance(raw, dict):
            continue
        product_name = naturalize(raw.get("product_name") or raw.get("topic") or "")
        project_name = naturalize(raw.get("project_name") or raw.get("entity") or "")
        expert_name = naturalize(raw.get("expert_name") or "")
        position_name = naturalize(raw.get("position_name") or "") or "Tenaga Ahli"
        if not product_name and not project_name:
            continue
        if not product_name:
            product_name = "Produk atau lingkup tidak tercatat"
        usable_records += 1

        if product_name not in grouped:
            grouped[product_name] = {
                "capability": product_name,
                "record_count": 0,
                "project_examples": [],
                "experts": set(),
                "role_coverage": [],
                "matched_terms": set(),
                "gap_flags": set(),
            }
            product_order.append(product_name)

        bucket = grouped[product_name]
        bucket["record_count"] += 1
        if project_name and project_name not in bucket["project_examples"]:
            bucket["project_examples"].append(project_name)
        if expert_name:
            bucket["experts"].add(expert_name)
        if position_name and position_name not in bucket["role_coverage"]:
            bucket["role_coverage"].append(position_name)

        haystack = _normalize_term(f"{product_name} {project_name}")
        for original, normalized in zip(focus_values, normalized_focus):
            if normalized and normalized in haystack:
                bucket["matched_terms"].add(original)

    cards: List[Dict[str, Any]] = []
    for product_name in product_order:
        bucket = grouped[product_name]
        project_count = len(bucket["project_examples"])
        expert_count = len(bucket["experts"])
        role_count = len(bucket["role_coverage"])
        if project_count >= 2 and expert_count == 0:
            bucket["gap_flags"].add("project_history_without_named_expert")
        if bucket["record_count"] >= 3 and role_count <= 1:
            bucket["gap_flags"].add("limited_role_diversity")
        matched_terms = sorted(bucket["matched_terms"], key=lambda item: item.lower())
        relevance_score = (
            int(bucket["record_count"]) * 2
            + project_count
            + expert_count
            + role_count
            + len(matched_terms) * 6
        )
        card = {
            "capability": product_name,
            "record_count": int(bucket["record_count"]),
            "project_count": project_count,
            "expert_count": expert_count,
            "role_count": role_count,
            "role_coverage": bucket["role_coverage"][:5],
            "project_examples": bucket["project_examples"][: max(1, int(limit_examples or 3))],
            "matched_terms": matched_terms[:6],
            "confidence": _confidence(int(bucket["record_count"]), project_count, expert_count, role_count),
            "relevance_score": relevance_score,
            "gap_flags": sorted(bucket["gap_flags"]),
        }
        card["safe_sentence"] = _sentence_for_card(card)
        cards.append(card)

    cards.sort(
        key=lambda item: (
            -int(item.get("relevance_score") or 0),
            -int(item.get("record_count") or 0),
            str(item.get("capability") or "").lower(),
        )
    )
    limited_cards = cards[: max(1, int(limit_cards or 8))]

    coverage_gaps = [
        {
            "capability": card["capability"],
            "gap_flags": card["gap_flags"],
            "record_count": card["record_count"],
        }
        for card in limited_cards
        if card.get("gap_flags")
    ]
    strongest = [
        f"{card['capability']} ({card['record_count']} catatan)"
        for card in limited_cards[:4]
    ]
    aggregate_summary = (
        f"Dari {usable_records} catatan riwayat tenaga ahli yang dapat dibaca, "
        f"cluster kapabilitas terkuat mencakup {', '.join(strongest)}."
        if strongest
        else ""
    )
    role_counter: Dict[str, int] = defaultdict(int)
    for card in cards:
        for role in card.get("role_coverage") or []:
            role_counter[str(role)] += int(card.get("record_count") or 0)
    strongest_roles = [
        {"position_name": role, "coverage_count": count}
        for role, count in sorted(role_counter.items(), key=lambda item: (-item[1], item[0].lower()))[:8]
    ]

    return {
        "available": bool(limited_cards),
        "total_record_count": total_records,
        "usable_record_count": usable_records,
        "aggregate_summary": aggregate_summary,
        "evidence_cards": limited_cards,
        "coverage_gaps": coverage_gaps,
        "strongest_roles": strongest_roles,
    }
