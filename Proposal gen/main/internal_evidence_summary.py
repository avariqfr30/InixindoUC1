"""Clean summaries for internally backed portfolio and capability evidence."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .reader_facing_hygiene import sanitize_reader_facing_sources
from .text_hygiene import clean_markup_artifacts


_BACKEND_TERMS = (
    "ConsultantProjectExpertHistory",
    "EmployeeExpertise",
    "ReferenceAccount",
    "APIDog",
    "dataset",
    "endpoint",
    "status",
    "record_count",
    "internal_api",
)


def _clean_text(value: Any, *, max_words: int = 18) -> str:
    text = clean_markup_artifacts(value)
    text = sanitize_reader_facing_sources(text)
    text = re.sub(r"^[A-Z]{1,5}\d{1,4}\s*[-–]\s*", "", text).strip()
    for term in _BACKEND_TERMS:
        text = re.sub(re.escape(term), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" ;,.-")
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(" ,;") + "..."
    return text


def _unique(items: Iterable[Any], *, max_items: int = 6, max_words: int = 18) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = _clean_text(item, max_words=max_words)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def _join(items: Iterable[str], *, fallback: str = "") -> str:
    values = [item for item in items if item]
    if not values:
        return fallback
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} dan {values[1]}"
    return f"{', '.join(values[:-1])}, dan {values[-1]}"


def _portfolio_section(bench_context: Dict[str, Any]) -> Dict[str, Any]:
    matrix = bench_context.get("product_expert_matrix") or []
    product_names: List[str] = []
    project_examples: List[str] = []
    positions: List[str] = []
    for product in matrix:
        if not isinstance(product, dict):
            continue
        product_names.extend(_unique([product.get("product_name")], max_items=1, max_words=8))
        project_examples.extend(_unique(product.get("project_examples") or [], max_items=2, max_words=10))
        for position in product.get("positions") or []:
            if isinstance(position, dict):
                positions.extend(_unique([position.get("position_name")], max_items=1, max_words=8))

    product_names = _unique(product_names, max_items=5, max_words=8)
    project_examples = _unique(project_examples, max_items=4, max_words=10)
    positions = _unique(positions, max_items=5, max_words=8)
    available = bool(product_names or project_examples or positions)
    bullets: List[str] = []
    if product_names:
        bullets.append(f"Pengalaman serupa yang paling relevan mencakup {_join(product_names)}.")
    if project_examples:
        bullets.append(f"Contoh lingkup kerja yang dapat menjadi pembanding: {_join(project_examples)}.")
    if positions:
        bullets.append(f"Komposisi peran yang tersedia dapat diarahkan melalui {_join(positions)}.")
    return {
        "available": available,
        "title": "Pengalaman serupa",
        "summary": (
            "Portofolio internal dipadatkan menjadi bukti pengalaman yang relevan dengan kebutuhan proposal."
            if available
            else "Bukti pengalaman akan muncul otomatis setelah riwayat internal atau file pendukung tersedia."
        ),
        "bullets": bullets[:3],
    }


def _credential_section(bench_context: Dict[str, Any]) -> Dict[str, Any]:
    rows = bench_context.get("employee_expertise_rows") or []
    certifications: List[str] = []
    projects: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        certifications.extend(_unique(row.get("certifications") or [], max_items=8, max_words=8))
        projects.extend(_unique(row.get("projects") or [], max_items=4, max_words=8))
    certifications = _unique(certifications, max_items=8, max_words=8)
    projects = _unique(projects, max_items=5, max_words=8)
    available = bool(certifications or projects or bench_context.get("employee_expertise_summary"))
    bullets: List[str] = []
    if certifications:
        bullets.append(f"Sertifikasi yang dapat ditonjolkan mencakup {_join(certifications)}.")
    if projects:
        bullets.append(f"Pengalaman tenaga ahli beririsan dengan {_join(projects)}.")
    if certifications and projects:
        bullets.append(
            "Kapabilitas ini dapat dipakai untuk memperkuat rancangan peran, kendali mutu, dan keyakinan eksekusi."
        )
    return {
        "available": available,
        "title": "Kapabilitas dan sertifikasi",
        "summary": (
            "Kapabilitas tenaga ahli diringkas dari bukti sertifikasi dan pengalaman yang dapat dipertanggungjawabkan."
            if available
            else "Bukti kapabilitas akan muncul otomatis setelah sumber internal atau file pendukung tersedia."
        ),
        "bullets": bullets[:3],
    }


def build_internal_evidence_summary(bench_context: Any) -> Dict[str, Any]:
    """Build non-technical evidence summaries for UI and proposal context."""
    context = bench_context if isinstance(bench_context, dict) else {}
    return {
        "portfolio": _portfolio_section(context),
        "credentials": _credential_section(context),
    }


def document_context_lines(summary: Dict[str, Any]) -> List[str]:
    """Flatten a UI-safe summary into prose-safe supporting context lines."""
    lines: List[str] = []
    for key in ("portfolio", "credentials"):
        section = summary.get(key) if isinstance(summary, dict) else {}
        if not isinstance(section, dict) or not section.get("available"):
            continue
        for value in [section.get("summary"), *(section.get("bullets") or [])]:
            text = _clean_text(value, max_words=34)
            if text:
                lines.append(text)
    return lines[:8]
