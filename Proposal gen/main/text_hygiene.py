"""Input cleanup helpers for user, settings, and KAK/TOR context."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


ACRONYMS = {
    "AI", "API", "B2B", "B2C", "BI", "BPK", "BUMN", "COBIT", "CRM", "DAMA",
    "ERP", "ISO", "IT", "ITIL", "KAK", "KPI", "NIST", "OJK", "OKR", "PDP",
    "PIC", "PMO", "POJK", "PT", "RACI", "SLA", "SOP", "SSO", "TI", "TOGAF",
    "TOR", "UAT", "UI", "UKM", "UU", "UX",
}

LOWERCASE_CONNECTORS = {
    "dan", "atau", "yang", "untuk", "dengan", "dari", "di", "ke", "pada",
    "dalam", "atas", "oleh", "serta", "sebagai", "terhadap",
}

FIELD_MAX_CHARS = {
    "nama_perusahaan": 120,
    "mode_proposal": 80,
    "jenis_proposal": 80,
    "jenis_proyek": 80,
    "konteks_organisasi": 900,
    "permasalahan": 1200,
    "klasifikasi_kebutuhan": 160,
    "estimasi_waktu": 120,
    "estimasi_biaya": 120,
    "potensi_framework": 500,
}


def normalize_spacing(text: Any) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r"([,.;:])(?=\S)", r"\1 ", value)
    value = re.sub(r"\.{2,}", ".", value)
    return value.strip()


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.isupper()) / len(letters)


def _format_word(word: str, index: int = 0) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", word)
    punctuation_prefix = word[: len(word) - len(word.lstrip("([{\"'"))]
    punctuation_suffix = word[len(word.rstrip(")]}\"'.,;:")) :]
    core = word[len(punctuation_prefix) : len(word) - len(punctuation_suffix) if punctuation_suffix else len(word)]
    if not core:
        return word
    normalized_core = re.sub(r"[^A-Za-z0-9]+", "", core).upper()
    if normalized_core in ACRONYMS:
        replacement = normalized_core
    elif re.fullmatch(r"PT\.?|CV\.?|TBK\.?", core, re.IGNORECASE):
        replacement = core.upper().replace("TBK", "Tbk").rstrip(".")
    elif index > 0 and core.lower() in LOWERCASE_CONNECTORS:
        replacement = core.lower()
    else:
        replacement = core[:1].upper() + core[1:].lower()
    return f"{punctuation_prefix}{replacement}{punctuation_suffix}"


def formalize_caps_text(text: Any) -> str:
    value = normalize_spacing(text)
    if not value:
        return ""
    formatted_lines: List[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            formatted_lines.append("")
            continue
        words = line.split()
        should_title = _uppercase_ratio(line) >= 0.72 and len(words) <= 28
        if should_title:
            line = " ".join(_format_word(word, idx) for idx, word in enumerate(words))
        formatted_lines.append(line)
    return normalize_spacing("\n".join(formatted_lines))


def normalize_field_value(key: str, value: Any) -> str:
    cleaned = formalize_caps_text(value)
    cleaned = re.sub(r"^\s*[-*•]\s*", "- ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*(\d+)[)]\s+", r"\1. ", cleaned, flags=re.MULTILINE)
    max_chars = FIELD_MAX_CHARS.get(key)
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip(" ,.;:-")
    return cleaned


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    for key in FIELD_MAX_CHARS:
        if key in normalized:
            normalized[key] = normalize_field_value(key, normalized.get(key))
    return normalized


def compact_context_lines(items: Iterable[Any], limit: int = 6) -> List[str]:
    seen = set()
    lines: List[str] = []
    for item in items:
        line = formalize_caps_text(item)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines
