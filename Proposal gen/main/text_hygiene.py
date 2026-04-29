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

KAK_REFERENCE_PATTERN = re.compile(
    r"^\s*(?:sesuai|mengacu|berdasarkan|ikut)\s+(?:dokumen\s+)?(?:kak|tor)\s*\.?\s*$",
    re.IGNORECASE,
)


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


def _format_sentence_word(word: str, capitalize: bool = False) -> str:
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
    else:
        replacement = core.lower()
        if capitalize:
            replacement = replacement[:1].upper() + replacement[1:]
    return f"{punctuation_prefix}{replacement}{punctuation_suffix}"


def _sentence_case_caps_line(line: str) -> str:
    tokens = line.split()
    formatted: List[str] = []
    capitalize_next = True
    for token in tokens:
        formatted_token = _format_sentence_word(token, capitalize=capitalize_next)
        formatted.append(formatted_token)
        if re.search(r"[.!?:]$", token):
            capitalize_next = True
        elif formatted_token:
            capitalize_next = False
    return " ".join(formatted)


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
        is_mostly_caps = _uppercase_ratio(line) >= 0.72
        looks_like_sentence = bool(re.search(r"[.!?]$", line)) and len(words) >= 5
        if is_mostly_caps and not looks_like_sentence and len(words) <= 28:
            line = " ".join(_format_word(word, idx) for idx, word in enumerate(words))
        elif is_mostly_caps:
            line = _sentence_case_caps_line(line)
        formatted_lines.append(line)
    return normalize_spacing("\n".join(formatted_lines))


def is_kak_reference(value: Any) -> bool:
    return bool(KAK_REFERENCE_PATTERN.match(str(value or "").strip()))


def normalize_duration_text(value: Any) -> str:
    cleaned = formalize_caps_text(value)
    cleaned = re.sub(r"\|", " ", cleaned)
    cleaned = re.sub(r"\(\s*[^)]*?\s*\)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;-,.")
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(hari|minggu|pekan|bulan|tahun)(?:\s+(kalender|kerja))?",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return normalize_spacing(cleaned)
    raw_number = match.group(1).replace(",", ".")
    number = raw_number[:-2] if raw_number.endswith(".0") else raw_number
    unit_map = {
        "hari": "Hari",
        "minggu": "Minggu",
        "pekan": "Minggu",
        "bulan": "Bulan",
        "tahun": "Tahun",
    }
    unit = unit_map.get(match.group(2).lower(), match.group(2).title())
    qualifier = match.group(3)
    return " ".join(part for part in [number, unit, qualifier.title() if qualifier else ""] if part)


def normalize_field_value(key: str, value: Any) -> str:
    if key == "estimasi_waktu":
        return normalize_duration_text(value)
    cleaned = formalize_caps_text(value)
    if key == "estimasi_biaya":
        cleaned = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", cleaned)
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
