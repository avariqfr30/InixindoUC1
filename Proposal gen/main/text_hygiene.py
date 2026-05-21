"""Input cleanup helpers for user, settings, and KAK/TOR context."""
from __future__ import annotations

import html
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


def clean_markup_artifacts(value: Any) -> str:
    """Remove escaped HTML and source-wire tokens before text becomes proposal prose."""
    text = str(value or "")
    if not text:
        return ""
    text = html.unescape(html.unescape(text)).replace("\\/", "/")
    text = re.sub(r"</?\s*[A-Za-z][A-Za-z0-9:-]*(?:\s+[^>]*)?>", " ", text)
    text = re.sub(
        r"(?i)\[(?:CHAPTER_RESEARCH_AGENT|CHAPTER_WRITER_AGENT|CHAPTER_HANDOFF|INVISIBLE_CHAPTER_PERSONA|MAIN_SYNTHESIS_AGENT|SPECIALIST_AGENT\s*:?\s*[A-Za-z0-9_-]*|EVIDENCE_CARD_SCHEMA|EVIDENCE_STAGE|RESEARCH_AGENT|INTERNAL_DATA_AGENT|COMMERCIAL_STRATEGY_AGENT|TECHNICAL_SOLUTION_AGENT|RISK_COMPLIANCE_AGENT|EDITOR_MAIN_AGENT|EFFICIENCY_POLICY)\]\s*",
        " ",
        text,
    )
    text = re.sub(r"(?i)\bfact\s*\|\s*why_it_matters\s*\|\s*source_lane\s*\|\s*confidence\s*\|\s*gap\.?", " ", text)
    text = re.sub(r"(?i)\bPrompt-only\s+(?:specialist\s+)?(?:research|writing|synthesis)\s+pass\.?", " ", text)
    text = re.sub(r"(?i)\bPrompt-only\s+lens;?\s*", " ", text)
    text = re.sub(r"(?im)^\s*Sumber\s+eksternal\s*\d*\s*:\s*", "", text)
    text = re.sub(r"(?i)\bDirangkum\s+dari\s+sumber(?:\s+publik)?(?:/OSINT)?\s*:?", "", text)
    text = re.sub(r"(?i)\bfakta\s*=\s*", "", text)
    text = re.sub(r"(?i)\s*\|\s*(?:sumber|url|sitasi_apa)\s*=\s*[^|\n]+", "", text)
    text = re.sub(r"(?im)^\s*(?:sumber|url|sitasi_apa)\s*=\s*.+$", "", text)
    text = re.sub(r"(?i)\bSumber\s+eksternal\s*\d*\b\s*:?", "", text)
    text = re.sub(r"\s+\.", ".", text)
    text = normalize_spacing(text)
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    return text


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
    canonical_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(bulan|month|months)\s*\((\d+(?:[.,]\d+)?)\s*(hari|day|days)(?:\s+(kalender|kerja))?\)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if canonical_match:
        month_number = canonical_match.group(1).replace(",", ".")
        day_number = canonical_match.group(3).replace(",", ".")
        qualifier = canonical_match.group(5)
        day_label = " ".join(part for part in [day_number, "Hari", qualifier.title() if qualifier else ""] if part)
        return f"{month_number} Bulan ({day_label})"
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
    raw_unit = match.group(2).lower()
    unit = unit_map.get(raw_unit, match.group(2).title())
    qualifier = match.group(3)
    canonical = " ".join(part for part in [number, unit, qualifier.title() if qualifier else ""] if part)
    try:
        numeric = float(raw_number)
    except ValueError:
        numeric = 0.0
    if raw_unit == "hari" and numeric >= 28:
        months = numeric / 30.0
        if abs(months - round(months)) < 0.05:
            month_label = str(int(round(months)))
        else:
            month_label = f"{months:.1f}".rstrip("0").rstrip(".")
        return f"{month_label} Bulan ({canonical})"
    return canonical


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


def naturalize_generation_text(value: Any, field: str = "", client_name: str = "") -> str:
    """Turn UI/API helper snippets into prose-safe proposal context."""
    text = clean_markup_artifacts(value)
    if not text:
        return ""
    field_key = str(field or "").strip().lower()
    client = normalize_spacing(client_name) or "klien"
    if field_key == "estimasi_waktu" and re.fullmatch(r"jangka\s+waktu\s+pelaksanaan", text, flags=re.IGNORECASE):
        return "periode pelaksanaan akan dikonfirmasi pada tahap klarifikasi"
    if field_key == "klasifikasi_kebutuhan":
        lowered = text.lower()
        selected = [token for token in ("problem", "opportunity", "directive") if token in lowered]
        if selected:
            if len(selected) == 3:
                return "kombinasi masalah, peluang, dan arahan prioritas yang perlu diterjemahkan menjadi rencana kerja"
            labels = {
                "problem": "masalah yang perlu ditangani",
                "opportunity": "peluang nilai bisnis yang bisa dikejar",
                "directive": "arahan prioritas yang perlu dipatuhi",
            }
            return ", ".join(labels[token] for token in selected)
    if field_key == "permasalahan" and re.search(r"\bmengadopsi\s+SPBE\b", text, flags=re.IGNORECASE):
        return (
            f"{client} perlu memperkuat tata kelola layanan digital dengan prinsip SPBE yang relevan sebagai "
            "disiplin arsitektur layanan, integrasi, keamanan, dan pengukuran kinerja; narasinya tetap disesuaikan "
            "dengan konteks organisasi sektor privat."
        )
    if field_key == "konteks_organisasi" and re.search(
        r"\bkonteks\s+akun\s+internal\s+menempatkan\b|\bGunakan\s+informasi\s+ini\s+sebagai\s+latar\b",
        text,
        flags=re.IGNORECASE,
    ):
        return (
            f"penajaman kebutuhan, ruang lingkup, dan roadmap kerja yang relevan bagi {client}; "
            "identitas akun internal dipakai sebagai konteks latar untuk memahami profil klien, bukan sebagai tujuan proyek."
        )

    text = re.sub(r"(?im)^\s*pembina\s+tk\.?\s*[ivxlcdm]+,\s*[ivxlcdm]+/[a-z]\s*$", "", text)
    reference_pattern = re.compile(
        r"Data internal ReferenceAccount mencatat\s+(?P<name>[^(\n.]+)"
        r"(?:\s*\((?P<details>[^)]*)\))?\.?",
        flags=re.IGNORECASE,
    )

    def replace_reference(match: re.Match[str]) -> str:
        name = normalize_spacing(match.group("name") or client_name or "")
        details = normalize_spacing(match.group("details") or "")
        location = ""
        classification = ""
        if details:
            location_match = re.search(r"lokasi\s+([^;]+)", details, flags=re.IGNORECASE)
            segment_match = re.search(r"segmentasi\s+(.+)$", details, flags=re.IGNORECASE)
            location = normalize_spacing(location_match.group(1)) if location_match else ""
            classification = normalize_spacing(segment_match.group(1)) if segment_match else ""
        parts = [f"Identitas akun internal mengonfirmasi {name}" if name else "Identitas akun internal mengonfirmasi klien"]
        qualifiers = []
        if location:
            qualifiers.append(f"di {location}")
        if classification:
            classification = re.sub(r"\s*/\s*", " / ", classification.lower())
            qualifiers.append(f"dengan segmentasi {classification}")
        if qualifiers:
            parts.append(" ".join(qualifiers))
        return " ".join(parts).strip() + "."

    text = reference_pattern.sub(replace_reference, text)
    if field_key == "konteks_organisasi" and re.search(
        r"\b(?:teridentifikasi|tercatat)\s+sebagai\s+klien\b",
        text,
        flags=re.IGNORECASE,
    ):
        return (
            f"penajaman kebutuhan, ruang lingkup, dan roadmap kerja yang relevan bagi {client}; "
            "detail identitas akun internal dipakai hanya sebagai konteks latar, bukan sebagai tujuan proyek."
        )
    text = re.sub(r"\bData internal\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bReferenceAccount\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,;:\n")
    return clean_markup_artifacts(text)


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    for key in FIELD_MAX_CHARS:
        if key in normalized:
            normalized[key] = naturalize_generation_text(
                normalize_field_value(key, normalized.get(key)),
                field=key,
                client_name=str(normalized.get("nama_perusahaan") or ""),
            )
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
