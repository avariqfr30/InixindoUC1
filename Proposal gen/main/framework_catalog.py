"""Framework catalogue and resolver for proposal framework choices."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


FALLBACK_FRAMEWORK_OPTIONS: List[Dict[str, Any]] = [
    {
        "value": "ITIL",
        "label": "ITIL",
        "aliases": ["itil", "itil v3", "itil v4"],
        "resolved": "ITIL 4",
        "description": "Manajemen layanan, SLA, incident/change, dan transisi operasi.",
    },
    {
        "value": "TOGAF",
        "label": "TOGAF",
        "aliases": ["togaf", "togaf 9", "togaf 10"],
        "resolved": "TOGAF",
        "description": "Arsitektur target, gap analysis, dan roadmap perubahan.",
    },
    {
        "value": "COBIT",
        "label": "COBIT",
        "aliases": ["cobit", "cobit 5", "cobit 2019"],
        "resolved": "COBIT 2019",
        "description": "Tata kelola, kontrol keputusan, RACI, KPI/KGI, dan quality gate.",
    },
    {
        "value": "ISO",
        "label": "ISO",
        "aliases": ["iso", "iso 27001", "iso 20000", "iso 9001"],
        "resolved": "ISO/IEC 27001:2022",
        "description": "Standar sistem manajemen yang dipilih sesuai konteks proyek.",
    },
    {
        "value": "Teori",
        "label": "Teori",
        "aliases": ["teori", "theory"],
        "resolved": "Teori perubahan organisasi dan adopsi teknologi",
        "description": "Landasan konseptual untuk perubahan, adopsi, dan pengambilan keputusan.",
    },
    {
        "value": "Regulasi",
        "label": "Regulasi",
        "aliases": ["regulasi", "kepatuhan", "compliance"],
        "resolved": "Regulasi sektoral yang berlaku",
        "description": "Pemetaan kewajiban, batas keputusan, risiko legal, dan bukti kepatuhan.",
    },
    {
        "value": "Praktik Baik",
        "label": "Praktik Baik",
        "aliases": ["best practice", "best practise", "praktik baik"],
        "resolved": "Praktik baik implementasi dan pengendalian mutu",
        "description": "Acuan praktis untuk checklist, acceptance, dan kontrol mutu delivery.",
    },
]


FALLBACK_FRAMEWORK_ALTERNATIVES: Dict[str, List[Dict[str, str]]] = {
    "iso": [
        {"value": "ISO/IEC 27001:2022", "label": "ISO/IEC 27001:2022", "description": "Keamanan informasi, risiko, kontrol, dan bukti audit."},
        {"value": "ISO/IEC 20000-1", "label": "ISO/IEC 20000-1", "description": "Manajemen layanan TI, SLA, operasi, dan transisi layanan."},
        {"value": "ISO 9001", "label": "ISO 9001", "description": "Manajemen mutu, proses, SOP, dan kontrol kualitas."},
        {"value": "ISO 22301", "label": "ISO 22301", "description": "Keberlanjutan bisnis dan kesiapan pemulihan layanan."},
    ],
    "regulasi": [
        {"value": "Perpres SPBE No. 95 Tahun 2018", "label": "Perpres SPBE No. 95 Tahun 2018", "description": "Acuan tata kelola pemerintahan digital/SPBE."},
        {"value": "UU Perlindungan Data Pribadi", "label": "UU Perlindungan Data Pribadi", "description": "Acuan perlindungan data pribadi dan privasi."},
        {"value": "ketentuan OJK/POJK yang relevan", "label": "Ketentuan OJK/POJK yang relevan", "description": "Acuan kepatuhan sektor jasa keuangan."},
        {"value": "Regulasi sektoral yang berlaku", "label": "Regulasi sektoral yang berlaku", "description": "Acuan kepatuhan sesuai sektor dan ruang lingkup klien."},
    ],
}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _split_frameworks(raw_value: str) -> List[str]:
    values = [
        re.sub(r"\s+", " ", item).strip(" -.;:")
        for item in re.split(r"[,;/\n]+|\s+ dan \s+", str(raw_value or ""), flags=re.IGNORECASE)
    ]
    return [item for item in values if item]


def _dedupe(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item or "").strip())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _context_text(context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(context, dict):
        return ""
    return " ".join(
        str(context.get(key) or "")
        for key in (
            "nama_perusahaan",
            "mode_proposal",
            "jenis_proposal",
            "jenis_proyek",
            "konteks_organisasi",
            "permasalahan",
            "klasifikasi_kebutuhan",
        )
    ).lower()


def _resolve_iso(context: Optional[Dict[str, Any]]) -> str:
    text = _context_text(context)
    if re.search(r"\b(keamanan|security|cyber|siber|soc|noc|pdp|privasi|risiko|risk|audit)\b", text):
        return "ISO/IEC 27001:2022"
    if re.search(r"\b(layanan|service|sla|incident|insiden|change|operasi|operasional|helpdesk|it service)\b", text):
        return "ISO/IEC 20000-1"
    if re.search(r"\b(mutu|quality|kualitas|sop|proses)\b", text):
        return "ISO 9001"
    return "ISO/IEC 27001:2022 atau ISO/IEC 20000-1 sesuai ruang lingkup"


def _resolve_regulation(context: Optional[Dict[str, Any]]) -> str:
    text = _context_text(context)
    values: List[str] = []
    if "spbe" in text:
        values.append("Perpres SPBE No. 95 Tahun 2018")
    if re.search(r"\b(pdp|data pribadi|privasi)\b", text):
        values.append("UU Perlindungan Data Pribadi")
    if re.search(r"\b(ojk|pojk|bank|bpr|bprs|keuangan|financial)\b", text):
        values.append("ketentuan OJK/POJK yang relevan")
    return ", ".join(_dedupe(values)) or "Regulasi sektoral yang berlaku"


class FrameworkCatalogService:
    """Resolves UI framework choices through API-backed catalogue data or fallback defaults."""

    def __init__(self, provider: Any = None) -> None:
        self.provider = provider

    @staticmethod
    def fallback_options() -> List[Dict[str, Any]]:
        return [dict(item) for item in FALLBACK_FRAMEWORK_OPTIONS]

    def options(self) -> Dict[str, Any]:
        api_options = self._provider_options()
        options = api_options or self.fallback_options()
        return {
            "source": "internal_api" if api_options else "fallback",
            "source_label": (
                "Standar internal dari API"
                if api_options
                else "Standar bawaan sampai katalog Internal API tersedia"
            ),
            "options": [
                {
                    "value": str(item.get("value") or item.get("label") or "").strip(),
                    "label": str(item.get("label") or item.get("value") or "").strip(),
                    "description": str(item.get("description") or "").strip(),
                    "available": bool(item.get("available", True)),
                }
                for item in options
                if str(item.get("value") or item.get("label") or "").strip()
            ],
        }

    def _provider_options(self) -> List[Dict[str, Any]]:
        provider = self.provider
        if provider is None or not hasattr(provider, "get_framework_catalog"):
            return []
        try:
            raw = provider.get_framework_catalog()
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        options = [item for item in raw if isinstance(item, dict)]
        return options

    def resolve(self, raw_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        selected = _split_frameworks(raw_value)
        if not selected:
            return ""
        options = self._provider_options() or self.fallback_options()
        lookup: Dict[str, Dict[str, Any]] = {}
        for option in options:
            aliases = [option.get("value"), option.get("label"), option.get("resolved"), *(option.get("aliases") or [])]
            for alias in aliases:
                key = _normalize_token(str(alias or ""))
                if key:
                    lookup[key] = option
        resolved: List[str] = []
        for item in selected:
            option = lookup.get(_normalize_token(item))
            if not option:
                resolved.append(item)
                continue
            value = str(option.get("resolved") or option.get("value") or item).strip()
            normalized_value = _normalize_token(value)
            normalized_item = _normalize_token(item)
            if normalized_item == "iso" or normalized_value == "iso":
                value = _resolve_iso(context)
            elif normalized_item in {"regulasi", "kepatuhan"} or normalized_value in {"regulasi", "kepatuhan"}:
                value = _resolve_regulation(context)
            resolved.append(value)
        return ", ".join(_dedupe(resolved))

    def confirmation_payload(self, raw_value: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        selected = _split_frameworks(raw_value)
        items: List[Dict[str, Any]] = []
        for item in selected:
            normalized = _normalize_token(item)
            if normalized not in FALLBACK_FRAMEWORK_ALTERNATIVES:
                continue
            alternatives = FALLBACK_FRAMEWORK_ALTERNATIVES[normalized]
            if normalized == "iso":
                recommended = _resolve_iso(context)
            elif normalized == "regulasi":
                recommended = _resolve_regulation(context).split(",")[0].strip()
            else:
                recommended = alternatives[0]["value"]
            if not any(option["value"] == recommended for option in alternatives):
                recommended = alternatives[0]["value"]
            items.append(
                {
                    "selection": item,
                    "recommended": recommended,
                    "options": alternatives,
                }
            )
        return {
            "requires_confirmation": bool(items),
            "items": items,
            "resolved_framework": self.resolve(raw_value, context=context),
        }
