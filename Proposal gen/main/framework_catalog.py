"""Framework catalogue and resolver for proposal framework choices."""
from __future__ import annotations

import json
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

    def __init__(self, provider: Any = None, researcher: Any = None) -> None:
        self.provider = provider
        self.researcher = researcher

    @staticmethod
    def fallback_options() -> List[Dict[str, Any]]:
        return [dict(item) for item in FALLBACK_FRAMEWORK_OPTIONS]

    def options(self) -> Dict[str, Any]:
        api_options = self._normalize_options(self._provider_options())
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
                    "issuer": str(item.get("issuer") or "").strip(),
                    "category": str(item.get("category") or "").strip(),
                    "versions": item.get("versions") if isinstance(item.get("versions"), list) else [],
                    "recommended_version": str(item.get("recommended_version") or item.get("resolved") or item.get("value") or "").strip(),
                    "osint_evidence": self._osint_evidence(item),
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

    @staticmethod
    def _truthy_active(value: Any) -> bool:
        text = str(value if value is not None else "1").strip().lower()
        return text not in {"0", "false", "no", "inactive", "tidak aktif"}

    @staticmethod
    def _parse_list_value(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item or "").strip()]
        text = str(value or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item or "").strip()]
        except Exception:
            pass
        return [item.strip() for item in re.split(r"[,;/\n]+", text) if item.strip()]

    def _normalize_version_item(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        value = str(
            raw.get("value")
            or raw.get("child_short_name")
            or raw.get("child_name")
            or raw.get("label")
            or raw.get("child_code")
            or ""
        ).strip()
        label = str(raw.get("label") or raw.get("child_name") or raw.get("child_short_name") or value).strip()
        version = str(raw.get("version") or raw.get("child_version") or "").strip()
        return {
            "value": value or label,
            "label": label or value,
            "description": str(raw.get("description") or raw.get("child_description") or "").strip(),
            "version": version,
            "issuer": str(raw.get("issuer") or raw.get("child_issuer") or "").strip(),
            "category": str(raw.get("category") or raw.get("child_category") or "").strip(),
            "available": self._truthy_active(raw.get("available", raw.get("child_is_active", True))),
        }

    def _normalize_options(self, raw_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw in raw_options:
            value = str(raw.get("value") or raw.get("parent_code") or raw.get("parent_short_name") or raw.get("label") or "").strip()
            label = str(raw.get("label") or raw.get("parent_short_name") or raw.get("parent_name") or value).strip()
            versions = [
                self._normalize_version_item(item)
                for item in (raw.get("versions") or raw.get("children") or [])
                if isinstance(item, dict)
            ]
            versions = [item for item in versions if item.get("value") and item.get("available", True)]
            resolved = str(raw.get("resolved") or raw.get("parent_version") or "").strip()
            recommended = versions[-1]["value"] if versions else (resolved or value or label)
            aliases = self._parse_list_value(raw.get("aliases"))
            aliases.extend([value, label, raw.get("parent_name"), raw.get("parent_short_name"), raw.get("parent_code")])
            normalized.append(
                {
                    "value": value or label,
                    "label": label or value,
                    "description": str(raw.get("description") or raw.get("parent_description") or "").strip(),
                    "resolved": recommended,
                    "recommended_version": recommended,
                    "aliases": _dedupe([str(item or "") for item in aliases]),
                    "issuer": str(raw.get("issuer") or raw.get("parent_issuer") or "").strip(),
                    "category": str(raw.get("category") or raw.get("parent_category") or "").strip(),
                    "versions": versions,
                    "available": self._truthy_active(raw.get("available", raw.get("parent_is_active", True))),
                }
            )
        return normalized

    def _osint_evidence(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        researcher = self.researcher
        if researcher is None or not hasattr(researcher, "search"):
            return []
        query_parts = [
            str(item.get("label") or item.get("value") or "").strip(),
            str(item.get("issuer") or "").strip(),
            "framework official guidance",
        ]
        query = " ".join(part for part in query_parts if part)
        if not query.strip():
            return []
        try:
            results = researcher.search(query, limit=3)
        except Exception:
            return []
        evidence: List[Dict[str, str]] = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "").strip()
            url = str(result.get("link") or result.get("url") or "").strip()
            snippet = str(result.get("snippet") or result.get("content") or "").strip()
            if title or url or snippet:
                evidence.append({"title": title, "url": url, "snippet": snippet})
            if len(evidence) >= 2:
                break
        return evidence

    def resolve(self, raw_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        selected = _split_frameworks(raw_value)
        if not selected:
            return ""
        options = self._normalize_options(self._provider_options()) or self.fallback_options()
        lookup: Dict[str, Dict[str, Any]] = {}
        for option in options:
            aliases = [option.get("value"), option.get("label"), option.get("resolved"), *(option.get("aliases") or [])]
            for version in option.get("versions") or []:
                if isinstance(version, dict):
                    aliases.extend([version.get("value"), version.get("label"), version.get("version")])
            for alias in aliases:
                key = _normalize_token(str(alias or ""))
                if key:
                    lookup[key] = option
        resolved: List[str] = []
        for item in selected:
            version_value = self._resolve_version_value(options, item)
            if version_value:
                resolved.append(version_value)
                continue
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

    @staticmethod
    def _resolve_version_value(options: List[Dict[str, Any]], item: str) -> str:
        item_key = _normalize_token(item)
        for option in options:
            for version in option.get("versions") or []:
                if not isinstance(version, dict):
                    continue
                candidates = [version.get("value"), version.get("label"), version.get("version")]
                if item_key in {_normalize_token(str(candidate or "")) for candidate in candidates}:
                    return str(version.get("value") or version.get("label") or item).strip()
        return ""

    def confirmation_payload(self, raw_value: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        selected = _split_frameworks(raw_value)
        options = self._normalize_options(self._provider_options()) or self.fallback_options()
        items: List[Dict[str, Any]] = []
        for item in selected:
            normalized = _normalize_token(item)
            catalog_option = self._catalog_option_for_selection(options, item)
            if catalog_option and catalog_option.get("versions"):
                items.append(
                    {
                        "selection": item,
                        "recommended": catalog_option.get("recommended_version") or catalog_option["versions"][0]["value"],
                        "options": catalog_option["versions"],
                    }
                )
                continue
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

    @staticmethod
    def _catalog_option_for_selection(options: List[Dict[str, Any]], item: str) -> Optional[Dict[str, Any]]:
        item_key = _normalize_token(item)
        for option in options:
            aliases = [option.get("value"), option.get("label"), option.get("resolved"), *(option.get("aliases") or [])]
            if item_key in {_normalize_token(str(alias or "")) for alias in aliases}:
                return option
        return None
