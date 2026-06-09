"""Deterministic proposal technique contract helpers."""
from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional


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


def _list_items(value: Any) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            raw_items = parsed if isinstance(parsed, list) else [text]
        except Exception:
            raw_items = re.split(r"[,;/\n]+", text)
    return _unique_items([str(item) for item in raw_items], limit=8)


def _contains_any(text: str, tokens: List[str]) -> bool:
    lowered = str(text or "").lower()
    return any(token.lower() in lowered for token in tokens)


def _build_solution_domains(
    frameworks: str,
    scope_inputs: List[str],
    goal_basis: str,
    safe_notes: str,
    safe_existing: str,
) -> List[Dict[str, str]]:
    context = " ".join([frameworks, goal_basis, safe_notes, safe_existing, " ".join(scope_inputs)])
    candidates = [
        {
            "name": "Strategic",
            "framework": "Business-IT Alignment",
            "complexity": "Medium",
            "duration": "6-8 minggu",
            "deliverable": "IT strategic plan, prioritas investasi, dan roadmap eksekutif",
            "need_terms": ["strategi", "strategic", "it master", "roadmap", "keselarasan", "alignment"],
        },
        {
            "name": "Governance",
            "framework": "COBIT 2019",
            "complexity": "Medium",
            "duration": "6-8 minggu",
            "deliverable": "governance assessment, operating model, KPI, risk, dan compliance framework",
            "need_terms": ["cobit", "governance", "tata kelola", "kpi", "risk", "compliance"],
        },
        {
            "name": "Architecture",
            "framework": "TOGAF",
            "complexity": "High",
            "duration": "10-14 minggu",
            "deliverable": "enterprise architecture blueprint, gap analysis, dan transition roadmap",
            "need_terms": ["togaf", "architecture", "arsitektur", "blueprint", "aplikasi", "data", "teknologi"],
        },
        {
            "name": "Security",
            "framework": "NIST CSF",
            "complexity": "Medium",
            "duration": "6-8 minggu",
            "deliverable": "cybersecurity assessment, risk register, security roadmap, dan resilience control",
            "need_terms": ["nist", "security", "keamanan", "cyber", "siber", "resilience", "bcp", "drp"],
        },
        {
            "name": "Digital Transformation",
            "framework": "Industry 4.0",
            "complexity": "High",
            "duration": "8-12 minggu",
            "deliverable": "digital transformation roadmap, smart operation use cases, analytics/AI readiness",
            "need_terms": ["industry 4.0", "smart", "manufacturing", "transformasi digital", "iot", "ai", "analytics", "automation"],
        },
    ]
    selected = [
        {key: value for key, value in candidate.items() if key != "need_terms"}
        for candidate in candidates
        if _contains_any(context, candidate["need_terms"])
    ]
    if not selected:
        selected = [
            {key: value for key, value in candidate.items() if key != "need_terms"}
            for candidate in candidates[:3]
        ]
    if _contains_any(context, ["it master plan", "itmp", "transformasi digital"]):
        selected_names = {item["name"] for item in selected}
        for candidate in candidates:
            if candidate["name"] not in selected_names:
                selected.append({key: value for key, value in candidate.items() if key != "need_terms"})
    return selected[:5]


def _build_psa_itmp_core(
    client_name: str,
    goal_basis: str,
    safe_notes: str,
    safe_existing: str,
    frameworks: str,
    scope_inputs: List[str],
    framework_basis: str,
    methodology_basis: str,
) -> Dict[str, Any]:
    solution_domains = _build_solution_domains(frameworks, scope_inputs, goal_basis, safe_notes, safe_existing)
    need_domains = _unique_items(
        [
            "keselarasan bisnis dan TI" if _contains_any(" ".join([goal_basis, safe_notes]), ["keselarasan", "alignment", "it master"]) else "",
            "tata kelola dan kontrol keputusan" if _contains_any(" ".join([frameworks, safe_notes]), ["cobit", "governance", "tata kelola"]) else "",
            "arsitektur enterprise dan integrasi data" if _contains_any(" ".join([frameworks, safe_existing]), ["togaf", "arsitektur", "data", "aplikasi"]) else "",
            "keamanan dan ketahanan layanan digital" if _contains_any(" ".join([frameworks, safe_notes]), ["nist", "security", "keamanan", "siber"]) else "",
            "kesiapan transformasi digital dan inovasi operasi" if _contains_any(" ".join([goal_basis, safe_notes]), ["transformasi", "smart", "industry", "ai", "manufacturing"]) else "",
            *scope_inputs[:3],
        ],
        limit=6,
    )
    if not need_domains:
        need_domains = ["penajaman kebutuhan, scope, dan keputusan eksekusi"]

    risk_model = [
        {
            "risk": "Keterbatasan Ketersediaan Stakeholder",
            "impact": "assessment, validasi kebutuhan, dan keputusan fase terlambat",
            "mitigation": "tetapkan jadwal workshop sejak awal, PIC tiap unit, dan escalation path sponsor",
        },
        {
            "risk": "Keterbatasan Data dan Dokumentasi",
            "impact": "kualitas analisis kondisi eksisting dan gap analysis menurun",
            "mitigation": "gunakan triangulasi melalui wawancara, observasi, dokumen pendukung, dan workshop validasi",
        },
        {
            "risk": "Perubahan Prioritas Organisasi",
            "impact": "ruang lingkup, timeline, atau prioritas roadmap bergeser",
            "mitigation": "pakai change control, impact assessment, dan persetujuan steering committee",
        },
        {
            "risk": "Rendahnya Keterlibatan Stakeholder",
            "impact": "rekomendasi sulit diterima dan implementabilitas menurun",
            "mitigation": "libatkan stakeholder sejak baseline, review berkala, dan sign-off tiap deliverable utama",
        },
    ]
    success_criteria = [
        {
            "category": "Business Success Criteria",
            "indicator": "roadmap, blueprint, governance framework, dan prioritas transformasi tersusun sesuai ruang lingkup",
        },
        {
            "category": "Strategic Success Criteria",
            "indicator": "strategi bisnis, strategi TI, investasi teknologi, dan prioritas program memiliki dasar keputusan yang jelas",
        },
        {
            "category": "Executive Success Criteria",
            "indicator": f"dokumen akhir disetujui bersama dan dapat menjadi acuan resmi {client_name} untuk pengembangan berikutnya",
        },
    ]
    commercial_drivers = [
        {
            "basis": "solution_domain",
            "driver": domain["name"],
            "complexity": domain["complexity"],
            "duration": domain["duration"],
            "estimate_basis": f"ditentukan oleh kedalaman {domain['deliverable']}",
        }
        for domain in solution_domains
    ]

    return {
        "visibility_rule": "gunakan sebagai logika tersembunyi; jangan tampilkan nama kerangka penalaran ini pada dokumen final",
        "problem_statement": _compact(" ".join([goal_basis, safe_notes, safe_existing]), f"kebutuhan prioritas {client_name}", max_words=42),
        "need_domains": need_domains,
        "scope_anchor": "; ".join(scope_inputs[:5]),
        "framework_rationale": framework_basis,
        "methodology_logic": methodology_basis,
        "solution_domains": solution_domains,
        "risk_model": risk_model,
        "success_criteria": success_criteria,
        "commercial_drivers": commercial_drivers,
        "chapter_map": {
            "executive_summary": ["executive_summary"],
            "problem_needs": ["c_1", "c_2", "c_7", "c_3"],
            "approach": ["c_4"],
            "methodology": ["c_5"],
            "solution_design": ["c_6"],
            "timeline": ["c_8"],
            "governance": ["c_9"],
            "team": ["c_11"],
            "risk": ["c_2", "c_9"],
            "success_criteria": ["c_5", "c_6", "c_9"],
            "commercial": ["c_12"],
        },
    }


def build_proposal_technique_contract(
    client: str,
    goals: str,
    customer_notes: str = "",
    existing_condition: str = "",
    frameworks: str = "",
    framework_context: Any = None,
    client_use_cases: Any = None,
    osint_facts: Any = None,
    kak_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the hidden contract that keeps chapter generation in the intended order."""

    client_name = _compact(client, "klien", max_words=8)
    goal_basis = _compact(goals, f"menajamkan tujuan proyek {client_name}", max_words=36)
    safe_notes = _without_account_metadata(customer_notes)
    safe_existing = _without_account_metadata(existing_condition)
    kak_contract = kak_contract if isinstance(kak_contract, dict) else {}
    kak_scope_contract = kak_contract.get("scope_contract") if isinstance(kak_contract.get("scope_contract"), dict) else {}
    kak_in_scope = _unique_items(_list_items(kak_scope_contract.get("in_scope")), limit=8)
    kak_out_of_scope = _unique_items(_list_items(kak_scope_contract.get("out_of_scope")), limit=6)
    kak_deliverables = _unique_items(
        [
            *_list_items(kak_contract.get("deliverables")),
            *_list_items(kak_scope_contract.get("deliverables")),
        ],
        limit=8,
    )
    kak_problems = _unique_items(_list_items(kak_contract.get("problems")), limit=4)
    kak_objectives = _unique_items(_list_items(kak_contract.get("objectives")), limit=4)

    framework_scope_inputs: List[str] = []
    framework_context_bits: List[str] = []
    framework_catalog: List[Dict[str, Any]] = []
    for item in framework_context or []:
        if not isinstance(item, dict):
            continue
        label = _compact(item.get("label") or item.get("value"), max_words=8)
        description = _compact(item.get("description"), max_words=12)
        issuer = _compact(item.get("issuer"), max_words=8)
        use_cases = _list_items(item.get("use_cases"))
        clean_versions: List[Dict[str, Any]] = []
        framework_scope_inputs.extend(use_cases)
        for version in item.get("versions") or []:
            if isinstance(version, dict):
                version_use_cases = _list_items(version.get("use_cases"))
                framework_scope_inputs.extend(version_use_cases)
                version_description = _compact(version.get("description"), max_words=12)
                if version_description:
                    framework_context_bits.append(version_description)
                clean_versions.append({
                    "value": _compact(version.get("value") or version.get("code") or version.get("label"), max_words=10),
                    "label": _compact(version.get("label") or version.get("value"), max_words=12),
                    "description": version_description,
                    "issuer": _compact(version.get("issuer"), max_words=10),
                    "use_cases": _unique_items(version_use_cases, limit=6),
                })
        if label or description or issuer:
            framework_context_bits.append(
                " ".join(part for part in [label, description, f"issuer {issuer}" if issuer else ""] if part)
            )
            framework_catalog.append({
                "value": _compact(item.get("value") or label, max_words=10),
                "label": label,
                "description": description,
                "issuer": issuer,
                "use_cases": _unique_items([str(use_case) for use_case in use_cases], limit=6),
                "versions": clean_versions,
            })
    for item in client_use_cases or []:
        if isinstance(item, dict):
            framework_scope_inputs.extend(
                str(item.get(key) or "")
                for key in ("product_name", "description", "proposal_use_guidance")
                if str(item.get(key) or "").strip()
            )
        elif str(item or "").strip():
            framework_scope_inputs.append(str(item))
    osint_scope_inputs = [
        _compact(item, max_words=14)
        for item in (osint_facts or [])
        if str(item or "").strip() and not re.search(r"https?://|source=|sumber=", str(item), flags=re.IGNORECASE)
    ]

    scope_inputs = _unique_items([
        *kak_in_scope,
        *kak_deliverables,
        *kak_objectives,
        *kak_problems,
        goal_basis,
        safe_notes,
        safe_existing,
        *framework_scope_inputs,
        *osint_scope_inputs,
    ], limit=7)
    if not scope_inputs:
        scope_inputs = [f"penajaman kebutuhan dan ruang lingkup prioritas {client_name}"]

    framework_basis = _compact(
        " ".join(item for item in [frameworks, *framework_context_bits[:2], scope_inputs[0] if scope_inputs else ""] if item),
        "kerangka kerja dipilih dari batas ruang lingkup dan kebutuhan proyek",
        max_words=34,
    )
    methodology_basis = _compact(
        " ".join(item for item in [scope_inputs[0], frameworks] if item),
        "metodologi diturunkan dari ruang lingkup dan kerangka kerja yang dipilih",
        max_words=34,
    )
    persuasion_scaffold = {
        "problem_agitation_solution": (
            "rumuskan tantangan klien, jelaskan konsekuensi bila tidak ditangani, "
            "lalu posisikan layanan sebagai cara kerja yang paling terkendali"
        ),
        "situation_problem_implication_need_payoff": (
            "baca situasi, masalah, implikasi, dan manfaat keputusan sebagai alur berpikir konsultatif"
        ),
        "visibility_rule": "gunakan sebagai logika tersembunyi; jangan tampilkan label kerangka ini pada dokumen final",
    }
    psa_itmp_core = _build_psa_itmp_core(
        client_name=client_name,
        goal_basis=goal_basis,
        safe_notes=safe_notes,
        safe_existing=safe_existing,
        frameworks=frameworks,
        scope_inputs=scope_inputs,
        framework_basis=framework_basis,
        methodology_basis=methodology_basis,
    )

    return {
        "client": client_name,
        "goals": goal_basis,
        "customer_notes": safe_notes,
        "existing_condition": safe_existing,
        "background_basis": goal_basis,
        "objective_basis": _compact("; ".join(kak_objectives) or goal_basis, goal_basis, max_words=42),
        "scope_basis": "; ".join(scope_inputs),
        "scope_decisions": scope_inputs[:5],
        "framework_context_summary": _compact("; ".join(framework_context_bits), "", max_words=40),
        "framework_catalog": framework_catalog,
        "scope_contract_seed": {
            "in_scope": scope_inputs,
            "out_of_scope": kak_out_of_scope or [
                "implementasi penuh atau perluasan objek kerja di luar scope hanya dilakukan melalui persetujuan perubahan ruang lingkup"
            ],
            "assumptions": _list_items(kak_scope_contract.get("assumptions"))[:6],
            "dependencies": _list_items(kak_scope_contract.get("dependencies"))[:6],
            "deliverables": kak_deliverables,
        },
        "kak_contract": kak_contract,
        "framework_basis": framework_basis,
        "methodology_basis": methodology_basis,
        "persuasion_scaffold": persuasion_scaffold,
        "psa_itmp_core": psa_itmp_core,
    }
