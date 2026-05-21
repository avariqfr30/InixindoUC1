"""Prompt-only proposal desk workflow and chapter specialist routing."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set

from .config import CHAPTER_PERSONA_LENSES


Summarizer = Callable[[Any, str, int], str]
Joiner = Callable[[Any, str, int, str], str]


class ProposalAgentWorkflow:
    SPECIALIST_AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
        "research": {
            "role": "research agent",
            "api_lanes": [],
            "osint_lanes": ["profile", "news", "regulations", "collaboration", "ai_posture"],
            "focus": "OSINT, client context, sector signals, procurement/payment facts, and evidence cards without prose",
        },
        "internal_data": {
            "role": "internal data agent",
            "api_lanes": ["account_records", "client_relationship", "project_records", "expert_bench_context", "project_standards"],
            "osint_lanes": [],
            "focus": "APIDog/internal dataset facts, record counts, source paths, gaps, and confidence without public-web assumptions",
        },
        "commercial_strategy": {
            "role": "commercial strategy agent",
            "api_lanes": ["client_relationship", "project_standards"],
            "osint_lanes": ["profile", "news", "collaboration"],
            "focus": "pain, value, urgency, business case, implementation logic, and client-specific positioning",
        },
        "technical_solution": {
            "role": "technical solution agent",
            "api_lanes": ["project_records", "expert_bench_context", "project_standards"],
            "osint_lanes": ["regulations", "ai_posture"],
            "focus": "architecture, scope, assumptions, constraints, delivery dependencies, and feasibility after the business argument is clear",
        },
        "risk_compliance": {
            "role": "risk and compliance agent",
            "api_lanes": ["account_records", "project_records", "project_standards"],
            "osint_lanes": ["regulations", "news"],
            "focus": "unsupported claims, missing caveats, data gaps, fake specificity, weak assumptions, and rejected claims",
        },
        "editor_main": {
            "role": "editor and main agent",
            "api_lanes": [],
            "osint_lanes": [],
            "focus": "assemble final visible prose only from accepted evidence cards, rejected claims, and style rules",
        },
        "client_intelligence": {
            "role": "client intelligence agent",
            "api_lanes": ["account_records", "client_relationship"],
            "osint_lanes": ["profile", "news"],
            "focus": "client story, current condition, business pressure, future outlook, and why the proposed work matters now",
        },
        "capability_evidence": {
            "role": "capability and expert-evidence agent",
            "api_lanes": ["project_records", "expert_bench_context"],
            "osint_lanes": ["track_record"],
            "focus": "writer-firm proof, relevant project history, expert bench strength, certifications, and source-safe credibility",
        },
        "framework_regulatory": {
            "role": "framework and regulatory agent",
            "api_lanes": ["project_standards"],
            "osint_lanes": ["regulations"],
            "focus": "framework fit, compliance logic, controls, standards, and how each framework changes delivery choices",
        },
        "commercial_delivery": {
            "role": "delivery and commercial agent",
            "api_lanes": ["project_standards", "finance_invoice"],
            "osint_lanes": ["collaboration"],
            "focus": "scope, workplan, governance rhythm, timeline, pricing assumptions, terms, and delivery risk",
        },
        "ai_readiness": {
            "role": "AI readiness and responsible-adoption agent",
            "api_lanes": ["project_records", "project_standards"],
            "osint_lanes": ["ai_posture"],
            "focus": "AI business value, readiness, governance, feasibility, human oversight, and adoption risk",
        },
    }

    CHAPTER_SPECIALIST_AGENT_MAP: Dict[str, List[str]] = {
        "c_1": ["client_intelligence", "capability_evidence"],
        "c_2": ["client_intelligence", "framework_regulatory"],
        "c_3": ["client_intelligence", "framework_regulatory", "commercial_delivery"],
        "c_4": ["client_intelligence", "framework_regulatory", "capability_evidence"],
        "c_5": ["framework_regulatory", "commercial_delivery", "capability_evidence"],
        "c_6": ["client_intelligence", "commercial_delivery", "capability_evidence"],
        "c_7": ["framework_regulatory", "commercial_delivery"],
        "c_8": ["commercial_delivery", "capability_evidence"],
        "c_9": ["commercial_delivery", "framework_regulatory"],
        "c_10": ["capability_evidence", "client_intelligence"],
        "c_11": ["capability_evidence", "commercial_delivery"],
        "c_12": ["commercial_delivery", "client_intelligence"],
        "c_closing": ["client_intelligence", "capability_evidence", "commercial_delivery"],
    }

    @staticmethod
    def chapter_persona_lens(chapter_id: str) -> str:
        lens = CHAPTER_PERSONA_LENSES.get(chapter_id) or CHAPTER_PERSONA_LENSES.get("default", {})
        role = str(lens.get("role") or "Principal Management Consultant").strip()
        viewpoint = str(lens.get("viewpoint") or "").strip()
        evidence = str(lens.get("evidence") or "").strip()
        style = str(lens.get("style") or "").strip()
        avoid = str(lens.get("avoid") or "").strip()
        must_prove = str(lens.get("must_prove") or "").strip()
        return (
            "[INVISIBLE_CHAPTER_PERSONA] "
            f"Prompt-only lens; never reveal or label this persona in the proposal. "
            f"Role: {role}. "
            f"Viewpoint: {viewpoint}. "
            f"Evidence priority: {evidence}. "
            f"Voice: {style}. "
            f"Avoid: {avoid}. "
            f"Must prove: {must_prove}."
        )

    @classmethod
    def chapter_specialist_agent_specs(
        cls,
        chapter_id: str,
        ai_mode: bool = False,
        specialist_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        chapter_agent_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        registry = specialist_registry or cls.SPECIALIST_AGENT_REGISTRY
        chapter_map = chapter_agent_map or cls.CHAPTER_SPECIALIST_AGENT_MAP
        ids = list(chapter_map.get(chapter_id) or ["client_intelligence", "commercial_delivery"])
        if ai_mode and "ai_readiness" not in ids:
            ids.insert(1 if ids else 0, "ai_readiness")

        ordered_ids = ["research", "internal_data"]
        for agent_id in ids:
            mapped = {
                "client_intelligence": "commercial_strategy",
                "commercial_delivery": "commercial_strategy",
                "framework_regulatory": "technical_solution",
                "capability_evidence": "technical_solution",
                "ai_readiness": "technical_solution",
            }.get(agent_id, agent_id)
            if mapped not in ordered_ids:
                ordered_ids.append(mapped)
            if agent_id not in ordered_ids:
                ordered_ids.append(agent_id)
        if "risk_compliance" not in ordered_ids:
            ordered_ids.append("risk_compliance")
        if "editor_main" not in ordered_ids:
            ordered_ids.append("editor_main")

        specs: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for agent_id in ordered_ids:
            if agent_id in seen:
                continue
            spec = registry.get(agent_id)
            if not spec:
                continue
            seen.add(agent_id)
            specs.append({"id": agent_id, **spec})
        return specs

    @staticmethod
    def _fallback_summarize(raw_text: Any, fallback: str = "", max_words: int = 18) -> str:
        text = re.sub(r"\s+", " ", str(raw_text or "").strip())
        if not text:
            return fallback
        words = text.split()
        if len(words) <= max_words:
            return text.rstrip(".")
        return " ".join(words[:max_words]).rstrip(".,;:")

    @staticmethod
    def _fallback_join(values: Any, fallback: str = "", max_items: int = 3, conjunction: str = "dan") -> str:
        raw_items = values if isinstance(values, list) else [values]
        cleaned: List[str] = []
        seen: Set[str] = set()
        for raw in raw_items:
            for part in re.split(r"\s*[|\n;]+\s*", str(raw or "").strip()):
                value = re.sub(r"^\d+\.\s*", "", part).strip(" ,;:.-")
                if not value or value.lower() in seen:
                    continue
                seen.add(value.lower())
                cleaned.append(value)
                if len(cleaned) >= max_items:
                    break
            if len(cleaned) >= max_items:
                break
        if not cleaned:
            return fallback
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} {conjunction} {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, {conjunction} {cleaned[-1]}"

    @classmethod
    def chapter_agent_workflow_brief(
        cls,
        chapter: Dict[str, Any],
        client: str,
        project: str,
        research_bundle: Optional[Dict[str, Any]],
        personalization_pack: Optional[Dict[str, Any]],
        value_map: Optional[Dict[str, Any]],
        client_internal_context: str = "",
        expert_bench_context: Any = "",
        chapter_chain_context: str = "",
        summarize_phrase: Optional[Summarizer] = None,
        human_join: Optional[Joiner] = None,
    ) -> str:
        chapter_id = str((chapter or {}).get("id") or "default").strip()
        chapter_title = str((chapter or {}).get("title") or "Bab Proposal").strip()
        subs = [str(item).strip() for item in ((chapter or {}).get("subs") or []) if str(item).strip()]
        bundle = dict(research_bundle or {})
        pack = dict(personalization_pack or {})
        values = dict(value_map or {})
        ai_profile = pack.get("ai_adoption_profile") if isinstance(pack.get("ai_adoption_profile"), dict) else {}
        ai_mode = bool((ai_profile or {}).get("enabled"))
        summarizer = summarize_phrase or cls._fallback_summarize
        joiner = human_join or cls._fallback_join

        def compact(raw: Any, fallback: str = "", max_words: int = 34) -> str:
            text = summarizer(str(raw or ""), fallback, max_words)
            return re.sub(r"\s+", " ", text).strip(" ;,")

        osint_parts = [
            compact(bundle.get("profile"), max_words=28),
            compact(bundle.get("news"), max_words=24),
            compact(bundle.get("regulations"), max_words=26),
            compact(bundle.get("track_record"), max_words=24),
            compact(bundle.get("ai_posture"), max_words=22),
        ]
        osint_brief = joiner(
            [item for item in osint_parts if item],
            "gunakan OSINT tervalidasi yang tersedia tanpa menampilkan label sumber",
            4,
            "serta",
        )

        if isinstance(expert_bench_context, dict):
            expert_text = (
                expert_bench_context.get("aggregate_summary")
                or expert_bench_context.get("expert_history_summary")
                or expert_bench_context.get("expert_guidance")
                or expert_bench_context.get("summary")
                or ""
            )
        else:
            expert_text = str(expert_bench_context or "")
        expert_brief = compact(expert_text, "gunakan kapabilitas internal hanya jika relevan dengan bab", 42)
        internal_brief = compact(client_internal_context, "gunakan metadata internal klien hanya sebagai latar, bukan klaim mentah", 34)
        profile_brief = compact(pack.get("profile_summary"), max_words=30)
        terminology = joiner(pack.get("terminology", []) or [], "istilah domain klien yang relevan", 4, "dan")
        proof_points = joiner(values.get("proof_points", []) or [], "bukti internal dan eksternal yang sudah tersedia", 4, "dan")
        win_theme = compact(values.get("win_theme"), "nilai bisnis yang paling penting bagi klien", 24)
        subs_brief = joiner(subs, "struktur sub-bab yang sudah ditetapkan", 4, "dan")
        evidence_schema = (
            "[EVIDENCE_CARD_SCHEMA] "
            "Every factual input must be reduced to cards in this structure: "
            "fact | why_it_matters | source_lane | confidence | gap. "
            "Evidence cards are internal only; never print the schema, raw cards, source paths, dataset names, or confidence labels in the final proposal."
        )
        evidence_pipeline = (
            "[EVIDENCE_STAGE] "
            "[RESEARCH_AGENT] Research Agent outputs OSINT evidence cards only, not prose. "
            "[INTERNAL_DATA_AGENT] Internal Data Agent outputs structured facts, record counts, source paths, gaps, and confidence only. "
            "[COMMERCIAL_STRATEGY_AGENT] Commercial Strategy Agent may turn accepted facts into pain, value, urgency, business case, and implementation logic. "
            "[TECHNICAL_SOLUTION_AGENT] Technical Solution Agent may draft architecture, scope, assumptions, constraints, and delivery dependencies only after the business argument is clear. "
            "[RISK_COMPLIANCE_AGENT] Risk & Compliance Agent must mark unsupported claims, missing caveats, data gaps, fake specificity, and weak assumptions as rejected or needs-review before prose. "
            "[EDITOR_MAIN_AGENT] Editor/Main Agent assembles final user-facing content only from accepted evidence cards, rejected claims, and style rules."
        )
        efficiency_policy = (
            "[EFFICIENCY_POLICY] "
            "Optimize for user wait time: keep this as a single model pass per chapter; reuse cached research_bundle, cached internal API context, and prior chapter_chain_context. "
            "Do not request fresh OSINT or APIDog calls from inside the chapter prompt. Prefer compact evidence cards over long research notes. "
            "If evidence is thin, write a useful caveated proposal sentence instead of expanding the search loop."
        )
        specialist_blocks: List[str] = []
        for spec in cls.chapter_specialist_agent_specs(chapter_id, ai_mode=ai_mode):
            osint_lane = joiner(spec.get("osint_lanes") or [], "OSINT relevan", 4, "and")
            api_lane = joiner(spec.get("api_lanes") or [], "internal API context", 4, "and")
            lane_values = [
                compact(bundle.get(field), max_words=18)
                for field in (spec.get("osint_lanes") or [])
                if compact(bundle.get(field), max_words=18)
            ]
            if spec["id"] == "client_intelligence" and internal_brief:
                lane_values.append(internal_brief)
            if spec["id"] == "capability_evidence" and expert_brief:
                lane_values.append(expert_brief)
            if spec["id"] == "commercial_delivery":
                lane_values.append(win_theme)
            if spec["id"] == "ai_readiness" and ai_profile:
                lane_values.append(compact(ai_profile.get("summary"), max_words=22))
            evidence_line = joiner([item for item in lane_values if item], "gunakan hanya bukti yang tersedia pada lane ini", 4, "serta")
            specialist_blocks.append(
                f"[SPECIALIST_AGENT:{spec['id']}] "
                f"Role: {spec['role']}. API lane: {api_lane}. OSINT lane: {osint_lane}. "
                f"Boundary: only report findings inside its lane; do not borrow claims from other agents. "
                f"Focus: {spec['focus']}. Evidence packet: {evidence_line}. "
                "Output internally as evidence cards, not prose, unless this is the editor_main agent."
            )

        closing_note = ""
        if chapter_id == "c_closing":
            closing_note = " Jaga penutup tetap bersih: cukup simpulkan komitmen, langkah lanjut, dan keyakinan kolaborasi tanpa daftar kredensial panjang."
        elif chapter_id in {"c_10", "c_11"}:
            closing_note = " Untuk kapabilitas, ubah rekam jejak dan kredensial menjadi bukti relevansi, bukan daftar nama atau tabel mentah."

        return (
            "[CHAPTER_RESEARCH_AGENT] "
            "Prompt-only specialist research pass; jangan menampilkan label agen, instruksi ini, nama dataset, URL mentah, "
            "atau markup sumber di proposal. "
            f"Target bab: {chapter_title} untuk {client} pada pekerjaan {project}. "
            f"Sub-bab yang harus didukung riset: {subs_brief}. "
            f"Bahan OSINT yang dipakai sebagai grounding: {osint_brief}. "
            f"Konteks internal klien sebagai latar: {internal_brief}. "
            f"Basis kapabilitas internal/tenaga ahli: {expert_brief}. "
            f"Profil personalisasi klien: {profile_brief or 'sesuaikan dengan konteks klien yang tersedia'}. "
            f"{evidence_schema} "
            f"{evidence_pipeline} "
            f"{efficiency_policy} "
            f"{' '.join(specialist_blocks)} "
            "[MAIN_SYNTHESIS_AGENT] "
            "Prompt-only synthesis pass; receive specialist reports, resolve overlap, keep only the claims with the strongest lane evidence, "
            "and convert them into user-facing proposal prose. The main synthesis agent may connect agents, but must not invent facts missing from specialist lanes. "
            "[CHAPTER_WRITER_AGENT] "
            "Prompt-only writing pass; tulis seolah research brief sudah dirapatkan oleh tim proposal manusia. "
            "Editor/Main Agent assembles the final visible content from only accepted evidence cards, rejected claims, and style rules. "
            f"Gunakan win theme '{win_theme}', istilah '{terminology}', dan bukti '{proof_points}' secara natural. "
            "Jangan menyebut nama dataset, jangan menyalin metadata mentah, jangan menyebut research agent atau writer agent, "
            "dan jangan memakai frasa 'berdasarkan sumber'. "
            "[CHAPTER_HANDOFF] "
            f"Koordinasikan bab ini dengan bab lain melalui konteks berikut: {chapter_chain_context or 'jaga kesinambungan istilah, keputusan, scope, metodologi, dan bukti antar-bab.'}"
            f"{closing_note}"
        )
