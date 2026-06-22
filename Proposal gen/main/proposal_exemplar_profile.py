"""Distilled UC1 proposal exemplar guidance.

The source PDFs are intentionally not stored in the app. This module keeps only
non-copying structural and editorial guidance extracted from the examples.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_UC1_PROPOSAL_EXEMPLAR_PROFILE: dict[str, Any] = {
    "version": "uc1-conditional-exemplar-profile-v3",
    "source_policy": "distilled_structure_only_no_raw_example_text",
    "role": "Calibrate UC1 proposal writing style and chapter continuity without using examples as factual evidence.",
    "source_mix": {
        "consultant_proposal_reference": "Use for compact proposal flow, execution clarity, risk framing, success indicators, and cost boundary discipline.",
        "english_architecture_reference": "Use for language-neutral reasoning depth only: framework sequence, architecture layers, migration planning, and governance logic.",
        "indonesian_master_plan_reference": "Use for Indonesian public-sector planning depth, current-state to target-state movement, policy alignment, and roadmap discipline.",
    },
    "document_moves": [
        "Start from the client's institutional context, then make the purpose and decision need explicit.",
        "Move from current condition to target condition before proposing methodology, work plan, governance, risk, success indicators, and cost.",
        "Use tables for inventories, risks, milestones, roles, and cost boundaries when they make a decision easier to scan.",
        "End each major chapter by preparing the next decision, not by repeating a generic summary.",
    ],
    "architecture_framework_moves": [
        "Separate business context, information needs, application/system needs, technology/infrastructure needs, and governance needs.",
        "When a framework is mentioned, explain the role of each phase in the proposal's work plan instead of naming the framework as decoration.",
        "Make migration or implementation planning explicit: baseline, target condition, gap, priority, dependency, and governance checkpoint.",
        "Use architecture logic to clarify scope and sequencing; do not convert it into a new chapter structure.",
    ],
    "planning_depth_rules": [
        "For public-sector or institutional clients, connect strategic direction, regulation/policy context, current condition, target condition, and feasible roadmap.",
        "Translate inventories and observations into implications; avoid leaving them as lists without a decision point.",
        "When discussing risks, pair each risk with mitigation, accountable owner or forum, and a measurable signal.",
        "When discussing success, define the indicator, how it will be observed, and which deliverable or milestone proves progress.",
    ],
    "language_transfer_rules": [
        "English references influence reasoning structure only, not final sentence rhythm.",
        "Final output must remain natural Indonesian and avoid translated-English stiffness.",
        "Indonesian references calibrate tone, formal register, and planning vocabulary without copying names or sentences.",
    ],
    "voice_rules": [
        "Use formal Indonesian that stays direct and accountable, not ceremonial.",
        "Prefer concrete nouns, active verbs, and named work products over broad consulting adjectives.",
        "Explain why a recommendation matters before listing activities.",
        "Keep paragraphs compact: one function, one reader decision, and one clear consequence.",
    ],
    "responsibility_pattern": [
        "context",
        "evidence_or_gap",
        "implication",
        "recommended_action",
        "measurement_or_owner",
    ],
    "hardcoded_structure_policy": "use_existing_universal_structure_only",
    "chapter_alignment": {
        "c_1": "Open with institutional context, policy or strategic pressure when relevant, and the reason a consulting decision is needed.",
        "c_2": "Translate context into current condition, explicit needs, gaps, risks, and solution requirements.",
        "c_7": "Make scope, assumptions, exclusions, deliverables, and decision boundaries scan-ready.",
        "c_3": "Classify the need before selecting the project direction.",
        "c_4": "Explain the selected principles, framework, or standard as the bridge to methodology.",
        "c_5": "Turn the approach into executable work steps, evidence collection, validation, and review points.",
        "c_6": "Describe the target output or solution design through baseline, target state, and acceptance logic the sponsor can approve.",
        "c_8": "Connect phases, milestones, deliverables, dependencies, and decision gates.",
        "c_9": "Make decision rights, controls, escalation, and monitoring explicit.",
        "c_11": "Show consultant roles, capability, and accountability without brochure language.",
        "c_12": "Tie cost, payment stages, scope boundary, and acceptance basis together.",
        "c_closing": "Close with concrete follow-up decisions and partnership commitment.",
    },
    "anti_template_rules": [
        "Do not reuse the same opening rhythm across chapters.",
        "Do not let every chapter end with 'dengan demikian' or an equivalent stock transition.",
        "Do not copy example wording, client names, page headers, table labels, or document titles.",
        "Do not turn the examples into factual claims about the generated client's condition.",
    ],
}


_UC1_CONVICTION_PROFILES: dict[str, dict[str, Any]] = {
    "kak_response": {
        "profile_id": "kak_response",
        "audience": "evaluation team, procurement owner, sponsor, and delivery controller",
        "conviction_goal": "Make compliance, traceability, delivery control, and acceptance readiness easy to verify.",
        "persuasion_priorities": [
            "Answer the stated requirement before adding broader strategic value.",
            "Tie every proposed activity to an explicit deliverable and acceptance basis.",
            "Turn assumptions, exclusions, and dependencies into clear decision boundaries.",
        ],
        "assurance_priorities": [
            "Keep scope, schedule, staffing, commercial stages, and acceptance criteria mutually consistent.",
            "Do not imply compliance when the supporting requirement or evidence is absent.",
        ],
    },
    "public_sector": {
        "profile_id": "public_sector",
        "audience": "institutional sponsor, policy owner, steering forum, and accountable delivery team",
        "conviction_goal": "Show that the proposal is policy-aware, governable, measurable, and feasible within institutional constraints.",
        "persuasion_priorities": [
            "Connect institutional direction and applicable policy to a concrete decision need.",
            "Explain current condition, target condition, gap, priority, and feasible roadmap in that order.",
            "Frame benefits as accountable service, governance, capability, and implementation outcomes.",
        ],
        "assurance_priorities": [
            "Name decision gates, accountable forums, risk controls, and observable success indicators.",
            "Keep recommendations proportionate to mandate, evidence, dependency, and implementation readiness.",
        ],
    },
    "architecture_transformation": {
        "profile_id": "architecture_transformation",
        "audience": "business sponsor, architecture owner, transformation leader, and implementation team",
        "conviction_goal": "Make the movement from current state to target state technically credible and executable.",
        "persuasion_priorities": [
            "Trace each recommendation from business need through information, application, technology, and governance implications.",
            "Explain why the selected framework changes the work sequence instead of presenting it as a label.",
            "Connect baseline, target state, gaps, priorities, dependencies, migration, and governance checkpoints.",
        ],
        "assurance_priorities": [
            "Separate confirmed facts, design decisions, assumptions, and items requiring validation.",
            "Keep solution domains, roadmap stages, ownership, and acceptance logic aligned.",
        ],
    },
    "commercial_enterprise": {
        "profile_id": "commercial_enterprise",
        "audience": "executive sponsor, business owner, finance decision maker, and delivery team",
        "conviction_goal": "Make the proposal commercially relevant, outcome-led, credible, and safe to approve.",
        "persuasion_priorities": [
            "Lead with the operational or strategic consequence of the client's current condition.",
            "Tie the recommendation to measurable business outcomes, urgency, and time-to-value.",
            "Use relevant delivery evidence to explain why the proposed team and method can produce the result.",
        ],
        "assurance_priorities": [
            "Make scope boundaries, dependencies, delivery controls, and commercial assumptions explicit.",
            "Avoid unsupported ROI, savings, performance, or implementation claims.",
        ],
    },
}


def select_uc1_proposal_profile(
    proposal_mode: str = "",
    service_type: str = "",
    project_type: str = "",
    client_context: str = "",
) -> dict[str, Any]:
    """Select one persuasion profile using existing request context only."""
    mode = str(proposal_mode or "").strip().lower()
    project_text = str(project_type or "").strip().lower()
    context = " ".join(
        str(value or "").strip().lower()
        for value in (service_type, project_type, client_context)
        if str(value or "").strip()
    )
    public_markers = (
        "kementerian", "pemerintah", "pemda", "dinas ", "kabupaten", "kota ",
        "provinsi", "lembaga negara", "badan nasional", "sektor publik",
    )
    architecture_markers = (
        "enterprise architecture", "arsitektur enterprise", "it master plan", "itmp",
        "target architecture", "target operating model", "migration roadmap", "peta jalan ti",
    )
    if mode == "kak_response" or "kerangka acuan kerja" in context or "term of reference" in context:
        profile_id = "kak_response"
    elif any(marker in context for marker in public_markers):
        profile_id = "public_sector"
    elif any(marker in project_text or marker in context for marker in architecture_markers):
        profile_id = "architecture_transformation"
    else:
        profile_id = "commercial_enterprise"
    return deepcopy(_UC1_CONVICTION_PROFILES[profile_id])


def scope_uc1_exemplar_profile(profile: dict[str, Any], chapter_id: str) -> dict[str, Any]:
    """Return compact calibration guidance for a single chapter."""
    source = dict(profile or {})
    chapter_key = str(chapter_id or "").strip()
    scoped = {
        key: deepcopy(source[key])
        for key in (
            "version", "source_policy", "role", "voice_rules", "responsibility_pattern",
            "hardcoded_structure_policy", "anti_template_rules", "selected_profile",
        )
        if key in source
    }
    chapter_guidance = (source.get("chapter_alignment") or {}).get(chapter_key)
    scoped["chapter_alignment"] = {chapter_key: chapter_guidance} if chapter_guidance else {}
    if chapter_key in {"c_4", "c_5", "c_6", "c_8", "c_9"}:
        scoped["architecture_framework_moves"] = deepcopy(source.get("architecture_framework_moves") or [])
    if chapter_key in {"c_1", "c_2", "c_6", "c_8", "c_9", "c_12"}:
        scoped["planning_depth_rules"] = deepcopy(source.get("planning_depth_rules") or [])
    return scoped


def build_uc1_exemplar_profile() -> dict[str, Any]:
    """Return a defensive copy of the distilled UC1 exemplar profile."""
    return deepcopy(_UC1_PROPOSAL_EXEMPLAR_PROFILE)
