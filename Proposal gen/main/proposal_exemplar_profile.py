"""Distilled UC1 proposal exemplar guidance.

The source PDFs are intentionally not stored in the app. This module keeps only
non-copying structural and editorial guidance extracted from the examples.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_UC1_PROPOSAL_EXEMPLAR_PROFILE: dict[str, Any] = {
    "version": "uc1-itmp-exemplar-profile-v2",
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


def build_uc1_exemplar_profile() -> dict[str, Any]:
    """Return a defensive copy of the distilled UC1 exemplar profile."""
    return deepcopy(_UC1_PROPOSAL_EXEMPLAR_PROFILE)
