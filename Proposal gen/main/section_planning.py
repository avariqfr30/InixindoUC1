"""App-owned writing planner for proposal sections."""
from __future__ import annotations

import re
import json
import hashlib
import copy
from typing import Any


def _compact(value: Any, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;") + "."


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ProposalSectionPlanner:
    """Build hidden guidance for mixed-reader proposal writing."""
    CACHE_VERSION = "proposal-section-plan-v2"
    _cache: dict[str, dict[str, Any]] = {}
    _stats = {"hits": 0, "misses": 0}

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        cls._stats = {"hits": 0, "misses": 0}

    @classmethod
    def cache_stats(cls) -> dict[str, int]:
        return {**cls._stats, "items": len(cls._cache)}

    @classmethod
    def _remember(cls, key: str, plan: dict[str, Any]) -> dict[str, Any]:
        if key in cls._cache:
            cls._cache.pop(key, None)
        cls._cache[key] = copy.deepcopy(plan)
        while len(cls._cache) > 256:
            cls._cache.pop(next(iter(cls._cache)))
        return copy.deepcopy(plan)

    @staticmethod
    def _select_candidate_angle(chapter_id: str, evidence_summary: str) -> tuple[list[str], str]:
        if chapter_id in {"c_4", "c_5", "c_6"}:
            candidates = [
                "mempercepat hasil bisnis melalui tahapan kerja yang mudah divalidasi",
                "mengendalikan risiko pelaksanaan melalui keputusan, kontrol, dan bukti yang jelas",
                "membangun kemampuan internal melalui kolaborasi dan transfer pengetahuan",
            ]
        elif chapter_id in {"c_10", "c_11"}:
            candidates = [
                "menunjukkan kecocokan pengalaman tim dengan situasi klien",
                "menjelaskan cara tim bekerja bersama dan menjaga mutu keputusan",
                "menghubungkan keahlian dengan risiko dan keluaran yang harus dipertanggungjawabkan",
            ]
        elif chapter_id == "c_12":
            candidates = [
                "membaca biaya sebagai investasi pada hasil dan pengendalian risiko",
                "menjelaskan hubungan ruang lingkup, upaya, asumsi, dan konsekuensi keputusan",
                "memberi pilihan komersial yang mudah dibandingkan tanpa menyamarkan batasan",
            ]
        else:
            candidates = [
                "membantu pembaca memahami keputusan utama tanpa harus menguasai detail teknis",
                "menghubungkan kondisi klien dengan dampak bisnis dan langkah yang realistis",
                "membangun keyakinan melalui bukti relevan dan batasan yang disampaikan terbuka",
            ]
        evidence_terms = set(re.findall(r"[a-z0-9]+", str(evidence_summary or "").lower()))
        selected = max(
            candidates,
            key=lambda angle: len(evidence_terms & set(re.findall(r"[a-z0-9]+", angle.lower()))),
        )
        return candidates, selected

    def build_plan(
        self,
        chapter: dict[str, Any],
        client: str,
        project: str,
        evidence_summary: str = "",
        data_version: str = "",
        document_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document_contract = document_contract or {}
        cache_key = _digest({
            "version": self.CACHE_VERSION,
            "chapter": chapter,
            "client": client,
            "project": project,
            "evidence_summary": _compact(evidence_summary, 900),
            "data_version": data_version,
            "document_contract_key": document_contract.get("cache_key"),
        })
        if cache_key in self._cache:
            self._stats["hits"] += 1
            return copy.deepcopy(self._cache[cache_key])
        self._stats["misses"] += 1
        title = str((chapter or {}).get("title") or "Bab Proposal").strip()
        chapter_id = str((chapter or {}).get("id") or "default").strip()
        subs = [str(item).strip() for item in ((chapter or {}).get("subs") or []) if str(item).strip()]
        if chapter_id in {"c_4", "c_5", "c_6"}:
            depth = "mulai dari manfaat bisnis, lalu jelaskan cara kerja operasional dan detail teknis hanya setelah konteksnya jelas"
            datasets = ["ReferenceFramework", "ConsultantProjectExpertHistory", "EmployeeExpertise"]
            narrative_angle = "proposal sebagai rancangan cara kerja yang memperpendek jarak antara kebutuhan, keputusan, dan eksekusi"
            paragraph_roles = ["tesis manfaat", "cara kerja", "bukti kapabilitas", "risiko yang dikendalikan", "keputusan pembaca"]
        elif chapter_id in {"c_10", "c_11"}:
            depth = "ubah bukti kapabilitas menjadi alasan relevansi, bukan daftar kredensial mentah"
            datasets = ["EmployeeExpertise", "ConsultantProjectExpertHistory"]
            narrative_angle = "proposal sebagai bukti kecocokan tim dan pengalaman untuk konteks klien"
            paragraph_roles = ["kebutuhan peran", "bukti pengalaman", "cara kolaborasi", "kendali mutu", "keyakinan eksekusi"]
        elif chapter_id == "c_12":
            depth = "jelaskan angka komersial melalui nilai, asumsi, batasan, dan konsekuensi keputusan"
            datasets = ["ReferenceAccount", "ConsultantProjectExpertHistory"]
            narrative_angle = "proposal sebagai keputusan investasi yang perlu jelas nilai, batasan, dan konsekuensinya"
            paragraph_roles = ["nilai yang dibeli", "asumsi biaya", "batasan scope", "konsekuensi keputusan", "langkah lanjut"]
        else:
            depth = "mulai dari pesan sederhana, lalu tambahkan konsekuensi bisnis dan detail pelaksanaan seperlunya"
            datasets = ["ReferenceAccount", "ConsultantProjectExpertHistory", "ReferenceFramework"]
            narrative_angle = "proposal sebagai cerita keputusan yang mudah dipahami pembaca lintas level"
            paragraph_roles = ["konteks spesifik", "masalah atau peluang", "implikasi", "rekomendasi", "bukti pendukung"]
        evidence = _compact(evidence_summary) or "gunakan bukti internal, OSINT, dan konteks klien yang tersedia tanpa menyebut label dataset."
        candidate_angles, selected_angle = self._select_candidate_angle(chapter_id, evidence)
        plan = {
            "cache_key": cache_key,
            "data_version": data_version,
            "use_case": "proposal",
            "reader": "mixed_business_technical",
            "section_title": title,
            "section_id": chapter_id,
            "section_goal": "membuat pembaca cepat memahami poin utama, alasan bisnis, cara pelaksanaan, dan bukti yang mendukung",
            "user_flow_context": f"proposal_generation client={client}; project={project}",
            "evidence_required": ["konteks klien", "bukti pengalaman", "framework atau kapabilitas relevan"],
            "protected_facts": [item for item in [client, project] if item],
            "tone_rules": ["accessible_senior_consultant", "persuasive_but_not_salesy", "technical_terms_explained_once"],
            "avoid_patterns": [
                "boilerplate consulting claims",
                "repeated openings",
                "raw dataset labels",
                "paragraf yang selalu dimulai dengan perlu/dapat/konteks/fokus",
                "kalimat proposal generik yang tidak menyebut keputusan, risiko, atau bukti",
            ],
            "quality_thresholds": {"max_repeated_openings": 2, "require_term_explanation": True},
            "narrative_angle": narrative_angle,
            "candidate_angles": candidate_angles,
            "selected_angle": selected_angle,
            "paragraph_roles": paragraph_roles,
            "reader_contract": [
                "satu gagasan utama per paragraf",
                "istilah teknis dijelaskan sekali dengan contoh keputusan atau dampak",
                "jangan menulis seolah semua pembaca punya latar teknis yang sama",
            ],
            "reader_ladder": [
                "kalimat pembuka harus bisa dipahami non-teknis",
                "kalimat berikutnya menjelaskan alasan bisnis",
                "baru setelah itu masukkan bukti teknis atau pengalaman",
            ],
            "table_policy": {
                "max_cell_words": 20,
                "prefer_notes_after_table": True,
                "dedupe_repeated_roles": True,
            },
            "retrieval_intent": {
                "goal": "find proposal evidence that supports the chapter argument",
                "preferred_datasets": datasets,
                "exclude": ["FinanceInvoice", "ProjectStandards"],
                "preferred_terms": [title, project, *subs[:4]],
            },
            "evidence_ledger": [
                {
                    "claim_role": "chapter argument",
                    "evidence_source": ", ".join(datasets),
                    "confidence": "use only if present in selected evidence",
                    "allowed_wording": "ubah menjadi manfaat, risiko, dependensi, indikator, atau keputusan klien",
                }
            ],
            "rationale_summary": {
                "main_reasoning": "Bab proposal harus dimulai dari kebutuhan pembaca campuran, lalu mengaitkan bukti ke keputusan bisnis dan kelayakan pelaksanaan.",
                "evidence_used": datasets,
                "caveats": ["jangan menambah klaim di luar evidence cards atau konteks terpilih"],
            },
            "depth_strategy": depth,
            "subsections": subs,
            "evidence_summary": evidence,
            "document_thesis": document_contract.get("document_thesis") or "",
            "chapter_contract": next(
                (
                    item for item in document_contract.get("chapter_contracts", [])
                    if item.get("section_id") == chapter_id
                ),
                {},
            ),
            "data_gap_register": document_contract.get("data_gap_register") or [],
            "editorial_contract": document_contract.get("editorial_contract") or {},
            "appendix_manifest": document_contract.get("appendix_manifest") or {},
        }
        return self._remember(cache_key, plan)

    def build_prompt_block_from_plan(self, plan: dict[str, Any]) -> str:
        subs = plan.get("subsections") or []
        section_list = ", ".join(str(item).strip() for item in subs[:6] if str(item).strip()) or "struktur H2 yang diminta"
        retrieval = plan.get("retrieval_intent") or {}
        ledger = plan.get("evidence_ledger") or []
        rationale = plan.get("rationale_summary") or {}
        return (
            "[SECTION_PLANNER] "
            f"[SECTION_PLAN_JSON] {plan} "
            f"Rencanakan bab '{plan.get('section_title')}' untuk konteks {plan.get('user_flow_context')}. "
            "Target pembaca campuran: eksekutif, manajer operasional, procurement, dan pembaca teknis. "
            f"Tujuan bab: {plan.get('section_goal')}. "
            f"Sub-bab yang harus dijaga: {section_list}. "
            f"Strategi kedalaman: {plan.get('depth_strategy')}. "
            f"Angle naratif: {plan.get('narrative_angle')}. "
            f"Alternatif angle yang sudah dipertimbangkan: {plan.get('candidate_angles')}. "
            f"Angle terpilih berdasarkan bukti: {plan.get('selected_angle')}. "
            f"Rotasi peran paragraf: {plan.get('paragraph_roles')}. "
            f"Kontrak keterbacaan: {plan.get('reader_contract')}. "
            f"Tangga pembaca: {plan.get('reader_ladder')}. Kebijakan tabel: {plan.get('table_policy')}. "
            "Jelaskan istilah teknis saat pertama kali dipakai dengan bahasa singkat dan wajar. "
            "Hindari pola pembuka dan alur argumen yang berulang antar-paragraf; jangan memulai banyak paragraf dengan 'perlu', 'dapat', 'konteks', atau 'fokus'. "
            "Setiap paragraf harus membawa fungsi berbeda: observasi, implikasi, trade-off, contoh, keputusan, atau tindakan. "
            f"Retrieval intent: {retrieval}. Evidence ledger: {ledger}. Rationale ringkas: {rationale}. "
            f"Bukti terpilih untuk dipakai secara natural: {plan.get('evidence_summary')}"
            f" Tesis dokumen: {plan.get('document_thesis')}. Kontrak bab: {plan.get('chapter_contract')}. "
            f"Kesenjangan data: {plan.get('data_gap_register')}. Kontrak editorial: {plan.get('editorial_contract')}."
        )

    def build_prompt_block(
        self,
        chapter: dict[str, Any],
        client: str,
        project: str,
        evidence_summary: str = "",
        document_contract: dict[str, Any] | None = None,
    ) -> str:
        plan = self.build_plan(chapter, client, project, evidence_summary, document_contract=document_contract)
        return self.build_prompt_block_from_plan(plan)
