"""Request-level orchestration for proposal generation routes."""
from __future__ import annotations

from typing import Any, Dict, List

from .internal_evidence_summary import build_internal_evidence_summary, document_context_lines
from .proposal_shared import logger
from .text_hygiene import normalize_payload


GENERATION_REQUIRED_FIELDS = [
    "nama_perusahaan",
    "konteks_organisasi",
    "estimasi_biaya",
    "mode_proposal",
    "jenis_proposal",
    "klasifikasi_kebutuhan",
    "jenis_proyek",
    "estimasi_waktu",
    "permasalahan",
    "potensi_framework",
]

BUDGET_REQUIRED_FIELDS = [
    "nama_perusahaan",
    "mode_proposal",
    "jenis_proposal",
    "jenis_proyek",
    "konteks_organisasi",
    "permasalahan",
    "klasifikasi_kebutuhan",
    "estimasi_waktu",
    "potensi_framework",
]

FIELD_LABELS = {
    "nama_perusahaan": "Nama perusahaan klien",
    "konteks_organisasi": "Konteks klien",
    "estimasi_biaya": "Estimasi anggaran",
    "mode_proposal": "Jenis dokumen proposal",
    "jenis_proposal": "Jenis layanan",
    "klasifikasi_kebutuhan": "Pendekatan / kebutuhan",
    "jenis_proyek": "Jenis proyek",
    "estimasi_waktu": "Durasi proyek",
    "permasalahan": "Permasalahan utama",
    "potensi_framework": "Potensi kerangka kerja",
}


def _compact_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


class ProposalRequestService:
    """Keeps route handlers thin while preserving existing JSON contracts."""

    def __init__(
        self,
        proposal_generator: Any,
        app_state_store: Any,
        client_context_service: Any,
        generation_queue: Any,
        framework_option_service: Any = None,
    ) -> None:
        self.proposal_generator = proposal_generator
        self.app_state_store = app_state_store
        self.client_context_service = client_context_service
        self.generation_queue = generation_queue
        self.framework_option_service = framework_option_service

    @staticmethod
    def missing_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        return [field for field in required_fields if not str(data.get(field, "")).strip()]

    def payload_with_kak_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.app_state_store.settings.resolve_kak_references_in_payload(
            normalize_payload(data or {}),
            company_candidates=self.client_context_service.company_candidates(),
        )
        raw_framework = str(payload.get("potensi_framework") or "").strip()
        if raw_framework and self.framework_option_service is not None:
            try:
                resolved = self.framework_option_service.resolve_selection(raw_framework, context=payload)
                if resolved:
                    payload["potensi_framework"] = resolved
                    payload["_framework_original_selection"] = raw_framework
            except Exception:
                logger.exception("Framework resolver failed; keeping original framework selection")
        return payload

    def warm_request_context(self, data: Dict[str, Any]) -> str:
        data = normalize_payload(data)
        client_name = str(data.get("nama_perusahaan", "")).strip()
        regulations = str(data.get("potensi_framework", "")).strip()
        ai_context = " ".join([
            str(data.get("konteks_organisasi", "")).strip(),
            str(data.get("permasalahan", "")).strip(),
            str(data.get("klasifikasi_kebutuhan", "")).strip(),
            str(data.get("jenis_proyek", "")).strip(),
            str(data.get("jenis_proposal", "")).strip(),
            str(data.get("mode_proposal", "")).strip(),
        ]).strip()
        if not client_name:
            return "skipped"
        return self.proposal_generator.prefetch_research_bundle(
            base_client=client_name,
            regulations=regulations,
            include_collaboration=self.proposal_generator.firm_api.uses_demo_logic(),
            ai_context=ai_context,
        )

    def preview_outline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.payload_with_kak_defaults(data)
        self.warm_request_context(payload)
        return {"outline": self.proposal_generator.build_preview_outline(payload)}

    def prefetch_context(self, data: Dict[str, Any]) -> Dict[str, str]:
        payload = self.payload_with_kak_defaults(data)
        return {"status": self.warm_request_context(payload)}

    def generation_precheck(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.payload_with_kak_defaults(data)
        missing = self.missing_required_fields(payload, GENERATION_REQUIRED_FIELDS)
        supporting_context = self.app_state_store.settings.build_generation_context(
            company_candidates=self.client_context_service.company_candidates()
        )
        client_name = str(payload.get("nama_perusahaan") or "").strip()
        client_internal_context = ""
        if client_name:
            client_internal_context = self.client_context_service.internal_context_text(client_name, payload)

        expert_bench_context: Dict[str, Any] = {}
        try:
            expert_bench_context = self.proposal_generator.firm_api.get_expert_bench_context(limit_products=8)
        except Exception:
            logger.exception("Internal expert bench context precheck failed")

        try:
            outline = self.proposal_generator.build_preview_outline(payload)
        except Exception:
            logger.exception("Proposal outline precheck failed")
            outline = []

        evidence_sources = [
            {
                "label": "Data akun internal",
                "status": "Siap" if client_internal_context else "Belum kuat",
                "detail": (
                    "Konteks klien dari data internal akan dipakai sebagai pembatas nama, profil, dan segmentasi."
                    if client_internal_context
                    else "Belum ada konteks internal tambahan yang terbaca untuk klien ini."
                ),
            },
            {
                "label": "Riwayat tenaga ahli internal",
                "status": "Siap" if expert_bench_context.get("available") or client_internal_context else "Belum kuat",
                "detail": (
                    "Kapabilitas dan pengalaman tenaga ahli akan diringkas secara kontekstual tanpa menyalin data mentah."
                    if expert_bench_context.get("available") or client_internal_context
                    else "Belum ada riwayat tenaga ahli yang cukup kuat untuk ditonjolkan."
                ),
            },
            {
                "label": "Riset publik OSINT",
                "status": "Disiapkan",
                "detail": "Riset publik akan digunakan sebagai konteks bantu, bukan sebagai kutipan mentah di proposal.",
            },
        ]
        if supporting_context.get("kak_available"):
            evidence_sources.append({
                "label": "KAK/TOR aktif",
                "status": "Siap",
                "detail": f"Dokumen aktif: {supporting_context.get('kak_source_document') or 'KAK/TOR terpilih'}.",
            })
        if supporting_context.get("portfolio_context"):
            evidence_sources.append({
                "label": "Dokumen portofolio",
                "status": "Siap",
                "detail": "Pengalaman perusahaan dari file pendukung akan dipakai sebagai bukti ringkas.",
            })
        if supporting_context.get("credential_context"):
            evidence_sources.append({
                "label": "Dokumen kapabilitas",
                "status": "Siap",
                "detail": "Sertifikasi dan kapabilitas dari file pendukung akan dipadatkan menjadi narasi bukti.",
            })

        expectation = []
        for chapter in outline[:5]:
            if not isinstance(chapter, dict):
                continue
            title = _compact_text(chapter.get("title"), 80)
            preview = _compact_text(chapter.get("preview"), 170)
            if title and preview:
                expectation.append(f"{title}: {preview}")
        if not expectation:
            expectation = [
                f"Proposal akan diarahkan untuk {client_name or 'klien terpilih'} dengan konteks, masalah, kerangka kerja, dan estimasi yang sudah diisi."
            ]

        return {
            "can_generate": not missing,
            "missing_fields": [{"field": item, "label": FIELD_LABELS.get(item, item)} for item in missing],
            "summary": {
                "client": client_name,
                "proposal_mode": payload.get("mode_proposal", ""),
                "service_type": payload.get("jenis_proposal", ""),
                "project_type": payload.get("jenis_proyek", ""),
                "need_type": payload.get("klasifikasi_kebutuhan", ""),
                "timeline": payload.get("estimasi_waktu", ""),
                "budget": payload.get("estimasi_biaya", ""),
                "framework_original": payload.get("_framework_original_selection") or payload.get("potensi_framework", ""),
                "framework_resolved": payload.get("potensi_framework", ""),
                "client_context": _compact_text(payload.get("konteks_organisasi"), 220),
                "main_problem": _compact_text(payload.get("permasalahan"), 220),
            },
            "evidence_sources": evidence_sources,
            "writing_expectation": expectation,
        }

    def submit_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.payload_with_kak_defaults(data)
        missing = self.missing_required_fields(payload, GENERATION_REQUIRED_FIELDS)
        if missing:
            return {"error": f"Missing required fields: {', '.join(missing)}", "status_code": 400}

        supporting_context = self.app_state_store.settings.build_generation_context(
            company_candidates=self.client_context_service.company_candidates()
        )
        client_internal_context = self.client_context_service.internal_context_text(
            payload.get("nama_perusahaan", ""),
            payload,
        )
        if client_internal_context:
            supporting_context["client_internal_context"] = client_internal_context
        try:
            expert_bench_context = self.proposal_generator.firm_api.get_expert_bench_context(limit_products=8)
            if expert_bench_context.get("available"):
                supporting_context["expert_bench_context"] = expert_bench_context
                internal_evidence = build_internal_evidence_summary(expert_bench_context)
                evidence_lines = document_context_lines(internal_evidence)
                if evidence_lines:
                    supporting_context["internal_evidence_summary"] = internal_evidence
                    supporting_context["internal_evidence_context"] = evidence_lines
                    merged_context = "\n".join(
                        item
                        for item in [
                            str(supporting_context.get("settings_context") or "").strip(),
                            *evidence_lines,
                        ]
                        if item
                    )
                    supporting_context["settings_context"] = merged_context
        except Exception:
            logger.exception("Internal expert bench context enrichment failed")

        payload["_supporting_context"] = supporting_context
        return self.generation_queue.submit(payload)
