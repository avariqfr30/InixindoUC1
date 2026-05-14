"""Request-level orchestration for proposal generation routes."""
from __future__ import annotations

from typing import Any, Dict, List

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


class ProposalRequestService:
    """Keeps route handlers thin while preserving existing JSON contracts."""

    def __init__(
        self,
        proposal_generator: Any,
        app_state_store: Any,
        client_context_service: Any,
        generation_queue: Any,
    ) -> None:
        self.proposal_generator = proposal_generator
        self.app_state_store = app_state_store
        self.client_context_service = client_context_service
        self.generation_queue = generation_queue

    @staticmethod
    def missing_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        return [field for field in required_fields if not str(data.get(field, "")).strip()]

    def payload_with_kak_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.app_state_store.settings.resolve_kak_references_in_payload(
            normalize_payload(data or {}),
            company_candidates=self.client_context_service.company_candidates(),
        )

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
        except Exception:
            logger.exception("Internal expert bench context enrichment failed")

        payload["_supporting_context"] = supporting_context
        return self.generation_queue.submit(payload)
