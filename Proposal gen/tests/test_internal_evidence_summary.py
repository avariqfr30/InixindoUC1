"""Regression coverage for clean internal evidence summaries."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SAMPLE_BENCH_CONTEXT = {
    "available": True,
    "record_count": 42,
    "source": "internal_api",
    "expert_history_summary": (
        "Riwayat proyek internal menunjukkan kapabilitas pada Arsitektur SPBE dan Manajemen SPBE."
    ),
    "product_expert_matrix": [
        {
            "product_name": "Arsitektur SPBE",
            "project_examples": [
                "PI06 - Kajian Arsitektur SPBE Domain Infrastruktur",
                "Peta Rencana SPBE Pemerintah Daerah",
            ],
            "positions": [
                {"position_name": "Project Manager", "expert_count": 3, "experts": ["Andi Yuniantoro"]},
                {"position_name": "Arsitek SPBE", "expert_count": 2, "experts": ["Citra Arfanudin"]},
            ],
        },
        {
            "product_name": "Manajemen SPBE",
            "project_examples": ["Pendampingan tata kelola layanan digital"],
            "positions": [{"position_name": "Tenaga Ahli", "expert_count": 4, "experts": []}],
        },
    ],
    "employee_expertise_summary": (
        "Data sertifikasi internal mencakup 5 tenaga ahli. Area sertifikasi yang dapat ditonjolkan: "
        "TOGAF 9 Foundations, COBIT 5, ISO/IEC 27001 Foundation, ITIL Foundation."
    ),
    "employee_expertise_rows": [
        {
            "employee_name": "Andi Yuniantoro",
            "certifications": ["TOGAF 9 Foundations", "COBIT 5", "ISO/IEC 27001 Foundation"],
            "projects": ["Arsitektur SPBE", "Masterplan TIK"],
        },
        {
            "employee_name": "Citra Arfanudin",
            "certifications": ["Lead Auditor ISO 27001", "CCNA Routing and Switching"],
            "projects": ["Manajemen SPBE"],
        },
    ],
    "employee_expertise_record_count": 5,
    "name_policy": {"source": "ConsultantProjectExpertHistory+EmployeeExpertise"},
}


class InternalEvidenceSummaryTest(unittest.TestCase):
    def test_ui_summary_uses_internal_evidence_without_backend_terms(self) -> None:
        from main.internal_evidence_summary import build_internal_evidence_summary

        payload = build_internal_evidence_summary(SAMPLE_BENCH_CONTEXT)
        rendered = str(payload)

        self.assertTrue(payload["portfolio"]["available"])
        self.assertTrue(payload["credentials"]["available"])
        self.assertIn("Arsitektur SPBE", rendered)
        self.assertIn("ISO/IEC 27001 Foundation", rendered)
        self.assertNotIn("ConsultantProjectExpertHistory", rendered)
        self.assertNotIn("EmployeeExpertise", rendered)
        self.assertNotIn("record_count", rendered)
        self.assertNotIn("endpoint", rendered.lower())
        self.assertNotIn("status", rendered.lower())
        self.assertNotIn("Andi Yuniantoro", rendered)
        self.assertNotIn("Citra Arfanudin", rendered)

    def test_document_context_is_plain_proposal_context_not_status_summary(self) -> None:
        from main.internal_evidence_summary import build_internal_evidence_summary, document_context_lines

        payload = build_internal_evidence_summary(SAMPLE_BENCH_CONTEXT)
        context = "\n".join(document_context_lines(payload))

        self.assertIn("Arsitektur SPBE", context)
        self.assertIn("TOGAF 9 Foundations", context)
        self.assertNotIn("API", context)
        self.assertNotIn("dataset", context.lower())
        self.assertNotIn("record", context.lower())
        self.assertNotIn("status", context.lower())
        self.assertNotIn("Andi Yuniantoro", context)

    def test_submit_generation_attaches_clean_internal_evidence_context(self) -> None:
        from main.proposal_request_service import ProposalRequestService

        store = Mock()
        store.settings.resolve_kak_references_in_payload.return_value = {
            "nama_perusahaan": "Ajinomoto Indonesia",
            "mode_proposal": "canvassing",
            "jenis_proposal": "Konsultan",
            "jenis_proyek": "Strategic",
            "konteks_organisasi": "Transformasi tata kelola digital.",
            "permasalahan": "Membutuhkan struktur kerja yang lebih terkendali.",
            "klasifikasi_kebutuhan": "Problem, Opportunity",
            "estimasi_waktu": "3 bulan",
            "estimasi_biaya": "Rp 215.784.000",
            "potensi_framework": "ISO, Regulasi",
        }
        store.settings.build_generation_context.return_value = {}
        client_context = Mock()
        client_context.company_candidates.return_value = ["Ajinomoto Indonesia"]
        client_context.internal_context_text.return_value = ""
        generator = Mock()
        generator.firm_api.get_expert_bench_context.return_value = SAMPLE_BENCH_CONTEXT
        queue = Mock()
        queue.submit.return_value = {"job_id": "job-1"}

        service = ProposalRequestService(
            proposal_generator=generator,
            app_state_store=store,
            client_context_service=client_context,
            generation_queue=queue,
            framework_option_service=None,
        )

        service.submit_generation({})
        submitted_payload = queue.submit.call_args.args[0]
        context = submitted_payload["_supporting_context"]
        rendered = "\n".join(context["internal_evidence_context"])

        self.assertIn("expert_bench_context", context)
        self.assertIn("Arsitektur SPBE", rendered)
        self.assertIn("TOGAF 9 Foundations", rendered)
        self.assertNotIn("ConsultantProjectExpertHistory", rendered)
        self.assertNotIn("EmployeeExpertise", rendered)
        self.assertNotIn("record", rendered.lower())

    def test_team_rows_use_employee_expertise_without_count_language(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        rows = ProposalSupportMixin._expert_rows_from_bench_context(
            SAMPLE_BENCH_CONTEXT,
            client="Ajinomoto Indonesia",
            project_type="Strategic",
            service_type="Konsultan",
            regulations="ISO, Regulasi",
            team_points=[],
        )
        rendered = str(rows)

        self.assertIn("TOGAF 9 Foundations", rendered)
        self.assertIn("ISO/IEC 27001 Foundation", rendered)
        self.assertNotIn("Data sertifikasi internal mencakup 5 tenaga ahli", rendered)
        self.assertNotIn("EmployeeExpertise", rendered)


if __name__ == "__main__":
    unittest.main()
