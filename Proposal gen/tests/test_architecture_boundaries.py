"""Regression coverage for cohesive service/facade boundaries."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ArchitectureBoundariesTest(unittest.TestCase):
    def test_state_facades_delegate_to_existing_store_contracts(self) -> None:
        from main.state_facades import AuthStateFacade, HistoryStateFacade, SettingsStateFacade

        store = Mock()
        store.get_user.return_value = {"username": "user@example.com"}
        store.get_settings.return_value = {"internal_portfolio": "Portfolio"}
        store.list_history.return_value = [{"id": "h1"}]

        auth = AuthStateFacade(store)
        settings = SettingsStateFacade(store)
        history = HistoryStateFacade(store)

        self.assertEqual(auth.get_user("user@example.com")["username"], "user@example.com")
        self.assertEqual(settings.get_settings()["internal_portfolio"], "Portfolio")
        self.assertEqual(history.list_history(limit=1), [{"id": "h1"}])
        store.list_history.assert_called_once_with(limit=1)

    def test_client_context_service_falls_back_to_knowledge_base_entities(self) -> None:
        from main.runtime_services import ClientContextService

        class FakeSeries:
            def __init__(self) -> None:
                self._values = ["Beta", "nan", "Alpha"]

            def dropna(self):
                return self

            def astype(self, _type):
                return self

            def str(self):
                return self

            def strip(self):
                return self

            def unique(self):
                return self

            def tolist(self):
                return list(self._values)

            @property
            def str(self):
                return self

        class FakeDataFrame:
            empty = False
            columns = ["entity", "topic"]

            def __getitem__(self, key):
                if key != "entity":
                    raise KeyError(key)
                return FakeSeries()

        generator = Mock()
        generator.firm_api.get_client_options.side_effect = RuntimeError("api unavailable")
        knowledge_base = Mock()
        knowledge_base.df = FakeDataFrame()

        service = ClientContextService(generator, knowledge_base, prefetch_research=lambda data: "ok")

        self.assertEqual(service.company_candidates(), ["Alpha", "Beta"])

    def test_client_context_service_returns_http_safe_unknown_client_payload(self) -> None:
        from main.runtime_services import ClientContextService

        generator = Mock()
        generator.firm_api.get_client_context.side_effect = LookupError("client not found")
        prefetch_research = Mock(return_value="queued")

        service = ClientContextService(generator, Mock(), prefetch_research=prefetch_research)

        payload = service.client_context_payload("Unknown Client")

        self.assertFalse(payload["available"])
        self.assertEqual(payload["client_name"], "Unknown Client")
        self.assertEqual(payload["account_summary"], "")
        self.assertEqual(payload["use_case_summary"], "")
        self.assertEqual(payload["use_cases"], [])
        self.assertEqual(payload["expert_guidance"], "")
        self.assertEqual(payload["osint_prefetch_status"], "queued")
        self.assertIn("OSINT", payload["osint_context_note"])
        self.assertNotIn("error", payload)
        prefetch_research.assert_called_once_with(
            {
                "nama_perusahaan": "Unknown Client",
                "potensi_framework": "",
                "konteks_organisasi": "Unknown Client",
            }
        )

    def test_generation_request_service_keeps_required_field_validation_out_of_route(self) -> None:
        from main.proposal_request_service import GENERATION_REQUIRED_FIELDS, ProposalRequestService

        service = ProposalRequestService(
            proposal_generator=Mock(),
            app_state_store=Mock(),
            client_context_service=Mock(),
            generation_queue=Mock(),
        )

        self.assertIn(
            "nama_perusahaan",
            service.missing_required_fields({}, GENERATION_REQUIRED_FIELDS),
        )

    def test_generation_request_service_delegates_framework_resolution(self) -> None:
        from main.proposal_request_service import ProposalRequestService

        store = Mock()
        store.settings.resolve_kak_references_in_payload.return_value = {
            "nama_perusahaan": "Klien",
            "potensi_framework": "ISO, Regulasi",
        }
        framework_service = Mock()
        framework_service.resolve_selection.return_value = "ISO/IEC 27001:2022, UU Perlindungan Data Pribadi"

        service = ProposalRequestService(
            proposal_generator=Mock(),
            app_state_store=store,
            client_context_service=Mock(company_candidates=Mock(return_value=[])),
            generation_queue=Mock(),
            framework_option_service=framework_service,
        )

        payload = service.payload_with_kak_defaults({"potensi_framework": "ISO, Regulasi"})

        self.assertEqual(payload["potensi_framework"], "ISO/IEC 27001:2022, UU Perlindungan Data Pribadi")
        self.assertEqual(payload["_framework_original_selection"], "ISO, Regulasi")
        framework_service.resolve_selection.assert_called_once()

    def test_generation_precheck_summarizes_resolved_inputs_and_evidence_sources(self) -> None:
        from main.proposal_request_service import ProposalRequestService

        store = Mock()
        store.settings.resolve_kak_references_in_payload.return_value = {
            "nama_perusahaan": "Ajinomoto Indonesia",
            "mode_proposal": "canvassing",
            "jenis_proposal": "Konsultan",
            "jenis_proyek": "Strategic",
            "konteks_organisasi": "Meningkatkan tata kelola layanan digital.",
            "permasalahan": "Butuh standar operasional yang lebih konsisten.",
            "klasifikasi_kebutuhan": "Problem, Opportunity",
            "estimasi_waktu": "3 bulan",
            "estimasi_biaya": "Rp 215.784.000",
            "potensi_framework": "ISO, Regulasi",
        }
        store.settings.build_generation_context.return_value = {
            "kak_available": True,
            "kak_source_document": "TOR Ajinomoto.docx",
            "portfolio_context": "Pengalaman proyek tata kelola layanan digital.",
            "credential_context": "Sertifikasi ISO/IEC 27001 dan ITSM tersedia.",
        }
        client_context = Mock()
        client_context.company_candidates.return_value = ["Ajinomoto Indonesia"]
        client_context.internal_context_text.return_value = "Konteks akun dan riwayat tenaga ahli internal tersedia."
        generator = Mock()
        generator.build_preview_outline.return_value = [
            {"title": "BAB I Pendahuluan", "preview": "Menjelaskan konteks klien.", "subsections": []}
        ]
        generator.firm_api.get_expert_bench_context.return_value = {"available": True, "products_count": 4}
        framework_service = Mock()
        framework_service.resolve_selection.return_value = "ISO/IEC 27001:2022, UU Perlindungan Data Pribadi"

        service = ProposalRequestService(
            proposal_generator=generator,
            app_state_store=store,
            client_context_service=client_context,
            generation_queue=Mock(),
            framework_option_service=framework_service,
        )

        result = service.generation_precheck({"potensi_framework": "ISO, Regulasi"})

        self.assertTrue(result["can_generate"])
        self.assertEqual(result["summary"]["client"], "Ajinomoto Indonesia")
        self.assertEqual(result["summary"]["framework_original"], "ISO, Regulasi")
        self.assertEqual(result["summary"]["framework_resolved"], "ISO/IEC 27001:2022, UU Perlindungan Data Pribadi")
        self.assertIn("BAB I Pendahuluan", result["writing_expectation"][0])
        labels = [item["label"] for item in result["evidence_sources"]]
        self.assertIn("Data akun internal", labels)
        self.assertIn("Riwayat tenaga ahli internal", labels)
        self.assertIn("KAK/TOR aktif", labels)
        self.assertIn("Dokumen kapabilitas", labels)


if __name__ == "__main__":
    unittest.main()
