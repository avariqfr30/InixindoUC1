import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.proposal_quality_pipeline import (
    ContextIntelligenceDesk,
    EvidenceDeckBuilder,
    ProposalQualityGate,
    ScopeContractExtractor,
)


class ProposalQualityPipelineTests(unittest.TestCase):
    def test_evidence_deck_keeps_raw_sources_out_of_prompt_text(self):
        deck = EvidenceDeckBuilder.build(
            client="PT Contoh",
            project="Transformasi Layanan Digital",
            project_goal="Problem",
            timeline="12 minggu",
            budget="Rp500 juta",
            research_bundle={"profile": "PT Contoh sedang memperkuat layanan digital. source=https://example.test"},
            internal_context="ReferenceAccount mencatat dataset_code=ConsultantProjectExpertHistory.",
            value_map={"value_statement": "tata kelola layanan lebih terkendali"},
        )

        chapter_text = deck.for_chapter("c_1")

        self.assertIn("PT Contoh", chapter_text)
        self.assertIn("tata kelola layanan lebih terkendali", chapter_text)
        for forbidden in ["ReferenceAccount", "dataset_code", "ConsultantProjectExpertHistory", "source="]:
            self.assertNotIn(forbidden, chapter_text)

    def test_context_intelligence_desk_synthesizes_raw_helpers_for_chapter_guidance(self):
        packet = ContextIntelligenceDesk.build(
            client="PT Contoh",
            project="Transformasi Layanan Digital",
            project_goal="Problem, Opportunity, Directive",
            notes="ReferenceAccount mencatat Identitas akun internal: source=/api/Resource/dataset.",
            research_bundle={
                "profile": "PT Contoh memperluas layanan digital untuk memperkuat operasi.",
                "regulations": "Regulasi SPBE dan ISO perlu dipakai sebagai acuan tata kelola.",
            },
            internal_context="APIDog ReferenceAccount dataset_code=ReferenceAccount; lokasi DIY; segmentasi SWASTA.",
            value_map={"value_statement": "tata kelola layanan lebih terkendali"},
            scope_contract={"in_scope": ["asesmen tata kelola dan roadmap prioritas"], "out_of_scope": ["implementasi aplikasi penuh"]},
        )

        guidance = packet.for_chapter("c_4")

        self.assertIn("PT Contoh", guidance)
        self.assertIn("tata kelola layanan", guidance.lower())
        self.assertIn("asesmen tata kelola", guidance.lower())
        for forbidden in ["ReferenceAccount", "APIDog", "dataset_code", "source=", "/api/Resource/dataset", "Identitas akun internal"]:
            self.assertNotIn(forbidden, guidance)

    def test_context_intelligence_desk_keeps_weak_osint_and_internal_confidence_silent(self):
        packet = ContextIntelligenceDesk.build(
            client="PT Contoh",
            project="Audit Tata Kelola",
            project_goal="Memperkuat tata kelola layanan",
            research_bundle={},
            internal_context="lokasi DIY; segmentasi SWASTA; status prospek aktif",
            value_map={"value_statement": "keputusan kerja lebih terukur"},
        )

        guidance = packet.for_chapter("c_1")
        team_guidance = packet.for_chapter("c_10")

        self.assertIn("tidak memaksakan klaim eksternal", guidance.lower())
        self.assertIn("data kemampuan internal", team_guidance.lower())
        for forbidden in ["keyakinan", "confidence", "dataset", "Internal API", "APIDog"]:
            self.assertNotIn(forbidden, guidance + team_guidance)

    def test_scope_contract_extracts_boundaries_for_later_chapters(self):
        scope_text = (
            "## 3.1 Lingkup Pekerjaan Utama\n"
            "Pekerjaan mencakup asesmen tata kelola, workshop prioritas, dan roadmap implementasi.\n\n"
            "## 3.2 Batasan Pekerjaan, Asumsi, dan Hal di Luar Cakupan\n"
            "Implementasi penuh aplikasi, pengadaan lisensi, dan integrasi sistem berada di luar cakupan."
        )

        contract = ScopeContractExtractor.extract(scope_text)

        self.assertIn("asesmen tata kelola", " ".join(contract["in_scope"]).lower())
        self.assertIn("implementasi penuh aplikasi", " ".join(contract["out_of_scope"]).lower())
        prompt_text = ScopeContractExtractor.to_prompt_text(contract)
        self.assertIn("Batas ruang lingkup", prompt_text)
        self.assertIn("di luar cakupan", prompt_text.lower())

    def test_final_quality_gate_flags_raw_helpers_and_scope_drift(self):
        scope_contract = {
            "in_scope": ["asesmen tata kelola", "roadmap prioritas"],
            "out_of_scope": ["implementasi penuh aplikasi"],
            "assumptions": [],
            "dependencies": [],
            "deliverables": ["dokumen rekomendasi"],
        }
        result = ProposalQualityGate.evaluate(
            chapter_outputs={
                "c_4": "ReferenceAccount mencatat source=/api/Resource/dataset. Identitas akun internal mengonfirmasi klien.",
                "c_6": "Solusi mencakup implementasi penuh aplikasi walau scope membatasinya.",
            },
            selected_chapters=[
                {"id": "c_4", "title": "BAB V – PENDEKATAN"},
                {"id": "c_6", "title": "BAB VII – SOLUTION DESIGN"},
            ],
            scope_contract=scope_contract,
            executive_summary="# Ringkasan Eksekutif\nBAB V menjelaskan semuanya.",
        )

        self.assertFalse(result["passes"])
        self.assertIn("raw_helper_text", result["categories"])
        self.assertIn("scope_drift", result["categories"])
        self.assertIn("executive_summary_literal_callback", result["categories"])

    def test_golden_fixture_encodes_reader_quality_contract(self):
        fixture_path = ROOT / "tests" / "fixtures" / "golden_proposal_scope_quality.json"
        payload = json.loads(fixture_path.read_text())

        self.assertEqual(payload["client"], "PT Contoh")
        self.assertIn("required_scope_terms", payload["expected"])
        self.assertIn("forbidden_reader_terms", payload["expected"])
        self.assertGreaterEqual(payload["expected"]["executive_summary_max_words"], 400)


if __name__ == "__main__":
    unittest.main()
