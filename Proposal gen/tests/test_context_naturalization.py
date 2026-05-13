"""Regression coverage for turning UI snippets into proposal-safe context."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ContextNaturalizationTest(unittest.TestCase):
    def test_generation_context_removes_internal_dataset_boilerplate_and_rank_noise(self) -> None:
        from main.text_hygiene import naturalize_generation_text

        raw = (
            "Pembina Tk. I, IV/b\n"
            "Data internal ReferenceAccount mencatat Accelbyte "
            "(lokasi Daerah Istimewa Yogyakarta; segmentasi SWASTA / Swasta)."
        )

        cleaned = naturalize_generation_text(raw, field="konteks_organisasi", client_name="Accelbyte")

        self.assertIn("Accelbyte", cleaned)
        self.assertIn("Yogyakarta", cleaned)
        self.assertIn("swasta", cleaned.lower())
        self.assertNotIn("ReferenceAccount", cleaned)
        self.assertNotIn("Pembina Tk", cleaned)
        self.assertNotIn("Data internal", cleaned)

    def test_literal_ui_tokens_are_reframed_as_business_context(self) -> None:
        from main.text_hygiene import naturalize_generation_text

        self.assertEqual(
            naturalize_generation_text("Jangka Waktu Pelaksanaan", field="estimasi_waktu"),
            "periode pelaksanaan akan dikonfirmasi pada tahap klarifikasi",
        )
        self.assertIn(
            "masalah, peluang, dan arahan",
            naturalize_generation_text("Problem, Opportunity, Directive", field="klasifikasi_kebutuhan"),
        )

    def test_source_markup_is_stripped_before_it_reaches_proposal_prose(self) -> None:
        from main.text_hygiene import naturalize_generation_text

        raw = (
            "Sumber eksternal 1: fakta=AccelByte adalah platform backend game "
            "yang mendukung layanan live game.&lt;/em&gt;&lt;/span&gt; "
            "| sumber=AccelByte | url=https://accelbyte.io | sitasi_apa=(accelbyte.io, n.d.)"
        )

        cleaned = naturalize_generation_text(raw, field="konteks_organisasi", client_name="AccelByte")

        self.assertIn("AccelByte", cleaned)
        self.assertIn("platform backend game", cleaned)
        self.assertNotIn("Sumber eksternal", cleaned)
        self.assertNotIn("fakta=", cleaned)
        self.assertNotIn("sumber=", cleaned)
        self.assertNotIn("url=", cleaned)
        self.assertNotIn("<", cleaned)
        self.assertNotIn("span", cleaned.lower())

    def test_prompt_only_agent_labels_are_stripped_from_prose_cleanup(self) -> None:
        from main.text_hygiene import clean_markup_artifacts

        raw = (
            "[CHAPTER_RESEARCH_AGENT] Prompt-only research pass. "
            "[EVIDENCE_CARD_SCHEMA] fact | why_it_matters | source_lane | confidence | gap. "
            "[EVIDENCE_STAGE] [RESEARCH_AGENT] [INTERNAL_DATA_AGENT] [COMMERCIAL_STRATEGY_AGENT] "
            "[TECHNICAL_SOLUTION_AGENT] [RISK_COMPLIANCE_AGENT] [EDITOR_MAIN_AGENT] "
            "[EFFICIENCY_POLICY] single model pass per chapter. "
            "[SPECIALIST_AGENT:client_intelligence] Role: client intelligence researcher. "
            "[MAIN_SYNTHESIS_AGENT] Prompt-only synthesis pass. "
            "[CHAPTER_WRITER_AGENT] Prompt-only writing pass. "
            "[CHAPTER_HANDOFF] Ajinomoto Indonesia membutuhkan tata kelola kerja yang konkret."
        )

        cleaned = clean_markup_artifacts(raw)

        self.assertIn("Ajinomoto Indonesia", cleaned)
        self.assertNotIn("CHAPTER_RESEARCH_AGENT", cleaned)
        self.assertNotIn("EVIDENCE_CARD_SCHEMA", cleaned)
        self.assertNotIn("INTERNAL_DATA_AGENT", cleaned)
        self.assertNotIn("RISK_COMPLIANCE_AGENT", cleaned)
        self.assertNotIn("EFFICIENCY_POLICY", cleaned)
        self.assertNotIn("SPECIALIST_AGENT", cleaned)
        self.assertNotIn("MAIN_SYNTHESIS_AGENT", cleaned)
        self.assertNotIn("CHAPTER_WRITER_AGENT", cleaned)
        self.assertNotIn("CHAPTER_HANDOFF", cleaned)
        self.assertNotIn("Prompt-only", cleaned)

    def test_numeric_currency_separators_survive_markup_cleanup(self) -> None:
        from main.text_hygiene import clean_markup_artifacts

        self.assertIn("Rp 85.000.000", clean_markup_artifacts("Rp 85.000.000"))

    def test_private_client_spbe_input_is_reframed_as_digital_governance(self) -> None:
        from main.text_hygiene import naturalize_generation_text

        cleaned = naturalize_generation_text(
            "Ingin mengadopsi SPBE",
            field="permasalahan",
            client_name="AccelByte",
        )

        self.assertIn("tata kelola layanan digital", cleaned.lower())
        self.assertIn("AccelByte", cleaned)
        self.assertNotIn("ingin mengadopsi", cleaned.lower())
        self.assertNotIn("instansi pemerintah", cleaned.lower())

    def test_reused_account_identity_is_not_treated_as_project_objective(self) -> None:
        from main.text_hygiene import naturalize_generation_text

        cleaned = naturalize_generation_text(
            "Ajinomoto Indonesia teridentifikasi sebagai klien dengan berbasis di Kota Jakarta Utara, "
            "DKI Jakarta; klasifikasi swasta / swasta.",
            field="konteks_organisasi",
            client_name="Ajinomoto Indonesia",
        )

        self.assertIn("roadmap kerja", cleaned)
        self.assertIn("Ajinomoto Indonesia", cleaned)
        self.assertIn("bukan sebagai tujuan proyek", cleaned)
        self.assertNotIn("teridentifikasi sebagai klien", cleaned)

    def test_spbe_problem_does_not_trigger_fake_channel_adoption_kpi(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        kpis = ProposalSupportMixin._build_kpi_blueprint(
            project_goal="Problem, Directive",
            notes="Ingin mengadopsi SPBE",
            timeline="Jangka Waktu Pelaksanaan",
            industry="Lintas Industri",
            client="Ajinomoto Indonesia",
        )

        joined = " ".join(kpis)
        self.assertIn("tata kelola", joined.lower())
        self.assertIn("arsitektur layanan digital", joined.lower())
        self.assertNotIn("MAU/Adopsi kanal digital", joined)

    def test_iso_and_regulasi_are_not_described_as_the_same_control(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        rows = ProposalSupportMixin._framework_reference_rows(
            "ISO, Regulasi",
            project_type="Strategic",
            context_hint="tata kelola layanan digital",
        )
        by_name = {row["acuan"].lower(): row for row in rows}

        self.assertIn("standar mutu", by_name["iso"]["ringkas"].lower())
        self.assertIn("kewajiban", by_name["regulasi"]["ringkas"].lower())
        self.assertNotEqual(by_name["iso"]["pembeda"], by_name["regulasi"]["pembeda"])


if __name__ == "__main__":
    unittest.main()
