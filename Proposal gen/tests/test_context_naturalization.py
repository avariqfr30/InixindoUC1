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
            "durasi pelaksanaan disepakati pada tahap klarifikasi",
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


if __name__ == "__main__":
    unittest.main()
