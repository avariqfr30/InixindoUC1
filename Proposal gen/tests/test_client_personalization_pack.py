"""Regression coverage for client-specific proposal personalization."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ClientPersonalizationPackTest(unittest.TestCase):
    def test_accelbyte_is_not_reclassified_as_government_because_spbe_was_entered(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        industry = ProposalSupportMixin._infer_industry(
            client="AccelByte",
            project="Strategic consulting",
            notes="Ingin mengadopsi SPBE",
            regulations="ISO, Regulasi",
        )

        self.assertEqual(industry, "Game Technology & Digital Platform")
        terms = ProposalSupportMixin._industry_terms(industry)
        self.assertIn("live service reliability", terms)
        self.assertNotIn("layanan publik", terms)

    def test_people_bio_search_noise_is_not_accepted_as_client_anchor_fact(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        cleaned = ProposalSupportMixin._sanitize_anchor_fact(
            "tata kelola kelembagaan serta peningkatan mutu akademik dan layanan "
            "AccelByte Teknologi Indonesia, Awardee New Zealand Scholarship.&lt;/em&gt;&lt;/span&gt;"
        )

        self.assertEqual(cleaned, "")


if __name__ == "__main__":
    unittest.main()
