"""Regression coverage for globally stronger OSINT relevance filtering."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OsintRelevanceFilteringTest(unittest.TestCase):
    def test_filters_person_bio_noise_even_when_entity_name_is_mentioned(self) -> None:
        from main.research import Researcher

        items = [
            {
                "title": "Profil awardee beasiswa",
                "snippet": "AccelByte Teknologi Indonesia, Awardee New Zealand Scholarship dan riwayat akademik pribadi.",
                "link": "https://example.edu/profile",
                "date": "2026",
            },
            {
                "title": "AccelByte Gaming Services",
                "snippet": "AccelByte menyediakan backend game, matchmaking, player identity, dan live service operations.",
                "link": "https://accelbyte.io/products",
                "date": "2026",
            },
        ]

        filtered = Researcher._filter_relevant_osint_results(
            items,
            entity_name="AccelByte",
            context_terms=["backend game", "live service", "matchmaking"],
            max_age_years=3,
        )

        self.assertEqual([item["title"] for item in filtered], ["AccelByte Gaming Services"])

    def test_ranks_official_and_context_relevant_sources_before_generic_mentions(self) -> None:
        from main.research import Researcher

        items = [
            {
                "title": "Forum mention",
                "snippet": "A short mention of ExampleBank without operational transformation detail.",
                "link": "https://random-blog.test/examplebank",
                "date": "2026",
            },
            {
                "title": "ExampleBank annual report",
                "snippet": "ExampleBank annual report discusses digital banking, risk governance, and customer experience.",
                "link": "https://examplebank.co.id/annual-report",
                "date": "2025",
            },
            {
                "title": "Old ExampleBank note",
                "snippet": "ExampleBank digital banking initiative from 2018.",
                "link": "https://examplebank.co.id/old",
                "date": "2018",
            },
        ]

        filtered = Researcher._filter_relevant_osint_results(
            items,
            entity_name="ExampleBank",
            context_terms=["digital banking", "risk governance"],
            max_age_years=3,
        )

        self.assertEqual(filtered[0]["title"], "ExampleBank annual report")
        self.assertNotIn("Old ExampleBank note", [item["title"] for item in filtered])

    def test_external_evidence_is_synthesized_before_document_use(self) -> None:
        from main.research import Researcher

        evidence = Researcher._format_source_evidence_item(
            {
                "title": "Raw Webpage Title That Should Not Become The Proof",
                "snippet": "ExampleBank melaporkan peningkatan adopsi layanan digital sebesar 35% untuk mempercepat respons pelanggan.",
                "link": "https://examplebank.co.id/reports/digital",
                "date": "2026",
            },
            index=1,
        )

        self.assertIn("Bukti eksternal 1:", evidence)
        self.assertIn("peningkatan adopsi layanan digital", evidence)
        self.assertIn("(examplebank.co.id, 2026)", evidence)
        self.assertNotIn("url=", evidence)
        self.assertNotIn("sumber=", evidence)
        self.assertNotIn("https://", evidence)
        self.assertNotIn("Raw Webpage Title", evidence)


if __name__ == "__main__":
    unittest.main()
