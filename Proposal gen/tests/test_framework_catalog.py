"""Regression coverage for framework catalogue fallback and resolver behavior."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FrameworkCatalogTest(unittest.TestCase):
    def test_fallback_options_are_available_without_internal_dataset(self) -> None:
        from main.framework_catalog import FrameworkCatalogService

        payload = FrameworkCatalogService().options()

        self.assertEqual(payload["source"], "fallback")
        values = [item["value"] for item in payload["options"]]
        self.assertIn("ISO", values)
        self.assertIn("COBIT", values)
        self.assertIn("Praktik Baik", values)

    def test_resolves_generic_iso_and_regulation_from_context(self) -> None:
        from main.framework_catalog import FrameworkCatalogService

        service = FrameworkCatalogService()

        resolved = service.resolve(
            "ISO, Regulasi",
            context={
                "konteks_organisasi": "Ingin mengadopsi SPBE dengan kontrol keamanan dan data pribadi.",
                "permasalahan": "Butuh tata kelola keamanan informasi dan kepatuhan PDP.",
            },
        )

        self.assertIn("ISO/IEC 27001:2022", resolved)
        self.assertIn("Perpres SPBE No. 95 Tahun 2018", resolved)
        self.assertIn("UU Perlindungan Data Pribadi", resolved)

    def test_preserves_unknown_manual_frameworks(self) -> None:
        from main.framework_catalog import FrameworkCatalogService

        resolved = FrameworkCatalogService().resolve("ISO, Kerangka Internal ABC")

        self.assertIn("ISO", resolved)
        self.assertIn("Kerangka Internal ABC", resolved)

    def test_provider_catalog_overrides_fallback_when_available(self) -> None:
        from main.framework_catalog import FrameworkCatalogService

        provider = Mock()
        provider.get_framework_catalog.return_value = [
            {
                "value": "SPBE",
                "label": "SPBE",
                "aliases": ["spbe"],
                "resolved": "Perpres SPBE No. 95 Tahun 2018",
                "description": "Tata kelola pemerintahan digital.",
            }
        ]

        service = FrameworkCatalogService(provider)

        self.assertEqual(service.options()["source"], "internal_api")
        self.assertEqual(service.resolve("SPBE"), "Perpres SPBE No. 95 Tahun 2018")


if __name__ == "__main__":
    unittest.main()
