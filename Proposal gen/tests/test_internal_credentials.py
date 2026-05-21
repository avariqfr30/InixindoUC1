"""Regression coverage for turning internal credential files into name-safe proof."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RAW_CREDENTIALS = """
NO
NAMA TENAGA AHLI
POSISI DIUSULKAN
TINGKAT PENDIDIKAN
SERTIFIKASI
PERAN DALAM PENUGASAN
LAMA PENGALAMAN (TAHUN)
1
Andi Yuniantoro
Project Manager
· S1, Elektronika dan Instrumentasi, Universitas Gadjah Mada Yogyakarta
· S2, Master of Information Technology, Swiss Germany University
· CAPM (Certified Associate Project Management)
· CompTIA Project+
· TOGAF 9 Foundations
· COBIT 5
· ITIL Foundation V3
· ISO/IEC 27001 (Information Security Foundation)
Bertugas mengatur dan mengawasi proses pelaksanaan proyek.
6 Tahun
2
Citra Arfanudin
Tenaga Ahli
· S1, Teknik Informatika, Universitas Islam Indonesia
· S2, Informatika, Universitas Islam Indonesia
· TOGAF
· Lead Auditor ISO 27001
· CCNA Routing and Switching
· CEH (Certified Ethical Hacker)
Memberikan masukan teknis dan review tata kelola.
8 Tahun
"""


class InternalCredentialsTest(unittest.TestCase):
    def test_extracts_named_experts_from_settings_table(self) -> None:
        from main.state_store import AppStateStore

        rows = AppStateStore._extract_internal_expert_rows(RAW_CREDENTIALS)

        self.assertEqual(rows[0]["name"], "Andi Yuniantoro")
        self.assertEqual(rows[0]["proposed_role"], "Project Manager")
        self.assertIn("CAPM", rows[0]["certifications"])
        self.assertIn("ISO/IEC 27001", rows[0]["certifications"])
        self.assertEqual(rows[1]["name"], "Citra Arfanudin")
        self.assertIn("Lead Auditor ISO 27001", rows[1]["certifications"])

    def test_summarizes_credentials_without_raw_headers_or_names(self) -> None:
        from main.state_store import AppStateStore

        summary = AppStateStore._summarize_credential_blob(RAW_CREDENTIALS, "fallback")

        self.assertNotIn("Andi Yuniantoro", summary)
        self.assertNotIn("Citra Arfanudin", summary)
        self.assertIn("Project Manager", summary)
        self.assertIn("Lead Auditor ISO 27001", summary)
        self.assertNotIn("NAMA TENAGA AHLI", summary)
        self.assertNotIn("NO ", summary)

    def test_generation_context_ignores_legacy_manual_settings(self) -> None:
        from main.state_store import AppStateStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AppStateStore(db_path=root / "state.db", asset_root=root / "assets")
            store._set_setting("internal_portfolio", "MANUAL PORTFOLIO SHOULD NOT LEAK")
            store.save_settings()
            store.save_supporting_documents(
                "portfolio",
                [("portfolio.txt", b"Portofolio unggahan: implementasi SPBE dan tata kelola layanan.")],
            )
            store.save_supporting_documents(
                "credentials",
                [("credentials.txt", b"Sertifikasi unggahan: ISO 27001, TOGAF, dan ITIL Foundation.")],
            )

            settings = store.get_settings()
            context = store.build_generation_context()
            profile = store.enrich_firm_profile(
                {
                    "portfolio_highlights": "Portofolio dari API internal untuk transformasi digital.",
                    "credential_highlights": "Kredensial dari API internal pada tata kelola TI.",
                }
            )

        self.assertEqual(settings["internal_portfolio"], "")
        self.assertEqual(settings["internal_credentials"], "")
        combined_context = "\n".join(
            [
                context.get("settings_context", ""),
                context.get("portfolio_context", ""),
                context.get("credential_context", ""),
                profile.get("portfolio_highlights", ""),
                profile.get("credential_highlights", ""),
            ]
        )
        self.assertNotIn("MANUAL PORTFOLIO SHOULD NOT LEAK", combined_context)
        self.assertIn("Portofolio unggahan", combined_context)
        self.assertIn("Sertifikasi unggahan", combined_context)
        self.assertIn("API internal", combined_context)


if __name__ == "__main__":
    unittest.main()
