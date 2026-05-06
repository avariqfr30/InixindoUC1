"""Regression coverage for turning pasted internal credential tables into persuasive proof."""
from __future__ import annotations

import sys
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

    def test_summarizes_credentials_without_raw_headers(self) -> None:
        from main.state_store import AppStateStore

        summary = AppStateStore._summarize_credential_blob(RAW_CREDENTIALS, "fallback")

        self.assertIn("Andi Yuniantoro", summary)
        self.assertIn("Citra Arfanudin", summary)
        self.assertIn("Project Manager", summary)
        self.assertIn("Lead Auditor ISO 27001", summary)
        self.assertNotIn("NAMA TENAGA AHLI", summary)
        self.assertNotIn("NO ", summary)


if __name__ == "__main__":
    unittest.main()
