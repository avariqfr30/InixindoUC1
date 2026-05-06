"""Regression coverage for DOCX rendering hygiene."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DocumentRenderingHygieneTest(unittest.TestCase):
    def test_renderer_strips_escaped_source_markup_and_demotes_invalid_pipe_rows(self) -> None:
        from main.document_rendering import DocumentBuilder

        doc = Document()
        raw = (
            "AccelByte membutuhkan tata kelola layanan digital yang selaras dengan operasi backend game "
            "&lt;/em&gt;&lt;/span&gt;.\n\n"
            "| Sumber eksternal 1 | fakta=raw snippet | sumber=Example | url=https://example.test |\n\n"
            "Narasi dilanjutkan sebagai paragraf biasa."
        )

        DocumentBuilder.process_content(doc, raw, (0, 51, 102), "BAB I")
        rendered_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

        self.assertIn("AccelByte membutuhkan tata kelola layanan digital", rendered_text)
        self.assertIn("Narasi dilanjutkan", rendered_text)
        self.assertNotIn("span", rendered_text.lower())
        self.assertNotIn("fakta=", rendered_text)
        self.assertEqual(len(doc.tables), 0)

    def test_renderer_keeps_valid_markdown_tables(self) -> None:
        from main.document_rendering import DocumentBuilder

        doc = Document()
        raw = (
            "| Area | Relevansi |\n"
            "| --- | --- |\n"
            "| Governance | Mengawal keputusan dan risiko |\n"
        )

        DocumentBuilder.process_content(doc, raw, (0, 51, 102), "BAB XI")

        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].cell(1, 0).text.strip(), "Governance")


if __name__ == "__main__":
    unittest.main()
