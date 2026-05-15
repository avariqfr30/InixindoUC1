import sys
import unittest
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.document_rendering import DocumentBuilder
from main.executive_summary import ExecutiveSummaryBuilder
from main.reader_facing_hygiene import sanitize_reader_facing_sources


class ReaderFacingDocumentContractTests(unittest.TestCase):
    def test_sanitizer_removes_internal_source_mode_terms(self):
        raw = "Proposal memakai APIDog, Internal API, Data Internal, dataset ReferenceAccount, dan URL https://example.test/api."
        clean = sanitize_reader_facing_sources(raw)
        for token in ["APIDog", "Internal API", "Data Internal", "dataset", "ReferenceAccount", "https://"]:
            self.assertNotIn(token, clean)
        self.assertIn("konteks yang tersedia", clean)

    def test_toc_static_fallback_is_visible(self):
        doc = Document()
        DocumentBuilder.add_table_of_contents(doc, ["Ringkasan Eksekutif", "BAB I - Pendekatan", "BAB II - Rencana Kerja"])
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        self.assertIn("DAFTAR ISI", text)
        self.assertIn("Ringkasan Eksekutif", text)
        self.assertIn("BAB I - Pendekatan", text)
        self.assertNotIn("Update Field", text)

    def test_executive_summary_is_decision_first_and_source_neutral(self):
        summary = ExecutiveSummaryBuilder.build(
            client="PT Contoh",
            project="Implementasi AI Adoption",
            project_goal="meningkatkan produktivitas layanan",
            timeline="12 minggu",
            budget="Rp500 juta",
            value_map={"primary_outcome": "adopsi berjalan terukur"},
        )
        self.assertIn("# Ringkasan Eksekutif", summary)
        self.assertIn("## Keputusan Utama", summary)
        self.assertIn("## Prioritas Eksekusi", summary)
        for token in ["APIDog", "Internal API", "Data Internal", "dataset"]:
            self.assertNotIn(token, summary)


if __name__ == "__main__":
    unittest.main()
