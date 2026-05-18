import sys
import unittest
import zipfile
from io import BytesIO
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
        raw = (
            "Proposal memakai APIDog, Internal API endpoint /api/Resource/dataset, "
            "dataset code ReferenceDataset dan dataset name ConsultantProjectExpertHistory, "
            "source-of-truth cache sync, agent workflow, evidence card, confidence label, "
            "[RESEARCH_AGENT] [INTERNAL_DATA_AGENT] [COMMERCIAL_STRATEGY_AGENT] "
            "[TECHNICAL_SOLUTION_AGENT] [RISK_COMPLIANCE_AGENT] [EDITOR_MAIN_AGENT], "
            "Data Internal ReferenceAccount, dan URL https://example.test/api."
        )
        clean = sanitize_reader_facing_sources(raw)
        forbidden_tokens = [
            "APIDog",
            "Internal API",
            "endpoint",
            "/api/Resource/dataset",
            "dataset",
            "dataset code",
            "dataset name",
            "ReferenceDataset",
            "ReferenceAccount",
            "ConsultantProjectExpertHistory",
            "source-of-truth",
            "cache",
            "sync",
            "agent",
            "workflow",
            "evidence card",
            "confidence label",
            "RESEARCH_AGENT",
            "INTERNAL_DATA_AGENT",
            "COMMERCIAL_STRATEGY_AGENT",
            "TECHNICAL_SOLUTION_AGENT",
            "RISK_COMPLIANCE_AGENT",
            "EDITOR_MAIN_AGENT",
            "Data Internal",
            "https://",
        ]
        for token in forbidden_tokens:
            self.assertNotIn(token, clean)
        self.assertIn("konteks yang tersedia", clean)

    def test_toc_does_not_emit_static_fallback_items(self):
        doc = Document()
        DocumentBuilder.add_table_of_contents(doc, ["Ringkasan Eksekutif", "BAB I - Pendekatan", "BAB II - Rencana Kerja"])
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        self.assertIn("DAFTAR ISI", text)
        self.assertNotIn("Ringkasan Eksekutif", text)
        self.assertNotIn("BAB I - Pendekatan", text)
        self.assertNotIn("BAB II - Rencana Kerja", text)
        self.assertNotIn("Update Field", text)

    def test_toc_uses_word_field_and_updates_on_open(self):
        doc = Document()
        DocumentBuilder.add_table_of_contents(doc, ["Ringkasan Eksekutif", "BAB I - Pendekatan"])

        output = BytesIO()
        doc.save(output)
        output.seek(0)

        with zipfile.ZipFile(output) as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8")
            settings_xml = archive.read("word/settings.xml").decode("utf-8")

        self.assertIn('TOC \\o "1-3" \\h \\z \\u', document_xml)
        self.assertIn('w:fldCharType="begin"', document_xml)
        self.assertIn('w:fldCharType="separate"', document_xml)
        self.assertIn('w:fldCharType="end"', document_xml)
        self.assertIn("w:updateFields", settings_xml)
        self.assertIn('w:val="true"', settings_xml)

    def test_sanitizer_translates_unnecessary_english_reader_labels(self):
        raw = (
            "Executive Summary dan Key Findings menampilkan Recommendation, "
            "Dashboard, Insight, Pain Points, Current State, Target State, "
            "Owner, Scope, Deliverables, Milestones, Roadmap, dan Quality Gate. "
            "Brand teknis seperti API, OSINT, RAG, UAT, Go-Live, Microsoft Azure, dan Power BI tetap relevan."
        )
        clean = sanitize_reader_facing_sources(raw)
        forbidden_tokens = [
            "Executive Summary",
            "Key Findings",
            "Recommendation",
            "Dashboard",
            "Insight",
            "Pain Points",
            "Current State",
            "Target State",
            "Owner",
            "Scope",
            "Deliverables",
            "Milestones",
            "Roadmap",
            "Quality Gate",
        ]
        for token in forbidden_tokens:
            self.assertNotIn(token, clean)
        for expected in [
            "Ringkasan Eksekutif",
            "Temuan Utama",
            "Rekomendasi",
            "dasbor",
            "wawasan",
            "Titik Masalah",
            "kondisi saat ini",
            "kondisi target",
            "penanggung jawab",
            "ruang lingkup",
            "keluaran kerja",
            "tonggak kerja",
            "peta jalan",
            "gerbang mutu",
        ]:
            self.assertIn(expected, clean)
        for expected in ["*API*", "*OSINT*", "*RAG*", "*UAT*", "*Go-Live*", "Microsoft Azure", "Power BI"]:
            self.assertIn(expected, clean)

    def test_executive_summary_is_decision_first_and_source_neutral(self):
        summary = ExecutiveSummaryBuilder.build(
            client="PT Contoh",
            project="Implementasi AI Adoption",
            project_goal="meningkatkan produktivitas layanan",
            timeline="12 minggu",
            budget="Rp500 juta",
            value_map={
                "primary_outcome": "adopsi berjalan terukur",
                "value_statement": "produktivitas layanan meningkat tanpa mengorbankan tata kelola",
                "proof_points": ["pengalaman implementasi tata kelola digital", "indikator keberhasilan disepakati sejak awal"],
            },
            personalization_pack={
                "industry": "Teknologi Layanan Digital",
                "relationship_mode": "new",
                "kpi_blueprint": ["waktu respons layanan", "adopsi proses kerja"],
            },
        )
        self.assertIn("# Ringkasan Eksekutif", summary)
        expected_sections = [
            "## Inti Keputusan",
            "## Situasi dan Masalah Klien",
            "## Solusi yang Direkomendasikan",
            "## Nilai dan Bukti",
            "## Prioritas Eksekusi",
            "## Risiko yang Perlu Dikendalikan",
            "## Keputusan Berikutnya",
        ]
        positions = [summary.index(section) for section in expected_sections]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("PT Contoh", summary)
        self.assertIn("Implementasi AI Adoption", summary)
        self.assertIn("Teknologi Layanan Digital", summary)
        self.assertIn("12 minggu", summary)
        self.assertIn("Rp500 juta", summary)
        for token in [
            "APIDog",
            "Internal API",
            "Data Internal",
            "dataset",
            "source-of-truth",
            "cache",
            "sync",
            "agent",
            "workflow",
            "evidence card",
            "confidence label",
            "method",
            "source",
            "process",
            "proses internal",
        ]:
            self.assertNotIn(token, summary)
        for heading in [
            "## BLUF",
            "## Bottom Line",
            "## Key Findings",
            "## Recommendation",
            "## Executive Summary",
            "## Dashboard",
            "## Insight",
        ]:
            self.assertNotIn(heading, summary)


if __name__ == "__main__":
    unittest.main()
