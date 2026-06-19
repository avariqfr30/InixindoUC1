import unittest

import pandas as pd
import json
from pathlib import Path
from unittest import mock


class ProposalEditorialQualityTests(unittest.TestCase):
    @staticmethod
    def _render_structured_quality_fixture(chapter_id):
        from main.config import UNIVERSAL_STRUCTURE
        from main.proposal_generator import ProposalGenerator
        from main.proposal_technique import build_proposal_technique_contract

        generator = ProposalGenerator.__new__(ProposalGenerator)
        chapter = next(item for item in UNIVERSAL_STRUCTURE if item["id"] == chapter_id)
        contract = build_proposal_technique_contract(
            client="BMKG",
            goals="Menyusun tata kelola data, koordinasi layanan digital, dan kesiapan analitik lintas unit",
            customer_notes="Koordinasi lintas unit belum konsisten dan keputusan sponsor perlu dipercepat.",
            existing_condition="Baseline tata kelola dan kualitas data belum seragam.",
            frameworks="SPBE, COBIT, Data Governance",
        )
        content = generator._render_structured_chapter(
            chapter=chapter,
            client="BMKG",
            project="Transformasi Digital dan Tata Kelola Data",
            budget="Rp 450.000.000",
            service_type="Konsultasi dan Implementasi",
            project_goal="Meningkatkan tata kelola data, koordinasi layanan digital, dan kesiapan analitik lintas unit",
            project_type="Implementasi",
            timeline="12 minggu",
            notes="Koordinasi lintas unit belum konsisten, keputusan sponsor perlu dipercepat, dan hasil analitik perlu dapat digunakan.",
            regulations="SPBE, COBIT, Data Governance",
            firm_data={
                "methodology": "Design, Configuration, UAT, Go-Live, Handover",
                "team": "Project Director, Project Manager, Lead Arsitektur, Quality Reviewer",
                "commercial": "professional services berbasis milestone",
            },
            firm_profile={
                "credential_highlights": "COBIT, TOGAF, dan manajemen proyek",
                "portfolio_highlights": "portofolio transformasi digital",
            },
            research_bundle={},
            personalization_pack={
                "terminology": ["SPBE", "tata kelola", "analitik"],
                "kpi_blueprint": ["kejelasan baseline", "keputusan sponsor", "kesiapan analitik"],
            },
            value_map={
                "value_statement": "Inixindo Jogja membantu BMKG menghasilkan keputusan yang lebih terukur",
                "value_hook": "mengubah kebutuhan bisnis menjadi rencana kerja terukur",
                "win_theme": "kejelasan keputusan dan kontrol delivery",
                "client_gains": ["baseline lebih jelas", "keputusan lebih cepat", "hasil analitik siap dipakai"],
                "proof_points": ["portofolio transformasi digital", "metodologi delivery internal", "tim inti delivery"],
                "differentiators": ["delivery terukur", "governance konsisten"],
            },
            proposal_mode="canvassing",
            supporting_context={
                "proposal_technique_contract": contract,
                "active_scope_contract": contract.get("scope_contract_seed", {}),
            },
        )
        target = generator._target_words(chapter)
        content = generator._tighten_structured_chapter(chapter, content, target)
        content = generator._postprocess_chapter_content(chapter, content, "BMKG")
        quality = generator._evaluate_chapter_quality(
            chapter=chapter,
            content=content,
            client="BMKG",
            target_words=target,
            allowed_external_citations=set(),
            personalization_pack={},
        )
        return generator, chapter, content, quality

    def test_structured_refinement_chapters_stay_inside_content_bounds(self):
        for chapter_id in ["c_1", "c_2", "c_3", "c_6"]:
            with self.subTest(chapter_id=chapter_id):
                _, _, _, quality = self._render_structured_quality_fixture(chapter_id)
                self.assertNotIn("missing_h2", quality["issues"])
                self.assertNotIn("too_short", quality["issues"])
                self.assertNotIn("too_long", quality["issues"])

    def test_team_chapter_uses_table_aware_content_bounds(self):
        _, _, _, quality = self._render_structured_quality_fixture("c_11")

        self.assertNotIn("too_short", quality["issues"])
        self.assertNotIn("too_long", quality["issues"])
        self.assertNotIn("list_structure", quality["issues"])

    def test_named_expert_rows_are_bounded_for_readable_team_table(self):
        from main.proposal_support import ProposalSupportMixin

        context = {
            "source": "internal_api",
            "name_policy": {
                "allow_named_specialists": True,
                "source": "ConsultantProjectExpertHistory",
            },
            "employee_expertise_rows": [
                {
                    "employee_name": f"Konsultan {index}",
                    "certifications": ["COBIT 2019", "TOGAF Enterprise Architecture", "Project Management Professional"],
                    "projects": ["Transformasi Digital Nasional", "Penyusunan Arsitektur SPBE", "Roadmap Tata Kelola Data"],
                }
                for index in range(1, 9)
            ],
        }

        rows = ProposalSupportMixin._individual_expert_capability_rows(
            context,
            client="BMKG",
            project_type="Implementasi",
            service_type="Konsultasi dan Implementasi",
            regulations="SPBE, COBIT, Data Governance",
            team_points=[],
        )

        self.assertLessEqual(len(rows), 4)
        limits = {"nama": 6, "posisi": 8, "kapabilitas": 20, "pengalaman": 18, "sertifikasi": 16, "pemanfaatan": 22}
        for row in rows:
            for field, limit in limits.items():
                self.assertLessEqual(len(row[field].split()), limit, f"{field}: {row[field]}")

    def test_problem_chapter_fits_comprehensive_document_allocation(self):
        from main.config import UNIVERSAL_STRUCTURE

        generator, chapter, content, _ = self._render_structured_quality_fixture("c_2")
        scaled_target = generator._chapter_word_targets(UNIVERSAL_STRUCTURE)["c_2"]
        content = generator._tighten_structured_chapter(chapter, content, scaled_target)
        quality = generator._evaluate_chapter_quality(
            chapter=chapter,
            content=content,
            client="BMKG",
            target_words=scaled_target,
            allowed_external_citations=set(),
            personalization_pack={},
        )

        self.assertNotIn("too_long", quality["issues"])

    def test_problem_matrix_cells_stay_concise(self):
        from main.proposal_support import ProposalSupportMixin

        matrices = [
            ProposalSupportMixin._problem_gap_matrix(
                "Transformasi Digital dan Tata Kelola Data",
                "meningkatkan tata kelola data lintas unit",
                "12 minggu",
                "SPBE, tata kelola, analitik",
            ),
            ProposalSupportMixin._problem_risk_matrix(
                "Transformasi Digital dan Tata Kelola Data",
                "koordinasi lintas unit belum konsisten",
                "kejelasan baseline dan keputusan sponsor",
            ),
            ProposalSupportMixin._problem_solution_matrix(
                "BMKG",
                "meningkatkan tata kelola data lintas unit",
                "kejelasan baseline dan keputusan sponsor",
                "SPBE, tata kelola, analitik",
                "baseline jelas dan keputusan lebih cepat",
            ),
        ]

        for rows in matrices:
            for row in rows:
                for value in row.values():
                    self.assertLessEqual(len(value.split()), 14, value)

    def test_numbered_contact_section_receives_only_missing_verified_lines(self):
        from main.proposal_support import ProposalSupportMixin

        profile = {
            "office_address": "Jalan Kenari 69, Yogyakarta, D.I. Yogyakarta",
            "email": "marketing@inixindojogja.co.id",
            "phone": "(0274) 515448, 554419",
            "whatsapp": "+62 823-2549-0909",
            "website": "https://inixindojogja.co.id/",
        }
        content = (
            "## 12.2 Informasi Kontak dan Langkah Lanjutan\n"
            "- Alamat kantor: Jalan Kenari 69, Yogyakarta, D.I. Yogyakarta\n"
            "- Email: marketing@inixindojogja.co.id\n"
            "- Telp: (0274) 515448, 554419\n"
            "- Website: https://inixindojogja.co.id/"
        )

        injected = ProposalSupportMixin._inject_verified_firm_contact(content, profile)

        self.assertEqual(injected.count("Informasi Kontak dan Langkah Lanjutan"), 1)
        self.assertEqual(injected.count("Alamat kantor:"), 1)
        self.assertEqual(injected.count("WhatsApp: +62 823-2549-0909"), 1)

    def test_verified_contact_line_matching_normalizes_phone_url_and_address(self):
        from main.proposal_support import ProposalSupportMixin

        profile = {
            "office_address": "Jalan Kenari 69, Yogyakarta, D.I. Yogyakarta",
            "email": "marketing@inixindojogja.co.id",
            "phone": "(0274) 515448, 554419",
            "whatsapp": "+62 823-2549-0909",
            "website": "https://inixindojogja.co.id/",
        }

        for line in [
            "Alamat kantor: Jalan Kenari 69, Yogyakarta, D.I. Yogyakarta",
            "Email: marketing@inixindojogja.co.id",
            "Telp: (0274) 515448, 554419",
            "WhatsApp: +62 823-2549-0909",
            "Website: https://inixindojogja.co.id/",
        ]:
            self.assertTrue(ProposalSupportMixin._contact_line_matches_profile(line, profile), line)
        self.assertFalse(ProposalSupportMixin._contact_line_matches_profile("Email: other@example.com", profile))

    def test_closing_quality_excludes_profile_section_rendered_elsewhere(self):
        from main.proposal_support import ProposalSupportMixin

        closing = (
            "## 12.1 Apresiasi dan Komitmen Kemitraan\nTerima kasih atas kesempatan yang diberikan.\n\n"
            "## 12.2 Informasi Kontak dan Langkah Lanjutan\nLangkah berikutnya adalah finalisasi ruang lingkup.\n\n"
            "## Tentang Mitra Penulis Proposal\n" + ("Profil perusahaan dan portofolio panjang. " * 100)
        )

        visible = ProposalSupportMixin._closing_content_for_quality(closing)

        self.assertIn("12.2 Informasi Kontak", visible)
        self.assertNotIn("Tentang Mitra Penulis Proposal", visible)

    def test_company_fit_recognizes_natural_paraphrases_of_proof_points(self):
        from main.proposal_generator import ProposalGenerator

        generator = ProposalGenerator.__new__(ProposalGenerator)
        content = (
            "Inixindo Jogja mendukung keputusan klien melalui portofolio transformasi digital, "
            "metodologi pelaksanaan yang terstruktur, dan tim delivery dengan kontrol mutu.\n"
            "1. Keputusan sponsor dijaga.\n- Bukti kerja dipakai dalam review."
        )
        report = generator._evaluate_proposal_acceptance(
            chapter_outputs={"c_10": content},
            selected_chapters=[{"id": "c_10", "title": "Profil", "subs": []}],
            chapter_targets={"c_10": 80},
            client="BMKG",
            project="Transformasi Digital",
            notes="Tata kelola data",
            firm_profile={},
            allowed_external_citations=set(),
            personalization_pack={},
            value_map={
                "positioning": "mitra konsultasi transformasi digital",
                "proposal_promise": "dokumen kerja untuk keputusan",
                "differentiators": ["kontrol mutu delivery"],
                "proof_points": [
                    "portofolio proyek transformasi digital yang relevan",
                    "metodologi delivery internal yang terstruktur",
                    "tim inti delivery dan quality control",
                ],
            },
        )

        self.assertGreaterEqual(report["company_fit_breakdown"]["proof_points"], 75)

    def test_roadmap_language_is_not_mistaken_for_a_street_address(self):
        from main.proposal_support import ProposalSupportMixin

        text = "Nilai utama mencakup peta jalan eksekusi yang lebih mudah disetujui."

        self.assertEqual(ProposalSupportMixin._find_contact_like_lines(text), [])

    def test_long_client_names_use_a_natural_followup_reference(self):
        from main.proposal_support import ProposalSupportMixin

        long_name = "DINAS KOMUNIKASI INFORMATIKA DAN PERSANDIAN KABUPATEN KENDAL"

        reference = ProposalSupportMixin._reader_client_reference(long_name)

        self.assertEqual(reference, "instansi ini")
        self.assertLess(len(reference.split()), len(long_name.split()))

    def test_closing_length_excludes_required_contact_metadata(self):
        from main.proposal_generator import ProposalGenerator

        generator = ProposalGenerator.__new__(ProposalGenerator)
        narrative = (
            "## 12.1 Apresiasi dan Komitmen Kemitraan\n"
            "BMKG dan Inixindo Jogja menjaga tindak lanjut yang terukur.\n"
            "1. Ruang lingkup dikonfirmasi bersama.\n"
            "- Agenda kickoff ditetapkan setelah persetujuan.\n\n"
            "## 12.2 Informasi Kontak dan Langkah Lanjutan\n"
            "Langkah berikutnya adalah finalisasi ruang lingkup dan PIC."
        )
        contacts = (
            "\n- Alamat kantor: Jalan Kenari 69, Yogyakarta, D.I. Yogyakarta"
            "\n- Email: marketing@inixindojogja.co.id"
            "\n- Telp: (0274) 515448, 554419"
            "\n- WhatsApp: +62 823-2549-0909"
            "\n- Website: https://inixindojogja.co.id/"
        )
        report = generator._evaluate_chapter_quality(
            chapter={
                "id": "c_closing",
                "title": "BAB XII – PENUTUP",
                "subs": ["12.1 Apresiasi dan Komitmen Kemitraan", "12.2 Informasi Kontak dan Langkah Lanjutan"],
            },
            content=narrative + contacts,
            client="BMKG",
            target_words=100,
            allowed_external_citations=set(),
            personalization_pack={},
        )

        self.assertEqual(report["word_count"], generator._word_count(narrative))

    def test_terminology_repairs_vary_by_chapter_purpose(self):
        from main.proposal_support import ProposalSupportMixin

        lines = {
            ProposalSupportMixin._terminology_repair_line(chapter_id, ["SPBE", "tata kelola", "analitik"])
            for chapter_id in ["c_1", "c_2", "c_3", "c_4", "c_5", "c_6", "c_7", "c_8", "c_9", "c_10", "c_11", "c_12", "c_closing"]
        }

        self.assertEqual(len(lines), 13)
        self.assertTrue(all("SPBE" in line for line in lines))

    def test_problem_chapter_uses_concise_direction_wording(self):
        _, _, content, _ = self._render_structured_quality_fixture("c_2")

        self.assertIn("arah kerja yang jelas, terukur, dan mudah diawasi", content)
        self.assertNotIn("lebih jelas, lebih terukur, dan lebih mudah diawasi", content)

    def test_generic_contact_pointer_is_not_treated_as_unverified_contact_data(self):
        from main.proposal_support import ProposalSupportMixin

        text = (
            "Komunikasi lanjutan dapat diarahkan melalui kanal resmi perusahaan "
            "yang tercantum pada bagian kontak. Alamat kantor di atas dapat digunakan "
            "sebagai rujukan apabila diperlukan pertemuan lanjutan."
        )

        self.assertEqual(ProposalSupportMixin._find_contact_like_lines(text), [])

    def test_company_fit_credits_verified_profile_rendered_after_team_chapter(self):
        from main.proposal_generator import ProposalGenerator

        generator = ProposalGenerator.__new__(ProposalGenerator)
        positioning = "mitra pembelajaran dan konsultasi dengan rencana eksekusi yang jelas dan kredibel"
        promise = "dokumen kerja siap dipakai untuk mengambil keputusan"
        proof_points = [
            "portofolio konsultasi transformasi digital",
            "metodologi delivery internal yang terstruktur",
        ]
        content = (
            "Inixindo Jogja hadir sebagai " + positioning + ". " + promise + ".\n\n"
            "## 10.1 Struktur Tim Proyek\n"
            "| Peran | Tanggung Jawab |\n| --- | --- |\n| Project Director | Menjaga keputusan sponsor |\n\n"
            "## 10.2 Kapabilitas Konsultan, Pengalaman, dan Sertifikasi\n"
            + ". ".join(proof_points)
        )

        report = generator._evaluate_proposal_acceptance(
            chapter_outputs={"c_11": content},
            selected_chapters=[{
                "id": "c_11",
                "title": "BAB X – STRUKTUR & TEAM PROYEK",
                "subs": ["10.1 Struktur Tim Proyek", "10.2 Kapabilitas Konsultan, Pengalaman, dan Sertifikasi"],
            }],
            chapter_targets={"c_11": 120},
            client="BMKG",
            project="Transformasi Digital",
            notes="Tata kelola data dan koordinasi layanan digital",
            firm_profile={
                "office_address": "Jalan Kenari 69, Yogyakarta",
                "email": "marketing@inixindojogja.co.id",
                "phone": "(0274) 515448",
                "website": "https://inixindojogja.co.id/",
            },
            allowed_external_citations=set(),
            personalization_pack={},
            value_map={
                "positioning": positioning,
                "proposal_promise": promise,
                "differentiators": ["rencana eksekusi yang jelas"],
                "proof_points": proof_points,
            },
        )

        self.assertEqual(report["company_fit_breakdown"]["verified_contacts"], 100)
        self.assertNotIn("verified_contact_missing", report["hard_failures"])
        self.assertNotIn("unverified_contact_detail", report["hard_failures"])

    def test_table_first_team_chapter_does_not_require_artificial_numbered_list(self):
        from main.proposal_generator import ProposalGenerator

        generator = ProposalGenerator.__new__(ProposalGenerator)
        chapter = {
            "id": "c_11",
            "title": "BAB X – STRUKTUR & TEAM PROYEK",
            "subs": ["10.1 Struktur Tim Proyek", "10.2 Kapabilitas Konsultan, Pengalaman, dan Sertifikasi"],
        }
        content = (
            "## 10.1 Struktur Tim Proyek\n"
            "| Peran | Tanggung Jawab |\n| --- | --- |\n| Project Director | Menjaga keputusan sponsor |\n\n"
            "## 10.2 Kapabilitas Konsultan, Pengalaman, dan Sertifikasi\n"
            "| Nama | Kapabilitas |\n| --- | --- |\n| Konsultan A | Tata kelola data |"
        )

        report = generator._evaluate_chapter_quality(
            chapter=chapter,
            content=content,
            client="BMKG",
            target_words=80,
            allowed_external_citations=set(),
            personalization_pack={},
        )

        self.assertNotIn("list_structure", report["issues"])
        self.assertEqual(report["word_count"], generator._word_count(content))
        self.assertIn("min_words", report)
        self.assertIn("max_words", report)

    def test_golden_fixture_contains_human_and_deterministic_closure_gates(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "golden_proposal_scope_quality.json").read_text())

        self.assertTrue(all(score == 4 for score in fixture["expected"]["human_rubric_minimums"].values()))
        self.assertEqual(fixture["expected"]["max_repeated_nontrivial_cell_count"], 2)
        self.assertLessEqual(fixture["expected"]["max_table_cell_words"], 22)

    def test_policy_excludes_irrelevant_datasets_and_serves_mixed_readers(self):
        from main.editorial_intelligence import EXCLUDED_DATASETS, proposal_voice_rules

        self.assertEqual(EXCLUDED_DATASETS, {"FinanceInvoice", "ProjectStandards"})
        rules = " ".join(proposal_voice_rules()).lower()
        self.assertIn("pembaca campuran", rules)
        self.assertIn("bukti", rules)

    def test_repeated_table_detail_stays_reader_facing_instead_of_blank(self):
        from main.editorial_intelligence import compact_markdown_table_rows

        rows = [["A", "Bukti pengalaman proyek yang sama"] for _ in range(3)]
        compacted = compact_markdown_table_rows(rows)

        self.assertTrue(compacted[2][1])
        self.assertNotIn("lihat uraian", " ".join(cell for row in compacted for cell in row).lower())

    def test_style_assessment_flags_stiff_template_language_and_repeated_openings(self):
        from main.editorial_intelligence import assess_proposal_style

        text = (
            "Dalam rangka pelaksanaan program, organisasi perlu melakukan asesmen.\n\n"
            "Dalam rangka pelaksanaan program, organisasi perlu menyusun roadmap.\n\n"
            "Dalam rangka pelaksanaan program, organisasi perlu melakukan evaluasi."
        )
        result = assess_proposal_style(text)

        self.assertFalse(result["passed"])
        self.assertIn("template_language", result["findings"])
        self.assertIn("repeated_openings", result["findings"])

    def test_evidence_card_preserves_provenance_without_exposing_raw_code_to_reader(self):
        from main.editorial_intelligence import build_evidence_card

        card = build_evidence_card(
            "ConsultantProjectExpertHistory",
            "Pengalaman implementasi tata kelola data",
            source_date="2026-05-10",
            confidence="high",
        )

        self.assertEqual(card["dataset_code"], "ConsultantProjectExpertHistory")
        self.assertEqual(card["source_date"], "2026-05-10")
        self.assertNotIn("ConsultantProjectExpertHistory", card["reader_text"])

    def test_final_quality_gate_rejects_stiff_template_language(self):
        from main.proposal_quality_pipeline import ProposalQualityGate

        result = ProposalQualityGate.evaluate(
            chapter_outputs={
                "c_1": (
                    "Dalam rangka pelaksanaan program, organisasi perlu melakukan asesmen.\n\n"
                    "Dalam rangka pelaksanaan program, organisasi perlu menyusun roadmap.\n\n"
                    "Dalam rangka pelaksanaan program, organisasi perlu melakukan evaluasi."
                )
            },
            selected_chapters=[{"id": "c_1", "title": "Konteks"}],
        )

        self.assertFalse(result["passes"])
        self.assertIn("editorial_style", result["categories"])

    def test_section_planner_selects_one_of_multiple_evidence_bound_angles(self):
        from main.section_planning import ProposalSectionPlanner

        plan = ProposalSectionPlanner().build_plan(
            {"id": "c_5", "title": "Metodologi", "subs": ["Tahapan", "Kontrol"]},
            client="PT Contoh",
            project="Transformasi Layanan",
            evidence_summary="Pengalaman proyek menunjukkan kebutuhan kontrol risiko dan transfer pengetahuan.",
        )

        self.assertGreaterEqual(len(plan["candidate_angles"]), 3)
        self.assertIn(plan["selected_angle"], plan["candidate_angles"])
        self.assertNotIn("FinanceInvoice", " ".join(plan["candidate_angles"]))

    def test_obsolete_project_standards_is_not_configured_or_requested(self):
        from main.internal_api_runtime import READINESS_ORDER
        from main.internal_api_setup import DEFAULT_DATASETS, build_internal_api_config
        from main.runtime_components import FirmAPIClient

        config = build_internal_api_config({"url": "https://example.test/api", "datasets": {}, "response_paths": {}})

        self.assertNotIn("project_standards", DEFAULT_DATASETS)
        self.assertNotIn("project_standards", config["resources"])
        self.assertNotIn("project_standards", READINESS_ORDER)

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.demo_mode = False
        with mock.patch.object(client, "_resolve_resource_payload", side_effect=AssertionError("obsolete dataset requested")):
            result = client.get_project_standards("Implementation")
        self.assertIn("methodology", result)

    def test_internal_data_client_keeps_delivery_guidance_engine_contract(self):
        from main.runtime_components import InternalDataClient

        client = InternalDataClient(force_source="demo")

        guidance = client.get_delivery_guidance("Implementation")

        self.assertIn("methodology", guidance)
        self.assertIn("team", guidance)
        self.assertIn("commercial", guidance)

    def test_vector_sync_reuses_existing_matching_embeddings(self):
        from main.project_knowledge_base import KnowledgeBase

        class FakeCollection:
            def __init__(self):
                self.upsert_calls = []

            def get(self, include=None):
                return {
                    "ids": ["0", "1"],
                    "metadatas": [
                        {"entity": "A", "topic": "Governance"},
                        {"entity": "B", "topic": "Security"},
                    ],
                }

            def delete(self, ids):
                raise AssertionError(f"Unexpected delete: {ids}")

            def upsert(self, **kwargs):
                self.upsert_calls.append(kwargs)

        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb.df = pd.DataFrame(
            [
                {"entity": "A", "topic": "Governance"},
                {"entity": "B", "topic": "Security"},
            ]
        )
        kb.collection = FakeCollection()
        kb.vector_ready = False
        kb.last_refresh_error = ""

        with mock.patch.object(kb, "_vector_store_current", return_value=False), mock.patch.object(kb, "_write_sync_state") as write_state:
            self.assertTrue(kb._sync_vector_store())

        self.assertEqual(kb.collection.upsert_calls, [])
        write_state.assert_called_once()

    def test_vector_sync_embeds_only_changed_rows(self):
        from main.project_knowledge_base import KnowledgeBase

        class FakeCollection:
            def __init__(self):
                self.upsert_calls = []

            def get(self, include=None):
                return {
                    "ids": ["0", "1"],
                    "metadatas": [
                        {"entity": "A", "topic": "Governance"},
                        {"entity": "B", "topic": "Old topic"},
                    ],
                }

            def delete(self, ids):
                raise AssertionError(f"Unexpected delete: {ids}")

            def upsert(self, **kwargs):
                self.upsert_calls.append(kwargs)

        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb.df = pd.DataFrame(
            [
                {"entity": "A", "topic": "Governance"},
                {"entity": "B", "topic": "Security"},
            ]
        )
        kb.collection = FakeCollection()
        kb.vector_ready = False
        kb.last_refresh_error = ""

        with mock.patch.object(kb, "_vector_store_current", return_value=False), mock.patch.object(kb, "_write_sync_state"):
            self.assertTrue(kb._sync_vector_store())

        self.assertEqual(len(kb.collection.upsert_calls), 1)
        self.assertEqual(kb.collection.upsert_calls[0]["ids"], ["1"])


if __name__ == "__main__":
    unittest.main()
