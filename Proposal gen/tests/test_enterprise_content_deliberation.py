import unittest

from main.proposal_deliberation import ProposalDeliberationBuilder
from main.section_planning import ProposalSectionPlanner
from main.proposal_quality_pipeline import ProposalQualityGate


class ProposalDeliberationTests(unittest.TestCase):
    def setUp(self):
        ProposalDeliberationBuilder.clear_cache()
        ProposalSectionPlanner.clear_cache()
        self.chapters = [
            {"id": "c_1", "title": "BAB I - Pendahuluan", "subs": ["Latar Belakang"]},
            {"id": "c_4", "title": "BAB IV - Solusi", "subs": ["Pendekatan"]},
            {"id": "c_8", "title": "BAB VIII - Jadwal", "subs": ["Tahapan"]},
        ]
        self.cards = [
            {
                "chapter_ids": ["c_1", "c_4"],
                "claim": "Klien membutuhkan penguatan tata kelola layanan.",
                "source_type": "input_pengguna",
                "confidence": "high",
                "implication": "Solusi harus memiliki kontrol dan hasil yang dapat divalidasi.",
            },
            {
                "chapter_ids": ["c_8"],
                "claim": "Durasi acuan pelaksanaan adalah 12 minggu.",
                "source_type": "kontrak_ruang_lingkup",
                "confidence": "high",
            },
        ]

    def test_builds_complete_cached_document_contract(self):
        builder = ProposalDeliberationBuilder()
        contract = builder.build(
            client="Contoh Klien",
            project="Transformasi Layanan",
            chapters=self.chapters,
            evidence_cards=self.cards,
            scope_contract={"assumptions": ["Akses narasumber tersedia"], "dependencies": ["Persetujuan jadwal"]},
            kak_contract={"deliverables": ["Peta jalan implementasi", "Rencana kontrol"]},
            data_version="snapshot-1",
        )

        self.assertEqual(
            {
                "cache_key", "data_version", "evidence_dossier", "research_plan",
                "document_thesis", "chapter_contracts", "claim_ledger",
                "data_gap_register", "editorial_contract", "appendix_manifest",
            },
            set(contract),
        )
        self.assertEqual("c_1", contract["chapter_contracts"][0]["section_id"])
        self.assertEqual("c_1", contract["chapter_contracts"][1]["depends_on"])
        self.assertEqual("c_8", contract["chapter_contracts"][1]["hands_off_to"])
        self.assertTrue(contract["research_plan"]["counterchecks"])
        self.assertIn("Bahasa Indonesia", " ".join(contract["editorial_contract"]["rules"]))
        self.assertEqual(
            "uc1-itmp-exemplar-profile-v2",
            contract["editorial_contract"]["exemplar_profile"]["version"],
        )
        self.assertEqual(
            "use_existing_universal_structure_only",
            contract["editorial_contract"]["exemplar_profile"]["hardcoded_structure_policy"],
        )
        self.assertIn("c_1", contract["editorial_contract"]["exemplar_profile"]["chapter_alignment"])
        self.assertIn("c_closing", contract["editorial_contract"]["exemplar_profile"]["chapter_alignment"])
        self.assertIn("source_mix", contract["editorial_contract"]["exemplar_profile"])
        self.assertIn("architecture_framework_moves", contract["editorial_contract"]["exemplar_profile"])
        self.assertIn("planning_depth_rules", contract["editorial_contract"]["exemplar_profile"])
        profile_payload = str(contract["editorial_contract"]["exemplar_profile"])
        self.assertNotIn("Banda Aceh", profile_payload)
        self.assertNotIn("INKA", profile_payload)
        self.assertNotIn("Lombok", profile_payload)
        self.assertNotIn("IJCTS", profile_payload)
        self.assertNotIn("Metro", profile_payload)
        self.assertEqual(contract, builder.build(
            client="Contoh Klien",
            project="Transformasi Layanan",
            chapters=self.chapters,
            evidence_cards=self.cards,
            scope_contract={"assumptions": ["Akses narasumber tersedia"], "dependencies": ["Persetujuan jadwal"]},
            kak_contract={"deliverables": ["Peta jalan implementasi", "Rencana kontrol"]},
            data_version="snapshot-1",
        ))
        self.assertEqual(1, builder.cache_stats()["hits"])

    def test_routes_supported_detail_and_gaps_to_reader_safe_appendix(self):
        builder = ProposalDeliberationBuilder()
        contract = builder.build(
            client="Contoh Klien",
            project="Transformasi Layanan",
            chapters=self.chapters,
            evidence_cards=self.cards,
            scope_contract={"assumptions": ["Akses narasumber tersedia"], "dependencies": ["Persetujuan jadwal"]},
            kak_contract={"deliverables": ["Peta jalan implementasi", "Rencana kontrol", "Kriteria penerimaan belum tersedia"]},
            data_version="snapshot-1",
        )

        appendix = builder.build_appendix_markdown(contract)

        self.assertIn("# Lampiran Bukti, Asumsi, dan Kesenjangan Data", appendix)
        self.assertIn("## A. Matriks Ketertelusuran", appendix)
        self.assertIn("## B. Asumsi dan Dependensi", appendix)
        self.assertIn("## C. Keterbatasan Bukti", appendix)
        self.assertNotIn("input_pengguna", appendix)
        self.assertNotIn("chain-of-thought", appendix.lower())

    def test_section_planner_inherits_document_thesis_and_chapter_obligations(self):
        contract = ProposalDeliberationBuilder().build(
            client="Contoh Klien",
            project="Transformasi Layanan",
            chapters=self.chapters,
            evidence_cards=self.cards,
            data_version="snapshot-1",
        )

        plan = ProposalSectionPlanner().build_plan(
            chapter=self.chapters[1],
            client="Contoh Klien",
            project="Transformasi Layanan",
            document_contract=contract,
        )

        self.assertEqual(contract["document_thesis"], plan["document_thesis"])
        self.assertEqual("c_1", plan["chapter_contract"]["depends_on"])
        self.assertIn("meaning_lock", plan["editorial_contract"])

    def test_final_quality_gate_requires_traceable_appendix_for_document_contract(self):
        builder = ProposalDeliberationBuilder()
        contract = builder.build(
            client="Contoh Klien",
            project="Transformasi Layanan",
            chapters=self.chapters,
            evidence_cards=self.cards,
            data_version="snapshot-1",
        )
        appendix = builder.build_appendix_markdown(contract)
        outputs = {
            "c_1": "Kebutuhan klien menjadi dasar keputusan solusi. Arah berikutnya masuk ke BAB IV - Solusi.",
            "c_4": "Melanjutkan BAB I - Pendahuluan, solusi mengikat kontrol dan hasil. Arah berikutnya masuk ke BAB VIII - Jadwal.",
            "c_8": "Melanjutkan BAB IV - Solusi, jadwal menjaga komitmen pelaksanaan selama 12 minggu.",
        }

        accepted = ProposalQualityGate.evaluate(
            chapter_outputs=outputs,
            selected_chapters=self.chapters,
            deliberation_contract=contract,
            appendix_content=appendix,
        )
        rejected = ProposalQualityGate.evaluate(
            chapter_outputs=outputs,
            selected_chapters=self.chapters,
            deliberation_contract=contract,
            appendix_content="",
        )

        self.assertNotIn("missing_deliberation_contract", accepted["categories"])
        self.assertNotIn("missing_traceability_appendix", accepted["categories"])
        self.assertIn("missing_traceability_appendix", rejected["categories"])


if __name__ == "__main__":
    unittest.main()
