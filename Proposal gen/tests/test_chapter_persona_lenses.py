"""Regression coverage for invisible chapter persona lenses."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ChapterPersonaLensesTest(unittest.TestCase):
    def test_chapter_one_prompt_gets_invisible_engagement_research_lens(self) -> None:
        from main.proposal_agents import ProposalAgentWorkflow
        from main.proposal_support import ProposalSupportMixin

        lens = ProposalSupportMixin._chapter_persona_lens("c_1")

        self.assertIn("INVISIBLE_CHAPTER_PERSONA", lens)
        self.assertIn("engagement strategist", lens.lower())
        self.assertIn("client story", lens.lower())
        self.assertIn("OSINT", lens)
        self.assertNotIn("tulis label persona", lens.lower())
        self.assertEqual(lens, ProposalAgentWorkflow.chapter_persona_lens("c_1"))

    def test_chapter_ten_and_eleven_have_distinct_evidence_lenses(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        profile_lens = ProposalSupportMixin._chapter_persona_lens("c_10")
        team_lens = ProposalSupportMixin._chapter_persona_lens("c_11")

        self.assertIn("writer-firm credibility", profile_lens.lower())
        self.assertIn("portfolio", profile_lens.lower())
        self.assertIn("internal expert", team_lens.lower())
        self.assertIn("assignment", team_lens.lower())
        self.assertNotEqual(profile_lens, team_lens)

    def test_unknown_chapter_uses_default_prompt_only_lens(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        lens = ProposalSupportMixin._chapter_persona_lens("unknown")

        self.assertIn("principal management consultant", lens.lower())
        self.assertIn("prompt-only", lens.lower())

    def test_chapter_agent_workflow_brief_separates_research_and_writer_work(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        brief = ProposalSupportMixin._chapter_agent_workflow_brief(
            chapter={
                "id": "c_11",
                "title": "Struktur Tim Proyek",
                "subs": ["Komposisi Tim", "Peran dan Kredensial"],
            },
            client="Ajinomoto Indonesia",
            project="Manajemen SPBE",
            research_bundle={
                "profile": "Ajinomoto Indonesia mengelola operasi manufaktur pangan.",
                "regulations": "SPBE menuntut tata kelola, layanan, dan pengukuran kematangan.",
            },
            personalization_pack={
                "profile_summary": "Klien membutuhkan struktur kerja yang konkret.",
                "terminology": ["operasi", "quality gate"],
            },
            value_map={
                "win_theme": "tim yang sudah terbukti pada tata kelola digital",
                "proof_points": ["rekam jejak pendampingan SPBE"],
            },
            client_internal_context="Akun klien berada di segmen SWASTA dan wilayah DIY.",
            expert_bench_context="Manajemen SPBE tercatat pada 5 tenaga ahli; contoh produk: Arsitektur SPBE.",
            chapter_chain_context="Pertahankan kesinambungan dari metodologi sebelumnya.",
        )

        self.assertIn("[CHAPTER_RESEARCH_AGENT]", brief)
        self.assertIn("[CHAPTER_WRITER_AGENT]", brief)
        self.assertIn("[CHAPTER_HANDOFF]", brief)
        self.assertIn("Ajinomoto Indonesia", brief)
        self.assertIn("Manajemen SPBE", brief)
        self.assertIn("rekam jejak pendampingan SPBE", brief)
        self.assertIn("Manajemen SPBE tercatat", brief)
        self.assertIn("jangan menampilkan label agen", brief.lower())
        self.assertIn("jangan menyebut nama dataset", brief.lower())

    def test_closing_agent_brief_keeps_narrative_clear_of_capability_dump(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        brief = ProposalSupportMixin._chapter_agent_workflow_brief(
            chapter={"id": "c_closing", "title": "Penutup", "subs": []},
            client="Accelbyte",
            project="Adopsi AI",
            research_bundle={"profile": "Accelbyte membangun platform teknologi gim."},
            personalization_pack={"profile_summary": "Konteks klien berbasis teknologi."},
            value_map={"win_theme": "langkah lanjut yang jelas", "proof_points": []},
            client_internal_context="",
            expert_bench_context="Lead auditor dan arsitek SPBE tersedia.",
            chapter_chain_context="Ringkas keputusan komersial dari bab sebelumnya.",
        )

        self.assertIn("jaga penutup tetap bersih", brief.lower())
        self.assertIn("tanpa daftar kredensial panjang", brief.lower())
        self.assertIn("Ringkas keputusan komersial", brief)

    def test_specialist_agents_have_bounded_api_and_osint_lanes(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        brief = ProposalSupportMixin._chapter_agent_workflow_brief(
            chapter={
                "id": "c_4",
                "title": "Pendekatan dan Framework",
                "subs": ["Kerangka Kerja", "Regulasi"],
            },
            client="Ajinomoto Indonesia",
            project="Manajemen SPBE",
            research_bundle={
                "profile": "Ajinomoto Indonesia mengelola operasi manufaktur pangan.",
                "regulations": "SPBE memerlukan tata kelola, layanan, dan pengukuran kematangan.",
                "track_record": "Inixindo memiliki rekam jejak konsultansi tata kelola digital.",
            },
            personalization_pack={
                "profile_summary": "Klien membutuhkan tata kelola yang dapat dipertanggungjawabkan.",
                "terminology": ["operasi", "quality gate"],
            },
            value_map={
                "win_theme": "framework yang relevan dan dapat dijalankan",
                "proof_points": ["rekam jejak pendampingan SPBE"],
            },
            client_internal_context="Akun klien berada di segmen SWASTA.",
            expert_bench_context={"expert_history_summary": "Arsitektur SPBE dan Manajemen SPBE tercatat dalam riwayat internal."},
            chapter_chain_context="Scope sudah dikunci pada tata kelola layanan digital.",
        )

        self.assertIn("[SPECIALIST_AGENT:client_intelligence]", brief)
        self.assertIn("[SPECIALIST_AGENT:framework_regulatory]", brief)
        self.assertIn("[SPECIALIST_AGENT:capability_evidence]", brief)
        self.assertIn("[MAIN_SYNTHESIS_AGENT]", brief)
        self.assertIn("API lane: account_records", brief)
        self.assertIn("API lane: project_records", brief)
        self.assertIn("OSINT lane: regulations", brief)
        self.assertIn("OSINT lane: track_record", brief)
        self.assertIn("only report findings inside its lane", brief)

    def test_specialist_agent_selection_changes_by_chapter(self) -> None:
        from main.proposal_agents import ProposalAgentWorkflow
        from main.proposal_support import ProposalSupportMixin

        commercial = ProposalSupportMixin._chapter_specialist_agent_specs("c_12")
        closing = ProposalSupportMixin._chapter_specialist_agent_specs("c_closing")

        self.assertEqual(commercial, ProposalAgentWorkflow.chapter_specialist_agent_specs("c_12"))
        self.assertIn("commercial_delivery", [item["id"] for item in commercial])
        self.assertIn("client_intelligence", [item["id"] for item in closing])
        self.assertIn("capability_evidence", [item["id"] for item in closing])
        self.assertNotIn("framework_regulatory", [item["id"] for item in closing])

    def test_mixin_specialist_agent_selection_preserves_subclass_overrides(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        class CustomSupport(ProposalSupportMixin):
            SPECIALIST_AGENT_REGISTRY = {
                **ProposalSupportMixin.SPECIALIST_AGENT_REGISTRY,
                "custom_agent": {
                    "role": "custom proof agent",
                    "api_lanes": ["custom_records"],
                    "osint_lanes": ["custom_osint"],
                    "focus": "custom scoped evidence",
                },
            }
            CHAPTER_SPECIALIST_AGENT_MAP = {"c_custom": ["custom_agent"]}

        specs = CustomSupport._chapter_specialist_agent_specs("c_custom")
        ids = [item["id"] for item in specs]

        self.assertIn("custom_agent", ids)
        self.assertIn("custom_records", specs[ids.index("custom_agent")]["api_lanes"])

    def test_proposal_agent_workflow_fallback_join_and_summarize_work_directly(self) -> None:
        from main.proposal_agents import ProposalAgentWorkflow

        brief = ProposalAgentWorkflow.chapter_agent_workflow_brief(
            chapter={"id": "c_1", "title": "Ringkasan", "subs": ["Satu", "Dua", "Tiga", "Empat"]},
            client="Client A",
            project="Project B",
            research_bundle={"profile": " ".join(["profil"] * 50)},
            personalization_pack={"terminology": ["alpha; beta", "alpha", "gamma"]},
            value_map={"proof_points": ["bukti satu|bukti dua|bukti tiga|bukti empat"]},
        )

        self.assertIn("Client A", brief)
        self.assertIn("Project B", brief)
        self.assertIn("bukti satu, bukti dua, bukti tiga, dan bukti empat", brief)
        self.assertIn("alpha, beta, dan gamma", brief)

    def test_optimized_workflow_uses_evidence_cards_and_single_pass_synthesis(self) -> None:
        from main.proposal_support import ProposalSupportMixin

        brief = ProposalSupportMixin._chapter_agent_workflow_brief(
            chapter={
                "id": "c_6",
                "title": "Solution Design",
                "subs": ["Target State", "Deliverable"],
            },
            client="Accelbyte",
            project="AI Governance",
            research_bundle={
                "profile": "Accelbyte menyediakan platform teknologi untuk live game services.",
                "ai_posture": "Adopsi AI perlu mengutamakan governance, readiness, dan human oversight.",
            },
            personalization_pack={
                "profile_summary": "Klien membutuhkan solusi yang scalable tetapi tetap terkendali.",
                "ai_adoption_profile": {"enabled": True, "summary": "AI use case membutuhkan validasi data dan kontrol risiko."},
            },
            value_map={
                "win_theme": "governance yang mempercepat implementasi tanpa melemahkan kontrol",
                "proof_points": ["kontrol risiko", "readiness assessment"],
            },
            client_internal_context="Akun klien sektor teknologi.",
            expert_bench_context={"expert_history_summary": "Riwayat internal mencakup arsitektur dan governance."},
            chapter_chain_context="Business case sudah dijelaskan di bab sebelumnya.",
        )

        self.assertIn("[EVIDENCE_CARD_SCHEMA]", brief)
        self.assertIn("fact | why_it_matters | source_lane | confidence | gap", brief)
        self.assertIn("[INTERNAL_DATA_AGENT]", brief)
        self.assertIn("[COMMERCIAL_STRATEGY_AGENT]", brief)
        self.assertIn("[TECHNICAL_SOLUTION_AGENT]", brief)
        self.assertIn("[RISK_COMPLIANCE_AGENT]", brief)
        self.assertIn("[EDITOR_MAIN_AGENT]", brief)
        self.assertIn("[EFFICIENCY_POLICY]", brief)
        self.assertIn("single model pass per chapter", brief)
        self.assertIn("reuse cached research_bundle", brief)
        self.assertIn("only accepted evidence cards", brief)
        self.assertIn("ai_readiness", [item["id"] for item in ProposalSupportMixin._chapter_specialist_agent_specs("c_6", ai_mode=True)])


if __name__ == "__main__":
    unittest.main()
