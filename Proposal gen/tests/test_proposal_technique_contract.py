"""Regression coverage for the proposal technique flow."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_goals_drive_background_objectives_and_scope_seed() -> None:
    from main.proposal_technique import build_proposal_technique_contract

    contract = build_proposal_technique_contract(
        client="Ajinomoto Indonesia",
        goals="Meningkatkan tata kelola layanan digital dan kepatuhan data.",
        customer_notes="Sponsor ingin prioritas kerja yang jelas.",
        existing_condition="Proses layanan masih berbeda antar unit.",
        frameworks="ISO, Regulasi",
    )

    assert "tata kelola layanan digital" in contract["background_basis"].lower()
    assert "kepatuhan data" in contract["objective_basis"].lower()
    assert "antar unit" in contract["scope_basis"].lower()
    assert contract["scope_contract_seed"]["in_scope"]
    assert contract["framework_basis"]
    assert contract["methodology_basis"]


def test_scope_seed_excludes_account_metadata_from_commitments() -> None:
    from main.proposal_technique import build_proposal_technique_contract

    contract = build_proposal_technique_contract(
        client="Accelbyte",
        goals="Menyusun roadmap AI support.",
        customer_notes="Konteks akun internal menempatkan Accelbyte di Yogyakarta.",
        existing_condition="Belum ada quality gate untuk eksperimen AI.",
        frameworks="Responsible AI",
    )

    joined_scope = " ".join(contract["scope_contract_seed"]["in_scope"]).lower()
    assert "quality gate" in joined_scope
    assert "yogyakarta" not in joined_scope


def test_framework_rows_include_scope_basis_when_scope_contract_is_available() -> None:
    from main.proposal_support import ProposalSupportMixin

    scope_contract = {
        "in_scope": ["asesmen tata kelola data", "roadmap prioritas", "quality gate"],
        "out_of_scope": ["implementasi penuh aplikasi"],
    }

    rows = ProposalSupportMixin._framework_reference_rows(
        "ISO, Regulasi",
        "Strategic",
        context_hint="kepatuhan data",
        scope_contract=scope_contract,
    )

    joined = " ".join(item["relevansi"] for item in rows).lower()
    assert "asesmen tata kelola data" in joined or "roadmap prioritas" in joined


def test_methodology_rows_use_scope_and_avoid_out_of_scope_commitments() -> None:
    from main.proposal_support import ProposalSupportMixin

    scope_contract = {
        "in_scope": ["asesmen tata kelola data", "roadmap prioritas", "quality gate"],
        "out_of_scope": ["implementasi penuh aplikasi"],
    }

    rows = ProposalSupportMixin._methodology_rows(
        "Strategic",
        "Konsultan",
        "3 bulan",
        scope_contract=scope_contract,
    )

    joined = " ".join(
        " ".join([item["tujuan"], item["keluaran"], item["quality_gate"]])
        for item in rows
    ).lower()
    assert "quality gate" in joined
    assert "implementasi penuh aplikasi" not in joined


def test_kak_methodology_render_adopts_reference_style_without_replacing_structure() -> None:
    from main.config import KAK_RESPONSE_STRUCTURE
    from main.proposal_support import ProposalSupportMixin
    from main.proposal_technique import build_proposal_technique_contract

    chapter = next(item for item in KAK_RESPONSE_STRUCTURE if item["id"] == "k_3")
    contract = build_proposal_technique_contract(
        client="KPK",
        goals="Menyusun tata kelola dan akselerasi implementasi transformasi digital.",
        customer_notes="KAK meminta tahapan kerja, framework, dan hasil yang dapat divalidasi.",
        existing_condition="Backlog dan pengendalian portofolio perlu ditajamkan.",
        frameworks="TOGAF, COBIT, ITIL, PMBOK",
    )

    content = ProposalSupportMixin()._render_structured_chapter(
        chapter=chapter,
        client="KPK",
        project="Tata Kelola dan Akselerasi Implementasi Transformasi Digital",
        budget="",
        service_type="konsultansi tata kelola",
        project_goal="Menyusun tata kelola dan akselerasi implementasi transformasi digital.",
        project_type="Strategic",
        timeline="5 bulan",
        notes="KAK meminta tahapan kerja, framework, backlog, quality gate, dan deliverable yang dapat divalidasi.",
        regulations="TOGAF, COBIT, ITIL, PMBOK",
        firm_data={},
        firm_profile={},
        research_bundle={},
        personalization_pack={},
        value_map={},
        proposal_mode="kak_response",
        supporting_context={"proposal_technique_contract": contract},
    )

    assert "## 3.1 Pemilihan Framework" in content
    assert "## 3.2 Metodologi Pekerjaan" in content
    assert "Kriteria pemilihan framework" in content
    assert "integrated framework stack" in content
    assert "| Kerangka / Acuan | Ringkasnya | Mengapa Dipakai pada Pekerjaan Ini | Pembeda Peran |" in content
    assert "Pengendalian mutu dan risiko pelaksanaan" in content
    assert "| Risiko Utama | Dampak terhadap Pekerjaan | Mitigasi Pelaksanaan |" in content
    assert "Ruang lingkup melebar" in content
    assert "Quality gate" in content or "quality gate" in content


def test_kak_facilities_and_innovations_are_operational_not_cosmetic() -> None:
    from main.proposal_support import ProposalSupportMixin

    facilities = ProposalSupportMixin._kak_facility_rows(
        {"credential_highlights": "sertifikasi COBIT dan pengalaman manajemen proyek"},
        "konsultansi tata kelola",
    )
    innovations = ProposalSupportMixin._kak_innovation_rows(
        "KPK",
        "Strategic",
    )

    facility_text = " ".join(" ".join(row.values()) for row in facilities).lower()
    innovation_text = " ".join(" ".join(row.values()) for row in innovations).lower()

    assert "kanban/action tracker" in facility_text
    assert "risk register" in facility_text
    assert "terenkripsi" in facility_text
    assert "indeks kesehatan program atau backlog" in innovation_text
    assert "knowledge transfer by design" in innovation_text


def test_kak_deliverables_include_acceptance_and_milestone_contract() -> None:
    from main.proposal_support import ProposalSupportMixin

    rows = ProposalSupportMixin._kak_deliverable_acceptance_rows(
        "Strategic",
        "konsultansi tata kelola",
        "5 bulan",
    )

    joined = " ".join(" ".join(row.values()) for row in rows).lower()

    assert "keputusan" in joined
    assert "milestone fase" in joined
    assert "persetujuan" in joined
    assert "notulen" in joined or "daftar hadir" in joined
