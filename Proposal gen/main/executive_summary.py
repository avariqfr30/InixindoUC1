"""Deterministic board-style executive summary for proposal documents."""
from .reader_facing_hygiene import sanitize_reader_facing_sources


class ExecutiveSummaryBuilder:
    @staticmethod
    def _value(value, fallback):
        cleaned = str(value or "").strip()
        return cleaned if cleaned else fallback

    @classmethod
    def build(cls, client, project, project_goal="", timeline="", budget="", value_map=None, personalization_pack=None, selected_chapters=None):
        value_map = value_map or {}
        outcome = cls._value(value_map.get("primary_outcome") or value_map.get("business_outcome"), project_goal or "hasil kerja yang terukur")
        summary = "\n\n".join([
            "# Executive Summary",
            "## Keputusan Utama\n" + f"{client} perlu menentukan apakah {project} layak dijalankan sekarang untuk mencapai {outcome} dengan tata kelola, ruang lingkup, dan komitmen eksekusi yang jelas.",
            "## Mengapa Ini Penting Sekarang\n" + f"Inisiatif ini berhubungan langsung dengan prioritas bisnis: memperjelas masalah, mengurangi risiko eksekusi, dan memastikan investasi menghasilkan manfaat yang dapat dilacak.",
            "## Prioritas Eksekusi\n- Sepakati ruang lingkup dan indikator keberhasilan sejak awal.\n- Tetapkan owner keputusan, owner teknis, dan ritme evaluasi.\n- Pastikan deliverable fokus pada outcome, bukan hanya aktivitas.",
            "## Risiko Jika Ditunda\n" + "Penundaan berisiko membuat kebutuhan bisnis semakin melebar, biaya koordinasi meningkat, dan momentum sponsor melemah sebelum fondasi implementasi siap.",
            "## Rekomendasi Langkah Berikutnya\n" + f"Mulai dengan alignment meeting, validasi scope, lalu finalisasi rencana kerja {timeline or 'sesuai jadwal yang disepakati'} dan anggaran {budget or 'sesuai kesepakatan komersial'}.",
        ])
        return sanitize_reader_facing_sources(summary)
