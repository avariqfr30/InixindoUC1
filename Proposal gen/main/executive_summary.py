"""Deterministic board-style executive summary for proposal documents."""
from .reader_facing_hygiene import sanitize_reader_facing_sources


class ExecutiveSummaryBuilder:
    @staticmethod
    def _value(value, fallback):
        cleaned = str(value or "").strip()
        return cleaned if cleaned else fallback

    @staticmethod
    def _join_items(items, fallback):
        cleaned = [str(item or "").strip() for item in (items or []) if str(item or "").strip()]
        if not cleaned:
            return fallback
        if len(cleaned) == 1:
            return cleaned[0]
        return ", ".join(cleaned[:-1]) + ", dan " + cleaned[-1]

    @classmethod
    def build(cls, client, project, project_goal="", timeline="", budget="", value_map=None, personalization_pack=None, selected_chapters=None):
        value_map = value_map or {}
        personalization_pack = personalization_pack or {}
        outcome = cls._value(value_map.get("primary_outcome") or value_map.get("business_outcome"), project_goal or "hasil kerja yang terukur")
        value_statement = cls._value(value_map.get("value_statement"), f"{outcome} dapat dicapai dengan ruang lingkup, tata kelola, dan ukuran keberhasilan yang jelas")
        industry = cls._value(personalization_pack.get("industry"), "konteks bisnis klien")
        proof_points = cls._join_items(
            value_map.get("proof_points") or personalization_pack.get("proof_points"),
            "pengalaman pelaksanaan, indikator keberhasilan yang disepakati, dan pengendalian mutu selama implementasi",
        )
        kpi_focus = cls._join_items(
            personalization_pack.get("kpi_blueprint"),
            "indikator keberhasilan bisnis dan operasional yang disepakati bersama",
        )
        selected_titles = cls._join_items(
            [
                chapter.get("title", "")
                for chapter in (selected_chapters or [])
                if isinstance(chapter, dict)
            ],
            "ruang lingkup proposal yang telah diprioritaskan",
        )
        timeline_text = cls._value(timeline, "jadwal yang disepakati")
        budget_text = cls._value(budget, "kesepakatan komersial")
        summary = "\n\n".join([
            "# Ringkasan Eksekutif",
            "## Inti Keputusan\n"
            + f"{client} sebaiknya menyetujui arah {project} sekarang bila tujuan utamanya adalah {outcome}. Keputusan ini perlu mengikat mandat sponsor, ruang lingkup, target nilai, jadwal {timeline_text}, dan anggaran {budget_text}.",
            "## Situasi dan Masalah Klien\n"
            + f"Dalam konteks {industry}, {client} membutuhkan rencana kerja yang menerjemahkan kebutuhan {project_goal or 'bisnis prioritas'} menjadi keputusan eksekusi yang konkret, bukan sekadar daftar aktivitas.",
            "## Solusi yang Direkomendasikan\n"
            + f"Rekomendasinya adalah menjalankan {project} sebagai program bertahap dengan fokus pada {selected_titles}, ukuran keberhasilan sejak awal, dan ritme pengambilan keputusan yang jelas.",
            "## Nilai dan Bukti\n"
            + f"Nilai utama yang dikejar adalah {value_statement}. Dasar penguatnya mencakup {proof_points}, dengan perhatian pada {kpi_focus}.",
            "## Prioritas Eksekusi\n"
            + "- Tetapkan sponsor, penanggung jawab harian, dan forum keputusan.\n"
            + "- Kunci ruang lingkup, keluaran kerja, dan indikator keberhasilan sebelum pekerjaan berjalan penuh.\n"
            + "- Jaga evaluasi berkala agar hasil tetap terhubung dengan kebutuhan bisnis klien.",
            "## Risiko yang Perlu Dikendalikan\n"
            + "Risiko utama berada pada pelebaran ruang lingkup, lambatnya keputusan lintas pihak, kesiapan data atau materi pendukung, dan perubahan prioritas sebelum hasil awal terlihat.",
            "## Keputusan Berikutnya\n"
            + f"Langkah berikutnya adalah menyetujui mandat awal, mengonfirmasi ruang lingkup final, lalu menetapkan jadwal {timeline_text} dan anggaran {budget_text} sebagai dasar pelaksanaan.",
        ])
        return sanitize_reader_facing_sources(summary)
