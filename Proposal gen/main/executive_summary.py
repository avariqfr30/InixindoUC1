"""Deterministic board-style executive summary for proposal documents."""
import re

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

    @staticmethod
    def _safe_project_label(project, fallback="inisiatif yang diusulkan"):
        text = " ".join(str(project or "").split())
        lowered = text.lower()
        raw_helper_markers = [
            "konteks akun internal",
            "gunakan informasi ini",
            "identitas akun internal",
            "bukan sebagai rumusan tujuan proyek",
            "data internal",
            "referenceaccount",
        ]
        if not text or any(marker in lowered for marker in raw_helper_markers):
            return fallback
        if len(text.split()) > 18:
            return fallback
        return text

    @staticmethod
    def _safe_goal_label(project_goal, fallback="kebutuhan prioritas yang perlu dipertegas"):
        text = " ".join(str(project_goal or "").split())
        lowered = text.lower()
        raw_choice_sets = [
            "problem, opportunity, directive",
            "problem opportunity directive",
            "problem",
            "opportunity",
            "directive",
        ]
        raw_helper_markers = [
            "pain points",
            "konteks klien",
            "konteks akun internal",
            "identitas akun internal",
            "data internal",
            "referenceaccount",
            "gunakan informasi ini",
        ]
        if not text:
            return fallback
        if lowered in raw_choice_sets or any(marker in lowered for marker in raw_helper_markers):
            return fallback
        if len(text.split()) > 22:
            return fallback
        return text

    @staticmethod
    def _scope_focus(selected_chapters):
        chapter_titles = [
            str(chapter.get("title") or "").strip()
            for chapter in (selected_chapters or [])
            if isinstance(chapter, dict) and str(chapter.get("title") or "").strip()
        ]
        if not chapter_titles:
            return "konteks klien, masalah utama, pendekatan kerja, rencana pelaksanaan, tata kelola, dan model pembiayaan"
        themes = []
        mapping = [
            ("KONTEKS", "pembacaan konteks klien"),
            ("PERMASALAHAN", "penajaman masalah utama"),
            ("KLASIFIKASI", "prioritas kebutuhan"),
            ("RUANG LINGKUP", "ruang lingkup pekerjaan"),
            ("PENDEKATAN", "pendekatan dan kerangka acuan"),
            ("METODOLOGI", "metodologi pelaksanaan"),
            ("SOLUTION", "desain solusi"),
            ("TIMELINE", "jadwal kerja"),
            ("TATA KELOLA", "tata kelola proyek"),
            ("PROFIL PERUSAHAAN", "bukti kapabilitas perusahaan"),
            ("TENAGA AHLI", "struktur tenaga ahli"),
            ("PEMBIAYAAN", "model pembiayaan"),
            ("PENUTUP", "langkah tindak lanjut"),
        ]
        for title in chapter_titles:
            normalized = title.upper()
            for token, label in mapping:
                if token in normalized and label not in themes:
                    themes.append(label)
                    break
        return ExecutiveSummaryBuilder._join_items(
            themes[:6],
            "konteks klien, masalah utama, pendekatan kerja, rencana pelaksanaan, tata kelola, dan model pembiayaan",
        )

    @staticmethod
    def _plain_text(markdown):
        text = re.sub(r"```.*?```", " ", str(markdown or ""), flags=re.DOTALL)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"[*_`>|]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _brief(markdown, max_words=42):
        text = ExecutiveSummaryBuilder._plain_text(markdown)
        if not text:
            return ""
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        generic_patterns = [
            r"pembahasan pada bagian ini",
            r"fokus utama harus tetap",
            r"risiko, dependensi, dan indikator hasil",
            r"bagian ini perlu menjelaskan",
        ]
        substantive_tokens = (
            "keluaran", "manfaat", "risiko", "batas", "milestone", "tonggak",
            "governance", "tata kelola", "kpi", "scope", "ruang lingkup",
            "arsitektur", "prioritas", "indikator", "keputusan",
        )
        ranked = []
        for index, sentence in enumerate(sentences):
            lowered = sentence.lower()
            if any(re.search(pattern, lowered) for pattern in generic_patterns):
                continue
            score = sum(1 for token in substantive_tokens if token in lowered)
            ranked.append((score, -index, sentence))
        ranked.sort(reverse=True)
        chosen = [item[2] for item in ranked[:2]] if ranked else sentences[:1]
        words = " ".join(chosen).split()
        if len(words) <= max_words:
            return " ".join(chosen).strip()
        return " ".join(words[:max_words]).rstrip(" ,;:") + "."

    @staticmethod
    def _chapter_digest(selected_chapters, chapter_outputs):
        outputs = chapter_outputs or {}
        digest = {}
        aliases = [
            ("scope", ("RUANG LINGKUP",)),
            ("problem", ("PERMASALAHAN",)),
            ("classification", ("KLASIFIKASI",)),
            ("approach", ("PENDEKATAN", "METODOLOGI")),
            ("solution", ("SOLUTION", "SOLUSI")),
            ("timeline", ("TIMELINE",)),
            ("governance", ("TATA KELOLA",)),
            ("team", ("STRUKTUR", "TEAM", "TENAGA AHLI")),
            ("commercial", ("PEMBIAYAAN",)),
        ]
        for chapter in selected_chapters or []:
            title = str(chapter.get("title") or "").upper() if isinstance(chapter, dict) else ""
            content = outputs.get(chapter.get("id")) if isinstance(chapter, dict) else ""
            brief = ExecutiveSummaryBuilder._brief(content)
            if not brief:
                continue
            for key, tokens in aliases:
                if key not in digest and any(token in title for token in tokens):
                    digest[key] = brief
                    break
        return digest

    @classmethod
    def build_from_chapters(cls, client, project, project_goal="", timeline="", budget="", value_map=None, personalization_pack=None, selected_chapters=None, chapter_outputs=None):
        digest = cls._chapter_digest(selected_chapters, chapter_outputs)
        if not digest:
            return cls.build(
                client=client,
                project=project,
                project_goal=project_goal,
                timeline=timeline,
                budget=budget,
                value_map=value_map,
                personalization_pack=personalization_pack,
                selected_chapters=selected_chapters,
            )

        value_map = value_map or {}
        personalization_pack = personalization_pack or {}
        project_label = cls._safe_project_label(project)
        outcome = cls._value(value_map.get("primary_outcome") or value_map.get("business_outcome"), project_goal or "hasil kerja yang terukur")
        value_statement = cls._value(value_map.get("value_statement"), f"{outcome} dicapai melalui ruang lingkup yang terkendali dan keluaran kerja yang dapat dievaluasi")
        timeline_text = cls._value(timeline, "jadwal yang disepakati")
        budget_text = cls._value(budget, "kesepakatan komersial")

        content_summary = []
        if digest.get("problem"):
            content_summary.append(f"- Masalah utama: {digest['problem']}")
        if digest.get("scope"):
            content_summary.append(f"- Ruang lingkup: {digest['scope']}")
        if digest.get("solution"):
            content_summary.append(f"- Solusi dan keluaran: {digest['solution']}")
        if digest.get("approach"):
            content_summary.append(f"- Pendekatan kerja: {digest['approach']}")
        if digest.get("timeline"):
            content_summary.append(f"- Jadwal: {digest['timeline']}")
        if digest.get("commercial"):
            content_summary.append(f"- Komersial: {digest['commercial']}")
        if not content_summary:
            content_summary.append(f"- Fokus proposal: {cls._scope_focus(selected_chapters)}.")

        execution_notes = []
        if digest.get("governance"):
            execution_notes.append(f"- Tata kelola: {digest['governance']}")
        if digest.get("team"):
            execution_notes.append(f"- Kapabilitas pelaksana: {digest['team']}")
        if not execution_notes:
            execution_notes = [
                "- Tetapkan sponsor, penanggung jawab harian, dan forum keputusan.",
                "- Kunci ruang lingkup, keluaran kerja, dan indikator keberhasilan sebelum pekerjaan berjalan penuh.",
            ]

        summary = "\n\n".join([
            "# Ringkasan Eksekutif",
            "## Inti Keputusan\n"
            + f"{client} dapat menggunakan {project_label} sebagai dasar keputusan eksekusi untuk mencapai {outcome}. Proposal ini menempatkan ruang lingkup, keluaran kerja, jadwal {timeline_text}, dan anggaran {budget_text} sebagai satu paket keputusan.",
            "## Ringkasan Isi Proposal\n" + "\n".join(content_summary[:6]),
            "## Nilai yang Dikejar\n"
            + f"Nilai utama yang dituju adalah {value_statement}. Narasi proposal disusun dari bab yang sudah selesai sehingga ringkasan ini mencerminkan isi dokumen, bukan pengulangan mentah dari isian formulir.",
            "## Prioritas Eksekusi\n" + "\n".join(execution_notes[:3]),
            "## Keputusan Berikutnya\n"
            + f"Langkah berikutnya adalah menyetujui mandat awal, mengonfirmasi batas pekerjaan, lalu menetapkan jadwal {timeline_text} dan anggaran {budget_text} sebagai dasar pelaksanaan.",
        ])
        return sanitize_reader_facing_sources(summary)

    @classmethod
    def build(cls, client, project, project_goal="", timeline="", budget="", value_map=None, personalization_pack=None, selected_chapters=None):
        value_map = value_map or {}
        personalization_pack = personalization_pack or {}
        project_label = cls._safe_project_label(project)
        goal_label = cls._safe_goal_label(project_goal)
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
        scope_focus = cls._scope_focus(selected_chapters)
        timeline_text = cls._value(timeline, "jadwal yang disepakati")
        budget_text = cls._value(budget, "kesepakatan komersial")
        summary = "\n\n".join([
            "# Ringkasan Eksekutif",
            "## Inti Keputusan\n"
            + f"{client} sebaiknya menyetujui arah {project_label} sekarang bila tujuan utamanya adalah {outcome}. Keputusan ini perlu mengikat mandat sponsor, ruang lingkup, target nilai, jadwal {timeline_text}, dan anggaran {budget_text}.",
            "## Situasi dan Masalah Klien\n"
            + f"Dalam konteks {industry}, {client} membutuhkan rencana kerja yang menerjemahkan {goal_label} menjadi keputusan eksekusi yang konkret, bukan sekadar daftar aktivitas.",
            "## Solusi yang Direkomendasikan\n"
            + f"Rekomendasinya adalah menjalankan {project_label} sebagai program bertahap dengan fokus pada {scope_focus}, ukuran keberhasilan sejak awal, dan ritme pengambilan keputusan yang jelas.",
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
