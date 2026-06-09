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
            "dokumen yang akan dibuat",
            "gunakan sebagai acuan",
        ]
        if "transformasi digital" in lowered and any(marker in lowered for marker in raw_helper_markers):
            return "Acuan Transformasi Digital BPK RI"
        if not text or any(marker in lowered for marker in raw_helper_markers):
            return fallback
        if len(text.split()) > 18:
            return fallback
        return text

    @staticmethod
    def _reader_summary_phrase(value, fallback, client=""):
        text = " ".join(str(value or "").split()).strip(" -;,.")
        lowered = text.lower()
        raw_fragments = [
            "manfaat keputusan yang perlu dibuktikan",
            "mandat kerja yang perlu diterjemahkan",
            "baseline kondisi berjalan",
            "prioritas peta jalan, dependensi",
            "eksekusi fase berjalan sesuai jadwal",
            "dokumen yang akan dibuat",
            "risiko delivery dapat",
        ]
        if not text or any(fragment in lowered for fragment in raw_fragments):
            if re.search(r"\b(bpk|badan pemeriksa keuangan)\b", client or "", flags=re.IGNORECASE):
                return fallback
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
    def _psa_phrase(value, fallback, client=""):
        return sanitize_reader_facing_sources(
            ExecutiveSummaryBuilder._reader_summary_phrase(value, fallback, client=client)
        )

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
        cleaned_lines = []
        for raw_line in str(markdown or "").splitlines():
            line = raw_line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            if re.match(r"^\s{0,3}#{1,6}\s+\d+\.\d+\s+", line):
                continue
            if re.match(r"^\s{0,3}#{1,6}\s+", line):
                continue
            if re.fullmatch(r"\[\[(GANTT|BAR|DONUT):.*\]\]", line, flags=re.IGNORECASE):
                continue
            if line.startswith("|"):
                continue
            if re.fullmatch(r"[-:\s|]+", line):
                continue
            cleaned_lines.append(raw_line)
        text = "\n".join(cleaned_lines)
        text = re.sub(r"\[\[(?:GANTT|BAR|DONUT):.*?\]\]", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\b(?:---\s+){1,}---\b", " ", text)
        text = re.sub(r"[*_`>|]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _is_structural_sentence(sentence):
        lowered = str(sentence or "").strip().lower()
        if not lowered:
            return True
        structural_patterns = [
            r"\b\d+\.\d+\b",
            r"^(\d+\.|[-*])\s+",
            r"\bsubbagian ini perlu\b",
            r"\blingkup kerja berikut sengaja ditegaskan\b",
            r"\bengagement lead\s*/\s*project director\b",
            r"\bstruktur tim proyek\b",
            r"\bwaktu pelaksanaan dan (?:deliverable|keluaran kerja)\b",
            r"\blingkup pekerjaan utama\b",
        ]
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in structural_patterns)

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
            if ExecutiveSummaryBuilder._is_structural_sentence(sentence):
                continue
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

    @staticmethod
    def _client_profile_clause(personalization_pack):
        profile = (personalization_pack or {}).get("client_profile") or {}
        if not isinstance(profile, dict):
            return ""
        location = str(profile.get("location") or "").strip()
        institution_type = str(profile.get("institution_type") or "").strip()
        parts = []
        if institution_type:
            parts.append(institution_type)
        if location:
            parts.append(f"berlokasi di {location}")
        if not parts:
            return ""
        return " Dalam konteks ini, profil klien dibaca sebagai " + " dan ".join(parts) + ", sehingga proposal perlu menjaga bahasa yang akuntabel, defensible, dan mudah diuji oleh sponsor."

    @staticmethod
    def _visual_projection_table(project_label, timeline_text, budget_text):
        rows = [
            ("Ruang lingkup", project_label),
            ("Estimasi waktu", timeline_text),
            ("Estimasi anggaran", budget_text),
            ("Fase awal", "Inisiasi, klarifikasi kebutuhan, dan baseline kondisi berjalan"),
            ("Fase rancangan", "Desain target, pendekatan kerja, dan validasi keputusan sponsor"),
            ("Fase finalisasi", "Penyelarasan keluaran kerja, kontrol mutu, dan rencana tindak lanjut"),
        ]
        body = "\n".join(
            f"| {component} | {sanitize_reader_facing_sources(summary)} |"
            for component, summary in rows
            if str(summary or "").strip()
        )
        return (
            "\n\nVisual Proyeksi Jadwal & Alokasi Anggaran\n"
            "| Komponen | Ringkasan |\n"
            "| --- | --- |\n"
            f"{body}"
        )

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
        profile_clause = cls._client_profile_clause(personalization_pack)

        content_summary = []
        bpk_context = bool(re.search(r"\b(bpk|badan pemeriksa keuangan)\b", client or "", flags=re.IGNORECASE))
        problem_fallback = (
            "Transformasi digital perlu memperkuat akuntabilitas publik, tata kelola data, ketertelusuran bukti, dan kontrol keputusan audit."
            if bpk_context else
            "Proposal perlu mengubah kebutuhan yang masih tersebar menjadi keputusan program yang jelas, terukur, dan siap dikendalikan."
        )
        scope_fallback = "Ruang lingkup dipusatkan pada baseline transformasi, desain target dan peta jalan prioritas, serta tata kelola eksekusi dengan quality gate yang eksplisit."
        approach_fallback = "Kerangka kerja dipakai sebagai alat bantu keputusan: arsitektur untuk target, tata kelola untuk kontrol, dan manajemen proyek untuk penerimaan hasil."
        problem_text = cls._reader_summary_phrase(digest.get("problem"), problem_fallback, client=client)
        scope_text = cls._reader_summary_phrase(digest.get("scope"), scope_fallback, client=client)
        approach_text = cls._reader_summary_phrase(digest.get("approach"), approach_fallback, client=client)
        solution_text = cls._reader_summary_phrase(digest.get("solution"), "", client=client) if digest.get("solution") else ""
        timeline_digest = cls._reader_summary_phrase(
            digest.get("timeline"),
            f"Jadwal {timeline_text} dibaca sebagai urutan fase yang harus menghasilkan keputusan, artefak, dan acceptance yang tidak tumpang tindih.",
            client=client,
        )
        commercial_text = cls._reader_summary_phrase(
            digest.get("commercial"),
            f"Basis komersial {budget_text} perlu dikaitkan dengan milestone, keluaran kerja, acceptance, dan change control agar nilainya transparan.",
            client=client,
        )
        problem_text = cls._psa_phrase(problem_text, problem_fallback, client=client)
        scope_text = cls._psa_phrase(scope_text, scope_fallback, client=client)
        approach_text = cls._psa_phrase(approach_text, approach_fallback, client=client)
        solution_text = cls._psa_phrase(solution_text, "", client=client) if solution_text else ""
        timeline_digest = cls._psa_phrase(timeline_digest, "", client=client)
        commercial_text = cls._psa_phrase(commercial_text, "", client=client)

        execution_notes = []
        if digest.get("governance"):
            execution_notes.append(f"- Tata kelola: {cls._psa_phrase(digest['governance'], '', client=client)}")
        execution_notes.extend([
            "- Kunci tiga fokus program sejak awal: baseline kondisi berjalan, rancangan target yang dapat diputuskan, dan mekanisme quality gate untuk penerimaan hasil.",
            "- Tetapkan sponsor keputusan, PIC harian, dan forum eskalasi agar perubahan prioritas tidak langsung mengubah ruang lingkup tanpa dasar tertulis.",
            "- Gunakan kapabilitas tim sebagai bukti kemampuan delivery, bukan sebagai daftar jabatan yang berdiri sendiri.",
        ])

        dashboard = cls._visual_projection_table(project_label, timeline_text, budget_text)

        summary = "\n\n".join([
            "# Ringkasan Eksekutif",
            "## Inti Keputusan\n"
            + f"{client} dapat menggunakan {project_label} sebagai dasar keputusan eksekusi untuk mencapai {outcome}. Proposal ini menempatkan ruang lingkup, keluaran kerja, jadwal {timeline_text}, dan anggaran {budget_text} sebagai satu paket keputusan.{profile_clause}"
            + dashboard,
            "## Situasi dan Masalah\n"
            + f"{problem_text} Karena itu, proposal perlu membuka pekerjaan dari konteks klien dan tekanan keputusan yang benar-benar perlu diselesaikan, bukan dari daftar aktivitas generik.",
            "## Solusi yang Direkomendasikan\n"
            + f"{scope_text} Pendekatan yang dipakai adalah {approach_text}."
            + (f" Keluaran yang dituju mencakup {solution_text}." if solution_text else ""),
            "## Nilai yang Dikejar\n"
            + f"Nilai utama yang dituju adalah {value_statement}. Bagi sponsor, nilai ini harus terasa pada prioritas yang lebih jelas, bukti keputusan yang lebih mudah diuji, dan ritme eksekusi yang tidak melebar dari ruang lingkup.",
            "## Prioritas Aksi\n"
            + "\n".join([
                f"- Jalankan jadwal {timeline_text} sebagai rangkaian fase yang menghasilkan keputusan, artefak, dan acceptance yang jelas: {timeline_digest}",
                f"- Kaitkan komersial {budget_text} dengan ruang lingkup, milestone, dan change control: {commercial_text}",
                *execution_notes[:2],
            ]),
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

        dashboard = cls._visual_projection_table(project_label, timeline_text, budget_text)

        summary = "\n\n".join([
            "# Ringkasan Eksekutif",
            "## Inti Keputusan\n"
            + f"{client} sebaiknya menyetujui arah {project_label} sekarang bila tujuan utamanya adalah {outcome}. Keputusan ini perlu mengikat mandat sponsor, ruang lingkup, target nilai, jadwal {timeline_text}, dan anggaran {budget_text}."
            + dashboard,
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
