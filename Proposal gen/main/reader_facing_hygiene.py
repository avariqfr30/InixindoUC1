"""Reader-facing source and language cleanup for generated proposals."""
import re

_REPLACEMENTS = (
    (r"\bNama\s+Perusahaan\s+Klien\s*:", "catatan klien:"),
    (r"\bKonteks\s+akun\s+internal\b", "profil klien yang tersedia"),
    (r"\b(?:detail\s+)?Identitas\s+akun\s+internal\b", "profil klien yang tersedia"),
    (r"\bGunakan\s+informasi\s+ini\b[^.]*\.?", ""),
    (r"\bReferenceAccount\s+mencatat\b", "catatan klien menunjukkan"),
    (r"\bsource\s*=\s*(?:https?://\S+|/api/[A-Za-z0-9_./-]+)?", ""),
    (r"\bdataset[_\s-]*code\s*=\s*ConsultantProjectExpertHistory\b", "riwayat pengalaman konsultan"),
    (r"\bDirangkum\s+dari\s+sumber[^:]*:\s*", "Berdasarkan catatan pendukung yang sudah dipadatkan: "),
    (r"\bProblem\s*,\s*Opportunity\s*,\s*Directive\b", "kebutuhan prioritas yang perlu dipertegas"),
    (r"\((?:Data Internal),\s*(?:\d{4}|n\.d\.)\)", ""),
    (r"\[[A-Z_]+(?:\s*:?\s*[A-Za-z0-9_-]*)?\]", ""),
    (r"https?://\S+", ""),
    (r"/api/[A-Za-z0-9_./-]+", ""),
    (r"APIDog", "konteks yang tersedia"),
    (r"Internal API", "konteks yang tersedia"),
    (r"Data Internal", "konteks yang tersedia"),
    (r"ReferenceDataset", "konteks yang tersedia"),
    (r"ReferenceAccount", "konteks akun"),
    (r"ConsultantProjectExpertHistory", "riwayat pengalaman konsultan"),
    (r"\bdataset\s+(?:code|name)\b", "konteks yang tersedia"),
    (r"\bdataset\b", "konteks yang tersedia"),
    (r"\bendpoint\b", "kanal masukan"),
    (r"\bsource-of-truth\b", "dasar rujukan"),
    (r"\bcache\b", "penyimpanan sementara"),
    (r"\bsync\b", "pemutakhiran"),
    (r"\bagents?\b", "tim penyusun"),
    (r"\bworkflow\b", "alur kerja"),
    (r"\bevidence\s+cards?\b", "catatan pendukung"),
    (r"\bconfidence\s+labels?\b", "penanda keyakinan"),
)

_LANGUAGE_REPLACEMENTS = (
    (r"\bExecutive Summary\b", "Ringkasan Eksekutif"),
    (r"\bKey Findings\b", "Temuan Utama"),
    (r"\bRecommendations?\b", "Rekomendasi"),
    (r"\bPain Points\b", "Titik Masalah"),
    (r"\bcurrent state\b", "kondisi saat ini"),
    (r"\btarget state\b", "kondisi target"),
    (r"\broot cause\b", "akar masalah"),
    (r"\bscope\b", "ruang lingkup"),
    (r"\bowner\b", "penanggung jawab"),
    (r"\boutcome\b", "hasil"),
    (r"\bdeliverables?\b", "keluaran kerja"),
    (r"\bmilestones?\b", "tonggak kerja"),
    (r"\bworkflow\b", "alur kerja"),
    (r"\bflow\b", "alur"),
    (r"\bdashboard\b", "dasbor"),
    (r"\binsights?\b", "wawasan"),
    (r"\bpreview\b", "pratinjau"),
    (r"\bgenerate\b", "susun"),
    (r"\bupload\b", "unggah"),
    (r"\bframework\b", "kerangka kerja"),
    (r"\bbest practice\b", "praktik baik"),
    (r"\bbest practise\b", "praktik baik"),
    (r"\bquality gate\b", "gerbang mutu"),
    (r"\broadmap\b", "peta jalan"),
)

_ITALIC_TECHNICAL_TERMS = (
    "API",
    "OSINT",
    "RAG",
    "UAT",
    "Go-Live",
    "cut-over",
)


def _protect_code_spans(text):
    protected = []

    def replace(match):
        protected.append(match.group(0))
        return f"@@CODE_SPAN_{len(protected) - 1}@@"

    return re.sub(r"`[^`]+`", replace, text), protected


def _restore_code_spans(text, protected):
    for index, value in enumerate(protected):
        text = text.replace(f"@@CODE_SPAN_{index}@@", value)
    return text


def sanitize_reader_facing_language(raw_text):
    text, protected = _protect_code_spans(str(raw_text or ""))
    for pattern, replacement in _LANGUAGE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for term in _ITALIC_TECHNICAL_TERMS:
        pattern = rf"(?<![\w*]){re.escape(term)}(?![\w*])"
        text = re.sub(pattern, f"*{term}*", text)
    return _restore_code_spans(text, protected)


def sanitize_reader_facing_sources(raw_text):
    text = str(raw_text or "")
    text = re.sub(r"https?://\S+", "", text)
    for pattern, replacement in _REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = sanitize_reader_facing_language(text)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()
