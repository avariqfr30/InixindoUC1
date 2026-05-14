"""Reader-facing source disclosure cleanup for generated proposals."""
import re

_REPLACEMENTS = (
    (r"\((?:Data Internal),\s*(?:\d{4}|n\.d\.)\)", ""),
    (r"APIDog", "konteks yang tersedia"),
    (r"Internal API", "konteks yang tersedia"),
    (r"Data Internal", "konteks yang tersedia"),
    (r"ReferenceAccount", "konteks akun"),
    (r"ConsultantProjectExpertHistory", "riwayat pengalaman konsultan"),
    (r"dataset", "konteks yang tersedia"),
)


def sanitize_reader_facing_sources(raw_text):
    text = str(raw_text or "")
    text = re.sub(r"https?://\S+", "", text)
    for pattern, replacement in _REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()
