"""Proposal-specific evidence, voice, and editorial quality helpers."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


EXCLUDED_DATASETS = {"FinanceInvoice", "ProjectStandards"}


def compact_text(value: Any, max_words: int = 18) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def dataset_role(dataset_code: str) -> str:
    return {
        "ConsultantProjectExpertHistory": "bukti pengalaman proyek dan tenaga ahli",
        "ReferenceAccount": "profil klien dan konteks relasi",
        "EmployeeExpertise": "kapabilitas orang yang bisa dipertanggungjawabkan",
        "ReferenceFramework": "kerangka kerja yang menjelaskan pendekatan",
    }.get(str(dataset_code or "").strip(), "konteks pendukung")


def build_evidence_card(
    dataset_code: str,
    fact: Any,
    *,
    source_date: str = "",
    confidence: str = "medium",
) -> dict[str, str]:
    """Keep provenance machine-readable while exposing only natural reader text."""
    code = str(dataset_code or "").strip()
    if code in EXCLUDED_DATASETS:
        return {}
    reader_text = compact_text(fact, 28)
    return {
        "dataset_code": code,
        "dataset_role": dataset_role(code),
        "reader_text": reader_text,
        "source_date": str(source_date or "").strip(),
        "confidence": str(confidence or "medium").strip().lower(),
    }


def compact_proposal_records(
    records: Iterable[dict[str, Any]] | None,
    dataset_code: str,
    limit: int = 6,
) -> list[dict[str, str]]:
    dataset_code = str(dataset_code or "").strip()
    if dataset_code in EXCLUDED_DATASETS:
        return []
    cards: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if dataset_code == "ConsultantProjectExpertHistory":
            key = (
                str(record.get("topic") or "").strip().lower(),
                str(record.get("position_name") or "").strip().lower(),
            )
            title = record.get("topic") or record.get("project_name") or record.get("entity")
            detail = (
                f"{record.get('position_name') or 'peran'} pada "
                f"{record.get('project_name') or record.get('entity') or 'proyek relevan'}"
            )
        elif dataset_code == "ReferenceAccount":
            key = tuple(
                str(record.get(name) or "").strip().lower()
                for name in ("company_segment", "company_sub_segment", "company_category_name")
            )
            title = record.get("company_name") or "profil klien"
            detail = " / ".join(
                str(record.get(name) or "").strip()
                for name in ("company_segment", "company_sub_segment", "company_province_name")
                if record.get(name)
            )
        elif dataset_code == "EmployeeExpertise":
            key = (str(record.get("employee_name") or "").strip().lower(),)
            title = record.get("employee_name") or "tenaga ahli"
            detail = "sertifikasi dan pengalaman dipadatkan sebagai bukti kapabilitas, bukan daftar mentah"
        elif dataset_code == "ReferenceFramework":
            key = (
                str(record.get("value") or "").strip().lower(),
                str(record.get("label") or "").strip().lower(),
            )
            title = record.get("label") or record.get("value") or "kerangka kerja"
            detail = record.get("description") or record.get("use_cases") or ""
        else:
            key = tuple(sorted((str(k), str(v)[:40]) for k, v in record.items())[:3])
            title = next(
                (record.get(k) for k in ("name", "label", "title", "project_name") if record.get(k)),
                dataset_code,
            )
            detail = " ".join(str(v) for v in list(record.values())[:3])
        if key in seen or not any(key):
            continue
        seen.add(key)
        cards.append(
            {
                "dataset": dataset_code,
                "role": dataset_role(dataset_code),
                "title": compact_text(title, 12),
                "detail": compact_text(detail, 22),
                "writing_use": "jadikan bukti, makna, atau batasan; jangan disalin sebagai isi tabel panjang",
            }
        )
        if len(cards) >= limit:
            break
    return cards


def proposal_voice_rules() -> list[str]:
    return [
        "Tulis seperti konsultan senior yang menjelaskan ke pembaca campuran, bukan seperti template tender.",
        "Mulai paragraf dari keputusan atau dampak pembaca, lalu baru bukti teknis.",
        "Jika ada istilah teknis, jelaskan sekali dengan konsekuensi bisnisnya.",
        "Hindari daftar kemampuan berulang; pilih bukti paling relevan dan jelaskan mengapa cocok.",
        "Gunakan OSINT untuk memperjelas konteks dan urgensi, bukan menggantikan bukti internal.",
    ]


def _paragraph_openings(text: Any, width: int = 4) -> Counter[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+", str(text or "")) if part.strip()]
    signatures = []
    for part in paragraphs:
        words = re.findall(r"[a-z0-9]+", part.lower())[:width]
        if not words or all(word.isdigit() for word in words):
            continue
        signatures.append(" ".join(words))
    return Counter(signatures)


def assess_proposal_style(text: Any) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", str(text or "").lower())
    findings: list[str] = []
    if any(phrase in normalized for phrase in ("dalam rangka pelaksanaan", "sehubungan dengan hal tersebut", "dengan demikian maka")):
        findings.append("template_language")
    if any(count >= 3 and opening for opening, count in _paragraph_openings(text).items()):
        findings.append("repeated_openings")
    return {"passed": not findings, "findings": findings}



SPINE_CONNECTOR_TERMS = (
    "melanjutkan", "menjadi dasar", "berangkat dari", "menghubungkan",
    "arah berikutnya", "dibaca sebagai kelanjutan", "diturunkan dari", "dengan dasar",
    "dari pijakan", "rangkaian", "pembaca yang", "keterhubungan",
    "masukan pada", "jembatan", "kerangka pada", "temuan dalam",
    "kaitan dengan", "dari titik", "pembahasan kemudian", "hasil pembacaan",
    "bab sesudahnya", "keterkaitan", "keputusan yang", "dampak bagian", "bagian berikutnya",
)


def _plain_document_text(value: Any) -> str:
    text = re.sub(r"\[\[(?:CHART|PIE|FLOW|GANTT|DASHBOARD):.*?\]\]", " ", str(value or ""))
    text = re.sub(r"[#*`>|_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_key(title: Any) -> str:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", str(title or "").lower())
    stop = {"dan", "atau", "the", "of", "untuk", "dengan", "yang", "ini"}
    return " ".join(word for word in words if word not in stop)[:80]


def _has_spine_connector(text: Any, *titles: Any) -> bool:
    lowered = _plain_document_text(text).lower()
    return any(term in lowered for term in SPINE_CONNECTOR_TERMS)


def _opening_signatures(text: Any, width: int = 3) -> Counter[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+", str(text or "")) if part.strip()]
    signatures = []
    for part in paragraphs:
        words = re.findall(r"[a-z0-9]+", part.lower())[:width]
        if not words or all(word.isdigit() for word in words):
            continue
        signatures.append(" ".join(words))
    return Counter(signatures)


def evaluate_proposal_document_spine(chapter_outputs: dict[str, str], selected_chapters: list[dict[str, Any]]) -> dict[str, Any]:
    categories: set[str] = set()
    findings: list[str] = []
    ordered = [chapter for chapter in selected_chapters or [] if chapter.get("id") in (chapter_outputs or {})]
    combined_openings = Counter()
    for chapter in ordered:
        combined_openings.update(_opening_signatures(chapter_outputs.get(chapter.get("id"), "")))
    repeated_global = [opening for opening, count in combined_openings.items() if opening and count >= 4]
    if repeated_global:
        categories.add("global_repeated_opening_spine")
        findings.append("Dokumen masih memakai pembuka paragraf berulang lintas bab: " + ", ".join(repeated_global[:4]) + ".")

    for index, chapter in enumerate(ordered):
        chapter_id = chapter.get("id")
        title = chapter.get("title") or chapter_id
        content = chapter_outputs.get(chapter_id, "")
        plain = _plain_document_text(content)
        if index > 0:
            previous_title = ordered[index - 1].get("title") or ordered[index - 1].get("id")
            if not _has_spine_connector(plain[:900], previous_title, title):
                categories.add("missing_previous_handoff")
                findings.append(f"{title} belum mengikat pembaca ke bab sebelumnya.")
        if index < len(ordered) - 1:
            next_title = ordered[index + 1].get("title") or ordered[index + 1].get("id")
            if not _has_spine_connector(plain[-900:], title, next_title):
                categories.add("missing_next_handoff")
                findings.append(f"{title} belum menyiapkan alasan menuju bab berikutnya.")
        if any(count >= 3 and opening for opening, count in _opening_signatures(content).items()):
            categories.add("repeated_opening_spine")
            findings.append(f"{title} masih membuka paragraf dengan pola yang berulang.")
    return {"passes": not categories, "categories": sorted(categories), "findings": findings}


def repair_proposal_document_spine(
    chapter_outputs: dict[str, str],
    selected_chapters: list[dict[str, Any]],
    *,
    compact: bool = False,
) -> dict[str, str]:
    repaired = dict(chapter_outputs or {})
    ordered = [chapter for chapter in selected_chapters or [] if chapter.get("id") in repaired]
    openers = (
        "Bagian ini melanjutkan pembacaan dari {previous}: {current} menunjukkan arti temuan sebelumnya bagi keputusan proposal.",
        "Berangkat dari {previous}, {current} mempersempit isu menjadi pilihan kerja yang bisa dipertanggungjawabkan.",
        "Setelah pembahasan {previous}, {current} membaca konsekuensi praktisnya agar bab ini tidak berdiri sendiri.",
        "Dari pijakan {previous}, {current} mengubah konteks menjadi rancangan yang lebih mudah diuji pembaca.",
        "Rangkaian dari {previous} berlanjut ke {current} untuk memperjelas hubungan antara kebutuhan dan pendekatan.",
        "Pembaca yang baru melewati {previous} dapat membaca {current} sebagai jawaban atas implikasi bab tersebut.",
        "Keterhubungan dengan {previous} membuat {current} berfungsi sebagai penajaman, bukan pengulangan konteks.",
        "Masukan pada {previous} diturunkan ke {current} agar keputusan proposal tetap mengikuti alur yang sama.",
        "Jembatan dari {previous} menuju {current} menjaga pembahasan tetap bergerak dari bukti ke pilihan kerja.",
        "Kerangka pada {previous} memberi dasar bagi {current} untuk menjelaskan langkah yang lebih operasional.",
        "Temuan dalam {previous} dibawa ke {current} supaya pembaca melihat kesinambungan antar-bab.",
        "Kaitan dengan {previous} menempatkan {current} sebagai lanjutan langsung dari analisis sebelumnya.",
    )
    closers = (
        "Dengan dasar tersebut, pembahasan bergerak ke {next} supaya pembaca melihat hubungan antara bukti, rancangan, dan keputusan lanjutan.",
        "Implikasi bagian ini menjadi dasar untuk {next}, bukan bab baru yang terlepas dari analisis sebelumnya.",
        "Arah berikutnya dibaca pada {next}, tempat konsekuensi dari bagian ini diterjemahkan menjadi rencana yang lebih operasional.",
        "Dari titik ini, {next} mengambil alih pembahasan agar konsekuensi proposal berubah menjadi keputusan yang lebih konkret.",
        "Pembahasan kemudian masuk ke {next}, sehingga alur dokumen tetap bergerak dari alasan menuju cara kerja.",
        "Hasil pembacaan ini mengantar pembaca ke {next}, tempat keputusan berikutnya dipilih dengan dasar yang lebih jelas.",
        "Bab sesudahnya, {next}, memakai pijakan ini untuk memperjelas konsekuensi teknis dan manajerial.",
        "Keterkaitan itu penting sebelum masuk ke {next}, karena bagian berikutnya membutuhkan dasar yang sudah disepakati.",
        "Rangkaian argumen berlanjut ke {next} agar dokumen tidak kembali memulai pembahasan dari nol.",
        "Keputusan yang muncul di sini menjadi pintu menuju {next}, tempat opsi tersebut diterjemahkan lebih lanjut.",
        "Dampak bagian ini dibawa ke {next} agar pembaca melihat kelanjutan antara analisis dan rencana.",
        "Bagian berikutnya, {next}, menggunakan dasar ini untuk memperkuat arah implementasi proposal.",
    )
    compact_openers = (
        "Melanjutkan {previous}, {current} menajamkan implikasi bab sebelumnya.",
        "Dari {previous}, {current} meneruskan alur ke keputusan yang lebih operasional.",
        "Kaitan dengan {previous} membuat {current} menjadi lanjutan, bukan awal baru.",
    )
    compact_closers = (
        "Bagian ini menjadi dasar untuk {next}.",
        "Arah berikutnya bergerak ke {next}.",
        "Dampak bagian ini diteruskan pada {next}.",
    )
    active_openers = compact_openers if compact else openers
    active_closers = compact_closers if compact else closers

    for index, chapter in enumerate(ordered):
        chapter_id = chapter.get("id")
        title = chapter.get("title") or chapter_id
        content = str(repaired.get(chapter_id, "") or "").strip()
        additions_before: list[str] = []
        additions_after: list[str] = []
        if index > 0:
            previous_title = ordered[index - 1].get("title") or ordered[index - 1].get("id")
            if not _has_spine_connector(content[:900], previous_title, title):
                additions_before.append(active_openers[(index - 1) % len(active_openers)].format(previous=previous_title, current=title))
        if index < len(ordered) - 1:
            next_title = ordered[index + 1].get("title") or ordered[index + 1].get("id")
            if not _has_spine_connector(content[-900:], title, next_title):
                additions_after.append(active_closers[index % len(active_closers)].format(next=next_title))
        repaired[chapter_id] = "\n\n".join([*additions_before, content, *additions_after]).strip()
    return repaired


def compact_table_cell(value: Any, max_words: int = 20) -> str:
    text = compact_text(value, max_words=max_words)
    return re.sub(r"\b(perlu|dapat)\s+(dilakukan|diperkuat)\b", "disarankan", text, flags=re.IGNORECASE)


def compact_markdown_table_rows(rows: list[list[str]], max_cell_words: int = 20) -> list[list[str]]:
    compacted: list[list[str]] = []
    seen_cells: Counter[str] = Counter()
    for row in rows:
        next_row: list[str] = []
        for cell in row:
            text = compact_table_cell(cell, max_words=max_cell_words)
            signature = re.sub(r"\s+", " ", text).strip().lower()
            seen_cells[signature] += 1
            next_row.append(text)
        compacted.append(next_row)
    return compacted
