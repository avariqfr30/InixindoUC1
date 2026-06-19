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
    return Counter(" ".join(re.findall(r"[a-z0-9]+", part.lower())[:width]) for part in paragraphs)


def assess_proposal_style(text: Any) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", str(text or "").lower())
    findings: list[str] = []
    if any(phrase in normalized for phrase in ("dalam rangka pelaksanaan", "sehubungan dengan hal tersebut", "dengan demikian maka")):
        findings.append("template_language")
    if any(count >= 3 and opening for opening, count in _paragraph_openings(text).items()):
        findings.append("repeated_openings")
    return {"passed": not findings, "findings": findings}


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
