"""Deterministic document-level deliberation for proposal generation."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .proposal_exemplar_profile import build_uc1_exemplar_profile


def _clean(value: Any, max_words: int = 36) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -;,.:")
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:") + "."
    return text


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ProposalDeliberationBuilder:
    """Build reusable planning artifacts without adding model or source calls."""

    CACHE_VERSION = "proposal-deliberation-v1"
    _cache: dict[str, dict[str, Any]] = {}
    _stats = {"hits": 0, "misses": 0}
    _source_labels = {
        "input_pengguna": "Kebutuhan yang disampaikan",
        "kontrak_ruang_lingkup": "Ruang lingkup yang disepakati",
        "sintesis_nilai": "Sintesis nilai proposal",
        "data_internal_tersintesis": "Bukti kapabilitas internal",
    }

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        cls._stats = {"hits": 0, "misses": 0}

    @classmethod
    def cache_stats(cls) -> dict[str, int]:
        return {**cls._stats, "items": len(cls._cache)}

    @classmethod
    def _remember(cls, key: str, value: dict[str, Any]) -> dict[str, Any]:
        cls._cache[key] = copy.deepcopy(value)
        while len(cls._cache) > 128:
            cls._cache.pop(next(iter(cls._cache)))
        return copy.deepcopy(value)

    @classmethod
    def build(
        cls,
        client: str,
        project: str,
        chapters: list[dict[str, Any]],
        evidence_cards: list[dict[str, Any]] | None = None,
        scope_contract: dict[str, Any] | None = None,
        kak_contract: dict[str, Any] | None = None,
        data_version: str = "",
    ) -> dict[str, Any]:
        cards = [dict(card) for card in (evidence_cards or []) if _clean(card.get("claim"))]
        scope = dict(scope_contract or {})
        kak = dict(kak_contract or {})
        exemplar_profile = build_uc1_exemplar_profile()
        cache_key = _digest({
            "version": cls.CACHE_VERSION,
            "exemplar_version": exemplar_profile.get("version"),
            "client": client,
            "project": project,
            "chapters": chapters,
            "cards": cards,
            "scope": scope,
            "kak": kak,
            "data_version": data_version,
        })
        if cache_key in cls._cache:
            cls._stats["hits"] += 1
            return copy.deepcopy(cls._cache[cache_key])
        cls._stats["misses"] += 1

        chapter_ids = [str(chapter.get("id") or "").strip() for chapter in chapters if chapter.get("id")]
        claim_ledger = []
        for index, card in enumerate(cards, start=1):
            supported = [item for item in card.get("chapter_ids", []) if item in chapter_ids or item == "all"]
            source_type = str(card.get("source_type") or "").strip()
            claim_ledger.append({
                "claim_id": f"P-{index:03d}",
                "claim": _clean(card.get("claim")),
                "implication": _clean(card.get("implication")),
                "supports": supported or ["all"],
                "confidence": str(card.get("confidence") or "medium").lower(),
                "reader_source": cls._source_labels.get(source_type, "Bukti pendukung yang telah ditelaah"),
            })

        chapter_contracts = []
        for index, chapter in enumerate(chapters):
            section_id = str(chapter.get("id") or "").strip()
            relevant = [
                item["claim_id"] for item in claim_ledger
                if section_id in item["supports"] or "all" in item["supports"]
            ]
            chapter_contracts.append({
                "section_id": section_id,
                "title": _clean(chapter.get("title"), 18),
                "purpose": "Mengubah kebutuhan dan bukti menjadi keputusan proposal yang dapat dipertanggungjawabkan.",
                "depends_on": chapter_ids[index - 1] if index else "",
                "hands_off_to": chapter_ids[index + 1] if index + 1 < len(chapter_ids) else "",
                "required_claim_ids": relevant,
                "closing_obligation": "Simpulkan konsekuensi bab ini dan siapkan keputusan yang dibutuhkan bagian berikutnya.",
            })

        gaps = []
        for chapter_contract in chapter_contracts:
            if not chapter_contract["required_claim_ids"]:
                gaps.append({
                    "area": chapter_contract["title"],
                    "gap": "Bukti khusus untuk bagian ini belum tersedia.",
                    "handling": "Gunakan batasan eksplisit; jangan menambah klaim spesifik.",
                })
        for values in kak.values():
            raw_values = values if isinstance(values, list) else [values]
            for value in raw_values:
                text = _clean(value)
                if text and any(term in text.lower() for term in ("belum tersedia", "belum ditentukan", "perlu dikonfirmasi")):
                    gaps.append({
                        "area": "KAK/TOR",
                        "gap": text,
                        "handling": "Konfirmasi bersama klien sebelum menjadi komitmen pelaksanaan.",
                    })

        assumptions = [_clean(item) for item in scope.get("assumptions", []) if _clean(item)]
        dependencies = [_clean(item) for item in scope.get("dependencies", []) if _clean(item)]
        deliverables = [_clean(item) for item in kak.get("deliverables", []) if _clean(item)]
        thesis = (
            f"Proposal untuk {client} harus menunjukkan bagaimana {project} mengubah kebutuhan menjadi "
            "hasil yang terukur, batas pelaksanaan yang jelas, dan keputusan yang dapat dipertanggungjawabkan."
        )
        contract = {
            "cache_key": cache_key,
            "data_version": data_version,
            "evidence_dossier": {
                "snapshot_policy": "immutable_per_generation",
                "accepted_claim_count": len(claim_ledger),
                "protected_facts": [_clean(client), _clean(project)],
            },
            "research_plan": {
                "questions": [
                    "Bukti apa yang membuat solusi relevan bagi kondisi klien?",
                    "Asumsi atau dependensi apa yang dapat mengubah komitmen proposal?",
                    "Apakah jadwal, tim, ruang lingkup, dan biaya menjanjikan hal yang konsisten?",
                ],
                "counterchecks": [
                    "Tolak janji yang tidak memiliki bukti atau batasan.",
                    "Bandingkan setiap deliverable dengan scope, jadwal, dan penanggung jawabnya.",
                    "Pisahkan konteks eksternal dari fakta internal klien.",
                ],
            },
            "document_thesis": thesis,
            "chapter_contracts": chapter_contracts,
            "claim_ledger": claim_ledger,
            "data_gap_register": gaps,
            "editorial_contract": {
                "voice": "konsultan senior yang jernih, tenang, dan bertanggung jawab",
                "rules": [
                    "Tulis langsung dalam Bahasa Indonesia yang alami dan profesional.",
                    "Gunakan satu fungsi argumentatif yang jelas untuk setiap paragraf.",
                    "Hubungkan bukti, implikasi, keputusan, dan tindakan tanpa pola template berulang.",
                    "Jelaskan istilah teknis sekali sebelum menggunakannya lebih lanjut.",
                    "Gunakan profil contoh UC1 hanya sebagai kalibrasi struktur dan gaya; jangan menyalin frasa atau menjadikannya bukti fakta.",
                ],
                "meaning_lock": ["nama", "angka", "tanggal", "nilai uang", "scope", "deliverable", "confidence"],
                "forbidden": ["label agen", "nama dataset", "prompt", "chain-of-thought", "klaim tanpa bukti"],
                "exemplar_profile": exemplar_profile,
            },
            "appendix_manifest": {
                "traceability": claim_ledger,
                "assumptions": assumptions,
                "dependencies": dependencies,
                "deliverables": deliverables,
                "data_gaps": gaps,
            },
        }
        return cls._remember(cache_key, contract)

    @staticmethod
    def for_chapter(contract: dict[str, Any], chapter_id: str) -> str:
        chapter_contract = next(
            (item for item in contract.get("chapter_contracts", []) if item.get("section_id") == chapter_id),
            {},
        )
        accepted_ids = set(chapter_contract.get("required_claim_ids", []))
        claims = [item for item in contract.get("claim_ledger", []) if item.get("claim_id") in accepted_ids]
        payload = {
            "document_thesis": contract.get("document_thesis"),
            "chapter_contract": chapter_contract,
            "accepted_claims": claims,
            "data_gaps": [item for item in contract.get("data_gap_register", []) if item.get("area") == chapter_contract.get("title")],
            "editorial_contract": contract.get("editorial_contract"),
        }
        return (
            "[DOCUMENT_DELIBERATION] Gunakan kontrak terstruktur ini secara internal. "
            "Jangan tampilkan label, struktur, confidence, atau proses berpikirnya. "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    @staticmethod
    def build_appendix_markdown(contract: dict[str, Any]) -> str:
        manifest = contract.get("appendix_manifest") or {}
        lines = [
            "# Lampiran Bukti, Asumsi, dan Kesenjangan Data",
            "Lampiran ini memisahkan rincian pendukung dari narasi utama agar keputusan tetap mudah dibaca dan dapat ditelusuri.",
            "",
            "## A. Matriks Ketertelusuran",
            "| ID | Klaim yang Didukung | Bagian Terkait | Dasar Bukti | Tingkat Keyakinan |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in manifest.get("traceability", []):
            supports = ", ".join(item.get("supports", [])) or "Seluruh dokumen"
            confidence = {"high": "Tinggi", "medium": "Sedang", "low": "Terbatas"}.get(item.get("confidence"), "Sedang")
            values = [item.get("claim_id"), item.get("claim"), supports, item.get("reader_source"), confidence]
            lines.append("| " + " | ".join(str(value or "-").replace("|", "/") for value in values) + " |")
        if not manifest.get("traceability"):
            lines.append("| - | Belum ada klaim pendukung yang dapat ditelusuri. | - | - | Terbatas |")

        lines.extend(["", "## B. Asumsi dan Dependensi"])
        assumptions = manifest.get("assumptions", [])
        dependencies = manifest.get("dependencies", [])
        if assumptions:
            lines.append("**Asumsi kerja**")
            lines.extend(f"- {item}" for item in assumptions)
        if dependencies:
            lines.append("**Dependensi pelaksanaan**")
            lines.extend(f"- {item}" for item in dependencies)
        if not assumptions and not dependencies:
            lines.append("- Asumsi dan dependensi khusus belum tersedia; keduanya perlu dikonfirmasi sebelum pelaksanaan.")

        lines.extend(["", "## C. Keterbatasan Bukti"])
        gaps = manifest.get("data_gaps", [])
        if gaps:
            lines.extend(f"- **{item.get('area')}:** {item.get('gap')} {item.get('handling')}" for item in gaps)
        else:
            lines.append("- Tidak ada kesenjangan bukti material yang teridentifikasi pada konteks yang tersedia.")
        return "\n".join(lines).strip()
