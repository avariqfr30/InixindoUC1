"""Deterministic document-level deliberation for proposal generation."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .proposal_exemplar_profile import (
    build_uc1_exemplar_profile,
    scope_uc1_exemplar_profile,
    select_uc1_proposal_profile,
)


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

    CACHE_VERSION = "proposal-deliberation-v2"
    _cache: dict[str, dict[str, Any]] = {}
    _stats = {"hits": 0, "misses": 0}
    _source_labels = {
        "input_pengguna": "Kebutuhan yang disampaikan",
        "kontrak_ruang_lingkup": "Ruang lingkup yang disepakati",
        "sintesis_nilai": "Sintesis nilai proposal",
        "data_internal_tersintesis": "Bukti kapabilitas internal",
    }
    _chapter_commitment_keys = {
        "c_1": ["dependencies"],
        "c_2": ["scope", "dependencies"],
        "c_3": ["scope"],
        "c_4": ["scope", "deliverables"],
        "c_5": ["scope", "deliverables", "acceptance_criteria"],
        "c_6": ["scope", "deliverables", "acceptance_criteria"],
        "c_7": ["scope", "deliverables", "assumptions", "dependencies"],
        "c_8": ["timeline", "deliverables", "dependencies"],
        "c_9": ["dependencies", "acceptance_criteria"],
        "c_10": ["dependencies"],
        "c_11": ["dependencies"],
        "c_12": ["commercial", "scope", "deliverables", "acceptance_criteria"],
        "c_closing": ["deliverables", "acceptance_criteria"],
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
        proposal_mode: str = "",
        service_type: str = "",
        project_type: str = "",
        client_context: str = "",
        timeline: str = "",
        budget: str = "",
        data_version: str = "",
    ) -> dict[str, Any]:
        cards = [dict(card) for card in (evidence_cards or []) if _clean(card.get("claim"))]
        scope = dict(scope_contract or {})
        kak = dict(kak_contract or {})
        exemplar_profile = build_uc1_exemplar_profile()
        exemplar_profile["selected_profile"] = select_uc1_proposal_profile(
            proposal_mode=proposal_mode,
            service_type=service_type,
            project_type=project_type,
            client_context=client_context,
        )
        cache_key = _digest({
            "version": cls.CACHE_VERSION,
            "exemplar_version": exemplar_profile.get("version"),
            "client": client,
            "project": project,
            "chapters": chapters,
            "cards": cards,
            "scope": scope,
            "kak": kak,
            "proposal_mode": proposal_mode,
            "service_type": service_type,
            "project_type": project_type,
            "client_context": client_context,
            "timeline": timeline,
            "budget": budget,
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
                "commitment_keys": list(cls._chapter_commitment_keys.get(section_id, [])),
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

        def unique_clean(values: Any, max_words: int = 36) -> list[str]:
            raw_values = values if isinstance(values, list) else [values]
            result = []
            seen = set()
            for item in raw_values:
                cleaned = _clean(item, max_words=max_words)
                key = cleaned.lower()
                if not cleaned or key in seen:
                    continue
                seen.add(key)
                result.append(cleaned)
            return result

        assumptions = unique_clean(scope.get("assumptions", []))
        dependencies = unique_clean(scope.get("dependencies", []))
        deliverables = unique_clean([
            *(scope.get("deliverables", []) or []),
            *(kak.get("deliverables", []) or []),
        ])
        commitment_map = {
            "scope": {
                "in_scope": unique_clean(scope.get("in_scope", [])),
                "out_of_scope": unique_clean(scope.get("out_of_scope", [])),
            },
            "deliverables": deliverables,
            "timeline": _clean(timeline, max_words=18),
            "commercial": _clean(budget, max_words=18),
            "assumptions": assumptions,
            "dependencies": dependencies,
            "acceptance_criteria": unique_clean(
                kak.get("acceptance_criteria", kak.get("acceptance", []))
            ),
        }
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
            "commitment_map": commitment_map,
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
        commitment_map = contract.get("commitment_map") or {}
        commitments = {
            key: copy.deepcopy(commitment_map.get(key))
            for key in chapter_contract.get("commitment_keys", [])
            if commitment_map.get(key)
        }
        editorial_contract = copy.deepcopy(contract.get("editorial_contract") or {})
        editorial_contract["exemplar_profile"] = scope_uc1_exemplar_profile(
            editorial_contract.get("exemplar_profile") or {},
            chapter_id,
        )
        payload = {
            "document_thesis": contract.get("document_thesis"),
            "chapter_contract": chapter_contract,
            "accepted_claims": claims,
            "commitments": commitments,
            "data_gaps": [item for item in contract.get("data_gap_register", []) if item.get("area") == chapter_contract.get("title")],
            "editorial_contract": editorial_contract,
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
