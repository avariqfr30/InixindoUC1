"""Evidence, scope, and final QA helpers for proposal generation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .reader_facing_hygiene import sanitize_reader_facing_sources


RAW_HELPER_PATTERNS = [
    r"\bReferenceAccount\b",
    r"\bReferenceDataset\b",
    r"\bConsultantProjectExpertHistory\b",
    r"\bAPIDog\b",
    r"\bInternal API\b",
    r"\bdataset(?:_code| code| name)?\b",
    r"\bsource\s*=",
    r"/api/Resource/dataset",
    r"\bendpoint\b",
    r"\bDirangkum dari sumber\b",
    r"\bKonteks akun internal\b",
    r"\b(?:detail\s+)?Identitas akun internal\b",
    r"\bGunakan informasi ini\b",
    r"\bPembahasan pada Pembahasan ini\b",
    r"\bPembahasan pada bagian ini perlu menjelaskan\b",
    r"\bFokus utama harus tetap pada apa yang perlu dipertegas\b",
]


def _clean_text(value: Any, max_words: int = 42) -> str:
    text = sanitize_reader_facing_sources(str(value or ""))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -;,.")
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:") + "."
    return text


@dataclass(frozen=True)
class EvidenceCard:
    chapter_ids: tuple
    claim: str
    source_type: str
    confidence: str = "medium"
    allowed_usage: str = "Gunakan sebagai konteks pendukung, bukan sebagai kutipan sumber."

    def to_prompt_line(self) -> str:
        claim = _clean_text(self.claim, max_words=36)
        usage = _clean_text(self.allowed_usage, max_words=20)
        if not claim:
            return ""
        return f"- {claim} ({self.source_type}; keyakinan {self.confidence}; {usage})"


class EvidenceDeck:
    def __init__(self, cards: Iterable[EvidenceCard]):
        self.cards = [card for card in cards if card.claim.strip()]

    def for_chapter(self, chapter_id: str, limit: int = 6) -> str:
        selected = []
        for card in self.cards:
            if chapter_id in card.chapter_ids or "all" in card.chapter_ids:
                line = card.to_prompt_line()
                if line:
                    selected.append(line)
            if len(selected) >= limit:
                break
        if not selected:
            return ""
        return "Evidence cards terkurasi untuk bab ini:\n" + "\n".join(selected)


class EvidenceDeckBuilder:
    @staticmethod
    def build(
        client: str,
        project: str,
        project_goal: str = "",
        timeline: str = "",
        budget: str = "",
        research_bundle: Optional[Dict[str, Any]] = None,
        internal_context: str = "",
        value_map: Optional[Dict[str, Any]] = None,
        scope_contract: Optional[Dict[str, List[str]]] = None,
    ) -> EvidenceDeck:
        research_bundle = research_bundle or {}
        value_map = value_map or {}
        scope_contract = scope_contract or {}
        cards: List[EvidenceCard] = [
            EvidenceCard(("c_1", "c_2", "all"), f"{client} menjadi pihak penerima proposal untuk {project}.", "input_pengguna", "high"),
            EvidenceCard(("c_2", "c_3", "c_4"), f"Kebutuhan awal diposisikan sebagai {project_goal or 'prioritas yang perlu dipertegas'}.", "input_pengguna", "medium"),
            EvidenceCard(("c_8", "c_12"), f"Durasi acuan adalah {timeline or 'jadwal yang disepakati'} dan estimasi komersial adalah {budget or 'menyesuaikan ruang lingkup final'}.", "input_pengguna", "medium"),
        ]
        if value_map.get("value_statement"):
            cards.append(EvidenceCard(("all",), value_map["value_statement"], "sintesis_nilai", "high"))
        for key, chapters in {
            "profile": ("c_1", "c_2"),
            "news": ("c_1", "c_2", "c_3"),
            "track_record": ("c_4", "c_5", "c_6"),
            "regulations": ("c_4", "c_5", "c_9"),
            "collaboration": ("c_10", "c_11", "c_closing"),
        }.items():
            if research_bundle.get(key):
                cards.append(EvidenceCard(chapters, research_bundle[key], f"osint_{key}", "medium"))
        if internal_context:
            cards.append(EvidenceCard(("c_1", "c_10", "c_11"), internal_context, "data_internal_tersintesis", "medium"))
        if scope_contract.get("in_scope"):
            cards.append(EvidenceCard(("c_4", "c_5", "c_6", "c_8", "c_9", "c_12"), "Ruang lingkup mencakup " + "; ".join(scope_contract["in_scope"][:4]), "kontrak_ruang_lingkup", "high"))
        if scope_contract.get("out_of_scope"):
            cards.append(EvidenceCard(("c_4", "c_5", "c_6", "c_8", "c_9", "c_12"), "Hal di luar cakupan: " + "; ".join(scope_contract["out_of_scope"][:4]), "kontrak_ruang_lingkup", "high"))
        return EvidenceDeck(cards)


@dataclass(frozen=True)
class ContextDeskPacket:
    context_brief: str
    chapter_guidance: Dict[str, str]
    risk_notes: List[str]
    evidence_deck: EvidenceDeck

    def for_chapter(self, chapter_id: str) -> str:
        parts = [self.context_brief, self.chapter_guidance.get(chapter_id, "")]
        cleaned = [_clean_text(part, max_words=70) for part in parts if str(part or "").strip()]
        if not cleaned:
            return ""
        return "\n".join(f"- {item}" for item in cleaned)


class ContextIntelligenceDesk:
    """Synthesizes raw UI, OSINT, and internal helper context into hidden writing guidance."""

    CHAPTER_FOCUS = {
        "c_1": "gunakan sebagai cerita pembuka tentang kondisi klien dan alasan kebutuhan muncul",
        "c_2": "hubungkan konteks klien dengan masalah yang perlu diselesaikan",
        "c_3": "klasifikasikan kebutuhan tanpa menyalin kategori input mentah",
        "c_4": "pilih pendekatan, standar, atau regulasi yang paling relevan dengan kebutuhan",
        "c_5": "jelaskan metodologi yang menjaga pekerjaan tetap terukur",
        "c_6": "turunkan solusi dari kebutuhan dan batas ruang lingkup",
        "c_7": "jadikan batas pekerjaan sebagai kontrak naratif untuk bab setelahnya",
        "c_8": "susun tahapan, aktivitas, dan keluaran yang realistis",
        "c_9": "jelaskan tata kelola, dependensi, dan kontrol keputusan proyek",
        "c_10": "tunjukkan kapabilitas tim tanpa menyalin biodata atau tabel mentah",
        "c_11": "ringkas bukti kemampuan dan sertifikasi menjadi alasan kepercayaan",
        "c_12": "ikat model biaya dengan scope, tahapan, dan asumsi kerja",
        "c_closing": "tutup dengan ajakan komunikasi yang bersih dan profesional",
    }

    @classmethod
    def build(
        cls,
        client: str,
        project: str,
        project_goal: str = "",
        notes: str = "",
        regulations: str = "",
        research_bundle: Optional[Dict[str, Any]] = None,
        internal_context: str = "",
        value_map: Optional[Dict[str, Any]] = None,
        scope_contract: Optional[Dict[str, List[str]]] = None,
        timeline: str = "",
        budget: str = "",
    ) -> ContextDeskPacket:
        research_bundle = research_bundle or {}
        value_map = value_map or {}
        scope_contract = scope_contract or {}
        source_text = " ".join(
            str(item or "")
            for item in [
                project_goal,
                notes,
                regulations,
                internal_context,
                research_bundle.get("profile"),
                research_bundle.get("news"),
                value_map.get("value_statement"),
            ]
        )
        synthesized_need = cls._synthesize_need(source_text)
        profile_signal = _clean_text(research_bundle.get("profile") or research_bundle.get("news") or "", max_words=28)
        value_signal = _clean_text(value_map.get("value_statement") or "", max_words=24)
        external_guard = "" if profile_signal else "tidak memaksakan klaim eksternal ketika pembanding publik belum cukup kuat"
        brief_parts = [
            f"{client} perlu dibaca sebagai penerima proposal untuk {project}",
            synthesized_need,
            profile_signal,
            value_signal,
            external_guard,
        ]
        context_brief = ". ".join(part for part in (_clean_text(item, max_words=32) for item in brief_parts) if part)

        chapter_guidance: Dict[str, str] = {}
        scope_text = ScopeContractExtractor.to_prompt_text(scope_contract)
        for chapter_id, focus in cls.CHAPTER_FOCUS.items():
            cues = [focus]
            if chapter_id in {"c_4", "c_5", "c_6", "c_8", "c_9", "c_12"} and scope_text:
                cues.append(scope_text)
            if chapter_id in {"c_4", "c_5"} and regulations:
                cues.append(f"Acuan yang relevan: {regulations}")
            if chapter_id in {"c_10", "c_11"} and internal_context:
                cues.append("pakai data kemampuan internal sebagai bukti kecocokan peran, bukan sebagai daftar mentah")
            chapter_guidance[chapter_id] = _clean_text(". ".join(cues), max_words=64)

        evidence_deck = EvidenceDeckBuilder.build(
            client=client,
            project=project,
            project_goal=synthesized_need or project_goal,
            timeline=timeline,
            budget=budget,
            research_bundle=research_bundle,
            internal_context=internal_context,
            value_map=value_map,
            scope_contract=scope_contract,
        )
        risk_notes = [
            "Jangan menampilkan nama dataset, endpoint, source path, atau label workflow internal.",
            "Jangan mengubah metadata akun menjadi pain point, KPI, atau scope jika tidak didukung konteks.",
        ]
        return ContextDeskPacket(context_brief=context_brief, chapter_guidance=chapter_guidance, risk_notes=risk_notes, evidence_deck=evidence_deck)

    @staticmethod
    def _synthesize_need(text: str) -> str:
        cleaned = _clean_text(text, max_words=44)
        lowered = cleaned.lower()
        needs = []
        if any(token in lowered for token in ("spbe", "tata kelola", "governance")):
            needs.append("penguatan tata kelola dan kesiapan eksekusi")
        if any(token in lowered for token in ("iso", "regulasi", "standard", "standar")):
            needs.append("keselarasan dengan standar dan regulasi yang relevan")
        if any(token in lowered for token in ("pain", "problem", "risiko", "keluhan")):
            needs.append("pengurangan risiko operasional yang paling terasa")
        if any(token in lowered for token in ("opportunity", "peluang", "directive", "arah")):
            needs.append("penajaman peluang dan arahan manajemen menjadi rencana kerja")
        if needs:
            return "Kebutuhan utama dibaca sebagai " + "; ".join(dict.fromkeys(needs))
        return cleaned


class ScopeContractExtractor:
    @staticmethod
    def _sentences(text: str) -> List[str]:
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", str(text or ""), flags=re.MULTILINE)
        cleaned = re.sub(r"\|", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return [item.strip(" -;:.") for item in re.split(r"(?<=[.!?])\s+", cleaned) if item.strip()]

    @classmethod
    def extract(cls, scope_text: str) -> Dict[str, List[str]]:
        contract = {"in_scope": [], "out_of_scope": [], "assumptions": [], "dependencies": [], "deliverables": []}
        for sentence in cls._sentences(scope_text):
            lowered = sentence.lower()
            cleaned = _clean_text(sentence, max_words=28)
            if not cleaned:
                continue
            if any(token in lowered for token in ["di luar cakupan", "di luar lingkup", "out-of-scope", "tidak termasuk", "batasan"]):
                contract["out_of_scope"].append(cleaned)
            elif any(token in lowered for token in ["asumsi", "mengasumsikan", "diasumsikan"]):
                contract["assumptions"].append(cleaned)
            elif any(token in lowered for token in ["dependency", "dependensi", "bergantung", "ketersediaan", "pic"]):
                contract["dependencies"].append(cleaned)
            elif any(token in lowered for token in ["keluaran", "deliverable", "dokumen", "roadmap", "rekomendasi"]):
                contract["deliverables"].append(cleaned)
                contract["in_scope"].append(cleaned)
            elif any(token in lowered for token in ["mencakup", "lingkup", "pekerjaan", "workshop", "asesmen"]):
                contract["in_scope"].append(cleaned)
        for key, values in list(contract.items()):
            deduped = []
            for item in values:
                if item and item not in deduped:
                    deduped.append(item)
            contract[key] = deduped[:6]
        return contract

    @staticmethod
    def to_prompt_text(contract: Optional[Dict[str, List[str]]]) -> str:
        contract = contract or {}
        if not any(contract.get(key) for key in ("in_scope", "out_of_scope", "assumptions", "dependencies", "deliverables")):
            return ""
        parts = ["Batas ruang lingkup yang wajib dijaga pada bab ini:"]
        labels = [
            ("in_scope", "Dalam cakupan"),
            ("out_of_scope", "Di luar cakupan"),
            ("assumptions", "Asumsi"),
            ("dependencies", "Dependensi"),
            ("deliverables", "Keluaran"),
        ]
        for key, label in labels:
            values = contract.get(key) or []
            if values:
                parts.append(f"- {label}: {'; '.join(values[:4])}")
        return "\n".join(parts)


class ProposalQualityGate:
    @staticmethod
    def _has_raw_helper(text: str) -> bool:
        return any(re.search(pattern, str(text or ""), flags=re.IGNORECASE) for pattern in RAW_HELPER_PATTERNS)

    @classmethod
    def evaluate(
        cls,
        chapter_outputs: Dict[str, str],
        selected_chapters: List[Dict[str, Any]],
        scope_contract: Optional[Dict[str, List[str]]] = None,
        executive_summary: str = "",
    ) -> Dict[str, Any]:
        categories = set()
        findings = []
        for chapter in selected_chapters or []:
            chapter_id = chapter.get("id")
            title = chapter.get("title", chapter_id)
            content = chapter_outputs.get(chapter_id, "")
            if not str(content or "").strip():
                categories.add("empty_chapter")
                findings.append(f"{title} kosong atau tidak dapat dirender.")
            if cls._has_raw_helper(content):
                categories.add("raw_helper_text")
                findings.append(f"{title} masih memuat label sumber/internal mentah.")
        out_scope_items = [item.lower() for item in (scope_contract or {}).get("out_of_scope", [])]
        if out_scope_items:
            combined_later = " ".join(
                str(chapter_outputs.get(chapter.get("id"), "") or "")
                for chapter in selected_chapters or []
                if chapter.get("id") not in {"c_7"}
            ).lower()
            for item in out_scope_items:
                if item and item in combined_later:
                    categories.add("scope_drift")
                    findings.append("Bab setelah ruang lingkup mengulang item yang sudah dinyatakan di luar cakupan.")
                    break
        if re.search(r"\bBAB\s+[IVXLC\d]+", str(executive_summary or ""), flags=re.IGNORECASE):
            categories.add("executive_summary_literal_callback")
            findings.append("Ringkasan eksekutif masih menyebut nomor bab secara literal.")
        if cls._has_raw_helper(executive_summary):
            categories.add("raw_helper_text")
            findings.append("Ringkasan eksekutif masih memuat label sumber/internal mentah.")
        return {"passes": not categories, "categories": sorted(categories), "findings": findings}
