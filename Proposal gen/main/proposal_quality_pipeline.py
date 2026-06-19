"""Evidence, scope, and final QA helpers for proposal generation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .editorial_intelligence import assess_proposal_style, evaluate_proposal_document_spine, proposal_voice_rules
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
    r"\bKPI\s+(?:outcome\s+)?hasil bisnis utama\b",
    r"\bpeluang nilai bisnis yang bisa dikejar\b",
    r"\barahan prioritas yang perlu dipatuhi\b",
    r"\bon-ruang lingkup\b",
    r"\bon-scope\b",
    r"\bPSA\b",
    r"\bProblem-Solution-Action\b",
    r"\bSlide Core\b",
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
    implication: str = ""
    recommended_angle: str = ""

    def to_prompt_line(self) -> str:
        claim = _clean_text(self.claim, max_words=36)
        usage = _clean_text(self.allowed_usage, max_words=20)
        implication = _clean_text(self.implication, max_words=28)
        angle = _clean_text(self.recommended_angle, max_words=28)
        if not claim:
            return ""
        parts = [f"- Fakta: {claim}"]
        if implication:
            parts.append(f"Makna: {implication}")
        if angle:
            parts.append(f"Arah tulisan: {angle}")
        parts.append(f"Sumber: {self.source_type}; keyakinan {self.confidence}; {usage}")
        return ". ".join(parts)


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
        return (
            "Evidence cards terkurasi untuk bab ini. Pakai sebagai bahan berpikir, bukan sebagai teks mentah:\n"
            + "\n".join(selected)
        )


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
            EvidenceCard(
                ("c_1", "c_2", "all"),
                f"{client} menjadi pihak penerima proposal untuk {project}.",
                "input_pengguna",
                "high",
                implication="proposal harus terasa ditulis untuk pembaca spesifik, bukan proposal generik",
                recommended_angle="hubungkan setiap manfaat dengan keputusan yang perlu diambil pembaca",
            ),
            EvidenceCard(
                ("c_2", "c_3", "c_4"),
                f"Kebutuhan awal diposisikan sebagai {project_goal or 'prioritas yang perlu dipertegas'}.",
                "input_pengguna",
                "medium",
                implication="masalah perlu diterjemahkan menjadi risiko, keputusan, dan dampak operasional",
                recommended_angle="buat pembaca paham mengapa pekerjaan ini perlu dimulai sekarang",
            ),
            EvidenceCard(
                ("c_8", "c_12"),
                f"Durasi acuan adalah {timeline or 'jadwal yang disepakati'} dan estimasi komersial adalah {budget or 'menyesuaikan ruang lingkup final'}.",
                "input_pengguna",
                "medium",
                implication="jadwal dan biaya perlu dibaca sebagai batas keputusan, bukan angka administratif",
                recommended_angle="jelaskan hubungan scope, effort, risiko, dan nilai yang dikejar",
            ),
        ]
        if value_map.get("value_statement"):
            cards.append(EvidenceCard(
                ("all",),
                value_map["value_statement"],
                "sintesis_nilai",
                "high",
                implication="nilai utama harus menjadi benang merah dokumen",
                recommended_angle="pakai nilai ini sebagai tesis proposal dan variasikan cara menjelaskannya per bab",
            ))
        for key, chapters in {
            "profile": ("c_1", "c_2"),
            "news": ("c_1", "c_2", "c_3"),
            "track_record": ("c_4", "c_5", "c_6"),
            "regulations": ("c_4", "c_5", "c_9"),
            "collaboration": ("c_10", "c_11", "c_closing"),
        }.items():
            if research_bundle.get(key):
                cards.append(EvidenceCard(
                    chapters,
                    research_bundle[key],
                    f"osint_{key}",
                    "medium",
                    implication="sinyal publik membantu menjelaskan tekanan eksternal atau konteks institusional",
                    recommended_angle="ubah OSINT menjadi alasan relevansi dan momentum, bukan paragraf latar belakang panjang",
                ))
        if internal_context:
            cards.append(EvidenceCard(
                ("c_1", "c_10", "c_11"),
                internal_context,
                "data_internal_tersintesis",
                "medium",
                implication="bukti internal menunjukkan kecocokan pengalaman dan kapabilitas pelaksana",
                recommended_angle="jadikan pengalaman sebagai alasan kelayakan, bukan daftar portofolio",
            ))
        if scope_contract.get("in_scope"):
            cards.append(EvidenceCard(
                ("c_4", "c_5", "c_6", "c_8", "c_9", "c_12"),
                "Ruang lingkup mencakup " + "; ".join(scope_contract["in_scope"][:4]),
                "kontrak_ruang_lingkup",
                "high",
                implication="scope menjadi pagar agar rekomendasi tetap realistis",
                recommended_angle="bedakan komitmen pekerjaan, asumsi, dan opsi pengembangan lanjutan",
            ))
        if scope_contract.get("out_of_scope"):
            cards.append(EvidenceCard(
                ("c_4", "c_5", "c_6", "c_8", "c_9", "c_12"),
                "Hal di luar cakupan: " + "; ".join(scope_contract["out_of_scope"][:4]),
                "kontrak_ruang_lingkup",
                "high",
                implication="batasan perlu mencegah ekspektasi berlebihan",
                recommended_angle="tulis batasan dengan bahasa tenang dan tetap membantu pembaca membuat keputusan",
            ))
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


class KakTorContractExtractor:
    """Extracts KAK/TOR requirements into a hidden proposal-planning contract."""

    SECTION_LABELS = {
        "problems": ["latar belakang", "permasalahan", "isu utama", "tantangan", "kondisi eksisting"],
        "objectives": ["maksud dan tujuan", "tujuan", "objective", "sasaran"],
        "in_scope": ["ruang lingkup", "lingkup pekerjaan", "scope of work", "cakupan pekerjaan"],
        "deliverables": ["keluaran", "deliverable", "hasil kerja", "output pekerjaan"],
        "staffing_requirements": ["tenaga ahli", "personil", "personel", "komposisi tim", "struktur tim"],
        "out_of_scope": ["di luar cakupan", "di luar lingkup", "tidak termasuk", "batasan pekerjaan"],
        "acceptance_criteria": ["acceptance", "kriteria penerimaan", "quality gate", "persetujuan hasil"],
    }

    STOP_LABELS = [
        "latar belakang", "permasalahan", "isu utama", "tantangan", "kondisi eksisting",
        "maksud dan tujuan", "tujuan", "objective", "sasaran",
        "ruang lingkup", "lingkup pekerjaan", "scope of work", "cakupan pekerjaan",
        "keluaran", "deliverable", "hasil kerja", "output pekerjaan",
        "tenaga ahli", "personil", "personel", "komposisi tim", "struktur tim",
        "jadwal", "jangka waktu", "durasi", "waktu pelaksanaan",
        "di luar cakupan", "di luar lingkup", "tidak termasuk", "batasan pekerjaan",
        "acceptance", "kriteria penerimaan", "quality gate", "persetujuan hasil",
    ]

    @classmethod
    def extract(
        cls,
        text: str,
        source_document: str = "",
        timeline_items: Optional[List[Dict[str, str]]] = None,
        frameworks: Optional[List[str]] = None,
        suggestions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        source = str(text or "").strip()
        suggestions = suggestions or {}
        if not source:
            return {
                "available": False,
                "source_document": source_document,
                "problems": [],
                "objectives": [],
                "scope_contract": {"in_scope": [], "out_of_scope": [], "assumptions": [], "dependencies": [], "deliverables": []},
                "deliverables": [],
                "staffing_requirements": [],
                "timeline_items": timeline_items or [],
                "frameworks": frameworks or [],
                "acceptance_criteria": [],
            }

        problems = cls._section_items(source, cls.SECTION_LABELS["problems"], max_items=4)
        objectives = cls._section_items(source, cls.SECTION_LABELS["objectives"], max_items=4)
        in_scope = cls._section_items(source, cls.SECTION_LABELS["in_scope"], max_items=8)
        deliverables = cls._section_items(source, cls.SECTION_LABELS["deliverables"], max_items=8)
        staffing = cls._section_items(source, cls.SECTION_LABELS["staffing_requirements"], max_items=8)
        out_of_scope = cls._section_items(source, cls.SECTION_LABELS["out_of_scope"], max_items=5)
        acceptance = cls._section_items(source, cls.SECTION_LABELS["acceptance_criteria"], max_items=5)

        if suggestions.get("permasalahan"):
            problems.insert(0, str(suggestions["permasalahan"]))
        if suggestions.get("konteks_organisasi"):
            objectives.insert(0, str(suggestions["konteks_organisasi"]))

        inferred_contract = ScopeContractExtractor.extract(source)
        scope_contract = {
            "in_scope": cls._dedupe([*in_scope, *inferred_contract.get("in_scope", [])], limit=8),
            "out_of_scope": cls._dedupe([*out_of_scope, *inferred_contract.get("out_of_scope", [])], limit=6),
            "assumptions": cls._dedupe(inferred_contract.get("assumptions", []), limit=6),
            "dependencies": cls._dedupe(inferred_contract.get("dependencies", []), limit=6),
            "deliverables": cls._dedupe([*deliverables, *inferred_contract.get("deliverables", [])], limit=8),
        }
        deliverables = cls._dedupe([*deliverables, *scope_contract["deliverables"]], limit=8)

        return {
            "available": True,
            "source_document": source_document,
            "problems": cls._dedupe(problems, limit=6),
            "objectives": cls._dedupe(objectives, limit=6),
            "scope_contract": scope_contract,
            "deliverables": deliverables,
            "staffing_requirements": cls._dedupe(staffing, limit=8),
            "timeline_items": [item for item in (timeline_items or []) if isinstance(item, dict)][:8],
            "frameworks": cls._dedupe([str(item) for item in (frameworks or []) if str(item or "").strip()], limit=8),
            "acceptance_criteria": cls._dedupe(acceptance, limit=6),
        }

    @classmethod
    def _section_items(cls, text: str, labels: List[str], max_items: int = 6) -> List[str]:
        section = cls._section_text(text, labels)
        if not section:
            return []
        items: List[str] = []
        for raw_line in section.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip(" -•\t|")
            line = re.sub(r"^\d+[\).\s-]+", "", line).strip()
            if not line or re.fullmatch(r"[-|:\s]+", line):
                continue
            cells = [cell.strip(" -") for cell in line.split("|") if cell.strip(" -")]
            candidate = " ".join(cells) if len(cells) > 1 else line
            if len(candidate.split()) < 2:
                continue
            items.append(_clean_text(candidate, max_words=28))
        if len(items) <= 1:
            items = [
                _clean_text(item, max_words=28)
                for item in re.split(r"(?<=[.;])\s+", section)
                if len(str(item).split()) >= 3
            ]
        return cls._dedupe(items, limit=max_items)

    @classmethod
    def _section_text(cls, text: str, labels: List[str]) -> str:
        source = str(text or "")
        start_pattern = "|".join(re.escape(item) for item in labels if item)
        stop_pattern = "|".join(re.escape(item) for item in cls.STOP_LABELS if item)
        if not start_pattern:
            return ""
        match = re.search(rf"(?:^|\n)\s*(?:{start_pattern})\s*[:\-]?\s*(.*)", source, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        remainder = str(match.group(1) or "")
        stop_match = re.search(rf"\n\s*(?:\d+[\).\s-]+)?(?:{stop_pattern})\s*[:\-]?", remainder, flags=re.IGNORECASE)
        if stop_match:
            remainder = remainder[:stop_match.start()]
        lines = []
        for raw_line in remainder.splitlines()[:14]:
            if re.search(r"^\s*(?:bab|pasal)\s+\w+", raw_line, flags=re.IGNORECASE):
                break
            lines.append(raw_line)
        return "\n".join(lines).strip()

    @staticmethod
    def _dedupe(items: List[str], limit: int = 6) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            cleaned = re.sub(r"\s+", " ", str(item or "")).strip(" -;:.")
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
            if len(result) >= limit:
                break
        return result


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
        proposal_technique_contract: Optional[Dict[str, Any]] = None,
    ) -> ContextDeskPacket:
        research_bundle = research_bundle or {}
        value_map = value_map or {}
        scope_contract = scope_contract or {}
        proposal_technique_contract = proposal_technique_contract or {}
        story_core = proposal_technique_contract.get("psa_itmp_core") if isinstance(proposal_technique_contract, dict) else {}
        if not isinstance(story_core, dict):
            story_core = {}
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
            if story_core:
                cues.append(cls._story_guidance_for_chapter(chapter_id, story_core))
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
            "Aturan suara proposal: " + "; ".join(proposal_voice_rules()),
        ]
        return ContextDeskPacket(context_brief=context_brief, chapter_guidance=chapter_guidance, risk_notes=risk_notes, evidence_deck=evidence_deck)

    @staticmethod
    def _story_guidance_for_chapter(chapter_id: str, story_core: Dict[str, Any]) -> str:
        need_domains = "; ".join((story_core.get("need_domains") or [])[:3])
        solution_domains = ", ".join(
            str(item.get("name") or "").strip()
            for item in (story_core.get("solution_domains") or [])[:4]
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        )
        risk_items = "; ".join(
            str(item.get("risk") or "").strip()
            for item in (story_core.get("risk_model") or [])[:3]
            if isinstance(item, dict) and str(item.get("risk") or "").strip()
        )
        success_items = "; ".join(
            str(item.get("category") or "").strip()
            for item in (story_core.get("success_criteria") or [])[:3]
            if isinstance(item, dict) and str(item.get("category") or "").strip()
        )
        commercial_items = "; ".join(
            str(item.get("driver") or "").strip()
            for item in (story_core.get("commercial_drivers") or [])[:4]
            if isinstance(item, dict) and str(item.get("driver") or "").strip()
        )
        mapping = {
            "c_1": f"buka dari konteks masalah dan domain kebutuhan: {need_domains}",
            "c_2": f"rumuskan masalah, risiko bila tidak ditangani, dan kebutuhan solusi: {need_domains}",
            "c_7": f"jadikan ruang lingkup sebagai anchor sebelum framework dan metodologi: {story_core.get('scope_anchor', '')}",
            "c_3": f"klasifikasikan kebutuhan menjadi domain keputusan: {need_domains}",
            "c_4": f"jelaskan framework sebagai pilihan pendekatan yang diturunkan dari scope: {story_core.get('framework_rationale', '')}",
            "c_5": f"turunkan framework menjadi metodologi, fase, output, dan acceptance: {story_core.get('methodology_logic', '')}",
            "c_6": f"ubah metodologi menjadi solution design dan domain solusi: {solution_domains}",
            "c_8": f"timeline mengikuti fase metodologi dan deliverable domain solusi: {solution_domains}",
            "c_9": f"tata kelola harus memuat risiko pengerjaan dan kriteria keberhasilan: risiko {risk_items}; kriteria keberhasilan {success_items}",
            "c_11": f"struktur tim perlu mengikuti tanggung jawab framework, metodologi, dan domain solusi: {solution_domains}",
            "c_12": f"model biaya berbasis domain solusi, kompleksitas, durasi, dan asumsi scope: {commercial_items}",
            "c_closing": "tutup dengan manfaat keputusan, komitmen tindak lanjut, dan penerimaan hasil",
        }
        return mapping.get(chapter_id, "")

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
    VISIBLE_REASONING_LABEL_PATTERNS = [
        r"\bProblem\s*,\s*Opportunity\s*,\s*Directive\b",
        r"\bProblem\s*-\s*Solution\s*-\s*Action\b",
        r"(?-i:\bPSA\b)",
        r"(?-i:\bPAS\b)",
        r"\b(?:adalah|kebutuhan|menuju|dasar|menjawab)\s+Problem\b",
        r"\b(?:sebagai|kategori|posisi)\s+(?:Opportunity|Directive)\b",
    ]
    REPETITION_MIN_WORDS = 4
    REPETITION_MAX_SENTENCE_WORDS = 18

    @staticmethod
    def _has_raw_helper(text: str) -> bool:
        return any(re.search(pattern, str(text or ""), flags=re.IGNORECASE) for pattern in RAW_HELPER_PATTERNS)

    @classmethod
    def _has_visible_reasoning_label(cls, text: str) -> bool:
        return any(
            re.search(pattern, str(text or ""), flags=re.IGNORECASE)
            for pattern in cls.VISIBLE_REASONING_LABEL_PATTERNS
        )

    @classmethod
    def _has_repetitive_sentence(cls, text: str) -> bool:
        counts: Dict[str, int] = {}
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
            normalized = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]", "", sentence).lower()
            normalized = re.sub(r"\s+", " ", normalized).strip()
            words = normalized.split()
            if not (
                cls.REPETITION_MIN_WORDS <= len(words) <= cls.REPETITION_MAX_SENTENCE_WORDS
            ):
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
            if counts[normalized] >= 3:
                return True
        return False

    @classmethod
    def evaluate(
        cls,
        chapter_outputs: Dict[str, str],
        selected_chapters: List[Dict[str, Any]],
        scope_contract: Optional[Dict[str, List[str]]] = None,
        executive_summary: str = "",
        deliberation_contract: Optional[Dict[str, Any]] = None,
        appendix_content: str = "",
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
            if cls._has_visible_reasoning_label(content):
                categories.add("visible_reasoning_label")
                findings.append(f"{title} masih memuat label kerangka berpikir internal.")
            if cls._has_repetitive_sentence(content):
                categories.add("narrative_repetition")
                findings.append(f"{title} masih memuat kalimat berulang yang terasa mekanis.")
            style_result = assess_proposal_style(content)
            if not style_result["passed"]:
                categories.add("editorial_style")
                findings.append(
                    f"{title} masih terasa kaku atau berpola: {', '.join(style_result['findings'])}."
                )
        spine_result = evaluate_proposal_document_spine(chapter_outputs, selected_chapters)
        if not spine_result["passes"]:
            categories.update(spine_result["categories"])
            findings.extend(spine_result["findings"])
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
        if cls._has_visible_reasoning_label(executive_summary):
            categories.add("visible_reasoning_label")
            findings.append("Ringkasan eksekutif masih memuat label kerangka berpikir internal.")
        if cls._has_repetitive_sentence(executive_summary):
            categories.add("narrative_repetition")
            findings.append("Ringkasan eksekutif masih memuat kalimat berulang yang terasa mekanis.")
        summary_style = assess_proposal_style(executive_summary)
        if executive_summary and not summary_style["passed"]:
            categories.add("editorial_style")
            findings.append(
                "Ringkasan eksekutif masih terasa kaku atau berpola: "
                + ", ".join(summary_style["findings"])
                + "."
            )
        contract = deliberation_contract or {}
        required_contract_keys = {
            "evidence_dossier", "research_plan", "document_thesis", "chapter_contracts",
            "claim_ledger", "data_gap_register", "editorial_contract", "appendix_manifest",
        }
        if deliberation_contract is not None and not required_contract_keys.issubset(contract):
            categories.add("missing_deliberation_contract")
            findings.append("Kontrak deliberasi dokumen belum lengkap pada finalisasi proposal.")
        elif deliberation_contract is not None and len(contract.get("chapter_contracts") or []) < len(selected_chapters or []):
            categories.add("missing_deliberation_contract")
            findings.append("Kontrak deliberasi belum mencakup seluruh bab yang dirender.")
        appendix = str(appendix_content or "")
        if contract and (
            "## A. Matriks Ketertelusuran" not in appendix
            or "## C. Keterbatasan Bukti" not in appendix
        ):
            categories.add("missing_traceability_appendix")
            findings.append("Lampiran ketertelusuran dan keterbatasan bukti belum lengkap.")
        if appendix and cls._has_visible_reasoning_label(appendix):
            categories.add("visible_reasoning_label")
            findings.append("Lampiran masih memuat label kerangka berpikir internal.")
        if re.search(r"\b(?:input_pengguna|kontrak_ruang_lingkup|SECTION_PLAN_JSON|DOCUMENT_DELIBERATION)\b", appendix):
            categories.add("raw_helper_text")
            findings.append("Lampiran masih memuat label perencanaan atau sumber internal mentah.")
        missing_claims = [
            item.get("claim_id") for item in contract.get("claim_ledger", [])
            if item.get("claim_id") and item.get("claim_id") not in appendix
        ]
        if appendix and missing_claims:
            categories.add("missing_traceability_appendix")
            findings.append("Lampiran belum memuat seluruh klaim yang diterima dalam ledger.")
        return {"passes": not categories, "categories": sorted(categories), "findings": findings}
