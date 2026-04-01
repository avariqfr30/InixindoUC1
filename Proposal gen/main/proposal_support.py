"""Proposal support logic: research assembly, structured chapters, quality checks, and acceptance."""

from .proposal_shared import *
from .runtime_components import (
    ChartEngine,
    DocumentBuilder,
    FinancialAnalyzer,
    FirmAPIClient,
    LogoManager,
    Researcher,
    SchemaMapper,
    StyleEngine,
)


class ProposalSupportMixin:
    def _get_cached_research_bundle(self, key: str) -> Optional[Dict[str, str]]:
        with self._cache_lock:
            cached = self._research_cache.get(key)
            cached_at = float(self._research_cache_times.get(key, 0.0) or 0.0)
            if cached and cached_at and (time.time() - cached_at) <= RESEARCH_CACHE_TTL_SECONDS:
                return dict(cached)
            if key in self._research_cache:
                self._research_cache.pop(key, None)
                self._research_cache_times.pop(key, None)
            return None

    def _store_research_bundle(self, key: str, bundle: Dict[str, str]) -> Dict[str, str]:
        normalized = dict(bundle or {})
        with self._cache_lock:
            self._cache_put(self._research_cache, key, normalized, max_size=96)
            self._research_cache_times[key] = time.time()
        return dict(normalized)

    @staticmethod
    def _fallback_research_bundle(base_client: str, include_collaboration: bool, ai_mode: bool = False) -> Dict[str, str]:
        return {
            "profile": f"Data profil terbaru {base_client} terbatas (sumber daring, n.d.).",
            "news": f"Data berita terbaru {base_client} terbatas (sumber daring, n.d.).",
            "track_record": f"Data track record {base_client} terbatas (sumber daring, n.d.).",
            "collaboration": (
                f"Data histori kolaborasi {WRITER_FIRM_NAME} dengan {base_client} terbatas (sumber daring, n.d.)."
                if include_collaboration else ""
            ),
            "regulations": "Data regulasi terbatas (sumber daring, n.d.).",
            "ai_posture": (
                f"Data publik tentang kesiapan atau inisiatif AI {base_client} terbatas; perlakukan detail kesiapan sebagai kebutuhan validasi awal (sumber daring, n.d.)."
                if ai_mode else ""
            ),
        }

    def _throughput_mode(self) -> bool:
        return self.generation_profile == "throughput"

    @staticmethod
    def _has_external_evidence(osint_block: str) -> bool:
        return any(line.strip().startswith("Sumber eksternal") for line in (osint_block or "").splitlines())

    @staticmethod
    def _extract_osint_facts(osint_block: str, max_items: int = 3) -> List[Dict[str, str]]:
        facts: List[Dict[str, str]] = []
        pattern = (
            r"fakta=(?P<fact>.*?)\s*\|\s*sumber=(?P<title>.*?)\s*\|\s*url=(?P<url>.*?)\s*"
            r"\|\s*sitasi_apa=(?P<citation>\([^)]+\))$"
        )
        for raw_line in (osint_block or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("Sumber eksternal"):
                continue
            match = re.search(pattern, line)
            if not match:
                continue
            fact = ProposalSupportMixin._sanitize_anchor_fact(match.group("fact") or "")
            title = re.sub(r"\s+", " ", (match.group("title") or "")).strip()
            url = (match.group("url") or "").strip()
            citation = (match.group("citation") or "").strip()
            if not fact or ProposalSupportMixin._is_low_signal_anchor_url(url):
                continue
            facts.append({
                "fact": fact[:240],
                "title": title,
                "url": url,
                "citation": citation,
            })
            if len(facts) >= max_items:
                break
        return facts

    @staticmethod
    def _infer_industry(client: str, project: str, notes: str, regulations: str) -> str:
        combined = " ".join([client or "", project or "", notes or "", regulations or ""]).lower()
        rules = [
            ("Perbankan", ["bank", "bri", "bca", "mandiri", "bni", "btn", "fintech", "kredit"]),
            ("Telekomunikasi", ["telkom", "telkomsel", "indosat", "xl", "axiata", "operator"]),
            ("Energi & Utilitas", ["pertamina", "pln", "energi", "listrik", "oil", "gas"]),
            ("Ritel & E-Commerce", ["tokopedia", "e-commerce", "retail", "marketplace", "goto"]),
            ("Transportasi & Aviasi", ["garuda", "aviation", "airline", "logistik", "transport"]),
            ("Pemerintah & BUMN", ["kementerian", "dinas", "pemprov", "pemkab", "bumn", "spbe"]),
            ("Manufaktur & Tambang", ["manufaktur", "adaro", "antam", "vale", "smelter", "mining"]),
        ]
        for label, tokens in rules:
            if any(token in combined for token in tokens):
                return label
        return "Lintas Industri"

    @staticmethod
    def _industry_terms(industry: str) -> List[str]:
        terms_map = {
            "Perbankan": ["nasabah", "core banking", "risk appetite", "kepatuhan POJK", "fraud control"],
            "Telekomunikasi": ["subscriber", "network availability", "service assurance", "NOC", "customer churn"],
            "Energi & Utilitas": ["operational reliability", "asset integrity", "outage window", "HSSE", "dispatch"],
            "Ritel & E-Commerce": ["conversion", "basket size", "checkout funnel", "retention cohort", "omnichannel"],
            "Transportasi & Aviasi": ["on-time performance", "flight disruption", "ground operation", "passenger experience"],
            "Pemerintah & BUMN": ["akuntabilitas", "SPBE", "tata kelola", "layanan publik", "audit trail"],
            "Manufaktur & Tambang": ["supply continuity", "production throughput", "downtime pabrik", "safety compliance"],
        }
        return terms_map.get(industry, ["operational excellence", "risk control", "business value", "governance"])

    @staticmethod
    def _extract_metric_keywords(kpi_blueprint: List[str]) -> List[str]:
        stopwords = {"dan", "yang", "untuk", "dengan", "dari", "pada", "target", "baseline", "dalam", "bulan"}
        keywords: List[str] = []
        seen: Set[str] = set()
        for line in kpi_blueprint or []:
            for token in re.findall(r"[A-Za-z]{3,}", (line or "").lower()):
                if token in stopwords or token in seen:
                    continue
                seen.add(token)
                keywords.append(token)
                if len(keywords) >= 10:
                    return keywords
        return keywords

    @staticmethod
    def _human_join(values: Any, fallback: str = "", max_items: int = 3, conjunction: str = "dan") -> str:
        raw_items = values if isinstance(values, list) else [values]
        cleaned: List[str] = []
        seen: Set[str] = set()
        for raw in raw_items:
            text = re.sub(r"\s+", " ", str(raw or "").strip())
            if not text:
                continue
            parts = re.split(r"\s*[|\n;]+\s*", text)
            for part in parts:
                value = re.sub(r"^\d+\.\s*", "", part or "").strip(" ,;:-")
                if not value:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(value)
                if len(cleaned) >= max_items:
                    break
            if len(cleaned) >= max_items:
                break

        if not cleaned:
            return fallback
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} {conjunction} {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, {conjunction} {cleaned[-1]}"

    @staticmethod
    def _is_low_signal_anchor_url(url: str) -> bool:
        domain = Researcher._source_name(url)
        blocked = (
            "instagram.com",
            "facebook.com",
            "x.com",
            "twitter.com",
            "tiktok.com",
            "linkedin.com",
            "youtube.com",
            "youtu.be",
        )
        return any(domain == item or domain.endswith(f".{item}") for item in blocked)

    @staticmethod
    def _sanitize_anchor_fact(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""
        text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\((?:[A-Za-z0-9.-]+\.[A-Za-z]{2,}|Data Internal),\s*(?:\d{4}|n\.d\.)\)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b(?:instagram|facebook|linkedin|twitter|tiktok|youtube|x)\.com\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" ,;:-.")

    @staticmethod
    def _extract_anchor_keywords(facts: List[Dict[str, str]], max_terms: int = 10) -> List[str]:
        stopwords = {
            "yang", "dengan", "untuk", "dari", "pada", "tahun", "bulan", "sebagai", "dan",
            "atau", "oleh", "this", "that", "from", "with", "into", "their"
        }
        seen: Set[str] = set()
        terms: List[str] = []
        for fact in facts or []:
            for token in re.findall(r"[A-Za-z]{4,}", (fact.get("fact", "") or "").lower()):
                if token in stopwords or token in seen:
                    continue
                seen.add(token)
                terms.append(token)
                if len(terms) >= max_terms:
                    return terms
        return terms

    @staticmethod
    def _anchor_required_chapters() -> Set[str]:
        return {"c_1", "c_2", "c_3", "k_2", "k_3"}

    @staticmethod
    def _text_has_signal(text: str, candidate: str) -> bool:
        value = re.sub(r"\s+", " ", str(candidate or "").strip().lower())
        if not value:
            return False
        pattern = re.escape(value)
        if " " not in value and value.isalpha():
            pattern = rf"\b{pattern}\b"
        return bool(re.search(pattern, (text or "").lower()))

    @classmethod
    def _ai_scope_signal_summary(cls, *values: Any) -> Dict[str, Any]:
        combined = re.sub(r"\s+", " ", " ".join(str(value or "") for value in values)).strip().lower()
        strong_hits = [
            token for token in (SPIRIT_OF_AI_RULES.get("strong_trigger_keywords") or [])
            if cls._text_has_signal(combined, token)
        ]
        supporting_hits = [
            token for token in (SPIRIT_OF_AI_RULES.get("supporting_signals") or [])
            if token not in strong_hits and cls._text_has_signal(combined, token)
        ]
        enabled = bool(strong_hits) or len(supporting_hits) >= 2
        return {
            "enabled": enabled,
            "strong_hits": strong_hits[:8],
            "supporting_hits": supporting_hits[:10],
        }

    @staticmethod
    def _ai_chapter_dimensions(chapter_id: str) -> List[str]:
        chapter_map = SPIRIT_OF_AI_RULES.get("chapter_dimension_map") or {}
        return [str(item).strip() for item in (chapter_map.get(chapter_id) or []) if str(item).strip()]

    @staticmethod
    def _ai_dimension_terms(dimension_id: str) -> List[str]:
        dimension_terms = SPIRIT_OF_AI_RULES.get("dimension_terms") or {}
        return [str(item).strip() for item in (dimension_terms.get(dimension_id) or []) if str(item).strip()]

    @classmethod
    def _chapter_ai_terms(cls, chapter_id: str) -> List[str]:
        kak_dimension_map = {
            "k_1": ["people_capability", "business_use_case"],
            "k_2": ["business_use_case", "governance", "culture_change"],
            "k_3": ["governance", "data_model_foundation", "infrastructure_architecture"],
            "k_4": ["people_capability", "culture_change"],
            "k_5": ["people_capability", "culture_change"],
            "k_6": ["business_use_case", "governance"],
            "k_7": ["infrastructure_architecture"],
            "k_8": ["business_use_case", "culture_change"],
        }
        if chapter_id in kak_dimension_map:
            terms: List[str] = []
            seen: Set[str] = set()
            for dimension_id in kak_dimension_map.get(chapter_id, []):
                for term in cls._ai_dimension_terms(dimension_id):
                    key = term.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    terms.append(term)
            return terms
        terms: List[str] = []
        seen: Set[str] = set()
        for dimension_id in cls._ai_chapter_dimensions(chapter_id):
            for term in cls._ai_dimension_terms(dimension_id):
                key = term.lower()
                if key in seen:
                    continue
                seen.add(key)
                terms.append(term)
        return terms

    @classmethod
    def _build_ai_adoption_profile(
        cls,
        client: str,
        project: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        research_bundle: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        research_bundle = research_bundle or {}
        ai_posture = str(research_bundle.get("ai_posture") or "").strip()
        signals = cls._ai_scope_signal_summary(project, project_goal, notes, regulations, ai_posture, project_type)
        if not signals["enabled"]:
            return {
                "enabled": False,
                "summary": "",
                "business_case": "",
                "data_foundation": "",
                "architecture_posture": "",
                "people_capability": "",
                "governance_posture": "",
                "culture_change": "",
                "delivery_guidance": [],
                "chapter_guidance": {},
                "quality_terms": [],
                "debug_terms": [],
            }

        objective_note = cls._summarize_phrase(
            project or project_goal,
            f"use case prioritas {client}",
            max_words=18
        )
        risk_note = cls._summarize_phrase(
            notes,
            "risiko implementasi dan adopsi yang masih perlu dikendalikan",
            max_words=18
        )
        regulatory_text = " ".join([regulations or "", ai_posture]).lower()
        if any(token in regulatory_text for token in ["on-prem", "on prem", "hybrid", "pdp", "pojk", "regulasi", "compliance", "bank", "audit"]):
            architecture_posture = (
                "Arsitektur perlu dirancang aman, scalable, dan tetap feasible terhadap integrasi, "
                "latency, serta kemungkinan kebutuhan hybrid atau kontrol lingkungan yang lebih ketat."
            )
        elif any(token in regulatory_text for token in ["cloud", "aws", "azure", "gcp", "saas"]):
            architecture_posture = (
                "Arsitektur dapat diarahkan ke pola cloud-based yang aman dan scalable, selama kontrol integrasi, "
                "akses, dan observability tetap dijaga sejak awal."
            )
        else:
            architecture_posture = (
                "Arsitektur perlu ditetapkan dari kebutuhan integrasi, keamanan, skala, dan ritme rollout, "
                "bukan dari pilihan teknologi semata."
            )

        data_foundation = (
            "Kesiapan data dan model perlu dibuktikan lebih dulu melalui validasi ketersediaan data, "
            "kualitas, ownership, dan mekanisme evaluasi model sebelum solusi diperluas."
        )
        if any(token in " ".join([project, notes, regulations]).lower() for token in ["rag", "knowledge", "analytics", "data", "model", "prediction", "forecast", "vision"]):
            data_foundation = (
                "Kesiapan data, kualitas sumber informasi, ownership, dan mekanisme validasi model harus dipetakan "
                "agar use case dapat berjalan stabil dan tidak berhenti di tahap proof-of-concept."
            )

        people_capability = (
            "Perlu kombinasi business owner, business translator, AI/engineering lead, governance reviewer, "
            "dan enablement pengguna agar solusi tidak berhenti pada build teknis."
        )
        if any(token in " ".join([project, notes, project_goal]).lower() for token in ["training", "adoption", "change", "user", "operational"]):
            people_capability = (
                "Kapabilitas tim perlu mencakup peran bisnis, engineering, governance, dan change enablement, "
                "karena adopsi AI sangat bergantung pada kesiapan cara kerja pengguna akhir."
            )

        governance_posture = (
            "Governance harus mencakup SOP penggunaan, approval, risk control, quality gate, "
            "dan mekanisme evaluasi untuk bias, hallucination, atau keputusan yang perlu human oversight."
        )
        culture_change = (
            "Adopsi AI perlu diperlakukan sebagai perubahan cara kerja: dimulai bertahap, "
            "punya ruang belajar, dan disertai mekanisme adopsi yang jelas agar organisasi tidak berhenti pada eksperimen."
        )
        business_case = (
            f"Use case ini perlu selalu diikat ke hasil bisnis yang ingin dicapai {client}, yaitu {objective_note.lower()}, "
            f"sementara tekanan utamanya berada pada {risk_note.lower()}."
        )

        delivery_guidance = [
            "Mulai dari validasi use case dan readiness, bukan langsung build solusi penuh.",
            "Gunakan tahapan pilot, kontrol kualitas, dan keputusan go/no-go yang eksplisit sebelum scale-up.",
            "Pastikan setiap fase menghubungkan business value, kesiapan data/model, dan kontrol risiko.",
            "Siapkan enablement pengguna dan perubahan cara kerja sebagai bagian dari delivery, bukan pekerjaan tambahan di akhir.",
        ]

        chapter_guidance = {
            "c_1": "Tegaskan mengapa use case AI ini penting secara bisnis dan kenapa sekarang relevan untuk diputuskan.",
            "c_2": "Jelaskan gap current state vs target state, termasuk gap kesiapan data, kontrol, atau operating model bila relevan.",
            "c_3": "Tunjukkan apakah kebutuhan ini menuntut intervensi adopsi AI yang lebih bertahap, terkontrol, atau segera.",
            "c_4": "Arahkan pendekatan ke responsible adoption: feasible, aman, sesuai regulasi, dan tidak solution-first.",
            "c_5": "Metodologi perlu mencakup readiness, validasi, pilot, rollout, dan learning loop.",
            "c_6": "Solution design harus terasa feasible untuk dioperasikan, dimonitor, diadopsi pengguna, dan jelas bentuk keluarannya.",
            "c_7": "Ruang lingkup harus realistis, fokus, dan menjaga agar use case yang dipilih tidak melebar tanpa kontrol.",
            "c_8": "Timeline harus menunjukkan dependency readiness, pilot, rollout, dan stabilisasi adopsi.",
            "c_9": "Governance perlu memuat stop/go criteria, approval, monitoring, dan accountability.",
            "c_10": "Profil perusahaan perlu memperlihatkan pengalaman yang relevan dengan use case, kontrol, dan delivery.",
            "c_11": "Struktur tim harus menutup gap bisnis, engineering, governance, dan change enablement.",
            "c_12": "Commercial model perlu mencerminkan effort readiness, control, pilot, dan adoption support.",
        }

        quality_terms = cls._semantic_terms(
            [business_case, data_foundation, architecture_posture, people_capability, governance_posture, culture_change] +
            (signals["strong_hits"] or []) + (signals["supporting_hits"] or []),
            max_terms=24
        )
        summary = (
            "Untuk proposal ini, narasi AI harus terasa dimulai dari use case bisnis, lalu dijaga oleh kesiapan data/model, "
            "arsitektur yang feasible, kapabilitas tim, governance yang bertanggung jawab, dan adopsi perubahan yang realistis."
        )

        return {
            "enabled": True,
            "summary": summary,
            "business_case": business_case,
            "data_foundation": data_foundation,
            "architecture_posture": architecture_posture,
            "people_capability": people_capability,
            "governance_posture": governance_posture,
            "culture_change": culture_change,
            "delivery_guidance": delivery_guidance,
            "chapter_guidance": chapter_guidance,
            "quality_terms": quality_terms,
            "debug_terms": signals["strong_hits"] + signals["supporting_hits"],
        }

    @classmethod
    def _build_kpi_blueprint(
        cls,
        project_goal: str,
        notes: str,
        timeline: str,
        industry: str,
        client: str
    ) -> List[str]:
        combined = " ".join([project_goal or "", notes or ""]).lower()
        months = FinancialAnalyzer._duration_to_months(timeline)
        horizon = f"{months:.0f} bulan" if months else (timeline or "periode proyek")

        kpis: List[str] = []
        if any(token in combined for token in ["mau", "monthly active", "active users", "adopsi"]):
            kpis.append(f"MAU/Adopsi kanal digital: target pertumbuhan terukur dalam {horizon}.")
        if any(token in combined for token in ["nps", "kepuasan", "customer experience", "csat"]):
            kpis.append(f"Customer sentiment (NPS/CSAT): target kenaikan konsisten dalam {horizon}.")
        if any(token in combined for token in ["churn", "retensi", "retention"]):
            kpis.append(f"Retention/Churn: turunkan churn dan naikkan retensi pada horizon {horizon}.")
        if any(token in combined for token in ["downtime", "incident", "availability", "sla", "mttr"]):
            kpis.append(f"Reliability operasi (Availability/MTTR/Incident): perbaikan stabil dalam {horizon}.")
        if any(token in combined for token in ["biaya", "cost", "efisiensi", "opex", "capex"]):
            kpis.append(f"Efisiensi biaya delivery: kontrol biaya per fase dan deviasi anggaran dalam {horizon}.")

        if not kpis:
            default_by_industry = {
                "Perbankan": [
                    f"Pertumbuhan transaksi digital nasabah dalam {horizon}.",
                    f"Kepatuhan & risiko operasional (audit findings) membaik dalam {horizon}.",
                    f"Ketersediaan layanan digital meningkat dan insiden prioritas menurun dalam {horizon}.",
                ],
                "Telekomunikasi": [
                    f"Service availability dan pengalaman pelanggan meningkat dalam {horizon}.",
                    f"Churn pelanggan menurun dan quality of service membaik dalam {horizon}.",
                    f"Efisiensi operasi network support meningkat dalam {horizon}.",
                ],
            }
            kpis.extend(default_by_industry.get(industry, [
                f"KPI outcome bisnis utama {client} bergerak positif dalam {horizon}.",
                f"Kualitas operasi (incident, SLA, lead time) membaik dalam {horizon}.",
                f"Efektivitas eksekusi proyek (on-time, on-budget, on-scope) tercapai dalam {horizon}.",
            ]))

        # Keep concise but actionable.
        return kpis[:4]

    @classmethod
    def _build_personalization_pack(
        cls,
        client: str,
        project: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        research_bundle: Dict[str, str],
        relationship_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ai_profile = cls._build_ai_adoption_profile(
            client=client,
            project=project,
            project_goal=project_goal,
            project_type=project_type,
            timeline=timeline,
            notes=notes,
            regulations=regulations,
            research_bundle=research_bundle,
        )
        industry = cls._infer_industry(client, project, notes, regulations)
        track_record = research_bundle.get("track_record", "")
        collaboration = research_bundle.get("collaboration", "")
        news = research_bundle.get("news", "")

        relationship_context = relationship_context or {}
        relationship_source = str(relationship_context.get("source") or "").strip().lower()
        relationship_mode = str(relationship_context.get("mode") or "").strip().lower()
        if relationship_mode not in {"existing", "new"}:
            relationship_has_evidence = cls._has_external_evidence(collaboration)
            relationship_mode = "existing" if relationship_has_evidence else "new"
            relationship_source = "osint" if relationship_has_evidence else relationship_source
        if relationship_source == "internal_api":
            relationship_guidance = (
                "Gunakan narasi continuity partnership hanya jika data internal memang menunjukkan hubungan kerja sebelumnya."
                if relationship_mode == "existing"
                else "Gunakan narasi kemitraan baru. Jangan klaim hubungan sebelumnya tanpa data internal."
            )
        else:
            relationship_guidance = (
                "Gunakan narasi continuity partnership berbasis bukti publik kolaborasi yang ada."
                if relationship_mode == "existing"
                else "Gunakan narasi kemitraan baru. Jangan klaim pernah bekerja sama tanpa bukti publik."
            )

        news_facts = cls._extract_osint_facts(news, max_items=2)
        track_facts = cls._extract_osint_facts(track_record, max_items=2)
        initiative_facts: List[Dict[str, str]] = []
        seen_citations: Set[str] = set()
        for item in news_facts + track_facts:
            citation = item.get("citation", "")
            if citation in seen_citations:
                continue
            seen_citations.add(citation)
            initiative_facts.append(item)
            if len(initiative_facts) >= 3:
                break

        kpi_blueprint = cls._build_kpi_blueprint(
            project_goal=project_goal,
            notes=notes,
            timeline=timeline,
            industry=industry,
            client=client
        )
        terminology = cls._industry_terms(industry)
        initiative_terms = cls._semantic_terms([project, notes, project_goal], max_terms=8)
        merged_terms: List[str] = []
        seen_terms: Set[str] = set()
        ai_terms = ai_profile.get("quality_terms", []) or []
        for term in terminology + initiative_terms + ai_terms:
            normalized = str(term or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            merged_terms.append(normalized)

        profile_summary = (
            f"Industri klien terdeteksi: {industry}. "
            f"Fokus inisiatif: {project}. "
            f"Model hubungan: {'kelanjutan kolaborasi' if relationship_mode == 'existing' else 'inisiasi kemitraan baru'}."
        )
        if ai_profile.get("enabled"):
            profile_summary += f" Konteks proposal juga menuntut pola adopsi AI yang lebih terstruktur: {ai_profile.get('summary', '')}"

        return {
            "industry": industry,
            "relationship_mode": relationship_mode,
            "relationship_source": relationship_source or "osint",
            "relationship_guidance": relationship_guidance,
            "kpi_blueprint": kpi_blueprint,
            "kpi_keywords": cls._extract_metric_keywords(kpi_blueprint),
            "terminology": merged_terms[:10],
            "initiative_facts": initiative_facts,
            "anchor_citations": [item.get("citation", "") for item in initiative_facts if item.get("citation")],
            "anchor_keywords": cls._extract_anchor_keywords(initiative_facts),
            "profile_summary": profile_summary,
            "ai_mode": bool(ai_profile.get("enabled")),
            "ai_adoption_profile": ai_profile,
        }

    @staticmethod
    def _playbook_for_project(project_type: str) -> Dict[str, Any]:
        key = SchemaMapper.normalize_key(project_type)
        return VALUE_PLAYBOOK.get(key, VALUE_PLAYBOOK.get("implementation", {}))

    @staticmethod
    def _industry_value_drivers(industry: str) -> List[str]:
        return INDUSTRY_VALUE_DRIVERS.get(industry, [
            "kejelasan keputusan",
            "kontrol risiko yang lebih baik",
            "mobilisasi delivery yang lebih cepat",
            "hasil bisnis yang lebih terukur",
        ])

    @classmethod
    def _build_value_map(
        cls,
        client: str,
        project: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        firm_data: Dict[str, str],
        firm_profile: Dict[str, str],
        personalization_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        industry = personalization_pack.get("industry", "Lintas Industri")
        ai_profile = personalization_pack.get("ai_adoption_profile", {}) or {}
        playbook = cls._playbook_for_project(project_type)
        kpis = personalization_pack.get("kpi_blueprint", []) or []
        drivers = cls._industry_value_drivers(industry)
        differentiators = COMPANY_DNA.get("differentiators", []) or []
        portfolio_note = cls._summarize_phrase(
            firm_profile.get("portfolio_highlights", ""),
            "kapabilitas delivery dan advisory yang relevan",
            max_words=22
        )
        methodology_note = cls._summarize_phrase(
            firm_data.get("methodology", ""),
            "metodologi delivery internal yang terstruktur",
            max_words=18
        )
        team_note = cls._summarize_phrase(
            firm_data.get("team", ""),
            "tim inti delivery dan quality control",
            max_words=18
        )
        customer_pressure = cls._summarize_phrase(
            notes,
            f"kebutuhan {project_goal or 'prioritas bisnis'} yang perlu ditangani lebih terukur",
            max_words=24
        )
        project_frame = cls._summarize_phrase(project, "inisiatif prioritas klien", max_words=18)
        value_hook = str(playbook.get("value_hook") or "mengubah kebutuhan klien menjadi rencana kerja yang lebih terkontrol").strip()
        capability = str(playbook.get("capability") or "advisory dan delivery management").strip()
        client_gains = playbook.get("client_gains", []) or []
        client_gains = [str(item).strip() for item in client_gains if str(item).strip()]
        ai_mode = bool(ai_profile.get("enabled"))
        if ai_mode:
            capability = (
                "advisory adopsi AI, delivery governance, readiness validation, dan rollout terkontrol"
            )
            value_hook = (
                "menghubungkan use case AI dengan hasil bisnis yang terukur sambil menjaga kesiapan data, kontrol risiko, dan adopsi organisasi"
            )
            ai_gains = [
                "keputusan use case yang lebih defensible",
                "rollout AI yang lebih terkontrol",
                "adopsi pengguna yang lebih siap",
            ]
            merged_gains: List[str] = []
            seen_gains: Set[str] = set()
            for item in ai_gains + client_gains:
                normalized = str(item or "").strip()
                key = normalized.lower()
                if not normalized or key in seen_gains:
                    continue
                seen_gains.add(key)
                merged_gains.append(normalized)
            client_gains = merged_gains
        value_statement = (
            f"{WRITER_FIRM_NAME} menempatkan engagement {client} sebagai upaya untuk {value_hook}, "
            f"dengan memanfaatkan kapabilitas {capability} agar inisiatif {project_frame.lower()} "
            f"bergerak ke hasil yang lebih terukur dalam horizon {timeline or 'proyek'}."
        )
        win_theme = (
            f"Nilai utama yang harus terasa bagi {client} adalah kombinasi antara {drivers[0]}, "
            f"{drivers[1] if len(drivers) > 1 else 'kontrol delivery'}, dan kemampuan "
            f"{WRITER_FIRM_NAME} untuk menjaga keputusan tetap selaras dengan eksekusi."
        )
        if ai_mode:
            value_statement = (
                f"{WRITER_FIRM_NAME} menempatkan engagement {client} sebagai upaya untuk memastikan inisiatif {project_frame.lower()} "
                f"bergerak dari use case AI yang menjanjikan menjadi program adopsi yang lebih terukur, feasible, dan bertanggung jawab dalam horizon {timeline or 'proyek'}."
            )
            win_theme = (
                f"Nilai utama yang harus terasa bagi {client} adalah kejelasan use case bisnis, kesiapan data dan delivery, "
                f"kontrol governance yang memadai, serta kemampuan {WRITER_FIRM_NAME} menjaga rollout tetap realistis dan mudah diadopsi organisasi."
            )
        human_touch_points = COMPANY_DNA.get("human_touch_review_points", []) or []
        return {
            "positioning": COMPANY_DNA.get("positioning", ""),
            "proposal_promise": COMPANY_DNA.get("proposal_promise", ""),
            "differentiators": differentiators[:3],
            "client_value_focus": COMPANY_DNA.get("client_value_focus", [])[:4],
            "industry_drivers": drivers[:4],
            "capability": capability,
            "value_hook": value_hook,
            "client_gains": client_gains[:3],
            "value_statement": value_statement,
            "win_theme": win_theme,
            "customer_pressure": customer_pressure,
            "proof_points": [item for item in [portfolio_note, methodology_note, team_note] if item],
            "kpi_bridge": kpis[:3],
            "human_touch_points": human_touch_points[:3],
            "review_note": (
                "Target draft adalah sekitar 80% siap pakai; sisakan ruang bagi reviewer manusia untuk "
                "menyempurnakan nuansa relasi, komersial, dan keputusan akhir."
            ),
            "ai_mode": ai_mode,
            "ai_summary": ai_profile.get("summary", ""),
            "ai_governance_posture": ai_profile.get("governance_posture", ""),
            "service_type": service_type,
            "project_type": project_type,
            "project_goal": project_goal,
            "regulations": regulations,
        }

    @staticmethod
    def _format_value_map(value_map: Optional[Dict[str, Any]]) -> str:
        data = value_map or {}
        lines = [
            f"Positioning: {data.get('positioning', '-')}",
            f"Proposal Promise: {data.get('proposal_promise', '-')}",
            f"Value Statement: {data.get('value_statement', '-')}",
            f"Win Theme: {data.get('win_theme', '-')}",
            f"Customer Pressure: {data.get('customer_pressure', '-')}",
            f"Capability Match: {data.get('capability', '-')}",
            f"Client Gains: {', '.join(data.get('client_gains', []) or []) or '-'}",
            f"Industry Drivers: {', '.join(data.get('industry_drivers', []) or []) or '-'}",
            f"Differentiators: {', '.join(data.get('differentiators', []) or []) or '-'}",
            f"Proof Points: {', '.join(data.get('proof_points', []) or []) or '-'}",
            f"KPI Bridge: {ProposalSupportMixin._human_join(data.get('kpi_bridge', []) or [], fallback='-')}",
            f"Human Review: {', '.join(data.get('human_touch_points', []) or []) or '-'}",
            f"AI Summary: {data.get('ai_summary', '-')}",
            f"Review Note: {data.get('review_note', '-')}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _normalize_external_citation(domain: str, year: str) -> str:
        domain_clean = (domain or "").strip().lower()
        year_clean = (year or "").strip().lower()
        if year_clean in {"nd", "n.d", "n.d"}:
            year_clean = "n.d."
        return f"({domain_clean}, {year_clean})"

    @classmethod
    def _extract_external_citations(cls, text: str) -> List[str]:
        pattern = r"\(([A-Za-z0-9.-]+\.[A-Za-z]{2,}),\s*(\d{4}|n\.d\.)\)"
        citations = []
        for domain, year in re.findall(pattern, text or "", flags=re.IGNORECASE):
            citations.append(cls._normalize_external_citation(domain, year))
        return citations

    @classmethod
    def _collect_allowed_external_citations(cls, research_bundle: Dict[str, str]) -> Set[str]:
        if not research_bundle:
            return set()
        merged = "\n".join([str(value) for value in research_bundle.values()])
        return set(cls._extract_external_citations(merged))

    @classmethod
    def _clean_external_citations(cls, content: str, allowed_external_citations: Set[str]) -> str:
        pattern = r"\(([A-Za-z0-9.-]+\.[A-Za-z]{2,}),\s*(\d{4}|n\.d\.)\)"
        internal_pattern = r"\((Data Internal),\s*(\d{4}|n\.d\.)\)"

        cleaned = re.sub(pattern, "", content or "", flags=re.IGNORECASE)
        cleaned = re.sub(internal_pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    @staticmethod
    def _verified_firm_contact_block(firm_profile: Optional[Dict[str, Any]]) -> str:
        contact_lines = FirmAPIClient.build_contact_lines(firm_profile)
        if not contact_lines:
            return ""
        return "\n".join([f"- {line}" for line in contact_lines])

    @classmethod
    def _inject_verified_firm_contact(
        cls,
        content: str,
        firm_profile: Optional[Dict[str, Any]]
    ) -> str:
        block = cls._verified_firm_contact_block(firm_profile)
        if not block:
            return content

        existing_lines = FirmAPIClient.build_contact_lines(firm_profile)
        if all(line in (content or "") for line in existing_lines):
            return content

        contact_section = f"### Kontak Resmi Terverifikasi\n{block}"
        heading_pattern = r"(?im)^\s*##\s*Informasi Kontak dan Langkah Lanjutan\s*$"
        match = re.search(heading_pattern, content or "")
        if match:
            insert_at = match.end()
            return (content or "")[:insert_at] + "\n\n" + contact_section + (content or "")[insert_at:]

        trimmed = (content or "").rstrip()
        if trimmed:
            trimmed += "\n\n"
        return (
            f"{trimmed}## Informasi Kontak dan Langkah Lanjutan\n\n"
            f"{contact_section}"
        )

    @staticmethod
    def _structured_chapter_ids() -> Set[str]:
        return {
            "c_1", "c_2", "c_3", "c_4", "c_5", "c_6", "c_7", "c_8", "c_9", "c_10", "c_11", "c_12", "c_closing",
            "k_1", "k_2", "k_3", "k_4", "k_5", "k_6", "k_7", "k_8",
        }

    def _use_structured_chapter(self, chapter_id: str) -> bool:
        return chapter_id in self._structured_chapter_ids()

    def _tighten_structured_chapter(
        self,
        chapter: Dict[str, Any],
        content: str,
        target_words: int
    ) -> str:
        text = (content or "").strip()
        if not text or self._word_count(text) <= target_words:
            return text

        lines = text.splitlines()
        protected_contact_tokens = ("Alamat kantor:", "Email:", "Telp:", "Website:")

        def current_text() -> str:
            return "\n".join(lines).strip()

        def line_kind(raw_line: str) -> str:
            line = raw_line.strip()
            if not line:
                return "blank"
            if line.startswith("## ") or line.startswith("### "):
                return "heading"
            if line.startswith("|"):
                return "table"
            if line.startswith("[["):
                return "visual"
            if re.match(r"^\d+\.\s+", line):
                return "numbered"
            if re.match(r"^[-*]\s+", line):
                return "bullet"
            return "plain"

        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx].strip()
            if line_kind(line) != "plain":
                continue
            if any(token in line for token in protected_contact_tokens):
                continue
            lines.pop(idx)
            if self._word_count(current_text()) <= target_words:
                return current_text()

        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx].strip()
            if line_kind(line) != "plain":
                continue
            if any(token in line for token in protected_contact_tokens):
                continue
            sentences = re.split(r"(?<=[.!?])\s+", line)
            if len(sentences) < 3:
                continue
            lines[idx] = " ".join(sentences[:2]).strip()
            if self._word_count(current_text()) <= target_words:
                return current_text()

        def count_lines(kind: str) -> int:
            return sum(1 for item in lines if line_kind(item) == kind)

        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx].strip()
            if line_kind(line) != "bullet":
                continue
            if count_lines("bullet") <= 1:
                break
            if any(token in line for token in protected_contact_tokens):
                continue
            lines.pop(idx)
            if self._word_count(current_text()) <= target_words:
                return current_text()

        for idx in range(len(lines) - 1, -1, -1):
            if line_kind(lines[idx]) != "numbered":
                continue
            if count_lines("numbered") <= 1:
                break
            lines.pop(idx)
            if self._word_count(current_text()) <= target_words:
                return current_text()

        return current_text()

    @staticmethod
    def _split_plain_points(raw_text: str, max_items: int = 5) -> List[str]:
        text = (raw_text or "").strip()
        if not text:
            return []
        matches = re.findall(r"(?:^|\n)\s*(?:\d+\.|[-*])\s*(.+)", text)
        if not matches:
            matches = re.split(r"[;|\n]+", text)
        cleaned: List[str] = []
        for item in matches:
            value = re.sub(r"\s+", " ", str(item or "")).strip(" -.;:")
            if not value:
                continue
            cleaned.append(value)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _summarize_phrase(raw_text: str, fallback: str, max_words: int = 18) -> str:
        text = ProposalSupportMixin._normalize_prose_fragment(raw_text)
        if not text:
            return fallback
        words = text.split()
        if len(words) <= max_words:
            return text.rstrip(".")
        return " ".join(words[:max_words]).rstrip(".,;:") + "..."

    @staticmethod
    def _normalize_prose_fragment(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""

        text = text.replace("\r", "\n")
        text = re.sub(
            r"\((?:[A-Za-z0-9.-]+\.[A-Za-z]{2,}|Data Internal),\s*(?:\d{4}|n\.d\.)\)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\n+\s*", " ", text)
        text = re.sub(r"[•▪◦●]+", ", ", text)
        text = re.sub(r"(?:(?<=^)|(?<=[\s,;:]))\d+\.\s+", ", ", text)
        text = re.sub(r"(?:(?<=^)|(?<=[\s,;:]))[-*]\s+", ", ", text)
        text = re.sub(r"\s*[;|]+\s*", ", ", text)
        text = re.sub(r"(\b[^()]{1,80}\([^()]+\))(?:\s+\1)+", r"\1", text)
        text = re.sub(
            r"\b([A-Za-z][A-Za-z0-9/&+\- ]{1,30})\s+\(\1\s+\(([^()]+)\)\)",
            r"\1 (\2)",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9/&+\- ]{1,40}\([^()]+\))\s+\1",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9/&+\- ]{1,40}\([^()]+\))\s+\(\1\)",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r",\s*,+", ", ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip(" ,;:.")

    @staticmethod
    def _extract_first_external_anchor(personalization_pack: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        facts = (personalization_pack or {}).get("initiative_facts", []) or []
        if not facts:
            return "", ""
        first = facts[0] or {}
        fact = ProposalSupportMixin._sanitize_anchor_fact(first.get("fact", "") or "")
        citation = str(first.get("citation", "") or "").strip()
        return fact, citation

    @classmethod
    def _chapter_anchor_line(
        cls,
        chapter_id: str,
        personalization_pack: Optional[Dict[str, Any]],
        prefix: str = "Sebagai konteks eksternal"
    ) -> str:
        if chapter_id not in cls._anchor_required_chapters():
            return ""
        fact, _ = cls._extract_first_external_anchor(personalization_pack)
        if not fact:
            return ""
        return f"{prefix}, {fact}."

    def _has_anchor_signal(self, content: str, personalization_pack: Optional[Dict[str, Any]]) -> bool:
        data = personalization_pack or {}
        citations = [item for item in (data.get("anchor_citations", []) or []) if item]
        if citations and any(citation in (content or "") for citation in citations):
            return True

        anchor_keywords = [item for item in (data.get("anchor_keywords", []) or []) if item]
        if anchor_keywords:
            required_hits = 1 if len(anchor_keywords) < 2 else 2
            if self._count_signal_hits(content or "", anchor_keywords, max_hits=required_hits) >= required_hits:
                return True

        for item in (data.get("initiative_facts", []) or [])[:2]:
            fact = self._sanitize_anchor_fact(item.get("fact", "") or "")
            tokens = fact.split()
            fragment = " ".join(tokens[: min(8, len(tokens))]).strip()
            if fragment and re.search(re.escape(fragment), content or "", re.IGNORECASE):
                return True

        return not bool(citations or anchor_keywords or data.get("initiative_facts"))

    @staticmethod
    def _normalize_proposal_mode(proposal_mode: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(proposal_mode or "").strip().lower()).strip("_")
        return "kak_response" if normalized in {"kak", "kak_response", "tanggapan_kak"} else "canvassing"

    def _structure_for_mode(self, proposal_mode: str) -> List[Dict[str, Any]]:
        return KAK_RESPONSE_STRUCTURE if self._normalize_proposal_mode(proposal_mode) == "kak_response" else UNIVERSAL_STRUCTURE

    @classmethod
    def _extract_need_candidates(cls, project_goal: str) -> List[str]:
        tokens = re.split(r"[,;/\n]+", str(project_goal or ""))
        normalized: List[str] = []
        mapping = {
            "problem": "Problem",
            "opportunity": "Opportunity",
            "directive": "Directive",
        }
        for token in tokens:
            clean = re.sub(r"[^a-z]+", "", token.lower())
            if clean in mapping and mapping[clean] not in normalized:
                normalized.append(mapping[clean])
        return normalized

    @classmethod
    def _resolve_primary_need(cls, project_goal: str, notes: str, regulations: str = "") -> Tuple[str, List[str]]:
        needs = cls._extract_need_candidates(project_goal)
        if not needs:
            return "Problem", []
        if len(needs) == 1:
            return needs[0], []

        context = f"{notes} {regulations}".lower()
        scores = {need: 0 for need in needs}
        priority = {"Problem": 3, "Directive": 2, "Opportunity": 1}
        signal_map = {
            "Problem": ["masalah", "gap", "kendala", "hambatan", "risiko", "belum", "kurang"],
            "Opportunity": ["peluang", "pertumbuhan", "peningkatan", "optimasi", "ekspansi", "adopsi"],
            "Directive": ["regulasi", "kepatuhan", "wajib", "mandat", "audit", "iso", "pojk"],
        }
        for need in needs:
            scores[need] += priority.get(need, 0)
            scores[need] += sum(2 for token in signal_map.get(need, []) if token in context)

        primary = max(needs, key=lambda item: (scores.get(item, 0), -needs.index(item)))
        secondary = [need for need in needs if need != primary]
        return primary, secondary

    @staticmethod
    def _deliverable_matrix(project_type: str, service_type: str, ai_mode: bool = False) -> List[Dict[str, str]]:
        normalized_type = (project_type or "").strip().lower()
        if ai_mode:
            return [
                {"kategori": "Dokumen", "bentuk": "readiness assessment, governance guideline, dan rekomendasi rollout", "tujuan": "memberi dasar keputusan yang aman sebelum adopsi diperluas"},
                {"kategori": "Pendampingan", "bentuk": "pendampingan validasi use case, kontrol, dan forum keputusan", "tujuan": "menjaga adopsi tetap realistis dan terkendali"},
                {"kategori": "Kegiatan", "bentuk": "workshop alignment, pilot review, dan sesi evaluasi berkala", "tujuan": "membangun kesepahaman lintas fungsi dan mempercepat koreksi arah"},
                {"kategori": "Implementation Support", "bentuk": "dukungan rollout terbatas, quality review, dan stabilization support", "tujuan": "memastikan hasil tidak berhenti pada eksperimen"},
            ]
        if normalized_type == "diagnostic":
            return [
                {"kategori": "Dokumen", "bentuk": "assessment report, baseline gap, dan rekomendasi prioritas", "tujuan": "memberi gambaran kondisi saat ini dan area intervensi utama"},
                {"kategori": "Pendampingan", "bentuk": "sesi validasi temuan dan penajaman prioritas", "tujuan": "menyatukan persepsi sponsor terhadap masalah inti"},
                {"kategori": "Kegiatan", "bentuk": "interview, workshop, dan review temuan", "tujuan": "mengumpulkan bukti dan memvalidasi akar persoalan"},
            ]
        if normalized_type == "strategic":
            return [
                {"kategori": "Dokumen", "bentuk": "target operating model, roadmap, dan decision paper", "tujuan": "mengarahkan sponsor pada keputusan strategi yang lebih defensible"},
                {"kategori": "Pendampingan", "bentuk": "facilitation steering discussion dan penajaman opsi strategi", "tujuan": "membantu sinkronisasi bisnis, teknologi, dan tata kelola"},
                {"kategori": "Kegiatan", "bentuk": "visioning workshop dan review roadmap", "tujuan": "membentuk urutan prioritas yang realistis"},
            ]
        if normalized_type == "transformation":
            return [
                {"kategori": "Dokumen", "bentuk": "blueprint transformasi, governance pack, dan rollout plan", "tujuan": "menjadi acuan perubahan yang lebih terstruktur"},
                {"kategori": "Pendampingan", "bentuk": "PMO advisory, quality gate review, dan change orchestration", "tujuan": "menjaga program berjalan disiplin dari awal sampai stabilisasi"},
                {"kategori": "Kegiatan", "bentuk": "kickoff, checkpoint, workshop alignment, dan review milestone", "tujuan": "menjaga keputusan dan eksekusi tetap seirama"},
                {"kategori": "Implementation Support", "bentuk": "support implementasi terbatas pada fase kritis", "tujuan": "membantu transisi hasil desain ke realisasi lapangan"},
            ]
        return [
            {"kategori": "Dokumen", "bentuk": "solution design, konfigurasi acuan, dan handover pack", "tujuan": "menjadi referensi kerja yang siap dieksekusi"},
            {"kategori": "Pendampingan", "bentuk": "pendampingan UAT, readiness review, dan issue resolution", "tujuan": "memastikan kesiapan implementasi dan kualitas hasil"},
            {"kategori": "Kegiatan", "bentuk": "workshop teknis, review design, dan sesi cut-over preparation", "tujuan": "menjaga kesiapan tim dan alignment antar pihak"},
            {"kategori": "Implementation Support", "bentuk": "go-live assistance dan early stabilization", "tujuan": "membantu transisi ke operasi berjalan lebih rapi"},
        ]

    @classmethod
    def _scope_matrix(cls, project_type: str, service_type: str, project: str, notes: str, ai_mode: bool = False) -> List[Dict[str, str]]:
        short_project = cls._summarize_phrase(project, "inisiatif prioritas klien", max_words=10)
        short_notes = cls._summarize_phrase(notes, "isu prioritas dan gap eksekusi", max_words=12)
        deliverables = cls._deliverable_matrix(project_type, service_type, ai_mode=ai_mode)
        rows = [
            {
                "lingkup": "Analisis dan alignment awal",
                "aktivitas": f"memastikan sasaran {short_project.lower()} dan ruang problem yang direspons sudah selaras",
                "keluaran": deliverables[0]["bentuk"],
                "batasan": "tidak termasuk pekerjaan di luar baseline kebutuhan yang telah disepakati",
            },
            {
                "lingkup": "Perancangan solusi dan prioritas kerja",
                "aktivitas": f"menerjemahkan isu seperti {short_notes.lower()} ke solusi, prioritas, dan kontrol delivery",
                "keluaran": deliverables[min(1, len(deliverables) - 1)]["bentuk"],
                "batasan": "bergantung pada ketersediaan data, sponsor, dan counterpart klien",
            },
            {
                "lingkup": "Forum kerja dan validasi",
                "aktivitas": "menjalankan workshop, review milestone, dan validasi arah bersama stakeholder utama",
                "keluaran": deliverables[min(2, len(deliverables) - 1)]["bentuk"],
                "batasan": "jadwal forum menyesuaikan kalender kerja dan kecepatan keputusan sponsor",
            },
        ]
        if len(deliverables) > 3:
            rows.append(
                {
                    "lingkup": "Dukungan implementasi terbatas",
                    "aktivitas": "memberi quality review dan support pada fase paling kritis dari eksekusi",
                    "keluaran": deliverables[3]["bentuk"],
                    "batasan": "bukan pelaksanaan implementasi penuh kecuali disepakati secara eksplisit",
                }
            )
        return rows

    @staticmethod
    def _framework_reference_rows(regulations: str, project_type: str, ai_mode: bool = False) -> List[Dict[str, str]]:
        raw_items = [re.sub(r"\s+", " ", item).strip(" -.;:") for item in re.split(r"[,;/\n]+", str(regulations or ""))]
        selected = [item for item in raw_items if item]
        if not selected:
            if ai_mode:
                selected = [
                    "Responsible AI",
                    "Tata kelola keamanan informasi",
                    "Arsitektur integrasi dan data",
                    "Kontrol mutu dan validasi",
                ]
            else:
                normalized = (project_type or "").strip().lower()
                if normalized == "strategic":
                    selected = ["TOGAF", "COBIT", "Tata kelola keputusan", "Kontrol mutu delivery"]
                elif normalized == "diagnostic":
                    selected = ["ITIL", "COBIT", "Standar asesmen", "Kontrol mutu temuan"]
                elif normalized == "transformation":
                    selected = ["TOGAF", "ITIL", "COBIT", "Kontrol perubahan dan quality gate"]
                else:
                    selected = ["Standar delivery internal", "Kontrol mutu", "Tata kelola proyek", "Acuan implementasi"]

        rows: List[Dict[str, str]] = []
        for item in selected[:4]:
            lowered = item.lower()
            if any(token in lowered for token in ["iso", "pojk", "ojk", "nist", "regulasi", "kepatuhan"]):
                peran = "memberi batas kepatuhan, kontrol minimum, dan ekspektasi kualitas yang wajib dijaga"
            elif any(token in lowered for token in ["cobit", "itil", "togaf", "tm forum", "dama", "framework", "standar"]):
                peran = "memberi struktur kerja agar desain, governance, dan prioritas implementasi tidak berjalan sporadis"
            elif "ai" in lowered or "model" in lowered or "data" in lowered:
                peran = "menjaga kesiapan data, akuntabilitas keputusan, dan kelayakan solusi sebelum diperluas"
            else:
                peran = "menjadi acuan praktis untuk menjaga kualitas keputusan, konsistensi langkah kerja, dan kontrol hasil"

            if ai_mode:
                relevansi = "relevan untuk memastikan adopsi tetap feasible, terkontrol, dan dapat dipertanggungjawabkan"
            else:
                relevansi = f"relevan untuk proyek {project_type.lower()} agar arah kerja tetap konsisten dari keputusan sampai eksekusi"

            rows.append(
                {
                    "acuan": item,
                    "peran": peran,
                    "relevansi": relevansi,
                }
            )
        return rows

    @classmethod
    def _methodology_rows(
        cls,
        project_type: str,
        service_type: str,
        timeline: str,
        ai_mode: bool = False,
    ) -> List[Dict[str, str]]:
        phase_plan = cls._build_ai_phase_plan(timeline) if ai_mode else cls._build_phase_plan(project_type, timeline)
        rows: List[Dict[str, str]] = []
        for idx, item in enumerate(phase_plan, start=1):
            if idx == 1:
                gate = "baseline kebutuhan dan agenda kerja disetujui sponsor"
            elif idx == len(phase_plan):
                gate = "keluaran akhir diterima dan langkah lanjutan disepakati"
            else:
                gate = "hasil fase tervalidasi sebelum masuk ke fase berikutnya"
            rows.append(
                {
                    "fase": item["phase"],
                    "periode": item["period"],
                    "tujuan": item["activity"],
                    "keluaran": item["deliverable"],
                    "quality_gate": gate,
                }
            )
        return rows

    @classmethod
    def _kak_understanding_points(
        cls,
        project: str,
        project_goal: str,
        notes: str,
        regulations: str,
    ) -> List[Tuple[str, str]]:
        short_project = cls._summarize_phrase(project, "pekerjaan yang diminta", max_words=16)
        short_goal = cls._summarize_phrase(project_goal, "maksud dan tujuan pekerjaan", max_words=18)
        short_notes = cls._summarize_phrase(notes, "ruang lingkup, tantangan, dan kebutuhan pelaksanaan", max_words=20)
        context_blob = f"{project_goal} {notes} {regulations}".lower()
        points: List[str] = [
            f"Pemahaman terhadap latar belakang pekerjaan {short_project.lower()} dan konteks kebutuhan yang mendasarinya.",
            f"Pemahaman terhadap maksud, tujuan, dan sasaran pekerjaan yang diarahkan untuk {short_goal.lower()}.",
            f"Pemahaman terhadap lingkup pekerjaan, keluaran, dan prioritas pelaksanaan yang tercermin dari {short_notes.lower()}.",
        ]
        if regulations.strip():
            points.append(
                f"Pemahaman terhadap acuan, standar, atau regulasi yang perlu dijaga, termasuk {cls._summarize_phrase(regulations, 'acuan kerja yang dipilih', max_words=14)}."
            )
        if any(token in context_blob for token in ["sasaran", "indikator keberhasilan", "target kinerja"]):
            points.append("Pemahaman terhadap sasaran pekerjaan dan indikator keberhasilan yang perlu dicapai selama pelaksanaan.")
        if any(token in context_blob for token in ["lokasi", "wilayah", "site", "unit kerja", "cabang"]):
            points.append("Pemahaman terhadap lokasi, cakupan area, atau unit kerja yang menjadi ruang pelaksanaan pekerjaan.")
        if any(token in context_blob for token in ["jangka waktu", "durasi", "timeline", "bulan", "minggu"]):
            points.append("Pemahaman terhadap jangka waktu pelaksanaan dan implikasinya terhadap ritme kerja, tahapan, serta pengendalian pekerjaan.")
        if any(token in context_blob for token in ["tenaga ahli", "kualifikasi", "personel", "kompetensi"]):
            points.append("Pemahaman terhadap kebutuhan tenaga ahli, kompetensi utama, dan tingkat keterlibatan personel yang disyaratkan.")
        limited_points = points[:6]
        return [(chr(97 + idx), text) for idx, text in enumerate(limited_points)]

    @classmethod
    def _kak_response_points(
        cls,
        project: str,
        project_goal: str,
        notes: str,
        timeline: str,
    ) -> List[Tuple[str, str]]:
        short_project = cls._summarize_phrase(project, "pekerjaan yang dimaksud", max_words=14)
        short_goal = cls._summarize_phrase(project_goal, "tujuan pekerjaan", max_words=16)
        short_notes = cls._summarize_phrase(notes, "lingkup pekerjaan dan keluaran kegiatan", max_words=18)
        timeline_label = timeline or "jangka waktu yang ditetapkan"
        context_blob = f"{project_goal} {notes} {timeline}".lower()
        points: List[str] = [
            f"Tanggapan terhadap urgensi pekerjaan {short_project.lower()} dan alasan mengapa pekerjaan ini perlu ditangani secara terstruktur.",
            f"Tanggapan terhadap maksud, tujuan, dan sasaran agar pekerjaan tetap terarah pada {short_goal.lower()}.",
            f"Tanggapan terhadap lingkup pekerjaan dan keluaran kegiatan, khususnya pada area {short_notes.lower()}.",
            f"Tanggapan terhadap kesiapan pelaksanaan, jadwal kerja, dan sumber daya agar pekerjaan tetap realistis dalam horizon {timeline_label}.",
        ]
        if any(token in context_blob for token in ["sasaran", "indikator keberhasilan", "target kinerja"]):
            points.append("Tanggapan terhadap sasaran dan indikator keberhasilan agar hasil kerja dapat diukur secara lebih objektif.")
        if any(token in context_blob for token in ["lokasi", "wilayah", "site", "unit kerja", "cabang"]):
            points.append("Tanggapan terhadap cakupan lokasi atau unit kerja agar pengaturan pelaksanaan dan koordinasi menjadi lebih jelas.")
        if any(token in context_blob for token in ["tenaga ahli", "kualifikasi", "personel", "kompetensi"]):
            points.append("Tanggapan terhadap kebutuhan tenaga ahli dan kualifikasi agar penugasan personel sejalan dengan kompleksitas pekerjaan.")
        if any(token in context_blob for token in ["fasilitas", "sarana", "prasarana", "dukungan"]):
            points.append("Tanggapan terhadap fasilitas pendukung agar pelaksanaan pekerjaan dapat berlangsung lebih tertib dan efektif.")
        limited_points = points[:6]
        return [(chr(97 + idx), text) for idx, text in enumerate(limited_points)]

    @staticmethod
    def _kak_company_structure_rows() -> List[Dict[str, str]]:
        return [
            {"unit": "Pimpinan / Partner", "peran": "menetapkan arah layanan, menjaga kualitas, dan memberi keputusan strategis", "fungsi": "pengendali mutu dan sponsor perusahaan"},
            {"unit": "Practice / Engagement Lead", "peran": "menerjemahkan kebutuhan klien ke pendekatan, metodologi, dan pengawasan pelaksanaan", "fungsi": "penanggung jawab delivery"},
            {"unit": "Project Management Office", "peran": "mengendalikan jadwal, administrasi proyek, dan koordinasi lintas pihak", "fungsi": "pengendali ritme dan dokumentasi kerja"},
            {"unit": "Tim Ahli / Subject Matter Team", "peran": "menyusun analisis, keluaran kerja, dan dukungan substansi sesuai lingkup pekerjaan", "fungsi": "pelaksana teknis dan substansi"},
        ]

    @classmethod
    def _kak_facility_rows(
        cls,
        firm_profile: Dict[str, Any],
        service_type: str,
    ) -> List[Dict[str, str]]:
        credentials = cls._summarize_phrase(
            firm_profile.get("credential_highlights", ""),
            "kapabilitas internal, sarana kolaborasi, dan dukungan dokumentasi",
            max_words=18,
        )
        return [
            {"fasilitas": "Perangkat kolaborasi dan rapat kerja", "dukungan": "rapat daring/luring, koordinasi rutin, dan dokumentasi hasil rapat", "manfaat": "mempercepat sinkronisasi keputusan selama pelaksanaan"},
            {"fasilitas": "Sarana pengolahan dokumen dan analisis", "dukungan": "penyusunan bahan kerja, reviu mutu, dan finalisasi deliverable", "manfaat": "menjaga konsistensi kualitas keluaran"},
            {"fasilitas": "Dukungan manajemen proyek", "dukungan": f"pengendalian jadwal, tindak lanjut, dan administrasi untuk layanan {service_type}", "manfaat": "menjaga ritme kerja tetap terukur"},
            {"fasilitas": "Kapabilitas pendukung perusahaan", "dukungan": credentials, "manfaat": "memastikan pelaksanaan pekerjaan didukung sumber daya perusahaan yang memadai"},
        ]

    @classmethod
    def _kak_innovation_rows(
        cls,
        client: str,
        project_type: str,
        ai_mode: bool = False,
    ) -> List[Dict[str, str]]:
        rows = [
            {"gagasan": "Penguatan quality gate pekerjaan", "nilai_tambah": f"membantu {client} menilai hasil tiap fase secara lebih objektif", "penerapan": "diterapkan pada titik review dan persetujuan keluaran"},
            {"gagasan": "Format deliverable yang lebih siap pakai", "nilai_tambah": "memudahkan sponsor dan tim inti menindaklanjuti hasil kerja", "penerapan": "dimuat pada dokumen, tabel kerja, dan bahan keputusan"},
            {"gagasan": "Ritme koordinasi yang lebih disiplin", "nilai_tambah": "menurunkan risiko miskomunikasi dan deviasi prioritas", "penerapan": "dipakai pada forum kerja, notulen, dan action tracker"},
        ]
        if ai_mode:
            rows.append(
                {"gagasan": "Validasi bertahap sebelum perluasan solusi", "nilai_tambah": "menjaga inovasi tetap realistis, aman, dan mudah diadopsi", "penerapan": "dipakai pada fase kesiapan, pilot, dan keputusan scale-up"}
            )
        elif (project_type or "").strip().lower() in {"transformation", "implementation"}:
            rows.append(
                {"gagasan": "Pemetaan kesiapan implementasi per fase", "nilai_tambah": "membantu pelaksanaan bergerak lebih stabil dan mudah dikendalikan", "penerapan": "dipakai sebelum transisi dari desain ke eksekusi"}
            )
        return rows[:4]

    @classmethod
    def _company_experience_rows(
        cls,
        project_type: str,
        service_type: str,
        firm_profile: Dict[str, str],
        value_map: Optional[Dict[str, Any]] = None,
        ai_mode: bool = False,
    ) -> List[Dict[str, str]]:
        value_map = value_map or {}
        internal_rows = [
            item for item in (firm_profile.get("internal_portfolio_rows") or [])
            if isinstance(item, dict)
        ]
        if internal_rows:
            normalized_rows: List[Dict[str, str]] = []
            for item in internal_rows[:4]:
                normalized_rows.append(
                    {
                        "area": str(item.get("area") or "Pengalaman internal yang relevan").strip(),
                        "relevansi": str(item.get("relevansi") or f"relevan untuk proyek {project_type.lower()} dengan layanan {service_type.lower()}").strip(),
                        "bukti": str(item.get("bukti") or "portofolio internal perusahaan penyusun").strip(),
                        "nilai_tambah": str(item.get("nilai_tambah") or "membantu klien melihat kesiapan eksekusi secara lebih konkret").strip(),
                    }
                )
            return normalized_rows

        profile_note = cls._summarize_phrase(
            firm_profile.get("portfolio_highlights", ""),
            "pengalaman internal pada advisory, delivery, dan penguatan tata kelola proyek",
            max_words=20,
        )
        proof_line = ", ".join((value_map.get("proof_points", []) or [])[:3]) or "metodologi delivery, governance, dan kontrol mutu"
        rows = [
            {
                "area": "Perumusan strategi, tata kelola, dan prioritas kerja",
                "relevansi": f"relevan untuk proyek {project_type.lower()} dengan layanan {service_type.lower()}",
                "bukti": profile_note,
                "nilai_tambah": "membantu klien mendapat arah kerja yang lebih tajam dan dapat diputuskan lebih cepat",
            },
            {
                "area": "Pengawalan delivery dan quality gate",
                "relevansi": "relevan ketika program memerlukan pengendalian milestone, deliverable, dan mekanisme review yang disiplin",
                "bukti": proof_line,
                "nilai_tambah": "membantu sponsor menjaga kualitas hasil dan mengurangi deviasi di lapangan",
            },
            {
                "area": "Pendampingan stakeholder dan penyiapan hasil kerja",
                "relevansi": "relevan untuk inisiatif yang menuntut sinkronisasi bisnis, operasi, dan tata kelola",
                "bukti": "pengalaman menyusun bahan keputusan, workshop kerja, dan dokumen tindak lanjut",
                "nilai_tambah": "membantu hasil proposal lebih mudah ditindaklanjuti oleh tim klien",
            },
        ]
        if ai_mode:
            rows.insert(
                1,
                {
                    "area": "Adopsi AI yang bertanggung jawab dan bertahap",
                    "relevansi": "relevan untuk use case yang membutuhkan keseimbangan antara nilai bisnis, kontrol, dan kesiapan adopsi",
                    "bukti": "pengalaman pada readiness, kontrol governance, dan validasi rollout bertahap",
                    "nilai_tambah": "membantu klien tidak berhenti pada eksperimen, tetapi bergerak ke implementasi yang terukur",
                }
            )
        return rows[:4]

    @classmethod
    def _expert_rows(
        cls,
        project_type: str,
        service_type: str,
        regulations: str,
        team_points: List[str],
        ai_mode: bool = False,
    ) -> List[Dict[str, str]]:
        rows = [
            {
                "peran": "Engagement Lead / Project Director",
                "fokus": "menjaga arah kemitraan, kualitas proposal, dan keputusan strategis sponsor",
                "kompetensi": "kepemimpinan delivery, komunikasi eksekutif, dan pengendalian isu kritis",
                "keterlibatan": "tetap pada fase kickoff, milestone utama, dan review keputusan",
            },
            {
                "peran": "Project Manager / PMO Lead",
                "fokus": "mengendalikan timeline, dependency, action log, dan quality gate proyek",
                "kompetensi": "manajemen proyek, governance, dan koordinasi lintas workstream",
                "keterlibatan": "aktif sepanjang siklus delivery",
            },
        ]
        if ai_mode:
            rows.extend([
                {
                    "peran": "Business Translator / Domain Lead",
                    "fokus": "menjembatani kebutuhan sponsor, use case prioritas, dan kebutuhan pengguna",
                    "kompetensi": "pemahaman domain bisnis, requirement shaping, dan change alignment",
                    "keterlibatan": "intens pada fase discovery, validasi, dan adoption checkpoint",
                },
                {
                    "peran": "AI / Solution Lead",
                    "fokus": "menjaga desain solusi, integrasi, dan kelayakan implementasi",
                    "kompetensi": "arsitektur solusi, validasi kontrol, dan implementation readiness",
                    "keterlibatan": "intens pada fase desain, pilot, dan rollout",
                },
            ])
        else:
            rows.extend([
                {
                    "peran": "Solution / Domain Lead",
                    "fokus": f"menerjemahkan kebutuhan {project_type.lower()} ke rancangan kerja dan deliverable utama",
                    "kompetensi": "analisis kebutuhan, desain solusi, dan penguatan target state",
                    "keterlibatan": "intens pada fase analisis, desain, dan review hasil",
                },
                {
                    "peran": "Quality & Governance Reviewer",
                    "fokus": "memastikan hasil kerja, dokumentasi, dan kontrol proyek sesuai standar",
                    "kompetensi": f"governance, review mutu, dan pemahaman framework/regulasi {regulations or 'yang dipilih'}",
                    "keterlibatan": "aktif pada quality gate dan sign-off deliverable utama",
                },
            ])
        for item in team_points[:2]:
            rows.append(
                {
                    "peran": "Spesialis Workstream",
                    "fokus": f"menjalankan area prioritas dengan baseline komposisi {item}",
                    "kompetensi": "pelaksanaan aktivitas teknis/fungsional dan penyusunan keluaran detail",
                    "keterlibatan": "menyesuaikan kebutuhan fase dan area kerja",
                }
            )
        return rows[:5]

    @staticmethod
    def _build_phase_plan(project_type: str, timeline: str) -> List[Dict[str, str]]:
        months = FinancialAnalyzer._duration_to_months(timeline)
        total_months = max(2, int(round(months or 6)))
        templates: Dict[str, List[Tuple[str, str, str, int]]] = {
            "diagnostic": [
                ("Discovery & Baseline", "Baca konteks bisnis, data, dan pain points prioritas.", "baseline assessment dan hipotesis awal", 1),
                ("As-Is Review", "Petakan proses, kontrol, dan gap pada kondisi berjalan.", "as-is findings dan prioritas gap", 1),
                ("Analisis Opsi", "Susun opsi intervensi yang realistis dan terukur.", "opsi perbaikan dan quick wins", 1),
                ("Final Recommendation", "Konsolidasikan rekomendasi, roadmap, dan keputusan lanjutan.", "report akhir dan arahan eksekusi", 1),
            ],
            "strategic": [
                ("Business Alignment", "Sinkronkan sasaran bisnis, sponsor, dan outcome yang diharapkan.", "alignment charter dan target state", 1),
                ("Target Design", "Rancang operating model, prinsip arsitektur, dan opsi strategi.", "target operating model", 2),
                ("Roadmap Prioritization", "Urutkan inisiatif berdasarkan manfaat, risiko, dan kesiapan.", "roadmap prioritas dan sequencing", 1),
                ("Executive Closure", "Finalisasi keputusan eksekutif dan arahan mobilisasi.", "board-ready recommendation", 1),
            ],
            "transformation": [
                ("Readiness Assessment", "Nilai kesiapan organisasi, proses, data, dan sponsor.", "readiness view dan daftar gap", 1),
                ("Blueprint & Mobilization", "Tetapkan rancangan solusi, governance, dan model delivery.", "blueprint, scope lock, dan mobilization plan", 2),
                ("Phased Rollout", "Eksekusi perubahan prioritas melalui workstream terukur.", "deliverable inti dan milestone rollout", 2),
                ("Hypercare & Stabilization", "Amankan adopsi, kualitas layanan, dan transisi operasi.", "stabilization report dan next-step backlog", 1),
            ],
            "implementation": [
                ("Design", "Finalisasi desain rinci, integrasi, dan acceptance criteria.", "solution design dan backlog implementasi", 1),
                ("Build & Configure", "Bangun komponen, konfigurasi, dan siapkan data/akses.", "build package dan konfigurasi inti", 2),
                ("UAT & Readiness", "Uji end-to-end, tutup defect penting, dan siapkan go-live.", "UAT sign-off dan readiness checklist", 1),
                ("Go-Live & Handover", "Jalankan cut-over, hypercare awal, dan transfer kontrol operasi.", "go-live report dan handover pack", 1),
            ],
        }
        blueprint = templates.get((project_type or "").strip().lower(), templates["implementation"])
        total_weight = sum(item[3] for item in blueprint) or 1
        remaining = total_months
        cursor = 0
        plan: List[Dict[str, str]] = []
        for idx, (phase, activity, deliverable, weight) in enumerate(blueprint):
            phases_left = len(blueprint) - idx
            allocated = max(1, round(total_months * weight / total_weight))
            allocated = min(allocated, remaining - (phases_left - 1))
            start = cursor
            end = min(total_months, start + allocated)
            if idx == len(blueprint) - 1:
                end = total_months
            label = (
                f"Bulan {start + 1}"
                if end - start <= 1
                else f"Bulan {start + 1}-{end}"
            )
            plan.append({
                "phase": phase,
                "activity": activity,
                "deliverable": deliverable,
                "start": str(start),
                "end": str(end),
                "period": label,
            })
            cursor = end
            remaining = max(0, total_months - cursor)
        return plan

    @staticmethod
    def _build_ai_phase_plan(timeline: str) -> List[Dict[str, str]]:
        months = FinancialAnalyzer._duration_to_months(timeline)
        total_months = max(3, int(round(months or 6)))
        blueprint: List[Tuple[str, str, str, int]] = [
            (
                "Use Case & Readiness",
                "Validasi use case bisnis, sponsor, baseline data, dan asumsi awal adopsi.",
                "use case charter, readiness checklist, dan baseline risiko",
                1,
            ),
            (
                "Data & Control Preparation",
                "Siapkan dependensi data, arsitektur, governance, dan kontrol kualitas solusi.",
                "data/control preparation pack dan decision log",
                1,
            ),
            (
                "Pilot & Validation",
                "Uji solusi pada lingkup terkendali, ukur hasil, dan validasi go/no-go.",
                "pilot result, validation findings, dan rekomendasi rollout",
                2,
            ),
            (
                "Controlled Rollout & Adoption",
                "Lakukan rollout bertahap, enablement pengguna, dan stabilisasi hasil di operasi.",
                "rollout deliverables, adoption pack, dan stabilization report",
                2,
            ),
        ]

        total_weight = sum(item[3] for item in blueprint) or 1
        remaining = total_months
        cursor = 0
        plan: List[Dict[str, str]] = []
        for idx, (phase, activity, deliverable, weight) in enumerate(blueprint):
            phases_left = len(blueprint) - idx
            allocated = max(1, round(total_months * weight / total_weight))
            allocated = min(allocated, remaining - (phases_left - 1))
            start = cursor
            end = min(total_months, start + allocated)
            if idx == len(blueprint) - 1:
                end = total_months
            label = f"Bulan {start + 1}" if end - start <= 1 else f"Bulan {start + 1}-{end}"
            plan.append({
                "phase": phase,
                "activity": activity,
                "deliverable": deliverable,
                "start": str(start),
                "end": str(end),
                "period": label,
            })
            cursor = end
            remaining = max(0, total_months - cursor)
        return plan

    @staticmethod
    def _build_payment_plan(project_type: str, budget: str) -> List[Tuple[str, str]]:
        presets: Dict[str, List[Tuple[str, str]]] = {
            "diagnostic": [
                ("Kickoff & data request", "35%"),
                ("Temuan awal & validasi", "35%"),
                ("Laporan akhir & penutupan", "30%"),
            ],
            "strategic": [
                ("Alignment & target design", "30%"),
                ("Roadmap & executive review", "40%"),
                ("Final recommendation & handover", "30%"),
            ],
            "transformation": [
                ("Mobilization & readiness", "20%"),
                ("Blueprint & governance setup", "30%"),
                ("Rollout milestone", "30%"),
                ("Hypercare & closure", "20%"),
            ],
            "implementation": [
                ("Kickoff & design sign-off", "20%"),
                ("Build/configuration milestone", "30%"),
                ("UAT & readiness sign-off", "30%"),
                ("Go-live & handover", "20%"),
            ],
        }
        return presets.get((project_type or "").strip().lower(), presets["implementation"])

    @staticmethod
    def _build_ai_payment_plan() -> List[Tuple[str, str]]:
        return [
            ("Readiness & scoping sign-off", "20%"),
            ("Control/data preparation milestone", "25%"),
            ("Pilot validation milestone", "30%"),
            ("Controlled rollout & adoption closure", "25%"),
        ]

    @staticmethod
    def _visual_marker(kind: str, title: str, items: List[Tuple[str, Any]], unit: str = "") -> str:
        safe_items: List[str] = []
        for label, value in items:
            clean_label = re.sub(r"[|;,\n]+", " ", str(label or "")).strip()
            clean_value = re.sub(r"[^\d.\-]", "", str(value or "")).strip()
            if not clean_label or not clean_value:
                continue
            safe_items.append(f"{clean_label},{clean_value}")
        if not safe_items:
            return ""
        normalized_kind = str(kind or "BAR").strip().upper()
        if normalized_kind == "DONUT":
            return f"[[DONUT: {title} | {'; '.join(safe_items)}]]"
        metric = unit or "Nilai"
        return f"[[BAR: {title} | {metric} | {'; '.join(safe_items)}]]"

    def _render_structured_chapter(
        self,
        chapter: Dict[str, Any],
        client: str,
        project: str,
        budget: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        firm_data: Dict[str, str],
        firm_profile: Dict[str, str],
        personalization_pack: Dict[str, Any],
        value_map: Optional[Dict[str, Any]] = None,
        proposal_mode: str = "canvassing",
    ) -> str:
        year = datetime.now().year
        value_map = value_map or {}
        ai_profile = personalization_pack.get("ai_adoption_profile", {}) or {}
        ai_mode = bool(ai_profile.get("enabled"))
        normalized_mode = self._normalize_proposal_mode(proposal_mode)
        terminology = personalization_pack.get("terminology", []) or []
        kpi_blueprint = personalization_pack.get("kpi_blueprint", []) or []
        term_line = ", ".join(terminology[:3]) if terminology else "tata kelola, pelaksanaan, pengendalian risiko"
        kpi_line = self._human_join(kpi_blueprint[:3], fallback=f"hasil bisnis utama {client}")
        short_notes = self._summarize_phrase(notes, "risiko delivery dan kesiapan stakeholder")
        short_project = self._summarize_phrase(project, "inisiatif prioritas klien")
        short_goal = self._summarize_phrase(project_goal, "kebutuhan inti klien")
        short_value = self._summarize_phrase(
            value_map.get("value_statement", ""),
            f"membantu {client} bergerak ke hasil yang lebih terukur",
            max_words=28
        )
        value_hook = self._summarize_phrase(
            value_map.get("value_hook", ""),
            "mewujudkan kebutuhan klien menjadi hasil kerja yang lebih terkontrol",
            max_words=18
        )
        win_theme = self._summarize_phrase(
            value_map.get("win_theme", ""),
            f"menjaga keputusan dan eksekusi {client} tetap selaras",
            max_words=24
        )
        proof_points = value_map.get("proof_points", []) or []
        proof_line = "; ".join(
            self._summarize_phrase(point, "", max_words=10)
            for point in proof_points[:3]
            if self._summarize_phrase(point, "", max_words=10)
        ) if proof_points else "kapabilitas internal, metodologi pelaksanaan, dan kontrol mutu"
        client_gains = value_map.get("client_gains", []) or []
        gains_line = self._human_join(
            client_gains[:3],
            fallback="kejelasan keputusan, kontrol risiko, dan hasil bisnis yang lebih terukur",
        )
        differentiators = value_map.get("differentiators", []) or []
        differentiator_line = self._human_join(
            differentiators[:2],
            fallback="pendekatan pelaksanaan yang rapi dan tetap relevan dengan kebutuhan sponsor",
        )
        standard_method = self._summarize_phrase(firm_data.get("methodology", ""), "metodologi pelaksanaan internal")
        value_hook_line = self._summarize_phrase(
            value_map.get("value_hook", ""),
            "mengubah kebutuhan bisnis menjadi rencana kerja yang lebih terukur",
            max_words=14,
        )
        team_points = self._split_plain_points(firm_data.get("team", ""), max_items=5)
        team_summary = self._human_join(
            team_points,
            fallback="tim pelaksana inti, peninjau mutu, dan pendukung substansi",
            max_items=5,
        )
        commercial_summary = self._summarize_phrase(firm_data.get("commercial", ""), "mekanisme komersial mengikuti baseline internal")
        payment_plan = self._build_ai_payment_plan() if ai_mode else self._build_payment_plan(project_type, budget)
        phase_plan = self._build_ai_phase_plan(timeline) if ai_mode else self._build_phase_plan(project_type, timeline)
        gantt_points = "; ".join(
            f"{item['phase']},{item['start']},{item['end']}"
            for item in phase_plan
        )
        ai_bridge = self._summarize_phrase(
            ai_profile.get("summary", ""),
            "inisiatif perlu terasa berangkat dari use case bisnis, lalu dijaga oleh readiness, governance, dan adoption",
            max_words=28,
        )
        credential_highlights = self._summarize_phrase(
            firm_profile.get("credential_highlights", ""),
            "kapabilitas inti dan sertifikasi relevan perusahaan penyusun",
            max_words=24,
        )
        ai_governance = self._summarize_phrase(
            ai_profile.get("governance_posture", ""),
            "kontrol penggunaan, gerbang mutu, dan mekanisme evaluasi yang bertanggung jawab",
            max_words=24,
        )
        ai_people = self._summarize_phrase(
            ai_profile.get("people_capability", ""),
            "kombinasi peran bisnis, engineering, governance, dan change enablement",
            max_words=22,
        )
        ai_change = self._summarize_phrase(
            ai_profile.get("culture_change", ""),
            "adopsi bertahap yang tetap mengubah cara kerja secara realistis",
            max_words=22,
        )
        relationship_mode = str(personalization_pack.get("relationship_mode", "new")).strip().lower()
        partnership_line = (
            f"{client} sudah memiliki konteks interaksi yang cukup untuk langsung masuk ke penajaman kebutuhan prioritas dan keputusan kerja."
            if relationship_mode == "existing"
            else f"Inisiatif ini membutuhkan mitra yang mampu membaca konteks awal secara rapi, proporsional, dan langsung mengarah pada prioritas kerja."
        )
        framework_rows = "\n".join(
            f"| {item['acuan']} | {item['peran']} | {item['relevansi']} |"
            for item in self._framework_reference_rows(regulations, project_type, ai_mode=ai_mode)
        )
        methodology_rows = "\n".join(
            f"| {item['fase']} | {item['periode']} | {item['tujuan']} | {item['keluaran']} | {item['quality_gate']} |"
            for item in self._methodology_rows(project_type, service_type, timeline, ai_mode=ai_mode)
        )

        if normalized_mode == "kak_response" and chapter["id"].startswith("k_"):
            verified_contact = self._verified_firm_contact_block(firm_profile)
            company_structure_rows = "\n".join(
                f"| {item['unit']} | {item['peran']} | {item['fungsi']} |"
                for item in self._kak_company_structure_rows()
            )
            experience_rows = "\n".join(
                f"| {item['area']} | {item['relevansi']} | {item['bukti']} | {item['nilai_tambah']} |"
                for item in self._company_experience_rows(
                    project_type=project_type,
                    service_type=service_type,
                    firm_profile=firm_profile,
                    value_map=value_map,
                    ai_mode=ai_mode,
                )
            )
            understanding_points = self._kak_understanding_points(project, project_goal, notes, regulations)
            response_points = self._kak_response_points(project, project_goal, notes, timeline)
            understanding_list = "\n".join(
                f"{label}) {text}"
                for label, text in understanding_points
            )
            response_list = "\n".join(
                f"{label}) {text}"
                for label, text in response_points
            )
            phase_rows = "\n".join(
                f"| {item['phase']} | {item['period']} | {item['activity']} | {item['deliverable']} |"
                for item in phase_plan
            )
            expert_rows = "\n".join(
                f"| {item['peran']} | {item['fokus']} | {item['kompetensi']} | {item['keterlibatan']} |"
                for item in self._expert_rows(project_type, service_type, regulations, team_points, ai_mode=ai_mode)
            )
            deliverable_rows = "\n".join(
                f"| {item['kategori']} | {item['bentuk']} | {item['tujuan']} |"
                for item in self._deliverable_matrix(project_type, service_type, ai_mode=ai_mode)
            )
            facility_rows = "\n".join(
                f"| {item['fasilitas']} | {item['dukungan']} | {item['manfaat']} |"
                for item in self._kak_facility_rows(firm_profile, service_type)
            )
            innovation_rows = "\n".join(
                f"| {item['gagasan']} | {item['nilai_tambah']} | {item['penerapan']} |"
                for item in self._kak_innovation_rows(client, project_type, ai_mode=ai_mode)
            )

            if chapter["id"] == "k_1":
                company_identity_lines = verified_contact or (
                    "- Nama perusahaan dan data kontak resmi akan disampaikan sesuai dokumen perusahaan yang telah diverifikasi.\n"
                    f"- Perusahaan penyusun diposisikan sebagai {self._summarize_phrase(value_map.get('positioning', ''), 'mitra konsultasi dan pelaksanaan yang terstruktur', max_words=18)}."
                )
                return (
                    f"Bab ini menyajikan data perusahaan penyusun sebagai dasar administratif dan profesional untuk mendukung penawaran tanggapan Kerangka Acuan Kerja {client}. "
                    f"Fokusnya adalah menunjukkan identitas perusahaan, struktur organisasi, serta pengalaman pekerjaan sejenis yang relevan dengan kebutuhan {short_project.lower()} (Data Internal, {year}).\n\n"
                    "## 1.1 Informasi Perusahaan\n"
                    f"{WRITER_FIRM_NAME} hadir sebagai perusahaan penyusun yang menempatkan kualitas pelaksanaan, disiplin tata kelola, dan ketepatan keluaran sebagai bagian utama dari komitmen layanan. "
                    f"Dalam konteks pekerjaan {client}, informasi perusahaan ditampilkan untuk memberi kepastian bahwa penawaran ini didukung kapabilitas yang memadai dan pengalaman yang relevan.\n"
                    "1. Informasi perusahaan dipakai untuk menunjukkan identitas penyedia, posisi layanan, dan kesiapan organisasi.\n"
                    "2. Informasi ini juga menjadi dasar untuk menilai kecocokan perusahaan penyusun dengan lingkup pekerjaan yang ditawarkan.\n"
                    f"{company_identity_lines}\n\n"
                    "## 1.2 Struktur Organisasi Perusahaan\n"
                    f"Struktur organisasi perusahaan disusun agar fungsi pengendalian mutu, pengelolaan proyek, dan dukungan substansi dapat bergerak seirama selama pekerjaan berlangsung.\n"
                    "| Unit / Fungsi | Peran Utama | Fungsi dalam Dukungan Pekerjaan |\n"
                    "| --- | --- | --- |\n"
                    f"{company_structure_rows}\n\n"
                    "## 1.3 Daftar Pengalaman Pekerjaan Sejenis\n"
                    f"Pengalaman pekerjaan sejenis berikut ditampilkan untuk menunjukkan relevansi kemampuan perusahaan penyusun terhadap kebutuhan {client}. "
                    f"Daftar ini menekankan area pengalaman, bukti kapabilitas, dan nilai tambah yang dapat dibawa pada pelaksanaan pekerjaan.\n"
                    f"| Area Pengalaman | Relevansi | Bukti Kapabilitas | Nilai Tambah |\n"
                    "| --- | --- | --- | --- |\n"
                    f"{experience_rows}\n"
                    "- Pengalaman sejenis dipakai sebagai dasar keyakinan bahwa pekerjaan dapat dijalankan secara terstruktur.\n"
                    "- Relevansi pengalaman tidak hanya dilihat dari tema pekerjaan, tetapi juga dari disiplin delivery, pengendalian mutu, dan kualitas hasil kerja."
                )

            if chapter["id"] == "k_2":
                anchor_line = self._chapter_anchor_line(
                    "k_2",
                    personalization_pack,
                    prefix="Sebagai konteks eksternal yang tervalidasi"
                )
                ai_note = (
                    f" Untuk konteks AI/adopsi, tanggapan juga memperhatikan kesiapan data, kontrol, dan kemampuan organisasi menyerap perubahan secara bertahap."
                    if ai_mode else ""
                )
                return (
                    f"Bab ini memuat pemahaman perusahaan penyusun terhadap Kerangka Acuan Kerja serta tanggapan dan saran yang dipandang perlu agar pelaksanaan pekerjaan {client} berlangsung lebih terarah, terukur, dan sesuai tujuan.{ai_note} {anchor_line}".strip() + f" (Data Internal, {year}).\n\n"
                    "## 2.1 Pemahaman terhadap Kerangka Acuan Kerja\n"
                    f"Pemahaman terhadap Kerangka Acuan Kerja disusun untuk memastikan bahwa pekerjaan {short_project.lower()} dibaca secara utuh, baik dari sisi latar belakang, tujuan, lingkup, maupun keluaran yang diharapkan.\n"
                    f"{understanding_list}\n"
                    f"- Pemahaman ini menempatkan kebutuhan {client} sebagai dasar utama sebelum pendekatan dan metodologi dirumuskan.\n"
                    f"- Dengan pembacaan yang runtut, proposal tidak berhenti pada pengulangan KAK, melainkan menjadi tanggapan yang menunjukkan kesiapan pelaksanaan.\n\n"
                    "## 2.2 Tanggapan dan Saran terhadap Kerangka Acuan Kerja\n"
                    f"Tanggapan dan saran berikut disampaikan untuk memperkuat ketepatan pelaksanaan, menjaga keselarasan ruang lingkup, serta membantu {client} memperoleh hasil kerja yang lebih siap ditindaklanjuti.\n"
                    f"{response_list}\n"
                    f"- Tanggapan ini diarahkan untuk menjaga hubungan antara urgensi pekerjaan, ruang lingkup, keluaran, dan kesiapan pelaksanaan.\n"
                    f"- Apabila dalam pelaksanaan ditemukan butir tambahan pada KAK yang memerlukan penajaman, perusahaan penyusun siap menyesuaikan rincian tanggapan sepanjang tetap sejalan dengan tujuan pekerjaan."
                )

            if chapter["id"] == "k_3":
                ai_kak_note = (
                    f"\n- Untuk konteks AI/adopsi, kerangka dan metodologi juga digunakan untuk menjaga {ai_bridge.lower()}."
                    if ai_mode else ""
                )
                return (
                    f"Pendekatan dan metodologi pada bab ini dirumuskan untuk menunjukkan bagaimana perusahaan penyusun menjawab KAK secara sistematis. "
                    f"Bagi {client}, bagian ini penting agar terdapat hubungan yang jelas antara acuan kerja, langkah pelaksanaan, dan kualitas hasil yang ditargetkan (Data Internal, {year}).\n\n"
                    "## 3.1 Pemilihan Framework\n"
                    f"Pemilihan kerangka acuan dilakukan untuk memastikan pekerjaan {short_project.lower()} memiliki dasar kerja yang jelas, terukur, dan dapat dipertanggungjawabkan. "
                    f"Kerangka yang dipilih juga harus mampu menjaga konsistensi antara kebutuhan {client}, ruang lingkup pekerjaan, dan bentuk keluaran yang dijanjikan.\n"
                    "1. Kerangka acuan dipilih berdasarkan relevansi terhadap tujuan pekerjaan, kompleksitas lingkup, dan kebutuhan tata kelola.\n"
                    "2. Kerangka acuan dipakai sebagai pedoman agar analisis, keputusan, dan keluaran pekerjaan mempunyai standar yang konsisten.\n"
                    "| Kerangka / Acuan | Peran dalam Pekerjaan | Relevansi bagi Klien |\n"
                    "| --- | --- | --- |\n"
                    f"{framework_rows}\n"
                    f"{ai_kak_note}\n\n"
                    "## 3.2 Metodologi Pekerjaan\n"
                    f"Metodologi pekerjaan disusun agar pelaksanaan dapat bergerak secara bertahap, mulai dari pemahaman kebutuhan sampai keluaran akhir yang diterima {client}. "
                    f"Dengan metodologi yang jelas, sponsor dapat melihat bagaimana pekerjaan dijalankan, kapan keputusan penting perlu diambil, dan keluaran apa yang dihasilkan pada tiap tahap.\n"
                    "1. Metodologi menjaga agar tiap fase mempunyai tujuan kerja, keluaran, dan titik pengendalian yang jelas.\n"
                    "2. Metodologi juga membantu memastikan bahwa tanggapan terhadap KAK turun menjadi program kerja yang dapat dijalankan.\n"
                    "| Fase | Periode | Tujuan Kerja | Keluaran Utama | Gerbang Mutu |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    f"{methodology_rows}\n"
                    f"- Dengan struktur ini, pekerjaan dapat dikendalikan secara lebih teratur dari awal sampai akhir.\n"
                    f"- Metodologi juga menjadi dasar bagi penyusunan program kerja, jadwal penugasan, dan pengendalian mutu pada bab berikutnya."
                )

            if chapter["id"] == "k_4":
                numbered_phases = "\n".join(
                    f"{idx}. **{item['phase']}** ({item['period']}): {item['activity']} Keluaran utama: {item['deliverable']}."
                    for idx, item in enumerate(phase_plan, start=1)
                )
                return (
                    f"Program kerja dan jadwal penugasan disusun agar pelaksanaan pekerjaan {client} dapat berlangsung terukur, disiplin, dan mudah dipantau sepanjang {timeline}. "
                    f"Bab ini memperlihatkan ritme kerja per fase sekaligus penempatan tenaga ahli sesuai kebutuhan pekerjaan (Data Internal, {year}).\n\n"
                    "## 4.1 Program Kerja / Timeline\n"
                    f"Program kerja dibangun dari urutan fase yang realistis terhadap kebutuhan {short_project.lower()}.\n"
                    f"{numbered_phases}\n"
                    f"- Setiap fase memiliki keluaran utama yang menjadi dasar evaluasi sebelum melanjutkan ke fase berikutnya.\n"
                    f"- Program kerja dijaga agar tetap selaras dengan tujuan pekerjaan dan prioritas keputusan {client}.\n"
                    f"[[GANTT: Jadwal Penugasan | Bulan | {gantt_points}]]\n"
                    "| Fase | Periode | Aktivitas Kunci | Keluaran |\n"
                    "| --- | --- | --- | --- |\n"
                    f"{phase_rows}\n\n"
                    "## 4.2 Tabel Penugasan Tenaga Ahli\n"
                    f"Penugasan tenaga ahli berikut disusun untuk memastikan tiap fase didukung peran yang tepat sesuai fokus pekerjaan.\n"
                    "| Peran Tenaga Ahli | Fokus Tanggung Jawab | Kompetensi Kunci | Keterlibatan |\n"
                    "| --- | --- | --- | --- |\n"
                    f"{expert_rows}\n"
                    f"- Penugasan dapat disesuaikan lebih rinci pada saat kickoff dan finalisasi jadwal bersama {client}.\n"
                    f"- Tabel ini dipakai untuk memastikan tidak ada fase yang berjalan tanpa dukungan peran yang memadai."
                )

            if chapter["id"] == "k_5":
                role_list = "\n".join(
                    [
                        "1. Pimpinan pelaksana menjaga arah umum, kualitas, dan pengambilan keputusan strategis.",
                        "2. Project manager atau PMO lead mengendalikan jadwal, koordinasi, dan tindak lanjut lintas pihak.",
                        "3. Lead substansi menjaga kualitas analisis, keluaran kerja, dan kesesuaian terhadap kebutuhan klien.",
                        "4. Reviewer mutu atau governance memastikan hasil pekerjaan memenuhi standar yang disepakati.",
                    ]
                )
                return (
                    f"Struktur organisasi pelaksana dan komposisi tim pada bab ini dimaksudkan untuk memperjelas tata kerja internal perusahaan penyusun dalam menjalankan pekerjaan {client}. "
                    f"Dengan struktur yang jelas, peran tiap personel dapat ditelusuri dan akuntabilitas pelaksanaan menjadi lebih kuat (Data Internal, {year}).\n\n"
                    "## 5.1 Struktur Organisasi Pelaksana\n"
                    f"{role_list}\n"
                    f"- Struktur pelaksana dirancang agar koordinasi strategis, kendali mutu, dan pelaksanaan harian tetap saling terhubung.\n"
                    f"- Susunan ini juga memudahkan {client} memahami jalur koordinasi dan eskalasi selama pekerjaan berlangsung.\n\n"
                    "## 5.2 Komposisi Tim Tenaga Ahli dan Penugasan\n"
                    f"Komposisi tim tenaga ahli disajikan agar peran, fokus tugas, dan tingkat keterlibatan tiap personel dapat dibaca secara jelas.\n"
                    "| Peran | Fokus Tanggung Jawab | Kompetensi / Pengalaman Kunci | Keterlibatan |\n"
                    "| --- | --- | --- | --- |\n"
                    f"{expert_rows}\n"
                    f"- Uraian tugas tiap peran menjadi dasar pengaturan tanggung jawab selama pelaksanaan.\n"
                    f"- Komposisi tim dapat diperinci lebih lanjut sesuai kebutuhan kickoff dan finalisasi penugasan bersama {client}."
                )

            if chapter["id"] == "k_6":
                return (
                    f"Bab hasil kerja menjelaskan keluaran utama yang akan diterima {client} dari pelaksanaan pekerjaan. "
                    f"Bagian ini dibuat agar ekspektasi terhadap bentuk hasil, fungsi praktis, dan relevansi tiap keluaran dapat dipahami sejak awal (Data Internal, {year}).\n\n"
                    "## 6.1 Hasil Kerja (Deliverable)\n"
                    f"| Kategori Keluaran | Bentuk Keluaran | Tujuan Praktis |\n"
                    "| --- | --- | --- |\n"
                    f"{deliverable_rows}\n\n"
                    "1. Setiap hasil kerja diarahkan untuk mendukung keputusan, pengendalian, atau tindak lanjut yang dibutuhkan klien.\n"
                    "2. Keluaran tidak hanya diposisikan sebagai dokumen, tetapi sebagai artefak kerja yang siap digunakan pada fase berikutnya.\n"
                    f"- Bentuk keluaran akhir akan dikalibrasi dengan kebutuhan rinci pekerjaan {client} tanpa melepaskan inti scope yang telah ditawarkan.\n"
                    f"- Dengan penjelasan ini, klien dapat menilai kecukupan hasil kerja secara lebih objektif."
                )

            if chapter["id"] == "k_7":
                return (
                    f"Fasilitas pendukung pelaksanaan pekerjaan disiapkan agar proses kerja, koordinasi, dan penyusunan keluaran dapat berjalan lebih lancar dan profesional. "
                    f"Bab ini menunjukkan dukungan perusahaan penyusun terhadap kelancaran pekerjaan {client} (Data Internal, {year}).\n\n"
                    "## 7.1 Fasilitas Pendukung Pelaksanaan Pekerjaan\n"
                    f"| Fasilitas | Dukungan yang Disediakan | Manfaat bagi Pelaksanaan |\n"
                    "| --- | --- | --- |\n"
                    f"{facility_rows}\n\n"
                    "1. Fasilitas pendukung disiapkan untuk membantu koordinasi, dokumentasi, dan pengendalian mutu selama pekerjaan berjalan.\n"
                    "2. Dukungan ini memastikan pekerjaan tidak hanya bergantung pada personel, tetapi juga pada sistem kerja yang memadai.\n"
                    f"- Pemanfaatan fasilitas akan disesuaikan dengan sifat pekerjaan dan cara kerja yang disepakati bersama {client}.\n"
                    f"- Dengan dukungan ini, pelaksanaan pekerjaan diharapkan berjalan lebih tertib dan responsif."
                )

            if chapter["id"] == "k_8":
                ai_innovation_note = (
                    f"\n- Pada konteks AI/adopsi, gagasan baru juga diarahkan agar tetap feasible, terkontrol, dan mudah diadopsi organisasi."
                    if ai_mode else ""
                )
                return (
                    f"Inovasi gagasan baru pada bab ini dimaksudkan sebagai nilai tambah perusahaan penyusun dalam melaksanakan pekerjaan {client}. "
                    f"Gagasan tersebut tidak dimaksudkan untuk mengubah tujuan pekerjaan, melainkan memperkuat cara pelaksanaannya agar hasil kerja menjadi lebih bernilai dan lebih mudah ditindaklanjuti (Data Internal, {year}).\n\n"
                    "## 8.1 Inovasi Gagasan Baru\n"
                    f"| Gagasan Baru | Nilai Tambah | Cara Penerapan |\n"
                    "| --- | --- | --- |\n"
                    f"{innovation_rows}\n\n"
                    "1. Gagasan baru dipilih dengan prinsip realistis, relevan terhadap lingkup kerja, dan dapat diterapkan tanpa mengganggu baseline pekerjaan.\n"
                    "2. Inovasi diarahkan untuk memperkuat kualitas hasil, disiplin koordinasi, dan kesiapan tindak lanjut setelah pekerjaan selesai.\n"
                    f"- Inovasi tidak diposisikan sebagai tambahan kosmetik, tetapi sebagai pengungkit untuk meningkatkan nilai manfaat pekerjaan bagi {client}.\n"
                    f"- Dengan demikian, penawaran ini tetap formal terhadap KAK, namun tetap memberi pembeda yang bernilai.{ai_innovation_note}"
                )

        if chapter["id"] == "c_1":
            anchor_line = self._chapter_anchor_line(
                "c_1",
                personalization_pack,
                prefix="Sebagai konteks eksternal yang sudah tervalidasi"
            )
            request_mode_line = (
                "Bagian ini menegaskan bahwa solusi yang diajukan berangkat dari kebutuhan kerja yang hendak dijawab secara formal."
                if normalized_mode == "kak_response"
                else "Penekanan utamanya adalah relevansi bisnis, urgensi, dan arah eksekusi yang realistis."
            )
            ai_context_line = (
                f" Pada konteks AI/adopsi, pengantar proposal juga harus menunjukkan bahwa {client} membutuhkan jalur adopsi yang menjaga {ai_bridge.lower()}."
                if ai_mode else ""
            )
            return (
                f"Bab pembuka untuk {client} perlu langsung memberi pegangan yang jelas mengenai konteks organisasi, tekanan bisnis, dan alasan mengapa jasa konsultasi dibutuhkan saat ini. "
                f"Tujuannya bukan sekadar memperkenalkan klien, melainkan membantu sponsor membaca bahwa inisiatif {short_project.lower()} memang memiliki kaitan langsung dengan hasil seperti {kpi_line}. "
                f"{partnership_line} {request_mode_line}{ai_context_line} {anchor_line}".strip() + f" (Data Internal, {year}).\n\n"
                "## 1.1 Latar Belakang Organisasi Klien\n"
                f"{client} saat ini berada pada fase yang menuntut keputusan kerja lebih tajam terhadap {short_project.lower()}. "
                f"Tekanan yang paling menonjol dapat dibaca dari {self._summarize_phrase(notes, 'kebutuhan menjaga hasil bisnis, kontrol pelaksanaan, dan koordinasi eksekusi', max_words=24).lower()}. "
                f"Karena itu, konteks organisasi perlu dibaca bukan hanya dari profil umum klien, tetapi dari tuntutan agar target seperti {kpi_line} tetap bergerak dalam horizon {timeline or 'proyek'}.\n"
                f"1. Prioritas pertama adalah memastikan keputusan sponsor tetap terhubung ke kebutuhan inti {short_goal.lower()}.\n"
                f"2. Prioritas kedua adalah menjaga bahasa kerja {term_line} agar forum bisnis dan pelaksana membaca arah yang sama.\n"
                f"- Konteks ini menunjukkan bahwa tantangan klien bukan hanya pada aktivitas operasional, tetapi pada bagaimana keputusan, kontrol, dan pelaksanaan dipertemukan dalam satu ritme kerja.\n"
                f"- Outcome awal yang perlu dijaga mencakup {gains_line}.\n"
                f"- Fokus keputusan sejak awal perlu tetap berada pada prioritas inti.\n\n"
                "## 1.2 Alasan Permintaan Jasa Konsultasi\n"
                f"Permintaan jasa konsultasi muncul karena {client} membutuhkan mitra yang mampu menerjemahkan tekanan bisnis menjadi arah kerja, metode, dan keluaran yang lebih siap dijalankan. "
                f"Dalam konteks ini, {WRITER_FIRM_NAME} diposisikan sebagai mitra yang membantu {client} {value_hook_line.lower()} melalui metodologi yang terstruktur, dukungan tim inti yang relevan, dan kontrol delivery yang jelas. "
                f"Artinya, konsultasi dibutuhkan untuk membantu sponsor mengubah kebutuhan yang masih tersebar menjadi program kerja yang lebih rapi, dapat dipertanggungjawabkan, dan dapat diawasi.\n"
                f"- Dukungan konsultasi diperlukan agar keputusan tentang ruang lingkup, prioritas, dan kontrol kerja tidak berubah-ubah ketika program mulai berjalan.\n"
                f"- Perusahaan penyusun juga diharapkan membantu menjaga kualitas terjemahan dari kebutuhan bisnis ke langkah pelaksanaan, terutama pada tema {win_theme.lower()}.\n"
                f"- Dengan keterlibatan yang tepat, {client} memperoleh mitra yang menjaga konsistensi dari konteks awal sampai bentuk keluaran yang benar-benar dapat dipakai.\n"
                f"Pada akhirnya, alasan permintaan jasa konsultasi adalah kebutuhan untuk memastikan bahwa inisiatif {short_project.lower()} tidak hanya terlihat baik di atas kertas, "
                f"tetapi sungguh mempunyai jalur kerja yang rapi untuk diwujudkan. Tujuan akhirnya adalah memberi sponsor {client} dasar keputusan yang lebih kuat sebelum organisasi bergerak lebih jauh."
            )

        if chapter["id"] == "c_2":
            anchor_line = self._chapter_anchor_line(
                "c_2",
                personalization_pack,
                prefix="Sebagai acuan konteks yang sudah tervalidasi"
            )
            ai_problem_line = (
                f" Untuk konteks AI/adopsi, permasalahan juga perlu dibaca dari sisi kesiapan data, kontrol, dan kemampuan organisasi menyerap perubahan."
                if ai_mode else ""
            )
            return (
                f"Bab permasalahan untuk {client} harus membantu sponsor melihat hubungan langsung antara kebutuhan yang terasa di lapangan, akar gap yang mendasarinya, "
                f"dan konsekuensi bisnis bila gap tersebut tidak segera ditangani. Karena itu, rumusan masalah dalam proposal ini sengaja dibuat runtut: dimulai dari kebutuhan klien, "
                f"lalu diperdalam ke konteks bisnis, tantangan utama, akar kesenjangan, implikasi risiko, dan akhirnya kebutuhan solusi.{ai_problem_line} {anchor_line}".strip() + f" (Data Internal, {year}).\n\n"
                "## 2.1 Kebutuhan atau Keinginan Klien\n"
                f"Kebutuhan utama {client} pada proposal ini berputar pada upaya untuk {short_goal.lower()}. "
                f"Di balik kebutuhan tersebut, sponsor sebenarnya sedang mencari cara agar keputusan, pelaksanaan, dan kontrol program tidak berjalan sendiri-sendiri. "
                f"Itulah sebabnya bab ini tidak cukup berhenti pada daftar titik sakit, tetapi harus memperlihatkan apa yang sesungguhnya perlu diperbaiki agar hasil seperti {kpi_line} dapat dikejar dengan lebih stabil.\n"
                f"- Kebutuhan klien harus dibaca sebagai kebutuhan akan arah kerja yang lebih jelas, lebih terukur, dan lebih mudah diawasi.\n"
                f"- Kebutuhan tersebut juga menuntut hubungan yang lebih rapi antara prioritas bisnis, ritme pelaksanaan, dan kualitas keputusan lintas fungsi.\n\n"
                "## 2.2 Konteks Bisnis\n"
                f"Dari sisi konteks bisnis, {client} sedang berada pada situasi yang menuntut kejelasan prioritas atas {short_project.lower()}. "
                f"Tekanan yang muncul tidak berdiri sendiri, melainkan terkait dengan harapan sponsor terhadap {gains_line}, kebutuhan menjaga ritme kerja {term_line}, dan tuntutan agar investasi atau upaya yang dikeluarkan benar-benar menghasilkan perbaikan yang bisa dirasakan. "
                f"Konteks ini menjelaskan mengapa proposal perlu berbicara dalam bahasa keputusan bisnis, bukan hanya bahasa aktivitas proyek.\n\n"
                "## 2.3 Tantangan Utama\n"
                f"Tantangan utama yang dihadapi {client} dapat diringkas ke dalam tiga hal yang saling terkait:\n"
                f"1. Menyatukan arah sponsor, tim inti, dan pelaksana kerja agar membaca tujuan yang sama terhadap {short_project.lower()}.\n"
                f"2. Menjaga agar keputusan kerja tetap cepat tanpa mengorbankan gerbang mutu, tata kelola, dan disiplin prioritas.\n"
                f"3. Mengubah kebutuhan yang masih tersebar menjadi model kerja yang cukup konkret untuk diterjemahkan menjadi keluaran dan pengendalian proyek.\n"
                f"Tantangan ini penting dicatat karena sering kali gejala di lapangan terlihat teknis atau operasional, padahal akar masalahnya justru berada pada sinkronisasi keputusan, kontrol, dan kesiapan eksekusi.\n\n"
                "## 2.4 Akar Kesenjangan\n"
                f"Akar kesenjangan pada proposal ini terletak pada jarak antara kondisi saat ini yang masih belum cukup terkendali dengan kondisi yang dituju, yang menuntut keputusan lebih tegas, ritme kerja lebih konsisten, dan hasil yang lebih terukur. "
                f"Gap tersebut biasanya muncul ketika organisasi sudah mengetahui arah yang diinginkan, tetapi belum memiliki mekanisme yang cukup disiplin untuk menjaga prioritas, tanggung jawab, dan acceptance hasil secara berurutan.\n"
                f"- Gap pertama berada pada penerjemahan kebutuhan bisnis ke langkah kerja yang benar-benar dapat dijalankan.\n"
                f"- Gap kedua berada pada pengendalian keputusan, terutama ketika ketergantungan kerja dan kepentingan lintas fungsi mulai bertemu.\n"
                f"- Gap ketiga berada pada kesiapan pelaksanaan untuk menjaga kualitas hasil sambil tetap bergerak dalam horizon {timeline or 'proyek'}.\n\n"
                "## 2.5 Implikasi / Risiko\n"
                f"Bila kesenjangan tersebut dibiarkan, {client} berisiko menghadapi deviasi terhadap KPI, penurunan kualitas koordinasi, serta bertambahnya tekanan pada sponsor ketika keputusan harus diambil cepat. "
                f"Dalam praktiknya, risiko yang timbul bukan hanya keterlambatan aktivitas, tetapi juga keputusan yang kurang presisi, ruang lingkup yang melebar, dan hasil kerja yang sulit diuji secara objektif. "
                f"Karena itu, pembacaan risiko pada bab ini perlu diposisikan sebagai dasar mengapa intervensi harus dilakukan secara lebih terstruktur.\n\n"
                "## 2.6 Kebutuhan Solusi\n"
                f"Dengan pola masalah seperti di atas, {client} membutuhkan solusi yang tidak hanya menjanjikan perbaikan, tetapi juga mampu menutup gap secara terukur. "
                f"Kebutuhan solusi pada proposal ini berarti kebutuhan akan pendekatan, metodologi, dan desain keluaran yang menjaga hubungan antara target bisnis, kontrol kerja, dan kelayakan implementasi. "
                f"Bab-bab setelah ini sengaja dibangun untuk menjawab kebutuhan tersebut secara bertahap: mulai dari pendekatan, metodologi, desain solusi, sampai model pelaksanaan dan tata kelola yang bisa dijalankan."
            )

        if chapter["id"] == "c_3":
            primary_need, secondary_needs = self._resolve_primary_need(project_goal, notes, regulations)
            secondary_line = self._human_join(
                secondary_needs,
                fallback="Tidak ada kebutuhan sekunder yang perlu diprioritaskan pada tahap awal.",
                max_items=3,
            )
            mode_line = (
                "Bab ini ditulis sebagai respons formal terhadap kebutuhan kerja yang perlu dijawab secara eksplisit."
                if normalized_mode == "kak_response"
                else "Bab ini ditulis sebagai penajaman arah penawaran agar sponsor segera melihat fokus kerja yang paling relevan."
            )
            need_mix = {
                "Problem": 20,
                "Opportunity": 20,
                "Directive": 20,
            }
            need_mix[primary_need] = 55
            for item in secondary_needs[:1]:
                if item in need_mix and item != primary_need:
                    need_mix[item] = 25
            need_chart = self._visual_marker(
                "DONUT",
                "Peta Fokus Kebutuhan Proposal",
                list(need_mix.items()),
            )
            return (
                f"Bab klasifikasi kebutuhan untuk {client} dipakai untuk mengerucutkan berbagai sinyal kebutuhan menjadi fokus kerja yang paling tepat diselesaikan lebih dahulu. "
                f"Dari kombinasi konteks proyek, pain points, dan tekanan bisnis yang muncul, kebutuhan yang paling tepat diposisikan sebagai **{primary_need}**. "
                f"Pilihan ini dibuat agar proposal tidak melebar, melainkan langsung mengunci prioritas yang paling berpengaruh pada outcome seperti {kpi_line}. {mode_line} (Data Internal, {year}).\n\n"
                "## 3.1 Penajaman Kebutuhan Utama yang Dipilih\n"
                f"| Opsi Kebutuhan | Posisi | Alasan Penetapan |\n"
                "| --- | --- | --- |\n"
                f"| Problem | {'Fokus utama' if primary_need == 'Problem' else 'Konteks pendukung'} | Menjawab gap, hambatan, dan isu yang mengganggu target bisnis atau delivery. |\n"
                f"| Opportunity | {'Fokus utama' if primary_need == 'Opportunity' else 'Konteks pendukung'} | Menangkap peluang peningkatan nilai, efisiensi, atau pertumbuhan yang relevan. |\n"
                f"| Directive | {'Fokus utama' if primary_need == 'Directive' else 'Konteks pendukung'} | Menjawab mandat, regulasi, atau kebutuhan kepatuhan yang tidak bisa diabaikan. |\n\n"
                f"{need_chart}\n\n"
                f"1. Fokus utama yang dipilih adalah **{primary_need}** agar arah kerja tetap terkendali sejak awal.\n"
                f"2. Kebutuhan sekunder yang masih diperhatikan adalah {secondary_line.lower()}.\n"
                f"3. Penajaman ini membantu {client} menjaga scope, metode kerja, dan bentuk keluaran tetap relevan terhadap outcome seperti {kpi_line}.\n"
                f"- Kebutuhan sekunder yang tetap diperhatikan: {secondary_line}.\n"
                f"- Penetapan fokus utama juga harus tetap membaca risiko seperti {short_notes.lower()} dan kaitannya dengan target bisnis {kpi_line}.\n\n"
                "## 3.2 Tujuan Utama dan Jenis Proyek\n"
                f"Dengan fokus kebutuhan yang sudah mengerucut, tujuan utama proyek dapat ditegaskan sebagai upaya membantu {client} mencapai {short_project.lower()} melalui model kerja **{project_type}**. "
                f"Jenis proyek ini dipandang paling tepat karena masih sejalan dengan baseline metodologi internal {standard_method}, sekaligus cukup kuat untuk menerjemahkan kebutuhan {primary_need.lower()} "
                f"menjadi deliverable, keputusan, dan quality gate yang dapat dijalankan.\n"
                f"- Tujuan utama proposal adalah mengubah kebutuhan {primary_need.lower()} menjadi program kerja yang lebih terstruktur, terukur, dan mudah dikendalikan.\n"
                f"- Jenis proyek **{project_type}** dipilih karena paling sesuai untuk menjaga outcome {gains_line} tetap realistis dicapai oleh {client}.\n"
                f"- Dengan penajaman ini, arah pendekatan, metodologi, dan solution design dapat difokuskan pada kebutuhan yang benar-benar diputuskan untuk ditangani."
            )

        if chapter["id"] == "c_4":
            ai_framework_note = (
                f" Pada konteks AI/adopsi, acuan ini juga dipakai untuk menjaga {ai_governance.lower()} dan mencegah proposal terjebak pada narasi yang terlalu solution-first."
                if ai_mode else ""
            )
            return (
                f"Pendekatan untuk {client} harus dibangun di atas acuan yang dapat menjelaskan mengapa langkah yang diusulkan layak, aman, dan relevan dengan kebutuhan kerja. "
                f"Oleh karena itu, bab ini tidak dimaksudkan sebagai daftar kerangka acuan semata, melainkan sebagai penjelasan mengenai prinsip dan standar yang dipakai untuk menjaga keputusan tetap konsisten dari awal sampai pelaksanaan. "
                f"Dengan cara ini, sponsor dapat melihat bahwa arah penyelesaian masalah berdiri di atas fondasi yang bisa dipertanggungjawabkan.{ai_framework_note} (Data Internal, {year}).\n\n"
                "## 4.1 Acuan Prinsip/Kerangka/Teori/Regulasi\n"
                f"Pemilihan acuan untuk {client} diarahkan agar kebutuhan {short_goal.lower()} dapat ditangani dengan bahasa kerja yang tetap konsisten terhadap {term_line}. "
                f"Artinya, acuan dipilih bukan karena popularitasnya, tetapi karena benar-benar membantu menyusun urutan keputusan, menjaga kualitas keluaran, dan menempatkan kontrol yang proporsional sejak awal.\n\n"
                "| Acuan | Peran dalam Proposal | Relevansi bagi Klien |\n"
                "| --- | --- | --- |\n"
                f"{framework_rows}\n\n"
                f"Dengan acuan seperti di atas, {client} memperoleh pegangan yang jelas mengenai standar apa yang dipakai untuk menguji apakah solusi yang diusulkan memang selaras dengan konteks, risiko, dan ekspektasi hasil bisnis.\n\n"
                "## 4.2 Standar Penyelesaian Masalah\n"
                f"Standar penyelesaian masalah pada proposal ini dijaga melalui alur kerja yang tertib, bukan lewat lompatan solusi. Bagi {client}, alur tersebut setidaknya harus mengikuti urutan berikut:\n"
                f"1. Memastikan definisi masalah, scope awal, dan target hasil benar-benar disepakati sponsor.\n"
                f"2. Memilih acuan yang paling relevan untuk menjaga kualitas keputusan, kontrol risiko, dan konsistensi pelaksanaan.\n"
                f"3. Menerjemahkan acuan tersebut ke langkah kerja, keluaran, dan gerbang mutu yang mudah diuji pada setiap fase.\n"
                f"4. Menjaga agar seluruh keputusan tetap kembali pada hasil seperti {kpi_line}, bukan melebar ke aktivitas yang tidak memberi dampak nyata.\n"
                f"- Standar ini membantu proposal tetap dapat dipertanggungjawabkan karena setiap langkah punya alasan, batas, dan ukuran hasil yang jelas.\n"
                f"- Standar ini juga menjaga agar diskusi dengan {client} tetap fokus pada apa yang perlu diputuskan, bukan hanya apa yang terdengar canggih.\n"
                f"- Dengan disiplin seperti ini, bab metodologi sesudahnya dapat langsung berbicara tentang cara kerja yang konkret tanpa kehilangan dasar logikanya."
            )

        if chapter["id"] == "c_5":
            ai_method_note = (
                f" Pada konteks AI/adopsi, metodologi juga harus secara eksplisit menjaga readiness, validasi, pilot terkontrol, dan kesiapan adopsi pengguna."
                if ai_mode else ""
            )
            return (
                f"Metodologi untuk {client} perlu menjawab dua hal sekaligus: mengapa cara kerja tertentu dipilih, dan bagaimana cara kerja itu menjaga hasil tetap dapat diuji dari fase ke fase. "
                f"Itulah sebabnya bab ini ditulis sebagai jalur pelaksanaan yang menghubungkan konteks masalah, pendekatan, dan bentuk keluaran secara utuh.{ai_method_note} (Data Internal, {year}).\n\n"
                "## 5.1 Alasan Pemilihan Metodologi\n"
                f"Metodologi dipilih dengan mempertimbangkan tipe proyek **{project_type}**, jenis layanan **{service_type}**, horizon **{timeline}**, serta kebutuhan {client} untuk menjaga {gains_line}. "
                f"Baseline internal yang digunakan adalah {standard_method.lower()}, namun penerapannya disesuaikan agar tetap relevan dengan konteks {short_project.lower()} dan tekanan seperti {short_notes.lower()}. "
                f"Bagi {client}, metodologi ini dipilih karena memberi keseimbangan antara kecepatan mobilisasi, kualitas hasil, dan kontrol keputusan.\n"
                f"1. Metodologi ini menjaga agar tiap fase punya tujuan yang jelas, bukan sekadar daftar aktivitas.\n"
                f"2. Metodologi ini memudahkan sponsor membaca hubungan antara ruang lingkup, keluaran, dan gerbang mutu sebelum program bergerak terlalu jauh.\n"
                f"3. Metodologi ini cukup fleksibel untuk merespons perubahan prioritas, tetapi tetap disiplin terhadap akuntabilitas hasil.\n"
                f"- Dengan metodologi seperti ini, forum kerja {client} bisa fokus pada keputusan penting, bukan menghabiskan energi untuk merapikan ulang arah kerja setiap saat.\n"
                f"- Metodologi ini juga memudahkan penempatan peninjau mutu, penerimaan keluaran, dan kontrol perubahan sejak awal.\n\n"
                "## 5.2 Langkah Kerja dengan Kerangka Acuan Terpilih\n"
                f"Langkah kerja berikut dipakai untuk menerjemahkan pendekatan ke ritme pelaksanaan yang lebih konkret:\n\n"
                "| Fase | Periode | Tujuan Kerja | Keluaran Utama | Gerbang Mutu |\n"
                "| --- | --- | --- | --- | --- |\n"
                f"{methodology_rows}\n\n"
                f"Struktur fase di atas memastikan bahwa {client} selalu memiliki titik evaluasi yang jelas sebelum bergerak ke tahap berikutnya. "
                f"Artinya, setiap keluaran menjadi bahan keputusan yang dipakai untuk menguji apakah program masih berada pada jalur yang mendukung KPI seperti {kpi_line}.\n"
                f"- Setiap fase diikat oleh gerbang mutu agar keputusan tentang lanjut, koreksi arah, atau penyesuaian ruang lingkup tidak dilakukan secara informal.\n"
                f"- Ritme kerja ini membantu {client} menjaga kesinambungan antara forum sponsor, tim inti, dan pelaksana kerja.\n"
                f"- Dengan pola ini, bab desain solusi setelahnya dapat langsung menjelaskan bentuk keluaran dan manfaat praktis tanpa harus mengulang dasar cara kerjanya."
            )

        if chapter["id"] == "c_6":
            deliverable_rows = "\n".join(
                f"| {item['kategori']} | {item['bentuk']} | {item['tujuan']} |"
                for item in self._deliverable_matrix(project_type, service_type, ai_mode=ai_mode)
            )
            ai_solution_note = (
                f"\n- Untuk konteks AI/adopsi, bentuk keluaran juga harus menjaga {ai_bridge.lower()}."
                if ai_mode else ""
            )
            return (
                f"Solution design untuk {client} tidak berhenti pada gambaran target state, tetapi juga harus jelas pada bentuk keluaran yang akan diterima sepanjang engagement. "
                f"Hal ini penting agar sponsor, counterpart kerja, dan tim delivery sama-sama memahami bahwa hasil proposal bukan sekadar ide, melainkan paket kerja yang bisa dipakai untuk menggerakkan {short_project.lower()} secara nyata. "
                f"Dengan pendekatan ini, desain solusi tetap terhubung ke KPI seperti {kpi_line}, sekaligus menjaga nilai proposal tetap terasa konkret dan dapat dijalankan (Data Internal, {year}).\n\n"
                "## 6.1 Solusi/Output Metodologi yang Dibangun\n"
                f"Solusi yang ditawarkan diarahkan untuk membantu {client} bergerak dari kebutuhan {short_goal.lower()} menuju kondisi target yang lebih terukur. "
                f"Karena itu, desain solusi tidak ditulis sebagai konsep yang berdiri sendiri, tetapi sebagai rangkaian output kerja yang diturunkan dari metodologi {standard_method}.\n"
                f"1. Output utama diterjemahkan dari kebutuhan bisnis ke bentuk kerja yang dapat diputuskan sponsor.\n"
                f"2. Bentuk keluaran dipilih agar mudah dipakai oleh tim inti, bukan berhenti sebagai dokumen presentasi.\n"
                f"3. Setiap output harus terhubung ke quality gate, keputusan fase, dan manfaat seperti {gains_line}.\n"
                f"- Output utama dirancang agar keputusan sponsor, kontrol delivery, dan koordinasi stakeholder tetap bergerak dengan istilah kerja {term_line}.\n"
                f"- Target state yang dibangun harus tetap selaras dengan manfaat yang dijanjikan, terutama {gains_line}.\n"
                f"- Tiap output harus bisa diterjemahkan menjadi deliverable, quality gate, dan bahan keputusan pada fase berikutnya.{ai_solution_note}\n\n"
                "## 6.2 Bentuk Keluaran dan Kesesuaian Solusi\n"
                f"| Kategori Keluaran | Bentuk Keluaran | Tujuan Praktis |\n"
                "| --- | --- | --- |\n"
                f"{deliverable_rows}\n\n"
                f"Dengan struktur keluaran seperti ini, {client} dapat melihat bahwa proposal mencakup kombinasi dokumen kerja, forum pendampingan, kegiatan validasi, dan dukungan implementasi yang proporsional. "
                f"Pendekatan ini membuat solusi terasa lebih kredibel karena bukan hanya menjelaskan apa yang akan dibangun, tetapi juga bagaimana bentuk hasilnya akan dipakai oleh sponsor dan tim inti."
            )

        if chapter["id"] == "c_7":
            scope_rows = "\n".join(
                f"| {item['lingkup']} | {item['aktivitas']} | {item['keluaran']} | {item['batasan']} |"
                for item in self._scope_matrix(project_type, service_type, project, notes, ai_mode=ai_mode)
            )
            return (
                f"Ruang lingkup pekerjaan untuk {client} perlu dijelaskan secara tegas agar proposal ini tidak dibaca sebagai komitmen yang kabur. "
                f"Bab ini menegaskan area kerja yang termasuk dalam engagement, bentuk keluaran yang akan diterima, serta batasan dasar yang menjadi asumsi bersama. "
                f"Dengan begitu, sponsor dan tim inti dapat melihat hubungan langsung antara scope kerja, bentuk keluaran, dan outcome seperti {kpi_line} (Data Internal, {year}).\n\n"
                "## 7.1 Lingkup Pekerjaan Utama\n"
                f"| Area Lingkup | Aktivitas Kunci | Bentuk Keluaran | Batasan / Asumsi |\n"
                "| --- | --- | --- | --- |\n"
                f"{scope_rows}\n\n"
                f"Ruang lingkup di atas dirancang agar tetap fokus pada kebutuhan {short_goal.lower()} dan tidak melebar ke area yang belum menjadi prioritas keputusan {client}. "
                f"Karena itu, setiap lingkup kerja selalu dikaitkan ke bentuk keluaran yang nyata, bukan sekadar aktivitas generik.\n"
                f"1. Lingkup kerja dibatasi pada area yang langsung mendukung keputusan dan delivery.\n"
                f"2. Setiap area lingkup harus menghasilkan output yang bisa diuji dan ditinjau sponsor.\n"
                f"3. Area di luar baseline hanya dibahas jika ada keputusan perubahan scope.\n\n"
                "## 7.2 Batasan Pekerjaan dan Asumsi\n"
                f"- Proposal ini mengasumsikan adanya sponsor, PIC inti, dan akses kerja yang cukup dari pihak {client} selama engagement berjalan.\n"
                "- Pekerjaan di luar ruang lingkup utama, termasuk perluasan objek review atau implementasi penuh, hanya dilakukan jika disepakati sebagai perubahan scope.\n"
                f"- Bentuk keluaran utama diarahkan untuk mendukung keputusan, eksekusi, dan quality gate terhadap {short_project.lower()}, bukan menggantikan seluruh fungsi operasional internal klien.\n"
                f"- Dengan batasan ini, ruang lingkup tetap profesional: cukup jelas untuk dijalankan, namun tetap memberi ruang kontrol ketika ada perubahan yang benar-benar material."
            )

        if chapter["id"] == "c_8":
            table_rows = "\n".join(
                f"| {item['phase']} | {item['period']} | {item['deliverable']} | Menjaga progres {short_goal.lower()} |"
                for item in phase_plan
            )
            numbered = "\n".join(
                f"{idx}. **{item['phase']}** ({item['period']}): {item['activity']} Output utama: {item['deliverable']}."
                for idx, item in enumerate(phase_plan, start=1)
            )
            ai_timeline_note = (
                f"- Untuk konteks AI/adopsi, fase kerja juga harus menjaga {ai_bridge.lower()}.\n"
                if ai_mode else ""
            )
            ai_timeline_tail = (
                "\n- Khusus konteks AI, checkpoint fase perlu menegaskan readiness, hasil validasi, dan kesiapan adopsi sebelum solusi diperluas."
                if ai_mode else ""
            )
            return (
                f"Untuk {client}, timeline pekerjaan disusun agar inisiatif {short_project} dapat bergerak stabil selama {timeline}. "
                f"Rencana ini mengikat ritme delivery ke KPI seperti {kpi_line}, memakai istilah kerja {term_line}, "
                f"dan menjaga agar keputusan fase selalu dapat ditelusuri terhadap outcome bisnis. Secara komersial dan delivery, ritme ini juga diarahkan untuk {value_hook} "
                f"sehingga manfaat seperti {gains_line} terasa sejak fase awal (Data Internal, {year}).\n\n"
                "## 8.1 Aktivitas per Fase\n"
                f"{numbered}\n"
                f"- Setiap fase memiliki owner, quality gate, dan exit criteria yang jelas untuk {client}.\n"
                f"- Risiko yang dipantau sejak awal mencakup {short_notes.lower()}, kesiapan data, dan kecepatan keputusan sponsor.\n"
                f"- Baseline metode kerja yang dipakai mengacu pada {standard_method}.\n"
                f"- Ritme eksekusi juga menjaga kesinambungan terhadap istilah kerja {term_line}, sehingga koordinasi antar-tim tidak kehilangan bahasa operasional yang sama.\n"
                f"{ai_timeline_note}"
                f"[[GANTT: Jadwal Pelaksanaan | Bulan | {gantt_points}]]\n\n"
                "## 8.2 Waktu Pelaksanaan dan Deliverable Tiap Fase\n"
                f"Pengaturan waktu tidak hanya membagi durasi, tetapi memastikan setiap deliverable langsung mendukung kebutuhan {short_goal.lower()} "
                f"dan memberi bahan keputusan yang rapi pada forum pengarah proyek {client}. Karena itu, deliverable dibangun berlapis: "
                f"mulai dari baseline, rancangan solusi, keputusan implementasi, sampai stabilisasi hasil. Untuk konteks {client}, pendekatan ini penting agar tim tidak sekadar menyelesaikan aktivitas, "
                f"tetapi juga menjaga relevansi terhadap KPI, dependensi, dan kesiapan adopsi pada tiap fase. Dengan demikian, bila terjadi perubahan kondisi lapangan atau kebutuhan sponsor, "
                f"tim masih punya ruang untuk melakukan penyesuaian yang terkontrol tanpa merusak keseluruhan jalur delivery.\n\n"
                "| Fase | Periode | Deliverable Utama | Kontribusi Bisnis |\n"
                "| --- | --- | --- | --- |\n"
                f"{table_rows}\n"
                f"- Review progres dilakukan berkala agar deviasi timeline dapat dikoreksi sebelum memengaruhi KPI inti seperti {kpi_line}.\n"
                "- Pergeseran jadwal hanya dilakukan melalui change control formal dan keputusan bersama sponsor proyek.\n"
                f"- Deliverable pada akhir fase menjadi dasar transisi ke fase berikutnya, sehingga tidak ada aktivitas {client} yang berjalan tanpa acceptance yang terukur.\n"
                f"- Setiap fase juga memuat checkpoint kesiapan stakeholder, kualitas data, dan keputusan operasional agar implementasi tetap feasible bagi lingkungan kerja {client}."
                f"{ai_timeline_tail}"
            )

        if chapter["id"] == "c_9":
            ai_governance_note = (
                f"\n- Untuk konteks AI, mekanisme kontrol juga harus mencakup {ai_governance.lower()}, sehingga keputusan scale-up selalu memiliki dasar yang dapat dipertanggungjawabkan."
                if ai_mode else ""
            )
            governance_rows = "\n".join(
                [
                    "| Steering Committee | Bulanan / saat keputusan material | Arah strategis, prioritas, dan keputusan lintas sponsor | keputusan strategis dan approval perubahan utama |",
                    "| Project Board | Dua mingguan | Review milestone, isu lintas workstream, dan acceptance deliverable | action log, keputusan korektif, dan status mitigasi |",
                    "| Working Session | Mingguan | Sinkronisasi kebutuhan, dependensi, kesiapan data, dan tindak lanjut harian | notulen kerja, risk update, dan daftar tindakan |",
                    "| Quality Gate Review | Akhir tiap fase | Validasi mutu hasil kerja sebelum fase berikutnya | sign-off fase, daftar perbaikan, dan readiness status |",
                ]
            )
            return (
                f"Tata kelola proyek untuk {client} dirancang agar keputusan strategis, kontrol eksekusi, dan penanganan risiko berjalan dalam satu sistem kerja yang konsisten. "
                f"Fokusnya adalah menjaga {short_project.lower()} tetap selaras dengan KPI seperti {kpi_line}, sambil memakai istilah operasional {term_line} "
                f"agar koordinasi lintas pihak tidak kehilangan konteks. Pola governance ini juga harus mendukung tema utama proposal: {win_theme} dan merefleksikan disiplin delivery {WRITER_FIRM_NAME} yang menghubungkan keputusan sponsor dengan kontrol eksekusi nyata (Data Internal, {year}).\n\n"
                "## 9.1 Mekanisme Pengambilan Keputusan\n"
                f"1. **Steering Committee** menetapkan arah, prioritas, dan keputusan yang berdampak pada scope, biaya, atau timeline program {client}.\n"
                "2. **Project Board** memutuskan isu lintas workstream, approval deliverable utama, dan tindakan korektif terhadap deviasi progres.\n"
                "3. **Working Session** mingguan dipakai untuk memvalidasi kebutuhan, dependency, dan kesiapan data atau user representative.\n"
                "4. **Escalation path** dibuat berjenjang agar isu operasional tidak langsung membebani forum eksekutif, tetapi tetap bisa naik cepat bila berdampak material.\n"
                f"- Keputusan yang mengubah arah program harus terdokumentasi dan ditautkan ke baseline kebutuhan {short_goal.lower()}.\n"
                "- Batas waktu keputusan dan owner tindak lanjut ditetapkan di akhir setiap forum agar tidak muncul keputusan menggantung.\n"
                f"- Semua notulen dan action log menjadi artefak kontrol yang dapat ditinjau ulang oleh sponsor {client}.\n"
                f"- Struktur keputusan juga menjaga agar diskusi tentang KPI, risk appetite, dan kepatuhan POJK tidak terpisah dari diskusi delivery harian.\n\n"
                "| Forum Tata Kelola | Frekuensi | Fokus Keputusan | Output Kontrol |\n"
                "| --- | --- | --- | --- |\n"
                f"{governance_rows}\n\n"
                "## 9.2 Mekanisme Pengendalian Proyek\n"
                f"Pengendalian proyek dilakukan lewat kombinasi dashboard progres, risk register, issue log, quality gate, dan acceptance deliverable. "
                f"Model kontrol ini memastikan aktivitas delivery tetap terkunci pada hasil bisnis, bukan sekadar penyelesaian aktivitas administratif.\n"
                f"Untuk {client}, pengendalian semacam ini penting karena program {short_project.lower()} berpotensi melibatkan banyak dependency dan keputusan cepat. "
                f"Oleh sebab itu, indikator kontrol tidak cukup hanya melihat status task, tetapi juga harus membaca dampaknya pada target bisnis, pengalaman nasabah, dan risiko operasional. "
                f"Setiap forum kontrol diarahkan untuk menjawab tiga hal sekaligus: apakah deliverable sudah sesuai standar, apakah risiko sudah dimitigasi, dan apakah hasil kerja masih bergerak ke KPI yang disepakati.\n"
                f"- Dashboard mingguan menyorot KPI inti, status milestone, isu utama, dan kebutuhan keputusan lanjutan yang memengaruhi {client}.\n"
                "- Setiap deliverable utama melewati review kualitas, validasi stakeholder, dan sign-off sebelum dinyatakan selesai.\n"
                f"- Change request yang berdampak pada ruang lingkup atau biaya akan dinilai terhadap manfaat bisnis, risiko, dan konsekuensi timeline {timeline}.\n"
                f"- Risiko prioritas seperti {short_notes.lower()} dipantau dengan mitigasi, owner, dan target penyelesaian yang eksplisit.\n"
                "- Jika ada deviasi, tim proyek melakukan recovery plan yang dibahas terbuka pada forum tata kelola terdekat.\n"
                f"- Paket kontrol ini menjaga supaya keputusan tentang perubahan, prioritas ulang, atau percepatan kerja tetap transparan bagi sponsor dan tim inti {client}.\n"
                f"- Dengan disiplin kontrol tersebut, governance berfungsi sebagai alat eksekusi nyata, bukan sekadar formalitas rapat proyek."
                f"{ai_governance_note}"
            )

        if chapter["id"] == "c_10":
            experience_rows = "\n".join(
                f"| {item['area']} | {item['relevansi']} | {item['bukti']} | {item['nilai_tambah']} |"
                for item in self._company_experience_rows(
                    project_type=project_type,
                    service_type=service_type,
                    firm_profile=firm_profile,
                    value_map=value_map,
                    ai_mode=ai_mode,
                )
            )
            return (
                f"Profil perusahaan pada proposal ini tidak dimaksudkan sebagai brosur umum, melainkan sebagai bukti bahwa {WRITER_FIRM_NAME} memiliki modal kerja yang relevan untuk membantu {client}. "
                f"Yang ditekankan bukan hanya deskripsi perusahaan, tetapi keterkaitan antara kapabilitas, pengalaman serupa, dan bentuk dukungan yang dibutuhkan oleh inisiatif {short_project.lower()} (Data Internal, {year}).\n\n"
                "## 10.1 Relevansi Profil dan Kapabilitas Perusahaan\n"
                f"{WRITER_FIRM_NAME} diposisikan sebagai {self._summarize_phrase(value_map.get('positioning', ''), 'mitra delivery dan konsultasi yang terstruktur', max_words=24)}. "
                f"Posisi ini diperkuat oleh kemampuan untuk menghubungkan metodologi, governance, dan quality control ke kebutuhan nyata klien, terutama pada konteks {term_line}. "
                f"Bagi {client}, kapabilitas semacam ini penting karena proposal yang baik harus bisa berubah menjadi keputusan dan pekerjaan yang sungguh berjalan, bukan berhenti pada narasi.\n"
                f"1. Relevansi utama kami terletak pada kemampuan menerjemahkan kebutuhan bisnis ke model delivery yang rapi.\n"
                f"2. Kapabilitas perusahaan diarahkan untuk menjaga keputusan sponsor, kualitas hasil, dan kontrol perubahan tetap selaras.\n"
                f"3. Profil perusahaan pada bab ini diposisikan sebagai bukti kesiapan kerja, bukan materi promosi umum.\n"
                f"- Modal kapabilitas yang ditonjolkan meliputi {proof_line}.\n"
                f"- Portofolio internal dan bahan pengalaman perusahaan penyusun dirangkum untuk menunjukkan bukti kerja yang lebih nyata terhadap kebutuhan {client}.\n"
                f"- Kapabilitas dan sertifikasi inti yang relevan: {credential_highlights}.\n"
                f"- Nilai tambah utama yang dibawa adalah {gains_line}.\n\n"
                "## 10.2 Pengalaman Serupa dan Nilai Tambah\n"
                "| Area Pengalaman | Relevansi terhadap Inisiatif | Bukti Kapabilitas | Nilai Tambah untuk Klien |\n"
                "| --- | --- | --- | --- |\n"
                f"{experience_rows}\n\n"
                f"Dengan penyajian seperti ini, {client} dapat melihat bahwa pengalaman {WRITER_FIRM_NAME} tidak diposisikan sebagai klaim umum, melainkan sebagai landasan untuk memberikan hasil yang lebih siap dijalankan. "
                f"Karena itu, bab ini sengaja diarahkan untuk menjawab pertanyaan sederhana sponsor: mengapa perusahaan penyusun ini layak dipercaya untuk mendampingi inisiatif yang diusulkan."
            )

        if chapter["id"] == "c_11":
            expert_rows = "\n".join(
                f"| {item['peran']} | {item['fokus']} | {item['kompetensi']} | {item['keterlibatan']} |"
                for item in self._expert_rows(project_type, service_type, regulations, team_points, ai_mode=ai_mode)
            )
            role_list = "\n".join(
                [
                    "1. **Engagement Lead / Project Director** menjaga arah kemitraan dan keputusan strategis sponsor.",
                    "2. **Project Manager / PMO Lead** mengendalikan ritme kerja, milestone, dan quality gate.",
                    "3. **Lead Fungsional / Solusi** menjaga kualitas desain solusi dan relevansi output kerja.",
                    "4. **Reviewer Governance / Quality** memastikan disiplin review, sign-off, dan kontrol risiko.",
                ]
            )
            if ai_mode:
                role_list += "\n5. **Business Translator / AI Lead** menjembatani use case bisnis, integrasi solusi, dan kesiapan adopsi."
            involvement_chart = self._visual_marker(
                "BAR",
                "Intensitas Keterlibatan Tim Inti",
                [
                    ("Engagement Lead", 65),
                    ("PMO Lead", 90),
                    ("Lead Solusi", 85),
                    ("Quality Reviewer", 70),
                ] + ([("AI Lead", 75)] if ai_mode else []),
                unit="Skor keterlibatan",
            )
            return (
                f"Struktur tim proyek untuk {client} dibentuk agar pengambilan keputusan, pengawasan mutu, dan eksekusi lapangan bergerak seirama. "
                f"Komposisi tim tidak diperlakukan sebagai daftar jabatan semata, tetapi sebagai mekanisme untuk menjaga kualitas output terhadap target {kpi_line} dan kebutuhan {short_goal.lower()} (Data Internal, {year}).\n\n"
                "## 11.1 Struktur Tim Proyek\n"
                f"{role_list}\n"
                f"- Komposisi inti yang direncanakan mengacu pada baseline internal: {team_summary}.\n"
                f"- Referensi kapabilitas dan sertifikasi internal yang mendukung tim: {credential_highlights}.\n"
                f"- Counterpart dari pihak {client} idealnya mencakup sponsor bisnis, PIC operasional, PIC teknologi/data, dan reviewer governance.\n"
                "- Alokasi resource dapat dinaikkan atau disesuaikan mengikuti fase kerja, tingkat risiko, dan kebutuhan keputusan cepat.\n\n"
                f"{involvement_chart}\n\n"
                "## 11.2 Tabel Tenaga Ahli dan Kualifikasi\n"
                "| Peran Tenaga Ahli | Fokus Tanggung Jawab | Kompetensi / Pengalaman Kunci | Keterlibatan |\n"
                "| --- | --- | --- | --- |\n"
                f"{expert_rows}\n\n"
                f"Dengan tabel ini, {client} memperoleh gambaran yang lebih konkret mengenai siapa yang menjaga keputusan, siapa yang mengawal delivery, dan siapa yang mengamankan mutu hasil kerja. "
                f"Pendekatan ini membuat bab tenaga ahli lebih mudah diaudit dan lebih meyakinkan bagi sponsor proyek."
            )

        if chapter["id"] == "c_12":
            payment_lines = "\n".join(
                f"{idx}. **{label}** sebesar {portion} dari estimasi investasi, ditagihkan setelah milestone terkait diterima."
                for idx, (label, portion) in enumerate(payment_plan, start=1)
            )
            ai_cost_note = (
                "\n- Untuk konteks AI, komponen biaya juga mencerminkan effort readiness, kontrol, validasi bertahap, dan enablement perubahan agar solusi tidak berhenti pada eksperimen."
                if ai_mode else ""
            )
            return (
                f"Model pembiayaan untuk {client} disusun agar komitmen biaya, mekanisme pembayaran, dan batas ruang lingkup tetap jelas sejak awal. "
                f"Tujuannya adalah menjaga proyek {short_project.lower()} tetap dapat dieksekusi tanpa ambiguitas komersial, sambil memberi ruang kontrol terhadap perubahan yang benar-benar material "
                f"dan memastikan investasi tetap tertaut pada manfaat yang dijanjikan, terutama {gains_line}. Struktur ini juga mencerminkan cara {WRITER_FIRM_NAME} menjaga disiplin komersial tetap selaras dengan delivery reality dan outcome klien (Data Internal, {year}).\n\n"
                "## 12.1 Biaya dan Tahapan Pembayaran\n"
                f"Estimasi investasi awal untuk engagement ini adalah **{budget or 'menyesuaikan scope final'}**, dengan tipe layanan **{service_type}** pada model proyek **{project_type}** selama **{timeline}**. "
                f"Baseline komersial internal yang menjadi acuan adalah {commercial_summary}. Dengan demikian, pembahasan biaya tidak berdiri sebagai angka semata, tetapi sebagai representasi dari komitmen kerja, quality gate, dan acceptance deliverable yang menjadi standar {WRITER_FIRM_NAME}.\n\n"
                "| Komponen | Rancangan Komersial |\n"
                "| --- | --- |\n"
                f"| Estimasi investasi | {budget or 'Menyesuaikan scope final'} |\n"
                f"| Model engagement | {service_type} - {project_type} |\n"
                f"| Durasi acuan | {timeline} |\n"
                f"| Basis commercial | {firm_data.get('commercial', 'Menyesuaikan kebijakan internal')} |\n"
                "| Mekanisme review | Review milestone, acceptance deliverable, dan approval perubahan scope |\n\n"
                f"{payment_lines}\n"
                f"- Nilai pembayaran dikaitkan dengan deliverable yang benar-benar selesai, bukan hanya dengan berjalannya waktu proyek.\n"
                "- Jika scope berubah secara material, dampak biaya akan dibahas melalui change request dan approval tertulis.\n\n"
                "## 12.2 Model Pekerjaan dan Batasan Pekerjaan\n"
                f"Model kerja dirancang agar ruang lingkup delivery tetap fokus pada kebutuhan {short_goal.lower()} dan KPI seperti {kpi_line}. "
                f"Karena itu, proposal komersial ini diposisikan sebagai baseline kerja yang transparan, bukan daftar komitmen tanpa batas.\n"
                f"Untuk {client}, disiplin komersial seperti ini penting agar keputusan investasi tetap berada dalam koridor manfaat bisnis dan risk appetite yang dapat dipertanggungjawabkan. "
                f"Model pembiayaan yang baik harus mampu menjaga fleksibilitas eksekusi tanpa membuat biaya berkembang secara tidak terkendali.\n"
                f"- Harga mengasumsikan ketersediaan sponsor, PIC, data, dan akses kerja dari pihak {client} sesuai jadwal yang disepakati.\n"
                "- Biaya pihak ketiga, lisensi, perangkat, perjalanan khusus, atau pekerjaan di luar ruang lingkup inti hanya dimasukkan jika dinyatakan eksplisit.\n"
                f"- Aktivitas tambahan yang muncul akibat perluasan kebutuhan, regulasi baru, atau permintaan percepatan di luar baseline {timeline} akan dievaluasi terpisah.\n"
                "- Dengan batasan ini, pembiayaan tetap profesional: cukup fleksibel untuk adaptasi, tetapi tetap disiplin terhadap kontrol biaya dan outcome.\n"
                f"- Nilai investasi dijaga tetap proporsional terhadap manfaat yang diharapkan, khususnya {gains_line}."
                f"{ai_cost_note}"
            )

        if chapter["id"] == "c_closing":
            verified_contact = self._verified_firm_contact_block(firm_profile)
            closing_contact = verified_contact or "- Kontak resmi writer firm akan diberikan sesuai data yang telah diverifikasi."
            visit_line = (
                "Alamat kantor di atas dapat digunakan sebagai rujukan apabila diperlukan kunjungan on-site, pertemuan lanjutan, atau verifikasi administratif secara langsung."
                if verified_contact else
                "Apabila dibutuhkan kunjungan on-site atau pertemuan lanjutan, lokasi dan detail kontak resmi akan dikonfirmasi melalui data perusahaan yang telah diverifikasi."
            )
            mode_opening = (
                f"Terima kasih atas kesempatan yang diberikan kepada {WRITER_FIRM_NAME} untuk menyampaikan tanggapan Kerangka Acuan Kerja bagi {client}. "
                if normalized_mode == "kak_response" else
                f"Terima kasih atas kesempatan yang diberikan kepada {WRITER_FIRM_NAME} untuk menyusun proposal bagi {client}. "
            )
            aspiration_line = (
                f"Kami berharap pekerjaan {short_project.lower()} dapat berjalan sebagai fondasi kolaborasi yang profesional, terukur, dan memberi hasil yang benar-benar dapat dipakai oleh {client}."
            )
            return (
                f"{mode_opening}"
                f"Kami memandang inisiatif {short_project.lower()} sebagai fondasi kemitraan profesional yang harus terasa rapi dan dapat dipertanggungjawabkan. "
                f"Komitmen kami adalah membantu {client} bergerak dari kebutuhan {short_goal.lower()} menuju hasil yang konkret, terukur, dan dapat dijalankan secara disiplin (Data Internal, {year}).\n\n"
                "## Apresiasi dan Komitmen Kemitraan\n"
                f"1. Kami mengapresiasi keterbukaan {client} dalam mengangkat konteks, tantangan, dan target bisnis yang menjadi dasar proposal ini.\n"
                f"2. Kami berkomitmen menjaga kualitas kolaborasi melalui cara kerja yang jelas, komunikasi yang responsif, dan deliverable yang dapat ditindaklanjuti, sejalan dengan positioning {WRITER_FIRM_NAME} sebagai {self._summarize_phrase(value_map.get('positioning', ''), 'mitra delivery dan konsultasi yang terstruktur', max_words=24)}.\n"
                f"3. Fokus awal kemitraan diarahkan pada prioritas seperti {kpi_line}, dengan tata kelola dan ritme eksekusi yang stabil sejak kickoff.\n"
                f"4. {aspiration_line}\n"
                f"- Nilai kemitraan yang ingin dibangun adalah kombinasi antara kecepatan delivery, {term_line}, dan manfaat seperti {gains_line}.\n"
                f"- Bila proposal ini disetujui, tahap berikutnya adalah finalisasi scope, konfirmasi tim inti, dan penetapan agenda kickoff bersama {client}.\n\n"
                "## Informasi Kontak dan Langkah Lanjutan\n"
                f"Untuk melanjutkan pembahasan, {WRITER_FIRM_NAME} siap menindaklanjuti review proposal, penajaman ruang lingkup, dan penyesuaian komersial yang diperlukan.\n"
                f"{closing_contact}\n"
                f"- {visit_line}\n"
                "- Agenda lanjutan yang disarankan: review scope final, konfirmasi sponsor dan PIC, lalu penjadwalan workshop kickoff.\n"
                "- Dengan fondasi ini, kemitraan dapat dimulai secara profesional dan siap dijalankan."
            )

        return ""

    def _resolve_chapters(self, chapter_id: Optional[str], proposal_mode: str = "canvassing") -> List[Dict[str, Any]]:
        structure = self._structure_for_mode(proposal_mode)
        normalized_id = (chapter_id or "").strip()
        if not normalized_id or normalized_id.lower() in {"all", "semua"}:
            return structure

        selected = [chapter for chapter in structure if chapter["id"] == normalized_id]
        if selected:
            return selected

        normalized = normalized_id.lower()
        selected = [chapter for chapter in structure if chapter["title"].strip().lower() == normalized]
        if selected:
            return selected

        raise ValueError(f"Unknown chapter_id: {normalized_id}")

    def build_preview_outline(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chapter_id = (data or {}).get("chapter_id")
        proposal_mode = self._normalize_proposal_mode((data or {}).get("mode_proposal", "canvassing"))
        try:
            chapters = self._resolve_chapters(chapter_id, proposal_mode=proposal_mode)
        except ValueError:
            chapters = self._structure_for_mode(proposal_mode)

        client = (data or {}).get("nama_perusahaan", "Klien")
        objective = (data or {}).get("konteks_organisasi", "").strip() or "tujuan proyek belum diisi"
        issues = (data or {}).get("permasalahan", "").strip() or "pain points belum diisi"
        need_type = (data or {}).get("klasifikasi_kebutuhan", "").strip() or "belum dipilih"
        project_type = (data or {}).get("jenis_proyek", "").strip() or "belum dipilih"
        service_type = (data or {}).get("jenis_proposal", "").strip() or "belum dipilih"
        frameworks = (data or {}).get("potensi_framework", "").strip() or "belum dipilih"
        timeline = (data or {}).get("estimasi_waktu", "").strip() or "belum ditentukan"
        budget = (data or {}).get("estimasi_biaya", "").strip() or "belum ditentukan"
        ai_profile = self._build_ai_adoption_profile(
            client=client,
            project=objective,
            project_goal=need_type,
            project_type=project_type,
            timeline=timeline,
            notes=issues,
            regulations=frameworks,
            research_bundle={},
        )
        ai_mode = bool(ai_profile.get("enabled"))

        preview_map = {
            "c_1": f"Menetapkan konteks organisasi {client} dan objektif inisiatif: {objective}.",
            "c_2": f"Mengurai kebutuhan dan akar masalah klien berdasarkan pain points: {issues}.",
            "c_3": f"Mengerucutkan pilihan kebutuhan dari {need_type} menjadi fokus utama yang benar-benar diselesaikan, lalu memvalidasi jenis proyek {project_type}.",
            "c_4": f"Menautkan kebutuhan klien dengan framework/regulasi utama: {frameworks}.",
            "c_5": f"Menjelaskan pemilihan metodologi delivery untuk layanan {service_type}.",
            "c_6": "Mendetailkan target state, solusi, serta bentuk keluaran seperti dokumen, pendampingan, kegiatan, atau dukungan implementasi.",
            "c_7": "Menegaskan ruang lingkup kerja, keluaran tiap area kerja, serta batasan dan asumsi utama pelaksanaan.",
            "c_8": f"Menyusun rencana fase, milestone, dan deliverable berdasarkan durasi {timeline}.",
            "c_9": "Merumuskan governance proyek: forum keputusan, eskalasi isu, dan quality gate.",
            "c_10": "Menunjukkan profil perusahaan penyusun dan pengalaman serupa yang relevan dengan kebutuhan klien.",
            "c_11": f"Mendetailkan struktur tim proyek dan tabel tenaga ahli yang dibutuhkan untuk model {service_type}.",
            "c_12": f"Mendefinisikan model pembiayaan, termin pembayaran, dan batasan scope dengan estimasi {budget}.",
            "c_closing": f"Menutup proposal dengan apresiasi kemitraan, kontak resmi {WRITER_FIRM_NAME}, dan langkah tindak lanjut bersama {client}.",
        }
        if ai_mode:
            preview_map.update({
                "c_1": f"Menetapkan konteks bisnis {client}, mengapa use case ini penting sekarang, dan outcome AI yang ingin dikejar: {objective}.",
                "c_2": f"Mengurai akar masalah, gap current state vs target state, serta hambatan kesiapan/adopsi berdasarkan pain points: {issues}.",
                "c_3": f"Mengkategorikan kebutuhan ke {need_type} sambil membaca tingkat kesiapan adopsi AI dan bentuk intervensi yang realistis.",
                "c_4": f"Menautkan kebutuhan klien ke prinsip responsible adoption, kontrol risiko, dan framework/regulasi utama: {frameworks}.",
                "c_5": "Menjelaskan metodologi bertahap: readiness, validasi, pilot/rollout, dan learning loop yang tetap terkontrol.",
                "c_6": "Mendetailkan solution design dan bentuk keluaran yang feasible untuk dioperasikan, dimonitor, dan diadopsi pengguna.",
                "c_7": "Menegaskan ruang lingkup kerja yang realistis untuk menjaga fokus use case, kontrol, dan kesiapan implementasi.",
                "c_8": f"Menyusun fase readiness, pilot/rollout, milestone, dan deliverable berdasarkan durasi {timeline}.",
                "c_9": "Merumuskan governance, approval, stop/go, monitoring, dan accountability yang menjaga adopsi tetap aman.",
                "c_10": "Menunjukkan kapabilitas perusahaan penyusun yang relevan dengan ruang lingkup AI, governance, dan delivery yang bertanggung jawab.",
                "c_11": f"Menetapkan struktur tim lintas bisnis, engineering, governance, dan change enablement untuk model {service_type}.",
                "c_12": f"Mendefinisikan model pembiayaan berdasarkan effort readiness, kontrol, rollout, dan change enablement dengan estimasi {budget}.",
            })

        if proposal_mode == "kak_response":
            preview_map = {
                "k_1": "Menyajikan informasi perusahaan, struktur organisasi perusahaan, dan pengalaman pekerjaan sejenis sebagai dasar kredibilitas penyedia.",
                "k_2": "Menunjukkan pemahaman terhadap KAK sekaligus tanggapan dan saran atas butir pekerjaan, tujuan, ruang lingkup, dan keluaran yang diminta.",
                "k_3": f"Menjelaskan pendekatan, pemilihan kerangka acuan, dan metodologi pekerjaan yang paling tepat untuk {client}.",
                "k_4": f"Menyusun program kerja, timeline, dan tabel penugasan tenaga ahli berdasarkan durasi {timeline}.",
                "k_5": "Menguraikan struktur organisasi pelaksana, komposisi tim, dan uraian tugas tiap peran utama.",
                "k_6": "Menegaskan hasil kerja atau deliverable yang akan diterima klien sesuai kebutuhan pekerjaan.",
                "k_7": "Menjelaskan fasilitas pendukung yang disiapkan untuk menunjang pelaksanaan pekerjaan secara profesional.",
                "k_8": "Menyampaikan inovasi atau gagasan baru yang relevan dan memberi nilai tambah pada pelaksanaan pekerjaan.",
                "c_closing": f"Menutup proposal dengan apresiasi kepada {client}, aspirasi pelaksanaan pekerjaan, serta kontak resmi {WRITER_FIRM_NAME} untuk tindak lanjut dan kunjungan on-site.",
            }

        return [
            {
                "id": chapter["id"],
                "title": chapter["title"],
                "preview": preview_map.get(chapter["id"], "Ringkasan konten bab akan disesuaikan dengan konteks klien."),
                "subsections": chapter["subs"],
            }
            for chapter in chapters
        ]

    def _build_research_bundle_uncached(
        self,
        base_client: str,
        regulations: str,
        include_collaboration: bool = True,
        ai_context: str = "",
    ) -> Dict[str, str]:
        ai_mode = self._ai_scope_signal_summary(ai_context, regulations).get("enabled", False)
        futures = {
            "profile": self.io_pool.submit(Researcher.get_entity_profile, base_client),
            "news": self.io_pool.submit(Researcher.get_latest_client_news, base_client),
            "track_record": self.io_pool.submit(Researcher.get_client_track_record, base_client),
            "regulations": self.io_pool.submit(Researcher.get_regulatory_data, regulations)
        }
        if ai_mode:
            futures["ai_posture"] = self.io_pool.submit(
                Researcher.get_client_ai_posture,
                base_client,
                ai_context,
            )
        if include_collaboration:
            futures["collaboration"] = self.io_pool.submit(
                Researcher.get_client_writer_collaboration, base_client, WRITER_FIRM_NAME
            )
        try:
            return {
                "profile": futures["profile"].result(timeout=8),
                "news": futures["news"].result(timeout=8),
                "track_record": futures["track_record"].result(timeout=8),
                "collaboration": futures["collaboration"].result(timeout=8) if include_collaboration else "",
                "regulations": futures["regulations"].result(timeout=8),
                "ai_posture": futures["ai_posture"].result(timeout=8) if ai_mode else "",
            }
        except Exception:
            return self._fallback_research_bundle(base_client, include_collaboration, ai_mode=ai_mode)

    def prefetch_research_bundle(
        self,
        base_client: str,
        regulations: str,
        include_collaboration: bool = True,
        ai_context: str = "",
    ) -> str:
        key = self._cache_key("research", base_client, regulations, str(include_collaboration), ai_context)
        if self._get_cached_research_bundle(key):
            logger.debug(f"Serper | Research bundle (cached) | client={base_client}")
            return "cached"

        with self._cache_lock:
            if key in self._research_inflight:
                logger.debug(f"Serper | Research bundle (warming) | client={base_client}")
                return "warming"
            event = threading.Event()
            self._research_inflight[key] = event

        logger.info(f"Serper | Research bundle (prefetching) | client={base_client} | ai_context={bool(ai_context)}")

        def worker() -> None:
            try:
                bundle = self._build_research_bundle_uncached(
                    base_client=base_client,
                    regulations=regulations,
                    include_collaboration=include_collaboration,
                    ai_context=ai_context,
                )
                self._store_research_bundle(key, bundle)
                has_external_data = any(
                    "Sumber eksternal" in str(v) or len(str(v)) > 50
                    for v in (bundle or {}).values()
                )
                logger.info(f"Serper | Research bundle (completed) | client={base_client} | external_data={has_external_data}")
            except Exception as e:
                logger.warning(f"Serper | Research bundle failed | client={base_client} | error={str(e)[:80]}")
            finally:
                with self._cache_lock:
                    inflight = self._research_inflight.pop(key, None)
                    if inflight:
                        inflight.set()

        threading.Thread(
            target=worker,
            name=f"research-prefetch-{SchemaMapper.normalize_key(base_client) or 'client'}",
            daemon=True
        ).start()
        return "warming"

    def _get_research_bundle(
        self,
        base_client: str,
        regulations: str,
        include_collaboration: bool = True,
        ai_context: str = "",
    ) -> Dict[str, str]:
        key = self._cache_key("research", base_client, regulations, str(include_collaboration), ai_context)
        cached = self._get_cached_research_bundle(key)
        if cached:
            return cached

        leader = False
        event: Optional[threading.Event] = None
        with self._cache_lock:
            event = self._research_inflight.get(key)
            if event is None:
                event = threading.Event()
                self._research_inflight[key] = event
                leader = True

        if not leader:
            event.wait(timeout=12)
            cached = self._get_cached_research_bundle(key)
            if cached:
                return cached

        try:
            bundle = self._build_research_bundle_uncached(
                base_client=base_client,
                regulations=regulations,
                include_collaboration=include_collaboration,
                ai_context=ai_context,
            )
            return self._store_research_bundle(key, bundle)
        finally:
            with self._cache_lock:
                inflight = self._research_inflight.pop(key, None)
                if inflight:
                    inflight.set()

    def _build_proposal_contract(
        self,
        client: str,
        project: str,
        budget: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        selected_chapters: List[Dict[str, Any]],
        research_bundle: Dict[str, str],
        firm_data: Dict[str, str],
        personalization_pack: Dict[str, Any],
        value_map: Dict[str, Any],
        proposal_mode: str = "canvassing",
    ) -> str:
        cache_key = self._cache_key(
            "contract", client, project, budget, service_type, project_goal,
            project_type, timeline, notes, regulations, "|".join([c["id"] for c in selected_chapters]),
            personalization_pack.get("industry", ""),
            personalization_pack.get("relationship_mode", ""),
            "|".join(personalization_pack.get("kpi_blueprint", []) or []),
            str(personalization_pack.get("ai_mode", False)),
            str((personalization_pack.get("ai_adoption_profile") or {}).get("summary", "")),
            value_map.get("value_statement", ""),
            value_map.get("win_theme", ""),
            "|".join(value_map.get("proof_points", []) or []),
        )
        cached = self._cache_get(self._proposal_contract_cache, cache_key)
        if cached:
            return cached

        chapter_titles = ", ".join([c["title"] for c in selected_chapters])
        normalized_mode = self._normalize_proposal_mode(proposal_mode)
        prompt = f"""
        Buat "Proposal Contract" ringkas untuk menjaga kualitas dan koherensi lintas bab.
        Konteks:
        - Klien: {client}
        - Inisiatif: {project}
        - Service Type: {service_type}
        - Jenis Proyek: {project_type}
        - Kebutuhan: {project_goal}
        - Durasi: {timeline}
        - Estimasi Biaya: {budget}
        - Pain Points: {notes}
        - Framework: {regulations}
        - Bab yang ditulis: {chapter_titles}
        - Baseline Metodologi: {firm_data.get('methodology', '')}
        - Baseline Team: {firm_data.get('team', '')}
        - Baseline Commercial: {firm_data.get('commercial', '')}
        - Profil OSINT: {research_bundle.get('profile', '')}
        - Berita OSINT: {research_bundle.get('news', '')}
        - Track Record OSINT: {research_bundle.get('track_record', '')}
        - Histori Kolaborasi {WRITER_FIRM_NAME} & {client}: {research_bundle.get('collaboration', '')}
        - Industry Pack: {personalization_pack.get('industry', 'Lintas Industri')}
        - Relationship Mode: {personalization_pack.get('relationship_mode', 'new')}
        - KPI Blueprint: {', '.join(personalization_pack.get('kpi_blueprint', []))}
        - Terminologi Prioritas: {', '.join(personalization_pack.get('terminology', []))}
        - AI Adoption Summary: {(personalization_pack.get('ai_adoption_profile') or {}).get('summary', '')}
        - Company DNA Positioning: {value_map.get('positioning', '')}
        - Proposal Promise: {value_map.get('proposal_promise', '')}
        - Value Statement: {value_map.get('value_statement', '')}
        - Win Theme: {value_map.get('win_theme', '')}
        - Client Gains: {', '.join(value_map.get('client_gains', []) or [])}
        - Proof Points: {', '.join(value_map.get('proof_points', []) or [])}
        - Mode Proposal: {normalized_mode}
        - Review Rule: Draft ini targetnya 80% siap pakai; sisakan ruang bagi reviewer manusia untuk menyempurnakan nuansa relasi, komersial, dan pesan penutup.
        - Aturan AI tersembunyi: bila konteks proposal terkait AI, narasi harus terasa dimulai dari use case bisnis, lalu dikendalikan oleh kesiapan data/model, arsitektur, governance, kapabilitas tim, dan perubahan cara kerja. Jangan menuliskan enam label itu secara eksplisit.

        OUTPUT WAJIB (tanpa markdown code block, <= 220 kata):
        1) Narasi Inti (1-2 kalimat)
        2) Terminologi Kanonis (maks 6 istilah)
        3) Prinsip Konsistensi Antarbab (maks 5 butir)
        4) Larangan Gaya Tulis (maks 3 butir)
        """
        contract = ""
        if not self._throughput_mode():
            try:
                res = self.ollama.chat(
                    model=LLM_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'num_ctx': 16384, 'num_predict': 700, 'temperature': 0.15}
                )
                contract = (res.get('message', {}).get('content', '') or '').strip()
            except Exception:
                contract = ""

        if not contract:
            contract = (
                f"Narasi Inti: {value_map.get('value_statement') or 'Proposal harus menjawab kebutuhan bisnis klien secara konkret, terukur, dan eksekutabel.'}\n"
                "Terminologi Kanonis: deliverable, milestone, target state, governance, quality gate, risiko.\n"
                "Prinsip Konsistensi Antarbab: istilah konsisten, alur masalah-ke-solusi jelas, "
                f"setiap bab menegaskan nilai untuk klien, timeline sinkron dengan deliverable, tata kelola tegas, hindari repetisi.\n"
                "Larangan Gaya Tulis: filler generik, klaim tanpa dasar, paragraf tanpa tindakan."
            )

        self._cache_store(self._proposal_contract_cache, cache_key, contract, max_size=96)
        return contract

    def _build_chapter_prompt(
        self,
        chapter: Dict[str, Any],
        client: str,
        project: str,
        budget: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        firm_data: Dict[str, str],
        firm_profile: Dict[str, str],
        research_bundle: Dict[str, str],
        personalization_pack: Dict[str, Any],
        value_map: Dict[str, Any],
        proposal_contract: str,
        proposal_mode: str,
        target_words: int
    ) -> Dict[str, Any]:
        try:
            current_year = datetime.now().year
            profile_data = research_bundle.get('profile', '')
            track_record_data = research_bundle.get('track_record', '')
            collaboration_data = research_bundle.get('collaboration', '')
            global_data = "\n".join([item for item in [profile_data, track_record_data, collaboration_data] if item])
            client_news = research_bundle.get('news', '')
            regulation_data = research_bundle.get('regulations', '')
            allowed_external_citations = self._collect_allowed_external_citations(research_bundle)

            ctx_key = self._cache_key("chapter_ctx", client, project, budget, chapter.get('id', ''), chapter.get('keywords', ''))
            cached_ctx = self._cache_get(self._chapter_context_cache, ctx_key)
            if cached_ctx:
                structured_row_data = cached_ctx.get("structured_row_data", "")
                rag_data = cached_ctx.get("rag_data", "")
            else:
                structured_row_data = self.kb.get_exact_context(client, project, budget)
                rag_data = self.kb.query(client, project, chapter['keywords'])
                self._cache_store(
                    self._chapter_context_cache,
                    ctx_key,
                    {"structured_row_data": structured_row_data, "rag_data": rag_data},
                    max_size=256
                )
            
            persona = PERSONAS.get(chapter.get('id', 'default'), PERSONAS['default'])
            subs = "\n".join([f"- {s}" for s in chapter['subs']])
            
            visual_prompt = ""
            if chapter.get('visual_intent') == "gantt":
                visual_prompt = f"Mandatory Timeline Visual: [[GANTT: Jadwal Pelaksanaan | Bulan | Fase 1,0,2; Fase 2,2,4]]. Total timeline: {timeline}."
            elif chapter.get('visual_intent') == "flowchart":
                visual_prompt = "Tambahkan alur tahapan metodologi dalam bentuk bullet bertingkat yang jelas (fase -> aktivitas -> output)."

            terminology_list = personalization_pack.get("terminology", []) or []
            kpi_blueprint = personalization_pack.get("kpi_blueprint", []) or []
            relationship_guidance = personalization_pack.get("relationship_guidance", "")
            relationship_mode = personalization_pack.get("relationship_mode", "new")
            relationship_source = personalization_pack.get("relationship_source", "osint")
            profile_summary = personalization_pack.get("profile_summary", "")
            ai_profile = personalization_pack.get("ai_adoption_profile", {}) or {}
            ai_mode = bool(ai_profile.get("enabled"))
            normalized_mode = self._normalize_proposal_mode(proposal_mode)
            ai_posture = research_bundle.get("ai_posture", "")
            ai_guidance = str((ai_profile.get("chapter_guidance") or {}).get(chapter.get("id", ""), "")).strip()
            initiative_facts = personalization_pack.get("initiative_facts", []) or []
            if initiative_facts:
                anchors_text = self._human_join([
                    self._sanitize_anchor_fact(item.get('fact', '') or "")
                    for item in initiative_facts[:3]
                ], fallback="", max_items=3)
            else:
                anchors_text = ""

            extra = (
                f"[PROPOSAL CONTRACT]\n{proposal_contract}\n"
                f"[COMPANY_DNA]\n{self._format_value_map(value_map)}\n"
                f"[GLOBAL] Proposal ini wajib mempertahankan kedalaman konten tingkat eksekutif dan total dokumen maksimal {MAX_PROPOSAL_PAGES} halaman. "
                "Setiap bab harus memiliki konteks spesifik klien, poin yang dapat ditindaklanjuti, dan tidak generik. Gunakan kombinasi numbering dan bullet yang rapi di setiap H2, namun tetap padat dan tidak banyak whitespace."
            )
            extra += (
                " [DRAFT_POLICY] Tulis sebagai draft proposal berkualitas tinggi yang siap dipakai sekitar 80%, "
                "dengan ruang review manusia tersisa untuk kalibrasi hubungan, penajaman komersial, dan sentuhan personal terakhir. "
                "Jangan terdengar seperti placeholder atau boilerplate."
            )
            if normalized_mode == "kak_response":
                extra += (
                    " [PROPOSAL_MODE] Ini adalah proposal penawaran tanggapan KAK. Tone harus lebih formal, "
                    "lebih tegas pada kesesuaian ruang lingkup, deliverable, asumsi, dan komitmen kerja. "
                    "Hindari gaya penjualan yang terlalu ringan."
                )
            else:
                extra += (
                    " [PROPOSAL_MODE] Ini adalah proposal penawaran pekerjaan canvasing. "
                    "Tone tetap profesional, namun harus terasa meyakinkan, persuasif, dan mudah diterima sponsor bisnis."
                )
            extra += f" [CLIENT_PROFILE_PACK] {profile_summary}"
            extra += f" [RELATIONSHIP_MODE] {relationship_mode}. {relationship_guidance}"
            extra += (
                f" [KPI_TAILORING] Gunakan KPI blueprint berikut sebagai baseline tailoring: "
                f"{self._human_join(kpi_blueprint, fallback='KPI belum spesifik, gunakan KPI operasional dan outcome bisnis yang terukur.', max_items=4)}"
            )
            extra += (
                f" [VALUE_PRIORITY] Nilai yang harus terasa bagi klien: {value_map.get('value_statement', '')}. "
                f"Win theme: {value_map.get('win_theme', '')}. Bukti yang boleh dipakai: {', '.join(value_map.get('proof_points', []) or []) or 'Data internal yang tersedia'}."
            )
            extra += (
                f" [TERMINOLOGI_ADAPTASI] Gunakan istilah domain klien berikut secara natural: "
                f"{', '.join(terminology_list) if terminology_list else 'operational excellence, governance, risk control'}."
            )
            extra += (
                f" [INITIATIVE_ANCHORS] Wajib menyisipkan minimal satu anchor inisiatif klien dari daftar berikut: {anchors_text}. "
                "Gunakan faktanya secara natural tanpa menyebut nama domain, URL, atau label sumber di tubuh paragraf."
            )
            if ai_mode:
                extra += (
                    f" [AI_ADOPTION_MODE] Proposal ini terkait AI/adopsi AI. Framework ini tidak boleh disebut eksplisit, "
                    f"tetapi isi bab harus terasa dimulai dari business value lalu dijaga oleh readiness, governance, feasibility delivery, dan adopsi perubahan. "
                    f"Jika ada detail yang belum terbukti, ubah menjadi asumsi, validation checkpoint, atau readiness requirement."
                )
                extra += f" [AI_SUMMARY] {ai_profile.get('summary', '')}"
                extra += (
                    f" [AI_PROFILE] Business: {ai_profile.get('business_case', '')} "
                    f"| Data/Model: {ai_profile.get('data_foundation', '')} "
                    f"| Arsitektur: {ai_profile.get('architecture_posture', '')} "
                    f"| People: {ai_profile.get('people_capability', '')} "
                    f"| Governance: {ai_profile.get('governance_posture', '')} "
                    f"| Change: {ai_profile.get('culture_change', '')}"
                )
                if ai_posture:
                    extra += f" [AI_OSINT] {ai_posture}"
                if ai_guidance:
                    extra += f" [AI_CHAPTER_GUIDANCE] {ai_guidance}"
            extra += f" [OSINT_TRACK_RECORD] {track_record_data}"
            extra += (
                f" [{'INTERNAL_RELATIONSHIP' if relationship_source == 'internal_api' else 'OSINT_KOLABORASI'}] {collaboration_data} "
                + (
                    "Jangan pernah mengklaim pernah bekerja sama sebelumnya jika tidak ada data internal yang jelas."
                    if relationship_source == "internal_api"
                    else "Jangan pernah mengklaim pernah bekerja sama sebelumnya jika tidak ada bukti publik yang jelas."
                )
            )
            if chapter['id'] == 'c_1':
                extra += f" [FOCUS] Fokus pada latar belakang organisasi '{client}' dan tujuan proyek: '{project}'. Soroti driver bisnis utama: [{project_goal}]."
                if ai_mode:
                    extra += " [AI_FOCUS] Tunjukkan mengapa use case AI ini relevan secara bisnis saat ini, bukan sekadar menarik secara teknologi."
            elif chapter['id'] == 'c_2':
                problem_rule = (CHAPTER_STANDARD_RULES.get("c_2") or {}).get("problem_definition_pattern", {})
                problem_note = str(problem_rule.get("focus_note") or "").strip()
                pattern_subsections = ", ".join(problem_rule.get("subsections", []) or [])
                extra += (
                    f" [FOCUS] Jabarkan kebutuhan/keinginan klien berdasarkan pain points berikut: '{notes}'. "
                    "Gunakan analisis masalah yang tajam dan ringkas."
                    f" [PROBLEM_PATTERN] {problem_note} "
                    f"Gunakan label fase tersebut secara eksplisit sebagai H2 wajib dalam urutan ini: {pattern_subsections}."
                )
                if ai_mode:
                    extra += " [AI_FOCUS] Rumusan masalah harus menunjukkan gap current state vs target state, termasuk gap kesiapan data, kontrol, atau operating model bila relevan."
            elif chapter['id'] == 'c_3':
                extra += f" [FOCUS] Klasifikasikan kebutuhan ke Problem/Opportunity/Directive berdasarkan input: '{project_goal}', lalu kerucutkan satu kebutuhan utama yang paling tepat untuk diselesaikan pada proposal ini. Tetapkan jenis proyek: '{project_type}'."
                if ai_mode:
                    extra += " [AI_FOCUS] Sertakan pembacaan readiness dan tingkat intervensi yang realistis; jangan langsung solution-first."
            elif chapter['id'] == 'c_4':
                extra += f" [FOCUS] Gunakan framework/regulasi terpilih berikut sebagai acuan utama: '{regulations}'. Petakan langsung ke kebutuhan klien."
                if ai_mode:
                    extra += " [AI_FOCUS] Pendekatan harus terasa responsible, feasible, aman, dan sesuai regulasi; hindari narasi AI yang terlalu optimistis tanpa kontrol."
            elif chapter['id'] == 'c_5':
                extra += f" [FOCUS] Jelaskan alasan pemilihan metodologi untuk engagement '{service_type}' dan gunakan baseline metodologi internal: {firm_data['methodology']}."
                if ai_mode:
                    extra += " [AI_FOCUS] Metodologi perlu memuat readiness assessment, validasi, pilot atau controlled rollout, dan learning loop."
            elif chapter['id'] == 'c_6':
                extra += f" [FOCUS] Turunkan metodologi menjadi solution design yang konkret: output, deliverable, target state, dan bentuk keluaran nyata seperti dokumen, pendampingan, kegiatan, atau implementation support."
                if ai_mode:
                    extra += " [AI_FOCUS] Solution design harus terasa feasible untuk integrasi, monitoring, human oversight, dan adopsi pengguna."
            elif chapter['id'] == 'c_7':
                extra += " [FOCUS] Tampilkan ruang lingkup kerja utama, keluaran tiap area kerja, asumsi, dan batasan ruang lingkup secara jelas."
            elif chapter['id'] == 'c_8':
                extra += f" [FOCUS] Timeline harus sinkron dengan durasi proyek: '{timeline}'. Tampilkan aktivitas per fase, milestone, dan deliverable yang terukur."
            elif chapter['id'] == 'c_9':
                extra += " [FOCUS] Definisikan model tata kelola proyek: forum keputusan, frekuensi rapat, eskalasi isu, quality gate, dan kontrol progres."
            elif chapter['id'] == 'c_10':
                extra += " [FOCUS] Uraikan profil perusahaan penyusun, relevansi kapabilitas, pengalaman serupa, dan nilai tambahnya terhadap inisiatif klien. Gunakan tabel markdown agar pengalaman lebih detail."
            elif chapter['id'] == 'c_11':
                extra += f" [FOCUS] Uraikan struktur tim proyek dan tabel tenaga ahli untuk model layanan '{service_type}' dengan kapabilitas kunci, pengalaman, dan sertifikasi relevan. Referensi komposisi inti: {firm_data['team']}."
            elif chapter['id'] == 'c_12':
                extra += f" [FOCUS] Wajib menyajikan model pembiayaan dengan angka estimasi: {budget}. Sertakan termin pembayaran, model kerja, asumsi, eksklusi, dan terms komersial: {firm_data['commercial']}. Gunakan tabel markdown."
            elif chapter['id'] == 'c_closing':
                verified_contact_block = self._verified_firm_contact_block(firm_profile)
                extra += (
                    f" [FOCUS] Ini adalah bab penutup proposal. Jangan pernah menulis label nomor bab apa pun pada bagian penutup. "
                    f"Tunjukkan apresiasi profesional kepada klien '{client}', tegaskan komitmen kolaborasi jangka panjang, "
                    f"dan berikan langkah tindak lanjut yang jelas dan actionable. "
                    f"Gunakan tone hangat, profesional, dan meyakinkan."
                )
                if verified_contact_block:
                    extra += (
                        " [CONTACT_POLICY] Cantumkan hanya detail kontak firma yang sudah terverifikasi berikut ini "
                        "secara persis, tanpa menambah atau memodifikasi data:\n"
                        f"{verified_contact_block}"
                    )
                else:
                    extra += (
                        " [CONTACT_POLICY] Selain nama firma, tidak ada detail kontak writer firm yang terverifikasi. "
                        "Jangan menulis alamat kantor, email, telepon, atau website yang tidak tersedia."
                    )

            if allowed_external_citations:
                allowed_list = ", ".join(sorted(allowed_external_citations))
                extra += (
                    f" [CITATION] Sitasi eksternal hanya boleh memakai daftar ini: {allowed_list}. "
                    "Gunakan daftar ini hanya untuk grounding dan pemilihan fakta, bukan untuk ditampilkan sebagai nama domain di paragraf. "
                    "Dilarang membuat domain/sitasi eksternal baru di luar daftar dan dilarang memakai placeholder sitasi."
                )
            else:
                extra += (
                    f" [CITATION] Tidak ada sumber eksternal tervalidasi untuk bab ini. "
                    "Dilarang menulis sitasi domain eksternal apa pun dan dilarang menampilkan label sumber internal di tubuh paragraf."
                )

            internal_citation_note = (
                "Gunakan data internal hanya sebagai grounding. Jangan tampilkan label sumber seperti "
                f"(Data Internal, {current_year}) di tubuh paragraf."
            )
            structured_row_data_with_note = (
                f"{structured_row_data}\n{internal_citation_note}" if structured_row_data else internal_citation_note
            )
            rag_data_with_note = f"{rag_data}\n{internal_citation_note}" if rag_data else internal_citation_note

            prompt = PROPOSAL_SYSTEM_PROMPT.format(
                client=client, 
                writer_firm=WRITER_FIRM_NAME, 
                persona=persona,
                global_data=global_data, 
                client_news=client_news, 
                regulation_data=regulation_data,
                structured_row_data=structured_row_data_with_note,
                rag_data=rag_data_with_note,
                current_year=current_year,
                visual_prompt=visual_prompt, 
                extra_instructions=extra,
                chapter_title=chapter['title'], 
                sub_chapters=subs, 
                length_intent=self._rewrite_length_intent(chapter.get('length_intent', ''), target_words)
            )
            return {"prompt": prompt, "success": True}
        except Exception as e:
            return {"prompt": "", "success": False, "error": str(e)}

    @staticmethod
    def _contains_client_reference(content: str, client: str) -> bool:
        tokens = [
            t for t in re.findall(r"[A-Za-z0-9]{3,}", client)
            if t.lower() not in {"pt", "cv", "tbk"}
        ]
        if not tokens:
            return True
        return any(re.search(rf"\b{re.escape(tok)}\b", content, re.IGNORECASE) for tok in tokens[:3])

    def _evaluate_chapter_quality(
        self,
        chapter: Dict[str, Any],
        content: str,
        client: str,
        target_words: Optional[int] = None,
        allowed_external_citations: Optional[Set[str]] = None,
        personalization_pack: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        target_words = int(target_words or self._target_words(chapter))
        floor = max(140, int(self._chapter_floor_words(chapter.get("id", ""), for_compression=False) * 0.8))
        min_words = max(floor, int(target_words * 0.72))
        max_words = max(min_words + 90, int(target_words * 1.25))
        word_count = self._word_count(content)
        missing_h2 = [
            sub for sub in chapter.get('subs', [])
            if not re.search(rf"(?im)^\s*##\s*{re.escape(sub)}\s*$", content)
        ]
        has_numbered_list = bool(re.search(r"(?m)^\s*\d+\.\s+\S+", content))
        has_bullet_list = bool(re.search(r"(?m)^\s*[-*]\s+\S+", content))
        has_client_ref = self._contains_client_reference(content, client)
        has_required_visual = True
        if chapter.get('visual_intent') == "gantt":
            has_required_visual = "[[GANTT:" in content

        issues = []
        if missing_h2:
            issues.append("missing_h2")
        if word_count < min_words:
            issues.append("too_short")
        if word_count > max_words:
            issues.append("too_long")
        if not has_numbered_list or not has_bullet_list:
            issues.append("list_structure")
        if not has_required_visual:
            issues.append("missing_visual")
        if not has_client_ref:
            issues.append("missing_client_ref")

        invalid_external_citations: List[str] = []
        if allowed_external_citations is not None:
            cited_external = sorted(set(self._extract_external_citations(content)))
            allowed = set(allowed_external_citations)
            if not allowed and cited_external:
                invalid_external_citations = cited_external
            elif allowed:
                invalid_external_citations = [citation for citation in cited_external if citation not in allowed]
            if invalid_external_citations:
                issues.append("citation_policy")

        missing_personalization_signals: List[str] = []
        if personalization_pack:
            terminology = personalization_pack.get("terminology", []) or []
            term_hit = any(
                re.search(rf"\b{re.escape(term)}\b", content, re.IGNORECASE)
                for term in terminology if term
            ) if terminology else True
            if not term_hit:
                missing_personalization_signals.append("terminology")

            kpi_keywords = personalization_pack.get("kpi_keywords", []) or []
            kpi_hit = any(
                re.search(rf"\b{re.escape(token)}\b", content, re.IGNORECASE)
                for token in kpi_keywords if token
            ) if kpi_keywords else True
            if not kpi_hit:
                missing_personalization_signals.append("kpi")

            if chapter.get("id", "") in self._anchor_required_chapters():
                anchor_hit = self._has_anchor_signal(content, personalization_pack)
                if not anchor_hit:
                    missing_personalization_signals.append("initiative_anchor")

            ai_profile = personalization_pack.get("ai_adoption_profile", {}) or {}
            if ai_profile.get("enabled"):
                ai_terms = self._chapter_ai_terms(chapter.get("id", ""))
                ai_hit = self._count_signal_hits(content, ai_terms, max_hits=3) if ai_terms else 1
                if ai_terms and ai_hit == 0:
                    missing_personalization_signals.append("ai_adoption")

            if missing_personalization_signals:
                issues.append("missing_personalization")

        return {
            "issues": issues,
            "word_count": word_count,
            "target_words": target_words,
            "min_words": min_words,
            "max_words": max_words,
            "missing_h2": missing_h2,
            "invalid_external_citations": invalid_external_citations,
            "missing_personalization_signals": missing_personalization_signals,
        }

    def _ensure_required_headings(self, chapter: Dict[str, Any], content: str) -> str:
        missing_h2 = [
            sub for sub in chapter.get('subs', [])
            if not re.search(rf"(?im)^\s*##\s*{re.escape(sub)}\s*$", content)
        ]
        if not missing_h2:
            return content

        patched = content.rstrip()
        for heading in missing_h2:
            patched += (
                f"\n\n## {heading}\n"
                "### Rincian Inti\n"
                "1. Aktivitas utama pada bagian ini disesuaikan langsung dengan kebutuhan bisnis klien.\n"
                "- Risiko, dependensi, dan indikator keberhasilan dijabarkan secara terukur untuk eksekusi.\n"
            )
        return patched

    def _ensure_list_structure(
        self,
        content: str,
        chapter: Dict[str, Any],
        client: str,
        personalization_pack: Optional[Dict[str, Any]] = None
    ) -> str:
        patched = (content or "").rstrip()
        if self._use_structured_chapter(str(chapter.get("id", "")).strip()):
            return patched
        personalization_pack = personalization_pack or {}
        kpi_line = self._human_join(
            (personalization_pack.get("kpi_blueprint", []) or [])[:2],
            fallback=f"outcome utama {client}",
        )
        term_line = ", ".join((personalization_pack.get("terminology", []) or [])[:2]) or "governance, risk control"

        if not re.search(r"(?m)^\s*\d+\.\s+\S+", patched):
            patched += (
                f"\n\n1. Prioritas implementasi untuk {client} diarahkan agar tetap selaras dengan {kpi_line}.\n"
                f"2. Keputusan delivery dan quality gate pada bab ini harus konsisten dengan istilah kerja {term_line}."
            )
        if not re.search(r"(?m)^\s*[-*]\s+\S+", patched):
            patched += (
                f"\n- Konteks {client} tetap menjadi acuan utama agar isi bab tidak bergeser menjadi generik.\n"
                f"- Setiap butir pada bab ini harus bisa ditelusuri ke KPI, risiko, atau keputusan kerja yang nyata."
            )
        return patched

    def _ensure_ai_adoption_signals(
        self,
        chapter: Dict[str, Any],
        content: str,
        personalization_pack: Optional[Dict[str, Any]] = None
    ) -> str:
        data = personalization_pack or {}
        ai_profile = data.get("ai_adoption_profile", {}) or {}
        if not ai_profile.get("enabled"):
            return content

        chapter_id = str(chapter.get("id") or "").strip()
        chapter_terms = self._chapter_ai_terms(chapter_id)
        if not chapter_terms:
            return content

        patched = (content or "").rstrip()
        if self._count_signal_hits(patched, chapter_terms, max_hits=4) >= 2:
            return patched

        additions: List[str] = []
        chapter_guidance = str((ai_profile.get("chapter_guidance") or {}).get(chapter_id, "")).strip()
        if chapter_guidance:
            additions.append(f"- {chapter_guidance}")

        ai_field_map = {
            "business_use_case": "business_case",
            "data_model_foundation": "data_foundation",
            "infrastructure_architecture": "architecture_posture",
            "people_capability": "people_capability",
            "governance": "governance_posture",
            "culture_change": "culture_change",
        }
        for dimension_id in self._ai_chapter_dimensions(chapter_id):
            field_name = ai_field_map.get(dimension_id)
            line = str(ai_profile.get(field_name, "") or "").strip()
            if line:
                additions.append(f"- {line}")
            if len(additions) >= 2:
                break

        if not additions:
            return patched
        return patched + "\n" + "\n".join(additions)

    def _ensure_problem_definition_pattern(
        self,
        chapter: Dict[str, Any],
        content: str,
        client: str,
        personalization_pack: Optional[Dict[str, Any]] = None
    ) -> str:
        chapter_rule = (CHAPTER_STANDARD_RULES.get(chapter.get("id", "")) or {}).get("problem_definition_pattern", {})
        if not chapter_rule:
            return content

        patched = (content or "").rstrip()
        subsections = [str(item).strip() for item in (chapter_rule.get("subsections") or []) if str(item).strip()]
        if subsections and all(
            re.search(rf"(?im)^\s*##\s*{re.escape(section)}\s*$", patched)
            for section in subsections
        ):
            return patched

        data = personalization_pack or {}
        kpi_line = self._human_join(
            (data.get("kpi_blueprint", []) or [])[:2],
            fallback=f"outcome utama {client}",
        )
        term_line = ", ".join((data.get("terminology", []) or [])[:2]) or "tata kelola, pengendalian risiko"
        anchor_line = ""
        if chapter.get("id", "") in self._anchor_required_chapters():
            fact, _ = self._extract_first_external_anchor(data)
            if fact:
                anchor_line = f" Acuan konteks yang tetap dipakai adalah {fact}."

        section_templates = {
            "2.2 Konteks Bisnis": (
                f"{client} sedang menjaga outcome seperti {kpi_line} dan membutuhkan arah kerja yang tetap terkoneksi dengan kebutuhan bisnis inti.{anchor_line}\n"
                f"1. Konteks bisnis harus dibaca dari target hasil, tekanan sponsor, dan arah transformasi yang sedang berjalan pada {client}.\n"
                f"- Pada tahap ini, proposal perlu menegaskan kondisi saat ini yang masih belum cukup stabil dibanding kondisi yang dituju."
            ),
            "2.3 Tantangan Utama": (
                f"Tantangan utama berada pada kemampuan menyelaraskan keputusan sponsor, ritme pelaksanaan, dan disiplin kerja {term_line} dalam satu alur yang konsisten.\n"
                f"1. Tantangan utama tidak boleh berhenti sebagai gejala permukaan, tetapi harus menunjukkan titik hambat yang benar-benar menahan {client} untuk bergerak lebih cepat.\n"
                f"- Hambatan koordinasi, kualitas keputusan, dan kesiapan eksekusi perlu dibaca sebagai satu masalah manajemen yang saling terkait."
            ),
            "2.4 Akar Kesenjangan": (
                f"Terdapat gap antara kondisi saat ini yang masih belum cukup terkontrol dengan kondisi yang dituju, yang menuntut keputusan lebih cepat, kontrol risiko lebih rapi, dan kualitas eksekusi yang lebih stabil.\n"
                f"1. Akar kesenjangan inilah yang menjadi inti definisi masalah untuk {client}, karena ia menjelaskan hal mendasar yang perlu ditutup.\n"
                f"- Dengan mengartikulasikan gap secara eksplisit, proposal dapat bergerak dari keluhan operasional menjadi dasar intervensi yang lebih dapat dipertanggungjawabkan."
            ),
            "2.5 Implikasi / Risiko": (
                f"Jika gap ini tidak ditutup, {client} berisiko mengalami deviasi KPI, penurunan kualitas layanan, dan bertambahnya tekanan pada governance maupun operasi.\n"
                f"1. Implikasi masalah harus diterjemahkan ke dampak bisnis, risiko eksekusi, dan konsekuensi keputusan yang mungkin muncul di level sponsor.\n"
                f"- Risiko ini juga menjadi alasan mengapa perbaikan tidak bisa ditunda atau diperlakukan sebagai isu administratif semata."
            ),
            "2.6 Kebutuhan Solusi": (
                f"Karena itu, rumusan masalah pada bab ini harus mengarah pada kebutuhan solusi yang mampu menutup gap tersebut secara terukur, defensible, dan siap diturunkan ke tahap pendekatan serta desain solusi.\n"
                f"1. Kebutuhan solusi perlu menjadi jembatan langsung ke bab pendekatan, metodologi, dan solution design.\n"
                f"- Dengan struktur ini, {client} dapat melihat bahwa solusi yang diusulkan memang lahir dari definisi masalah yang runtut, bukan dari asumsi generik."
            ),
        }

        additions: List[str] = []
        for section in subsections:
            if re.search(rf"(?im)^\s*##\s*{re.escape(section)}\s*$", patched):
                continue
            body = section_templates.get(section, f"{section} perlu dijelaskan secara konkret untuk {client}.")
            additions.append(f"## {section}\n{body}")

        if not additions:
            return patched
        return patched + "\n\n" + "\n\n".join(additions)

    def _ensure_visual_requirements(self, chapter: Dict[str, Any], content: str, timeline: str = "") -> str:
        patched = (content or "").rstrip()
        if chapter.get("visual_intent") != "gantt" or "[[GANTT:" in patched:
            return patched

        months = max(4, int(round(FinancialAnalyzer._duration_to_months(timeline) or 6)))
        breakpoints = [0, max(1, months // 4), max(2, months // 2), max(3, (months * 3) // 4), months]
        phase_names = ["Discovery", "Design", "Execution", "Stabilization"]
        parts = []
        for idx, name in enumerate(phase_names):
            start = min(breakpoints[idx], months - 1)
            end = max(start + 1, min(months, breakpoints[idx + 1]))
            parts.append(f"{name},{start},{end}")
        return patched + f"\n[[GANTT: Jadwal Pelaksanaan | Bulan | {'; '.join(parts)}]]"

    def _ensure_personalization_signals(
        self,
        content: str,
        client: str,
        personalization_pack: Optional[Dict[str, Any]] = None,
        chapter: Optional[Dict[str, Any]] = None
    ) -> str:
        data = personalization_pack or {}
        patched = (content or "").rstrip()
        additions: List[str] = []
        chapter_id = str((chapter or {}).get("id") or "").strip()

        if not self._contains_client_reference(patched, client):
            additions.append(f"- Untuk {client}, isi bab ini tetap diarahkan pada keputusan dan langkah kerja yang konkret.")

        terminology = data.get("terminology", []) or []
        if terminology:
            term_hits = self._count_signal_hits(patched, terminology, max_hits=3)
            if term_hits < 2:
                additions.append(f"- Istilah kerja yang tetap dijaga pada bab ini mencakup {', '.join(terminology[:3])}.")

        kpis = data.get("kpi_blueprint", []) or []
        if kpis:
            kpi_keywords = data.get("kpi_keywords", []) or []
            kpi_hits = self._count_signal_hits(patched, kpis + kpi_keywords, max_hits=3)
            if kpi_hits < 2:
                kpi_reference = self._human_join(kpis[:2], fallback=f"outcome utama {client}")
                additions.append(f"- KPI acuan yang tetap dijaga pada bab ini adalah {kpi_reference}.")

        if chapter_id in self._anchor_required_chapters():
            anchors = data.get("initiative_facts", []) or []
            anchor_hit = self._has_anchor_signal(patched, data)
            if anchors and not anchor_hit:
                anchor = anchors[0]
                anchor_fact = self._sanitize_anchor_fact(anchor.get("fact", "") or "")
                if anchor_fact:
                    additions.append(f"- Anchor inisiatif yang tetap dirujuk adalah {anchor_fact}.")

        if not additions:
            return patched
        return patched + "\n" + "\n".join(additions)

    def _ensure_minimum_substance(
        self,
        chapter: Dict[str, Any],
        content: str,
        client: str,
        target_words: Optional[int] = None,
        personalization_pack: Optional[Dict[str, Any]] = None,
        timeline: str = ""
    ) -> str:
        target = int(target_words or self._target_words(chapter))
        floor = max(140, int(self._chapter_floor_words(chapter.get("id", ""), for_compression=False) * 0.8))
        min_words = max(floor, int(target * 0.72))
        patched = (content or "").rstrip()
        if self._word_count(patched) >= min_words:
            return patched

        data = personalization_pack or {}
        kpi_line = self._human_join(
            (data.get("kpi_blueprint", []) or [])[:2],
            fallback=f"outcome utama {client}",
        )
        term_line = ", ".join((data.get("terminology", []) or [])[:2]) or "governance, risk control"
        anchor_line = ""
        if chapter.get("id", "") in self._anchor_required_chapters():
            fact, _ = self._extract_first_external_anchor(data)
            if fact:
                anchor_line = f" Rujukan konteks yang tetap dipakai adalah {fact}."

        chapter_specific_blocks = {
            "c_2": [
                (
                    f"Definisi masalah untuk {client} perlu dibangun dengan pola yang konsisten agar sponsor dapat melihat hubungan langsung antara konteks bisnis, tantangan utama, gap yang mendasari, "
                    f"risiko yang muncul, dan kebutuhan solusi. Dengan pendekatan ini, bab permasalahan tidak berhenti pada daftar pain point, tetapi benar-benar menjelaskan mengapa gap antara current state "
                    f"dan target state perlu ditutup dengan intervensi yang lebih terukur.{anchor_line}"
                ),
                (
                    f"- Pola definisi masalah membantu {client} membedakan gejala operasional dari akar gap yang perlu diprioritaskan.\n"
                    f"- Untuk menjaga relevansi eksekusi, rumusan masalah harus tetap ditautkan ke KPI seperti {kpi_line} dan istilah kerja {term_line}."
                ),
                (
                    f"Dengan demikian, bab ini memberi dasar yang lebih kuat untuk bab berikutnya: klasifikasi kebutuhan, pendekatan, metodologi, dan solution design dapat diturunkan dari rumusan masalah yang sama, "
                    f"bukan dari asumsi yang berubah-ubah. Itu membuat proposal lebih defensible, lebih personal terhadap kondisi {client}, dan lebih berguna sebagai dokumen keputusan."
                ),
            ],
            "c_3": [
                (
                    f"Pada praktiknya, klasifikasi kebutuhan tidak boleh berhenti sebagai label administratif. Bagi {client}, "
                    f"klasifikasi ini menentukan apakah fokus kerja harus menutup gap yang sudah nyata, menangkap peluang yang bisa mempercepat hasil, "
                    f"atau memenuhi directive yang tidak bisa ditawar. Dengan kerangka seperti ini, keputusan scope, prioritas eksekusi, dan governance "
                    f"bisa diarahkan lebih presisi terhadap KPI seperti {kpi_line}.{anchor_line}"
                ),
                (
                    f"- Dampak utama dari klasifikasi yang benar adalah keputusan proyek menjadi lebih cepat, lebih defensible, dan lebih mudah diturunkan ke delivery plan.\n"
                    f"- Untuk {client}, bahasa kerja seperti {term_line} perlu tetap hadir agar klasifikasi kebutuhan tidak terlepas dari konteks operasionalnya."
                ),
                (
                    f"Karena itu, hasil klasifikasi juga perlu dipakai sebagai dasar untuk menetapkan sponsor keputusan, scope awal, dan bentuk intervensi yang paling realistis. "
                    f"Dengan pendekatan ini, {client} tidak hanya memperoleh label kebutuhan, tetapi juga dasar yang lebih kuat untuk menyelaraskan tujuan proyek, kontrol delivery, dan ukuran keberhasilan sejak fase awal."
                ),
            ],
            "c_7": [
                (
                    f"Bab ruang lingkup harus membantu {client} melihat secara tegas pekerjaan apa yang termasuk, keluaran apa yang diterima, dan area apa yang berada di luar baseline proposal. "
                    f"Tanpa kejelasan seperti ini, diskusi delivery akan mudah melebar dan keputusan sponsor terhadap KPI seperti {kpi_line} menjadi kurang presisi."
                ),
                (
                    f"- Ruang lingkup perlu ditulis dalam bahasa kerja yang konkret, termasuk aktivitas, keluaran, dan asumsi utama.\n"
                    f"- Untuk {client}, istilah kerja seperti {term_line} juga perlu muncul agar scope tetap menempel pada konteks operasional yang nyata."
                ),
                (
                    f"Dengan pendekatan tersebut, ruang lingkup tidak hanya menjadi daftar aktivitas, tetapi menjadi alat kontrol bersama antara {client} dan {WRITER_FIRM_NAME}. "
                    f"Sponsor dapat membaca batas pekerjaan, bentuk keluaran, dan asumsi dasar secara lebih cepat sebelum proposal bergerak ke timeline dan model pembiayaan."
                ),
            ],
            "c_8": [
                (
                    f"Penjadwalan tidak hanya membagi durasi {timeline or 'proyek'} ke dalam fase, tetapi juga memastikan dependensi, keputusan sponsor, "
                    f"dan kesiapan stakeholder bergerak dalam ritme yang sama. Untuk {client}, pengaturan ini penting agar progres tidak sekadar terlihat aktif, "
                    f"melainkan benar-benar menjaga jalur pencapaian KPI seperti {kpi_line}."
                ),
                (
                    f"- Setiap fase perlu punya quality gate yang jelas agar perubahan prioritas tidak langsung merusak baseline jadwal.\n"
                    f"- Koordinasi timeline juga harus menjaga konsistensi istilah kerja {term_line} supaya forum eksekusi dan forum pengarah membaca progres dengan bahasa yang sama."
                ),
                (
                    f"Dengan cara ini, timeline berfungsi sebagai alat steering, bukan hanya kalender aktivitas. Sponsor {client} dapat melihat kapan keputusan penting harus diambil, "
                    f"kapan deliverable harus ditinjau, dan kapan penyesuaian perlu dilakukan agar program tetap bergerak ke outcome yang dituju tanpa kehilangan kontrol atas risiko dan dependensi."
                ),
            ],
            "c_10": [
                (
                    f"Bab profil perusahaan perlu memperjelas mengapa {WRITER_FIRM_NAME} relevan untuk mendampingi {client}, bukan hanya menampilkan deskripsi umum perusahaan. "
                    f"Yang penting adalah kaitan antara kapabilitas, pengalaman serupa, dan kebutuhan proyek {client}."
                ),
                (
                    f"- Pengalaman sebaiknya ditulis dalam bentuk yang mudah dibaca sponsor, idealnya tabel dengan area pengalaman, relevansi, dan nilai tambah.\n"
                    f"- Bukti kapabilitas perlu tetap terkait dengan bahasa kerja {term_line} agar profil perusahaan tidak terasa seperti brosur generik."
                ),
                (
                    f"Dengan pola itu, profil perusahaan membantu membangun kepercayaan lebih cepat. {client} dapat melihat bahwa perusahaan penyusun bukan hanya memahami metodologi, "
                    f"tetapi juga punya modal pengalaman untuk mengubah kebutuhan menjadi keputusan dan deliverable yang dapat dijalankan."
                ),
            ],
            "c_11": [
                (
                    f"Bab tenaga ahli perlu menunjukkan bahwa tim yang ditawarkan benar-benar memadai untuk ruang lingkup {client}, bukan hanya daftar jabatan. "
                    f"Tabel tenaga ahli yang detail membantu sponsor membaca fokus tanggung jawab, kompetensi, dan peran setiap personel sejak awal."
                ),
                (
                    f"- Struktur tim harus memperlihatkan hubungan antara kepemimpinan delivery, quality gate, dan eksekusi harian.\n"
                    f"- Detail tenaga ahli perlu menyinggung kompetensi, pengalaman relevan, dan keterlibatan per fase secara lebih konkret."
                ),
                (
                    f"Dengan demikian, komposisi tim tidak sekadar membuat proposal terlihat lengkap, tetapi juga memberi kepastian bahwa hasil kerja akan dijaga oleh peran yang jelas dan bisa dipertanggungjawabkan. "
                    f"Hal ini penting agar {client} percaya bahwa scope, timeline, dan deliverable akan dikawal oleh orang yang tepat."
                ),
            ],
            "c_12": [
                (
                    f"Bab pembiayaan tidak boleh berhenti sebagai angka dan termin pembayaran saja. Untuk {client}, bagian ini perlu menunjukkan bahwa struktur biaya, tahapan pembayaran, "
                    f"dan batas pekerjaan disusun agar manfaat bisnis seperti {kpi_line} tetap dapat dikejar tanpa membuat komitmen komersial menjadi kabur. Dalam praktiknya, disiplin seperti ini "
                    f"juga memperlihatkan cara {WRITER_FIRM_NAME} menjaga hubungan antara acceptance deliverable, quality gate, dan kontrol perubahan ruang lingkup."
                ),
                (
                    f"- Setiap komponen biaya perlu dikaitkan dengan milestone dan output yang nyata agar sponsor {client} dapat menilai nilai investasinya secara lebih objektif.\n"
                    f"- Bahasa komersial pada bab ini juga harus tetap konsisten dengan istilah kerja {term_line}, sehingga diskusi biaya tidak terlepas dari konteks delivery dan governance."
                ),
                (
                    f"Dengan pendekatan tersebut, model pembiayaan menjadi alat pengendali bersama antara klien dan penyedia jasa. {client} memperoleh kejelasan mengenai apa yang dibayar, "
                    f"hasil apa yang diterima, serta kondisi apa yang dapat memicu perubahan komersial. Bagi {WRITER_FIRM_NAME}, pola ini membantu menjaga proposal tetap profesional, defensible, "
                    f"dan sejalan dengan outcome yang dijanjikan sejak awal engagement."
                ),
            ],
        }
        generic_blocks = [
            (
                f"Untuk {client}, isi bab ini harus dibaca sebagai dasar keputusan kerja yang dapat ditindaklanjuti, bukan sekadar penjelasan konseptual. "
                f"Karena itu, isi bab tetap diarahkan untuk menjaga relevansi terhadap KPI seperti {kpi_line} dan istilah kerja {term_line}."
            ),
            (
                f"- Implikasi eksekusinya harus tetap jelas agar sponsor dan tim delivery dapat menurunkan isi bab ini menjadi tindakan yang konkret.\n"
                f"- Dengan pendekatan ini, narasi proposal tetap terhubung pada kebutuhan bisnis sekaligus tidak kehilangan kontrol implementasi."
            ),
            (
                f"Secara praktis, hal ini membantu {client} membaca setiap bagian proposal sebagai bahan keputusan yang lebih operasional: apa yang harus diprioritaskan, "
                f"siapa yang perlu mengambil keputusan, risiko apa yang harus dijaga, dan indikator apa yang dipakai untuk menguji bahwa arah kerja masih benar."
            ),
        ]

        blocks = chapter_specific_blocks.get(chapter.get("id", ""), []) + generic_blocks
        for block in blocks:
            if self._word_count(patched) >= min_words:
                break
            patched += "\n\n" + block

        if self._word_count(patched) < min_words:
            patched += (
                f"\n\nParagraf penegasan ini menjaga agar isi bab tetap cukup substantif untuk dibaca oleh sponsor {client}. "
                f"Artinya, isi bab tidak berhenti pada deskripsi, tetapi tetap menunjukkan keterkaitan antara keputusan kerja, KPI {kpi_line}, "
                f"bahasa operasional {term_line}, dan konteks inisiatif yang sedang dijalankan."
            )
        booster_count = 0
        while self._word_count(patched) < min_words and booster_count < 3:
            patched += (
                f"\n\nPenajaman tambahan ini memastikan bab tetap berguna untuk pengambilan keputusan {client}. "
                f"Isi bab harus tetap menjelaskan implikasi kerja, konsekuensi pada KPI {kpi_line}, "
                f"dan bagaimana langkah eksekusinya dijaga tetap konsisten dengan istilah operasional {term_line}."
            )
            booster_count += 1
        return patched

    def _apply_draft_repairs(
        self,
        chapter: Dict[str, Any],
        content: str,
        client: str,
        allowed_external_citations: Optional[Set[str]] = None,
        personalization_pack: Optional[Dict[str, Any]] = None
        ,
        target_words: Optional[int] = None,
        timeline: str = ""
    ) -> str:
        allowed = set(allowed_external_citations or set())
        repaired = self._clean_external_citations(content or "", allowed)
        repaired = self._ensure_required_headings(chapter, repaired)
        repaired = self._ensure_list_structure(repaired, chapter, client, personalization_pack=personalization_pack)
        repaired = self._ensure_problem_definition_pattern(chapter, repaired, client, personalization_pack=personalization_pack)
        repaired = self._ensure_visual_requirements(chapter, repaired, timeline=timeline)
        repaired = self._ensure_personalization_signals(
            repaired,
            client,
            personalization_pack=personalization_pack,
            chapter=chapter,
        )
        repaired = self._ensure_ai_adoption_signals(
            chapter=chapter,
            content=repaired,
            personalization_pack=personalization_pack,
        )
        repaired = self._ensure_minimum_substance(
            chapter,
            repaired,
            client,
            target_words=target_words,
            personalization_pack=personalization_pack,
            timeline=timeline
        )
        return self._clean_external_citations(repaired, allowed)

    @staticmethod
    def _semantic_terms(values: Any, max_terms: int = 12) -> List[str]:
        stopwords = {
            "yang", "untuk", "dengan", "dari", "pada", "dan", "atau", "agar", "dalam",
            "lebih", "tetap", "harus", "jadi", "sebagai", "melalui", "terhadap", "karena",
            "the", "and", "with", "from", "into", "this", "that", "your", "their",
        }
        raw_items = values if isinstance(values, list) else [values]
        seen: Set[str] = set()
        terms: List[str] = []
        for raw in raw_items:
            text = re.sub(r"\s+", " ", str(raw or "").strip())
            if not text:
                continue
            lowered = text.lower()
            if 2 <= len(lowered.split()) <= 5 and lowered not in seen:
                seen.add(lowered)
                terms.append(text)
            for token in re.findall(r"[A-Za-z]{4,}", lowered):
                if token in stopwords or token in seen:
                    continue
                seen.add(token)
                terms.append(token)
                if len(terms) >= max_terms:
                    return terms
            if len(terms) >= max_terms:
                break
        return terms[:max_terms]

    @staticmethod
    def _count_signal_hits(content: str, candidates: List[str], max_hits: int = 6) -> int:
        hits = 0
        seen: Set[str] = set()
        text = content or ""
        for candidate in candidates or []:
            value = re.sub(r"\s+", " ", str(candidate or "").strip())
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            pattern = re.escape(value)
            if " " not in value and value.isalpha():
                pattern = rf"\b{pattern}\b"
            if re.search(pattern, text, re.IGNORECASE):
                hits += 1
                if hits >= max_hits:
                    return hits
        return hits

    @staticmethod
    def _chapter_usefulness_terms(chapter_id: str) -> List[str]:
        mapping = {
            "c_1": ["latar belakang", "konteks", "prioritas", "permintaan"],
            "c_2": ["masalah", "hambatan", "gap", "risiko", "konteks bisnis", "tantangan utama", "akar kesenjangan", "kebutuhan solusi"],
            "c_3": ["problem", "opportunity", "directive", "fokus utama", "tujuan", "jenis proyek"],
            "c_4": ["framework", "regulasi", "standar", "acuan"],
            "c_5": ["metodologi", "langkah kerja", "fase", "output"],
            "c_6": ["solusi", "deliverable", "target state", "dokumen", "pendampingan", "kegiatan"],
            "c_7": ["ruang lingkup", "keluaran", "batasan", "asumsi"],
            "c_8": ["deliverable", "milestone", "fase", "gantt"],
            "c_9": ["keputusan", "eskalasi", "quality gate", "kontrol"],
            "c_10": ["profil perusahaan", "pengalaman", "kapabilitas", "relevansi"],
            "c_11": ["tenaga ahli", "tim", "kapabilitas", "pengalaman", "sertifikasi"],
            "c_12": ["biaya", "pembayaran", "scope", "batasan"],
            "c_closing": ["terima kasih", "langkah lanjutan", "kemitraan"],
            "k_1": ["informasi perusahaan", "struktur organisasi", "pengalaman pekerjaan sejenis", "kapabilitas"],
            "k_2": ["kerangka acuan kerja", "pemahaman", "tanggapan", "saran", "lingkup pekerjaan", "keluaran"],
            "k_3": ["framework", "kerangka acuan", "metodologi", "fase", "gerbang mutu"],
            "k_4": ["program kerja", "timeline", "penugasan", "tenaga ahli", "gantt"],
            "k_5": ["struktur organisasi pelaksana", "komposisi tim", "uraian tugas", "tenaga ahli"],
            "k_6": ["hasil kerja", "deliverable", "keluaran", "output"],
            "k_7": ["fasilitas pendukung", "pelaksanaan pekerjaan", "dukungan"],
            "k_8": ["inovasi", "gagasan baru", "nilai tambah"],
        }
        return mapping.get(chapter_id, ["deliverable", "risiko", "kpi"])

    @staticmethod
    def _chapter_output_by_aliases(chapter_outputs: Dict[str, str], *aliases: str) -> str:
        for alias in aliases:
            text = chapter_outputs.get(alias, "")
            if text:
                return text
        return ""

    @staticmethod
    def _safe_score(value: float) -> int:
        return max(0, min(100, int(round(value))))

    @classmethod
    def _find_contact_like_lines(cls, content: str) -> List[str]:
        suspects: List[str] = []
        patterns = [
            r"@",
            r"https?://|www\.",
            r"\b(?:telp|telepon|phone|whatsapp|wa)\b",
            r"\b(?:jl\.|jalan|office|kantor)\b",
            r"\(\d{3,4}\)",
        ]
        for raw_line in (content or "").splitlines():
            line = re.sub(r"^[#*\-\d.\s]+", "", raw_line).strip()
            if not line:
                continue
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                suspects.append(line)
        return suspects

    def _evaluate_proposal_acceptance(
        self,
        chapter_outputs: Dict[str, str],
        selected_chapters: List[Dict[str, Any]],
        chapter_targets: Dict[str, int],
        client: str,
        project: str,
        notes: str,
        firm_profile: Dict[str, Any],
        allowed_external_citations: Set[str],
        personalization_pack: Dict[str, Any],
        value_map: Dict[str, Any],
    ) -> Dict[str, Any]:
        chapter_map = {chapter["id"]: chapter for chapter in selected_chapters}
        kak_mode = any(chapter["id"].startswith("k_") for chapter in selected_chapters)
        full_text = "\n\n".join(
            chapter_outputs.get(chapter["id"], "")
            for chapter in selected_chapters
            if chapter_outputs.get(chapter["id"], "")
        )
        generated_words = sum(self._word_count(text) for text in chapter_outputs.values() if text)
        estimated_pages = self._estimated_pages(generated_words)

        chapter_findings: Dict[str, Dict[str, Any]] = {}
        hard_failures: List[str] = []
        soft_findings: List[str] = []
        invalid_external_citations: List[str] = []
        personalization_weight = 0.0
        personalization_score_total = 0.0

        company_terms = self._semantic_terms(
            [value_map.get("positioning", ""), value_map.get("proposal_promise", "")] +
            (value_map.get("differentiators", []) or []) +
            (value_map.get("proof_points", []) or []),
            max_terms=18
        )
        persuasion_terms = self._semantic_terms(
            [value_map.get("value_statement", ""), value_map.get("value_hook", ""), value_map.get("win_theme", "")] +
            (value_map.get("client_gains", []) or []) +
            (value_map.get("industry_drivers", []) or []),
            max_terms=18
        )
        ai_profile = personalization_pack.get("ai_adoption_profile", {}) or {}
        ai_mode = bool(ai_profile.get("enabled"))
        pressure_terms = self._semantic_terms([project, notes, value_map.get("customer_pressure", "")], max_terms=12)
        anchor_terms = personalization_pack.get("anchor_keywords", []) or []
        if not anchor_terms:
            anchor_terms = self._semantic_terms(
                [item.get("fact", "") for item in personalization_pack.get("initiative_facts", []) or []],
                max_terms=8
            )

        for chapter in selected_chapters:
            chapter_id = chapter["id"]
            content = (chapter_outputs.get(chapter_id) or "").strip()
            target_words = chapter_targets.get(chapter_id, self._target_words(chapter))
            if not content:
                hard_failures.append(f"{chapter['title']}: empty")
                chapter_findings[chapter_id] = {
                    "title": chapter["title"],
                    "issues": ["empty"],
                    "score": 0,
                    "weaknesses": ["missing_content"],
                }
                continue

            quality = self._evaluate_chapter_quality(
                chapter=chapter,
                content=content,
                client=client,
                target_words=target_words,
                allowed_external_citations=allowed_external_citations,
                personalization_pack=personalization_pack,
            )
            issues = quality.get("issues", [])
            invalid_external_citations.extend(quality.get("invalid_external_citations", []))

            if any(issue in issues for issue in {"missing_h2", "too_short", "missing_visual", "citation_policy"}):
                hard_failures.extend([f"{chapter['title']}: {issue}" for issue in issues if issue in {"missing_h2", "too_short", "missing_visual", "citation_policy"}])
            elif issues:
                soft_findings.extend([f"{chapter['title']}: {issue}" for issue in issues])

            client_ref_hit = self._contains_client_reference(content, client)
            term_hits = self._count_signal_hits(content, personalization_pack.get("terminology", []) or [], max_hits=3)
            kpi_hits = self._count_signal_hits(
                content,
                (personalization_pack.get("kpi_blueprint", []) or []) + (personalization_pack.get("kpi_keywords", []) or []),
                max_hits=3
            )
            anchor_expected = chapter_id in self._anchor_required_chapters()
            anchor_hits = self._count_signal_hits(content, anchor_terms, max_hits=2) if anchor_expected else 1
            value_hits = self._count_signal_hits(content, persuasion_terms, max_hits=4)
            usefulness_hits = self._count_signal_hits(content, self._chapter_usefulness_terms(chapter_id), max_hits=3)
            ai_hits = self._count_signal_hits(content, self._chapter_ai_terms(chapter_id), max_hits=4) if ai_mode else 0

            business_rank = self._chapter_business_rank(chapter_id)
            chapter_personalization = (
                (1.0 if client_ref_hit else 0.0) +
                min(1.0, term_hits / 2.0) +
                min(1.0, kpi_hits / 2.0) +
                min(1.0, anchor_hits / 1.0)
            ) / 4.0
            personalization_score_total += chapter_personalization * business_rank
            personalization_weight += business_rank

            chapter_score = 100
            chapter_score -= 16 * len([issue for issue in issues if issue in {"missing_h2", "too_short", "missing_visual", "citation_policy"}])
            chapter_score -= 8 * len([issue for issue in issues if issue not in {"missing_h2", "too_short", "missing_visual", "citation_policy"}])
            chapter_score += min(8, value_hits * 2)
            chapter_score += min(8, usefulness_hits * 2)
            chapter_score = self._safe_score(chapter_score)

            weaknesses: List[str] = []
            if not client_ref_hit:
                weaknesses.append("client_specificity")
            if term_hits == 0 or kpi_hits == 0:
                weaknesses.append("personalization")
            if anchor_expected and anchor_hits == 0:
                weaknesses.append("external_or_internal_anchor")
            if value_hits == 0:
                weaknesses.append("business_value")
            if usefulness_hits == 0:
                weaknesses.append("actionability")
            if ai_mode and self._chapter_ai_terms(chapter_id) and ai_hits == 0:
                weaknesses.append("ai_adoption_balance")
            if "too_short" in issues:
                weaknesses.append("substance")

            chapter_findings[chapter_id] = {
                "title": chapter["title"],
                "issues": issues,
                "score": chapter_score,
                "client_ref_hit": client_ref_hit,
                "term_hits": term_hits,
                "kpi_hits": kpi_hits,
                "anchor_hits": anchor_hits,
                "value_hits": value_hits,
                "usefulness_hits": usefulness_hits,
                "ai_hits": ai_hits,
                "weaknesses": weaknesses,
            }

        format_penalty = 0
        if estimated_pages > MAX_PROPOSAL_PAGES:
            hard_failures.append(f"page_limit_exceeded:{estimated_pages}>{MAX_PROPOSAL_PAGES}")
            format_penalty += min(40, (estimated_pages - MAX_PROPOSAL_PAGES) * 10)
        format_penalty += min(40, len(hard_failures) * 4)
        format_penalty += min(20, len(soft_findings) * 2)
        format_fidelity = self._safe_score(100 - format_penalty)

        personalization_score = self._safe_score(
            100 * (personalization_score_total / max(personalization_weight, 1.0))
        )

        company_fit_hits = [
            1.0 if WRITER_FIRM_NAME.lower() in full_text.lower() else 0.0,
            min(1.0, self._count_signal_hits(full_text, company_terms, max_hits=6) / 4.0),
            min(1.0, self._count_signal_hits(full_text, value_map.get("proof_points", []) or [], max_hits=3) / 2.0),
        ]
        verified_contact_lines = FirmAPIClient.build_contact_lines(firm_profile)
        if verified_contact_lines and chapter_map.get("c_closing"):
            closing_text = chapter_outputs.get("c_closing", "")
            extracted_contact = FirmAPIClient._extract_contact_fields(closing_text)
            expected_contact = {
                "office_address": str(firm_profile.get("office_address") or "").strip(),
                "email": str(firm_profile.get("email") or "").strip(),
                "phone": str(firm_profile.get("phone") or "").strip(),
                "website": str(firm_profile.get("website") or "").strip(),
            }
            missing_contact_lines = [
                field_name for field_name, expected_value in expected_contact.items()
                if expected_value
                and expected_value not in closing_text
                and expected_value not in str(extracted_contact.get(field_name) or "")
            ]
            if missing_contact_lines:
                hard_failures.append("verified_contact_missing")
            company_fit_hits.append(1.0 if not missing_contact_lines else 0.0)
        company_fit = self._safe_score(100 * (sum(company_fit_hits) / max(len(company_fit_hits), 1)))

        persuasion_components = [
            min(1.0, self._count_signal_hits(full_text, persuasion_terms, max_hits=8) / 5.0),
            min(1.0, self._count_signal_hits(full_text, pressure_terms, max_hits=6) / 3.0),
        ]
        persuasion_aliases = (
            [("k_2",), ("k_3",), ("k_4",), ("k_5",), ("k_6",), ("k_8",)]
            if kak_mode else
            [("c_2",), ("c_3",), ("c_5",), ("c_6",), ("c_10",), ("c_11",), ("c_12",)]
        )
        for aliases in persuasion_aliases:
            chapter_text = self._chapter_output_by_aliases(chapter_outputs, *aliases)
            if not chapter_text:
                continue
            persuasion_components.append(
                min(1.0, self._count_signal_hits(chapter_text, persuasion_terms + pressure_terms, max_hits=5) / 2.0)
            )
        persuasion_score = self._safe_score(100 * (sum(persuasion_components) / max(len(persuasion_components), 1)))

        if kak_mode:
            usefulness_checks = [
                1.0 if "[[GANTT:" in chapter_outputs.get("k_4", "") else 0.0 if chapter_map.get("k_4") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("k_3", ""), self._chapter_usefulness_terms("k_3"), max_hits=4) / 2.0) if chapter_map.get("k_3") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("k_4", ""), self._chapter_usefulness_terms("k_4"), max_hits=4) / 2.0) if chapter_map.get("k_4") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("k_5", ""), self._chapter_usefulness_terms("k_5"), max_hits=4) / 2.0) if chapter_map.get("k_5") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("k_6", ""), self._chapter_usefulness_terms("k_6"), max_hits=4) / 2.0) if chapter_map.get("k_6") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("k_1", ""), self._chapter_usefulness_terms("k_1"), max_hits=4) / 2.0) if chapter_map.get("k_1") else 1.0,
            ]
        else:
            usefulness_checks = [
                1.0 if "[[GANTT:" in chapter_outputs.get("c_8", "") else 0.0 if chapter_map.get("c_8") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("c_5", ""), self._chapter_usefulness_terms("c_5"), max_hits=4) / 2.0) if chapter_map.get("c_5") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("c_6", ""), self._chapter_usefulness_terms("c_6"), max_hits=4) / 2.0) if chapter_map.get("c_6") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("c_10", ""), self._chapter_usefulness_terms("c_10"), max_hits=4) / 2.0) if chapter_map.get("c_10") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("c_11", ""), self._chapter_usefulness_terms("c_11"), max_hits=4) / 2.0) if chapter_map.get("c_11") else 1.0,
                min(1.0, self._count_signal_hits(chapter_outputs.get("c_12", ""), self._chapter_usefulness_terms("c_12"), max_hits=4) / 2.0) if chapter_map.get("c_12") else 1.0,
            ]
        usefulness_score = self._safe_score(100 * (sum(usefulness_checks) / max(len(usefulness_checks), 1)))

        factual_penalty = 0
        if invalid_external_citations:
            factual_penalty += min(50, len(set(invalid_external_citations)) * 20)
        closing_contact_lines = self._find_contact_like_lines(chapter_outputs.get("c_closing", "")) if chapter_map.get("c_closing") else []
        if verified_contact_lines and chapter_map.get("c_closing"):
            allowed_contact_values = set()
            for line in verified_contact_lines:
                extracted = FirmAPIClient._extract_contact_fields(line)
                allowed_contact_values.update(
                    value.strip() for value in extracted.values() if str(value).strip()
                )
            unverified_contact_lines = [
                line for line in closing_contact_lines
                if not any(value and value in line for value in allowed_contact_values)
            ]
        else:
            unverified_contact_lines = closing_contact_lines
        if unverified_contact_lines:
            factual_penalty += min(40, len(unverified_contact_lines) * 10)
            hard_failures.append("unverified_contact_detail")
        factual_safety = self._safe_score(100 - factual_penalty)

        categories = {
            "format_fidelity": format_fidelity,
            "personalization": personalization_score,
            "company_fit": company_fit,
            "persuasion": persuasion_score,
            "usefulness": usefulness_score,
            "factual_safety": factual_safety,
        }
        ai_adoption_fit = 100
        if ai_mode:
            quality_signals = SPIRIT_OF_AI_RULES.get("quality_signals") or {}
            business_terms = self._semantic_terms(
                quality_signals.get("business_value", []) + [ai_profile.get("business_case", ""), value_map.get("value_statement", "")],
                max_terms=20
            )
            readiness_terms = self._semantic_terms(
                quality_signals.get("readiness_realism", []) + [ai_profile.get("data_foundation", ""), ai_profile.get("architecture_posture", "")],
                max_terms=20
            )
            governance_terms = self._semantic_terms(
                quality_signals.get("governance_control", []) + [ai_profile.get("governance_posture", "")],
                max_terms=20
            )
            delivery_terms = self._semantic_terms(
                quality_signals.get("delivery_feasibility", []) + (ai_profile.get("delivery_guidance", []) or []),
                max_terms=20
            )
            change_terms = self._semantic_terms(
                quality_signals.get("change_adoption", []) + [ai_profile.get("culture_change", ""), ai_profile.get("people_capability", "")],
                max_terms=20
            )
            if kak_mode:
                readiness_text = "\n".join([
                    chapter_outputs.get("k_2", ""),
                    chapter_outputs.get("k_3", ""),
                    chapter_outputs.get("k_4", ""),
                ])
                governance_text = "\n".join([
                    chapter_outputs.get("k_2", ""),
                    chapter_outputs.get("k_3", ""),
                    chapter_outputs.get("k_5", ""),
                    chapter_outputs.get("k_7", ""),
                ])
                delivery_text = "\n".join([
                    chapter_outputs.get("k_3", ""),
                    chapter_outputs.get("k_4", ""),
                    chapter_outputs.get("k_5", ""),
                    chapter_outputs.get("k_6", ""),
                ])
                change_text = "\n".join([
                    chapter_outputs.get("k_4", ""),
                    chapter_outputs.get("k_5", ""),
                    chapter_outputs.get("k_8", ""),
                    full_text,
                ])
            else:
                readiness_text = "\n".join([
                    chapter_outputs.get("c_2", ""),
                    chapter_outputs.get("c_3", ""),
                    chapter_outputs.get("c_5", ""),
                    chapter_outputs.get("c_6", ""),
                ])
                governance_text = "\n".join([
                    chapter_outputs.get("c_4", ""),
                    chapter_outputs.get("c_9", ""),
                    chapter_outputs.get("c_12", ""),
                ])
                delivery_text = "\n".join([
                    chapter_outputs.get("c_5", ""),
                    chapter_outputs.get("c_6", ""),
                    chapter_outputs.get("c_7", ""),
                    chapter_outputs.get("c_8", ""),
                ])
                change_text = "\n".join([
                    chapter_outputs.get("c_8", ""),
                    chapter_outputs.get("c_11", ""),
                    full_text,
                ])
            ai_components = [
                min(1.0, self._count_signal_hits(full_text, business_terms, max_hits=8) / 4.0),
                min(1.0, self._count_signal_hits(readiness_text, readiness_terms, max_hits=8) / 4.0),
                min(1.0, self._count_signal_hits(governance_text, governance_terms, max_hits=8) / 4.0),
                min(1.0, self._count_signal_hits(delivery_text, delivery_terms, max_hits=8) / 4.0),
                min(1.0, self._count_signal_hits(change_text, change_terms, max_hits=8) / 4.0),
            ]
            ai_adoption_fit = self._safe_score(100 * (sum(ai_components) / max(len(ai_components), 1)))
            if ai_adoption_fit < self.PROPOSAL_CATEGORY_FLOOR:
                soft_findings.append("ai_adoption_fit_low")
        total_score = self._safe_score(
            (format_fidelity * 0.20) +
            (personalization_score * 0.25) +
            (company_fit * 0.20) +
            (persuasion_score * 0.15) +
            (usefulness_score * 0.10) +
            (factual_safety * 0.10)
        )
        if ai_mode:
            total_score = self._safe_score((total_score * 0.85) + (ai_adoption_fit * 0.15))
        low_categories = [name for name, score in categories.items() if score < self.PROPOSAL_CATEGORY_FLOOR]

        return {
            "score": total_score,
            "categories": categories,
            "ai_adoption_fit": ai_adoption_fit,
            "estimated_pages": estimated_pages,
            "generated_words": generated_words,
            "hard_failures": sorted(set(hard_failures)),
            "soft_findings": sorted(set(soft_findings)),
            "low_categories": low_categories,
            "chapter_findings": chapter_findings,
            "passes": (
                not hard_failures
                and total_score >= self.PROPOSAL_ACCEPTANCE_TARGET
                and not low_categories
                and (not ai_mode or ai_adoption_fit >= self.PROPOSAL_CATEGORY_FLOOR)
            ),
        }

    def _select_improvement_chapters(
        self,
        acceptance_report: Dict[str, Any],
        selected_chapters: List[Dict[str, Any]]
    ) -> List[str]:
        limit = 2 if self._throughput_mode() else 3
        candidates: List[Tuple[int, int, str]] = []
        for chapter in selected_chapters:
            chapter_id = chapter["id"]
            if chapter_id == "c_closing" or self._use_structured_chapter(chapter_id):
                continue
            finding = acceptance_report.get("chapter_findings", {}).get(chapter_id, {})
            issues = finding.get("issues", []) or []
            weaknesses = finding.get("weaknesses", []) or []
            score = int(finding.get("score", 100))
            if score >= 82 and not issues and not weaknesses:
                continue
            priority = self._chapter_business_rank(chapter_id)
            candidates.append((score, -priority, chapter_id))
        candidates.sort()
        return [chapter_id for _, _, chapter_id in candidates[:limit]]

    def _improve_weak_chapter(
        self,
        chapter: Dict[str, Any],
        prompt: str,
        content: str,
        client: str,
        target_words: int,
        acceptance_report: Dict[str, Any],
        personalization_pack: Dict[str, Any],
        value_map: Dict[str, Any],
        allowed_external_citations: Set[str],
        timeline: str = ""
    ) -> str:
        finding = acceptance_report.get("chapter_findings", {}).get(chapter["id"], {})
        weaknesses = finding.get("weaknesses", []) or ["personalization", "business_value", "actionability"]
        focus_terms = ", ".join(personalization_pack.get("terminology", [])[:3]) or "governance, delivery, risk control"
        kpi_line = self._human_join(
            personalization_pack.get("kpi_blueprint", [])[:3],
            fallback=f"hasil bisnis terukur untuk {client}",
        )
        proof_line = ", ".join(value_map.get("proof_points", [])[:3]) or "kapabilitas delivery dan kontrol mutu yang tersedia"
        ai_profile = personalization_pack.get("ai_adoption_profile", {}) or {}
        ai_guidance = str((ai_profile.get("chapter_guidance") or {}).get(chapter["id"], "")).strip()
        ai_extra = ""
        if ai_profile.get("enabled"):
            ai_extra = (
                "- untuk konteks AI, perkuat nuansa use case bisnis, readiness, governance, feasibility, dan adoption tanpa menyebut framework enam pilar secara eksplisit\n"
                f"- arahan AI khusus bab ini: {ai_guidance or 'jaga keseimbangan antara nilai bisnis, kontrol, dan kesiapan eksekusi'}\n"
            )
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": (
                        f"Perkuat draft {chapter['title']} agar lebih layak sebagai proposal 80% siap pakai.\n"
                        f"Fokus perbaikan: {', '.join(weaknesses)}.\n"
                        f"Wajib:\n"
                        f"- lebih spesifik ke konteks {client}\n"
                        f"- lebih jelas menghubungkan isi bab dengan KPI {kpi_line}\n"
                        f"- lebih terasa nilai dan kredibilitas {WRITER_FIRM_NAME} lewat bukti {proof_line}\n"
                        f"- lebih actionable untuk sponsor, decision maker, dan delivery team\n"
                        f"- tetap faktual, jangan menambah klaim baru di luar draft/sumber yang sudah ada\n"
                        f"- pertahankan H2 wajib, numbering, bullet, dan disiplin panjang sekitar {target_words} kata\n"
                        "- jangan sebut nama domain, URL, atau label sumber seperti Data Internal di tubuh paragraf\n"
                        f"- gunakan istilah domain secara natural: {focus_terms}\n"
                        f"{ai_extra}\n"
                        f"DRAFT SAAT INI:\n{content}"
                    )},
                ],
                options=self._chapter_generation_options(target_words, purpose="retry")
            )
            revised = (res.get("message", {}).get("content", "") or "").strip()
        except Exception:
            revised = ""

        repaired = self._apply_draft_repairs(
            chapter=chapter,
            content=revised or content,
            client=client,
            allowed_external_citations=allowed_external_citations,
            personalization_pack=personalization_pack,
            target_words=target_words,
            timeline=timeline,
        )
        return repaired

    def _stabilize_chapter_outputs(
        self,
        chapter_outputs: Dict[str, str],
        selected_chapters: List[Dict[str, Any]],
        chapter_targets: Dict[str, int],
        client: str,
        timeline: str,
        allowed_external_citations: Set[str],
        personalization_pack: Dict[str, Any]
    ) -> Dict[str, str]:
        stabilized = dict(chapter_outputs)
        for chapter in selected_chapters:
            chapter_id = chapter["id"]
            content = stabilized.get(chapter_id, "")
            if not content:
                continue
            report = self._evaluate_chapter_quality(
                chapter=chapter,
                content=content,
                client=client,
                target_words=chapter_targets.get(chapter_id, self._target_words(chapter)),
                allowed_external_citations=allowed_external_citations,
                personalization_pack=personalization_pack,
            )
            if not any(issue in report["issues"] for issue in {"missing_h2", "too_short", "list_structure", "missing_visual", "missing_personalization"}):
                continue
            stabilized[chapter_id] = self._apply_draft_repairs(
                chapter=chapter,
                content=content,
                client=client,
                allowed_external_citations=allowed_external_citations,
                personalization_pack=personalization_pack,
                target_words=chapter_targets.get(chapter_id, self._target_words(chapter)),
                timeline=timeline,
            )
        return stabilized

    # ========== ENHANCED CLOSING CHAPTER METHODS ==========
    # Helper methods for enhancing closing chapters with OSINT firm information
    
    @staticmethod
    def _build_firm_information_section(firm_name: str, office_location: str = "") -> str:
        """Build comprehensive firm information section using OSINT."""
        try:
            firm_profile = Researcher.build_comprehensive_firm_profile(firm_name, office_location)
            return ProposalSupportMixin._format_firm_section(firm_name=firm_name, firm_profile=firm_profile)
        except Exception as e:
            logger.error(f"Error building enhanced firm information: {e}")
            return ""
    
    @staticmethod
    def _format_firm_section(firm_name: str, firm_profile: Dict[str, str]) -> str:
        """Format firm information into proposal closing text."""
        sections = []
        sections.append("## Tentang Penulis Proposal")
        sections.append("")
        
        # Opening paragraph about the firm
        sections.append(
            f"{firm_name} adalah mitra konsultasi dan delivery yang fokus pada solusi "
            "strategis, transformasi terstruktur, dan hasil bisnis yang terukur."
        )
        sections.append("")
        
        # Values and approach
        if firm_profile.get("values_approach"):
            sections.append(f"**Pendekatan Kami:** {firm_profile['values_approach']}")
            sections.append("")
        
        # Team expertise
        if firm_profile.get("team_expertise"):
            sections.append(f"**Keahlian Tim:** {firm_profile['team_expertise']}")
            sections.append("")
        
        # Portfolio and scale
        if firm_profile.get("portfolio_scale"):
            sections.append(f"**Portofolio & Pengalaman:** {firm_profile['portfolio_scale']}")
            sections.append("")
        
        # Certifications and credentials
        if firm_profile.get("certifications"):
            sections.append(f"**Sertifikasi & Kredensial:** {firm_profile['certifications']}")
            sections.append("")
        
        # Recognition and accolades
        if firm_profile.get("accolades"):
            sections.append(f"**Pengakuan Industri:** {firm_profile['accolades']}")
            sections.append("")
        
        # How to reach out
        if firm_profile.get("key_contacts"):
            sections.append(f"**Hubungi Kami:** {firm_profile['key_contacts']}")
            sections.append("")
        
        return "\n".join(sections)
    
    @staticmethod
    def _build_firm_information_section_from_osint(
        firm_name: str,
        comprehensive_profile: Optional[Dict[str, str]] = None,
        firm_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build firm information section for closing chapter using OSINT data.
        Integrates external numbers from Serper, certifications, portfolio scale, and credentials
        to demonstrate firm credibility and persuade client they're hiring the right team.
        
        Args:
            firm_name: Name of the firm
            comprehensive_profile: OSINT-gathered firm profile data (values, team, certifications, portfolio, contacts, accolades)
            firm_profile: Internal firm profile data
            
        Returns:
            Formatted firm information section with external OSINT data
        """
        if not comprehensive_profile:
            return ""
        
        sections = []
        sections.append("## Tentang Mitra Penulis Proposal")
        sections.append("")
        
        # Opening paragraph about the firm with credibility framing
        sections.append(
            f"{firm_name} adalah mitra konsultasi dan delivery yang fokus pada solusi "
            "strategis, transformasi terstruktur, dan hasil bisnis yang terukur. Kami berkomitmen "
            "untuk memberikan nilai jangka panjang dan membantu klien mencapai tujuan transformasi mereka."
        )
        sections.append("")
        
        # Firm values and approach from OSINT
        if comprehensive_profile.get("values_approach"):
            sections.append(f"**Pendekatan & Nilai Kami:** {comprehensive_profile['values_approach']}")
            sections.append("")
        
        # Team expertise from OSINT
        if comprehensive_profile.get("team_expertise"):
            sections.append(f"**Keahlian Tim:** {comprehensive_profile['team_expertise']}")
            sections.append("")
        
        # Portfolio and scale (external numbers from OSINT)
        if comprehensive_profile.get("portfolio_scale"):
            sections.append(f"**Portofolio & Skala Pengalaman:** {comprehensive_profile['portfolio_scale']}")
            sections.append("")
        
        # Certifications and credentials from OSINT
        if comprehensive_profile.get("certifications"):
            sections.append(f"**Sertifikasi & Kredensial:** {comprehensive_profile['certifications']}")
            sections.append("")
        
        # Industry recognition and accolades
        if comprehensive_profile.get("accolades"):
            sections.append(f"**Pengakuan Industri:** {comprehensive_profile['accolades']}")
            sections.append("")
        
        # Contact information
        if comprehensive_profile.get("key_contacts"):
            sections.append(f"**Hubungi Kami:** {comprehensive_profile['key_contacts']}")
            sections.append("")
        
        return "\n".join(sections)
    
    @staticmethod
    def _build_firm_information_section(firm_name: str, firm_profile: Dict[str, str]) -> str:
        """Build comprehensive firm information section using OSINT."""
        try:
            firm_profile_comprehensive = Researcher.build_comprehensive_firm_profile(firm_name, firm_profile.get("office_address", "") if firm_profile else "")
            return ProposalSupportMixin._format_firm_section(firm_name=firm_name, firm_profile=firm_profile_comprehensive)
        except Exception as e:
            logger.error(f"Error building enhanced firm information: {e}")
            return ""
    
    @staticmethod
    def _format_firm_section(firm_name: str, firm_profile: Dict[str, str]) -> str:
        """Format firm information into proposal closing text."""
        sections = []
        sections.append("## Tentang Penulis Proposal")
        sections.append("")
        
        # Opening paragraph about the firm
        sections.append(
            f"{firm_name} adalah mitra konsultasi dan delivery yang fokus pada solusi "
            "strategis, transformasi terstruktur, dan hasil bisnis yang terukur."
        )
        sections.append("")
        
        # Values and approach
        if firm_profile.get("values_approach"):
            sections.append(f"**Pendekatan Kami:** {firm_profile['values_approach']}")
            sections.append("")
        
        # Team expertise
        if firm_profile.get("team_expertise"):
            sections.append(f"**Keahlian Tim:** {firm_profile['team_expertise']}")
            sections.append("")
        
        # Portfolio and scale
        if firm_profile.get("portfolio_scale"):
            sections.append(f"**Portofolio & Pengalaman:** {firm_profile['portfolio_scale']}")
            sections.append("")
        
        # Certifications and credentials
        if firm_profile.get("certifications"):
            sections.append(f"**Sertifikasi & Kredensial:** {firm_profile['certifications']}")
            sections.append("")
        
        # Recognition and accolades
        if firm_profile.get("accolades"):
            sections.append(f"**Pengakuan Industri:** {firm_profile['accolades']}")
            sections.append("")
        
        # How to reach out
        if firm_profile.get("key_contacts"):
            sections.append(f"**Hubungi Kami:** {firm_profile['key_contacts']}")
            sections.append("")
        
        return "\n".join(sections)
    
    @staticmethod
    def _enhance_closing_with_firm_details(
        closing_content: str,
        firm_name: str,
        firm_profile: Optional[Dict[str, Any]] = None,
        office_location: str = ""
    ) -> str:
        """Keep closing content concise; the structured firm profile is rendered in DOCX assembly."""
        result = (closing_content or "").strip()
        if result:
            return result

        profile = firm_profile or {}
        summary = str(profile.get("profile_summary") or "").strip()
        contact_lines = FirmAPIClient.build_contact_lines(profile)

        if not summary and not contact_lines:
            return ""

        parts = []
        if summary:
            parts.append(summary)
        if contact_lines:
            parts.append("## Informasi Kontak dan Langkah Lanjutan\n\n" + "\n".join(f"- {line}" for line in contact_lines))
        return "\n\n".join(parts)
