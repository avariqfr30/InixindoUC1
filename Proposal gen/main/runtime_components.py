"""Runtime components for data access, research, pricing, and document rendering."""

from .proposal_shared import *


class SchemaMapper:
    @classmethod
    def flatten_payload(cls, payload: Any, prefix: str = "") -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        flattened: Dict[str, Any] = {}
        for key, value in payload.items():
            raw_key = str(key or "").strip()
            if not raw_key:
                continue
            next_prefix = f"{prefix}_{raw_key}" if prefix else raw_key
            if isinstance(value, dict):
                flattened.update(cls.flatten_payload(value, next_prefix))
                continue
            if isinstance(value, list) and value and all(not isinstance(item, (dict, list)) for item in value):
                flattened[next_prefix] = ", ".join(str(item) for item in value if str(item).strip())
                continue
            flattened[next_prefix] = value
        return flattened

    @staticmethod
    def normalize_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    @classmethod
    def _token_set(cls, value: Any) -> Set[str]:
        return set(filter(None, cls.normalize_key(value).split("_")))

    @classmethod
    def _resolve_alias(cls, raw_key: Any, alias_map: Dict[str, List[str]]) -> Optional[str]:
        normalized_key = cls.normalize_key(raw_key)
        if not normalized_key:
            return None

        direct_hits: List[str] = []
        fuzzy_hits: List[Tuple[int, str]] = []
        raw_tokens = cls._token_set(raw_key)

        for canonical, aliases in (alias_map or {}).items():
            for alias in aliases or []:
                alias_key = cls.normalize_key(alias)
                if not alias_key:
                    continue
                if normalized_key == alias_key:
                    direct_hits.append(canonical)
                    break
                alias_tokens = cls._token_set(alias)
                if raw_tokens and alias_tokens and (alias_tokens <= raw_tokens or raw_tokens <= alias_tokens):
                    fuzzy_hits.append((len(alias_tokens & raw_tokens), canonical))

        if direct_hits:
            return direct_hits[0]
        if fuzzy_hits:
            fuzzy_hits.sort(reverse=True)
            return fuzzy_hits[0][1]
        return None

    @classmethod
    def normalize_record(
        cls,
        payload: Optional[Dict[str, Any]],
        alias_map: Dict[str, List[str]],
        passthrough_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        raw = dict(payload or {})
        flattened = cls.flatten_payload(raw)
        raw.update({k: v for k, v in flattened.items() if k not in raw})
        normalized: Dict[str, Any] = {}
        passthrough = passthrough_keys or []

        for key, value in raw.items():
            mapped_key = cls._resolve_alias(key, alias_map)
            target_key = mapped_key or cls.normalize_key(key)
            if mapped_key and normalized.get(mapped_key) not in (None, "", [], {}):
                continue
            normalized[target_key] = value

        for key in passthrough:
            if key in raw and key not in normalized:
                normalized[key] = raw[key]
        return normalized

    @classmethod
    def remap_dataframe(cls, df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> pd.DataFrame:
        renamed = df.copy()
        rename_map: Dict[str, str] = {}
        taken_targets = set(renamed.columns)

        for column in renamed.columns:
            target = cls._resolve_alias(column, alias_map)
            if not target or column == target or target in taken_targets:
                continue
            rename_map[column] = target
            taken_targets.add(target)

        if rename_map:
            renamed = renamed.rename(columns=rename_map)
        return renamed


# Internal API adapter.
class FirmAPIClient:
    def __init__(self) -> None:
        self.demo_mode = DEMO_MODE
        self.data_acquisition_mode = DATA_ACQUISITION_MODE
        self.base_url = FIRM_API_URL
        self.headers = {"Authorization": f"Bearer {API_AUTH_TOKEN}"}

    def uses_demo_logic(self) -> bool:
        return self.demo_mode or self.data_acquisition_mode == "demo"

    @staticmethod
    def _normalize_project_standards(raw_payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
        normalized = SchemaMapper.normalize_record(raw_payload, PROJECT_STANDARD_FIELD_ALIASES)
        methodology = str(normalized.get("methodology") or "").strip() or "TBD"
        team = str(normalized.get("team") or "").strip() or "TBD"
        commercial = str(normalized.get("commercial") or "").strip() or "TBD"
        return {
            "methodology": methodology,
            "team": team,
            "commercial": commercial,
        }

    @staticmethod
    def _normalize_relationship_mode(value: Any) -> str:
        normalized = SchemaMapper.normalize_key(value)
        existing_markers = {
            "existing", "active", "returning", "repeat", "renewal", "incumbent",
            "prior", "previous", "historical", "yes", "true", "1"
        }
        return "existing" if normalized in existing_markers else "new"

    @classmethod
    def _normalize_client_relationship(cls, raw_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = SchemaMapper.normalize_record(raw_payload, CLIENT_RELATIONSHIP_FIELD_ALIASES)
        status_value = normalized.get("status", "")
        summary = str(normalized.get("summary") or "").strip()
        mode = cls._normalize_relationship_mode(status_value)
        return {
            "summary": summary,
            "mode": mode,
            "source": "internal_api",
            "verified": bool(summary or str(status_value).strip()),
        }

    @staticmethod
    def _empty_relationship_context(source: str = "internal_api") -> Dict[str, Any]:
        return {
            "summary": "",
            "mode": "new",
            "source": source,
            "verified": False,
        }

    @staticmethod
    def _has_osint_evidence(summary: str) -> bool:
        return any(line.strip().startswith("Sumber eksternal") for line in (summary or "").splitlines())

    def get_project_standards(self, project_type: str) -> Dict[str, str]:
        if self.demo_mode:
            logger.info("Using demo standards for project type: %s", project_type)
            demo_standards = MOCK_FIRM_STANDARDS.get(project_type, MOCK_FIRM_STANDARDS.get("Implementation"))
            return self._normalize_project_standards(demo_standards)
        try:
            res = requests.get(f"{self.base_url}/standards/{project_type}", headers=self.headers, timeout=5)
            res.raise_for_status()
            return self._normalize_project_standards(res.json())
        except requests.RequestException as e:
            logger.error(f"Internal API Error: {e}")
            return self._normalize_project_standards({})

    @classmethod
    def _default_firm_profile(cls) -> Dict[str, str]:
        base_profile = dict(MOCK_FIRM_PROFILE)
        base_profile["office_address"] = base_profile.get("office_address") or WRITER_FIRM_OFFICE_ADDRESS
        base_profile["email"] = base_profile.get("email") or WRITER_FIRM_EMAIL
        base_profile["phone"] = base_profile.get("phone") or WRITER_FIRM_PHONE
        base_profile["website"] = base_profile.get("website") or WRITER_FIRM_WEBSITE
        return cls._normalize_firm_profile(base_profile)

    def get_firm_profile(self) -> Dict[str, str]:
        if self.demo_mode:
            return self._default_firm_profile()
        try:
            res = requests.get(f"{self.base_url}/firm-profile", headers=self.headers, timeout=5)
            res.raise_for_status()
            return self._normalize_firm_profile(res.json())
        except requests.RequestException as e:
            logger.error(f"Internal API Error: {e}")
            if self.uses_demo_logic():
                return self._build_profile_from_osint()
            return self._default_firm_profile()

    def get_client_relationship(self, client_name: str) -> Dict[str, Any]:
        if self.uses_demo_logic():
            summary = Researcher.get_client_writer_collaboration(client_name, WRITER_FIRM_NAME)
            has_evidence = self._has_osint_evidence(summary)
            return {
                "summary": summary,
                "mode": "existing" if has_evidence else "new",
                "source": "osint",
                "verified": has_evidence,
            }

        try:
            res = requests.get(
                f"{self.base_url}/client-relationship",
                params={"client_name": client_name},
                headers=self.headers,
                timeout=5
            )
            res.raise_for_status()
            return self._normalize_client_relationship(res.json())
        except requests.RequestException as e:
            logger.error(f"Internal API Error: {e}")
            return self._empty_relationship_context()

    @staticmethod
    def _extract_first(pattern: str, text: str) -> str:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        return match.group(0).strip() if match else ""

    @staticmethod
    def _clean_contact_value(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", (value or "").replace("\xa0", " ")).strip()
        return cleaned.strip(" ,;|-")

    @classmethod
    def _looks_like_address(cls, value: str) -> bool:
        cleaned = cls._clean_contact_value(value)
        lowered = cleaned.lower()
        if len(cleaned) < 14:
            return False
        if "@" in cleaned or re.search(r"https?://|www\.", lowered):
            return False
        address_markers = (
            "jl.", "jalan", "alamat", "gedung", "tower", "ruko", "komplek", "kompleks",
            "kawasan", "suite", "lantai", "yogyakarta", "sleman", "bantul", "jakarta",
            "bandung", "surabaya", "indonesia"
        )
        return any(marker in lowered for marker in address_markers)

    @classmethod
    def _extract_address_candidates(cls, text: str) -> List[str]:
        if not text:
            return []
        candidates: List[str] = []
        patterns = [
            r"(?:alamat|address)\s*[:\-]?\s*([^|\n]{12,180})",
            r"((?:jl\.|jalan)\s+[^|\n]{12,180})",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidate = match.group(1) if match.lastindex else match.group(0)
                candidate = re.split(r"(?:\s+[|]\s+| email\s*:| telp\s*:| phone\s*:| website\s*:)", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
                candidate = cls._clean_contact_value(candidate)
                if cls._looks_like_address(candidate):
                    candidates.append(candidate)

        unique_candidates: List[str] = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            unique_candidates.append(candidate)
            seen.add(key)
        return unique_candidates

    @classmethod
    def _extract_contact_fields(cls, text: str) -> Dict[str, str]:
        office_candidates = cls._extract_address_candidates(text)
        website = cls._extract_first(r"(?:https?://|www\.)[A-Za-z0-9./_%#?=&-]+\.[A-Za-z]{2,}", text)
        return {
            "office_address": office_candidates[0] if office_candidates else "",
            "email": cls._extract_first(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text),
            "phone": cls._extract_first(r"(?:\+62|62|0)\d[\d\-\s()]{7,}\d", text),
            "website": website.rstrip(".,;)"),
        }

    @classmethod
    def build_contact_lines(cls, firm_profile: Optional[Dict[str, Any]]) -> List[str]:
        profile = firm_profile or {}
        lines = []
        office_address = cls._clean_contact_value(str(profile.get("office_address") or ""))
        email = cls._clean_contact_value(str(profile.get("email") or ""))
        phone = cls._clean_contact_value(str(profile.get("phone") or ""))
        website = cls._clean_contact_value(str(profile.get("website") or ""))

        if office_address:
            lines.append(f"Alamat kantor: {office_address}")
        if email:
            lines.append(f"Email: {email}")
        if phone:
            lines.append(f"Telp: {phone}")
        if website:
            if not re.match(r"^https?://", website, flags=re.IGNORECASE):
                website = f"https://{website.lstrip('/')}"
            lines.append(f"Website: {website}")
        return lines

    @classmethod
    def _normalize_firm_profile(cls, raw_profile: Optional[Dict[str, Any]]) -> Dict[str, str]:
        source = SchemaMapper.normalize_record(raw_profile, FIRM_PROFILE_FIELD_ALIASES)
        raw_contact = str(source.get("contact_info") or "").strip()
        parsed = cls._extract_contact_fields(raw_contact)

        office_address = cls._clean_contact_value(str(
            source.get("office_address")
            or source.get("address")
            or source.get("office_location")
            or source.get("location")
            or parsed.get("office_address")
            or ""
        ))
        email = cls._clean_contact_value(str(
            source.get("email")
            or source.get("official_email")
            or parsed.get("email")
            or ""
        ))
        phone = cls._clean_contact_value(str(
            source.get("phone")
            or source.get("telephone")
            or source.get("telp")
            or source.get("mobile")
            or parsed.get("phone")
            or ""
        ))
        website = cls._clean_contact_value(str(
            source.get("website")
            or source.get("url")
            or parsed.get("website")
            or ""
        ))
        portfolio = cls._clean_contact_value(str(source.get("portfolio_highlights") or ""))
        contact_lines = cls.build_contact_lines(
            {
                "office_address": office_address,
                "email": email,
                "phone": phone,
                "website": website,
            }
        )

        return {
            "office_address": office_address,
            "email": email,
            "phone": phone,
            "website": website,
            "contact_info": "\n".join(contact_lines),
            "portfolio_highlights": portfolio or "Kapabilitas layanan menyesuaikan kebutuhan proyek klien.",
        }

    def _build_profile_from_osint(self) -> Dict[str, str]:
        current_year = datetime.now().year
        query = (
            f'"{WRITER_FIRM_NAME}" Yogyakarta kontak email telp alamat profil '
            f'{current_year} OR {current_year - 1}'
        )
        raw_hits = Researcher.search(query, limit=8, recency_bucket="year")
        hits = Researcher._filter_recent_entity_results(
            raw_hits,
            entity_name=WRITER_FIRM_NAME,
            max_age_years=4,
            strict_entity=True
        )
        if not hits:
            fallback_query = f'"{WRITER_FIRM_NAME}" Yogyakarta pelatihan konsultasi kontak resmi'
            fallback_hits = Researcher.search(fallback_query, limit=8)
            hits = Researcher._filter_recent_entity_results(
                fallback_hits,
                entity_name=WRITER_FIRM_NAME,
                max_age_years=6,
                strict_entity=True
            )
        address_query = f'"{WRITER_FIRM_NAME}" alamat kantor Yogyakarta OR "Jl." OR "Jalan"'
        address_hits = Researcher.search(address_query, limit=8, recency_bucket="year")
        address_hits = Researcher._filter_recent_entity_results(
            address_hits,
            entity_name=WRITER_FIRM_NAME,
            max_age_years=6,
            strict_entity=True
        )

        merged_hits = hits + [item for item in address_hits if item not in hits]
        merged_text = " ".join(
            [
                " ".join(
                    [
                        str(item.get("title", "")),
                        str(item.get("snippet", "")),
                        str(item.get("link", "")),
                    ]
                )
                for item in merged_hits
            ]
        )
        parsed = self._extract_contact_fields(merged_text)

        if not parsed.get("office_address"):
            for item in merged_hits:
                item_text = " ".join(
                    [
                        str(item.get("title", "")),
                        str(item.get("snippet", "")),
                        str(item.get("link", "")),
                    ]
                )
                parsed_item = self._extract_contact_fields(item_text)
                if parsed_item.get("office_address"):
                    parsed["office_address"] = parsed_item["office_address"]
                    break

        return self._normalize_firm_profile(
            {
                "office_address": parsed.get("office_address", ""),
                "email": parsed.get("email", ""),
                "phone": parsed.get("phone", ""),
                "website": parsed.get("website", ""),
                "portfolio_highlights": "Kapabilitas layanan menyesuaikan kebutuhan proyek klien.",
            }
        )


# Knowledge base and vector index.
class KnowledgeBase:
    def __init__(self, db_uri: str) -> None:
        self.engine = create_engine(db_uri)
        self.chroma = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings", model_name=EMBED_MODEL
        )
        self.collection = self.chroma.get_or_create_collection(
            name="projects_db", embedding_function=self.embed_fn
        )
        self.df: Optional[pd.DataFrame] = None
        self.refresh_data()

    @staticmethod
    def _required_project_fields() -> Tuple[str, ...]:
        return ("entity", "topic")

    @classmethod
    def _normalize_projects_df(cls, raw_df: pd.DataFrame) -> pd.DataFrame:
        normalized = raw_df.copy()
        normalized.columns = [str(col).strip() for col in normalized.columns]
        normalized = SchemaMapper.remap_dataframe(normalized, PROJECT_DATA_FIELD_ALIASES)
        return normalized

    @classmethod
    def _has_required_project_fields(cls, df: Optional[pd.DataFrame]) -> bool:
        if df is None:
            return False
        return all(field in df.columns for field in cls._required_project_fields())

    def refresh_data(self) -> bool:
        try:
            self.df = self._normalize_projects_df(pd.read_sql("SELECT * FROM projects", self.engine))
        except Exception:
            if not PROJECT_CSV_PATH.exists():
                return False
            raw_df = pd.read_csv(PROJECT_CSV_PATH)
            normalized_df = self._normalize_projects_df(raw_df)
            normalized_df.to_sql("projects", self.engine, index=False, if_exists='replace')
            self.df = normalized_df

        if not self._has_required_project_fields(self.df):
            logger.warning(
                "Project data schema is missing required fields. Expected aliases for: %s. Available columns: %s",
                ", ".join(self._required_project_fields()),
                ", ".join(self.df.columns.astype(str).tolist()) if self.df is not None else "-",
            )
            return False
            
        existing_ids = set(self.collection.get()['ids'])
        new_ids_map = {str(idx): row for idx, row in self.df.iterrows()}
        new_ids_set = set(new_ids_map.keys())
        
        ids_to_delete = list(existing_ids - new_ids_set)
        
        if ids_to_delete: 
            self.collection.delete(ids_to_delete)

        all_ids = list(new_ids_set)
        if all_ids:
            for i in range(0, len(all_ids), 500):
                batch_ids = all_ids[i:i + 500]
                docs = [" | ".join([f"{col}: {val}" for col, val in new_ids_map[b].items()]) for b in batch_ids]
                metas = [new_ids_map[b].astype(str).to_dict() for b in batch_ids]
                self.collection.upsert(documents=docs, metadatas=metas, ids=batch_ids)
                
        return True

    def get_exact_context(self, entity: str, topic: str, budget: Optional[str] = None) -> str:
        if self.df is None or self.df.empty or not self._has_required_project_fields(self.df):
            return "No data."
        try:
            match = self.df[(self.df['entity'] == entity) & (self.df['topic'] == topic)]
            if budget and 'budget' in self.df.columns and not match.empty:
                match = match[match['budget'] == budget]
            if not match.empty:
                return "".join([f"- {k.capitalize()}: {v}\n" for k, v in match.iloc[0].to_dict().items()])
            return "No data."
        except Exception:
            return ""

    def query(self, client: str, project: str, context_keywords: str = "") -> str:
        try:
            res = self.collection.query(query_texts=[f"{project} for {client} {context_keywords}"], n_results=2)
            if res['documents'] and len(res['documents'][0]) > 0:
                return "\n".join(res['documents'][0])
            return ""
        except Exception:
            return ""


# Public research helper (Serper).
class Researcher:
    @staticmethod
    def _has_serper_key() -> bool:
        key = (SERPER_API_KEY or "").strip()
        if not key:
            return False
        placeholder_keys = {"YOUR_SERPER_API_KEY", "YOUR_SERPER", "SERPER_API"}
        return key not in placeholder_keys

    @staticmethod
    @lru_cache(maxsize=256)
    def search(query: str, limit: int = 5, recency_bucket: str = "") -> List[Dict[str, Any]]:
        """General web search using Serper.dev"""
        if not Researcher._has_serper_key():
            return []
        url = "https://google.serper.dev/search"
        payload_data = {"q": query, "gl": "id", "num": limit}
        recency_map = {
            "week": "qdr:w",
            "month": "qdr:m",
            "year": "qdr:y",
        }
        if recency_bucket in recency_map:
            payload_data["tbs"] = recency_map[recency_bucket]
        payload = json.dumps(payload_data)
        headers = {
            'X-API-KEY': (SERPER_API_KEY or "").strip(),
            'Content-Type': 'application/json'
        }
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=8)
            response.raise_for_status()
            return response.json().get('organic', [])
        except requests.RequestException as e:
            logger.warning(f"Serper API Error: {e}")
            return []

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _entity_tokens(entity_name: str) -> List[str]:
        legal_tokens = {"pt", "cv", "tbk", "inc", "ltd", "co", "corp", "persero", "company"}
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", (entity_name or "").lower())
            if len(token) >= 3 and token not in legal_tokens
        ]
        # Keep order while dropping duplicates.
        ordered = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            ordered.append(token)
            seen.add(token)
        return ordered

    @staticmethod
    def _is_entity_match(item: Dict[str, Any], entity_name: str, strict: bool = False) -> bool:
        if not entity_name:
            return True

        merged = " ".join([
            str(item.get("title", "")),
            str(item.get("snippet", "")),
            str(item.get("link", "")),
        ])
        normalized_merged = Researcher._normalize_text(merged)
        phrase = Researcher._normalize_text(entity_name)
        tokens = Researcher._entity_tokens(entity_name)

        if phrase and phrase in normalized_merged:
            return True

        if not tokens:
            return False

        merged_tokens = set(normalized_merged.split())
        hits = sum(1 for token in tokens if token in merged_tokens)
        if strict:
            return hits == len(tokens)
        if len(tokens) == 1:
            return hits == 1
        if len(tokens) == 2:
            return hits == 2
        return hits >= (len(tokens) - 1)

    @staticmethod
    def _extract_month(text: str) -> Optional[int]:
        if not text:
            return None
        month_map = {
            "jan": 1, "januari": 1, "january": 1,
            "feb": 2, "februari": 2, "february": 2,
            "mar": 3, "maret": 3, "march": 3,
            "apr": 4, "april": 4,
            "mei": 5, "may": 5,
            "jun": 6, "juni": 6, "june": 6,
            "jul": 7, "juli": 7, "july": 7,
            "agu": 8, "agustus": 8, "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "okt": 10, "oct": 10, "oktober": 10, "october": 10,
            "nov": 11, "november": 11,
            "des": 12, "dec": 12, "desember": 12, "december": 12,
        }
        lowered = (text or "").lower()
        for key, month in month_map.items():
            if re.search(rf"\b{re.escape(key)}\b", lowered):
                return month
        return None

    @staticmethod
    def _extract_day(text: str) -> Optional[int]:
        if not text:
            return None
        match = re.search(r"\b([0-2]?\d|3[01])\b", text)
        if not match:
            return None
        day = int(match.group(1))
        if 1 <= day <= 31:
            return day
        return None

    @staticmethod
    def _published_sort_key(item: Dict[str, Any]) -> int:
        merged = " ".join([
            str(item.get("date", "")),
            str(item.get("title", "")),
            str(item.get("snippet", "")),
        ])
        year = Researcher._extract_year(merged) or 0
        month = Researcher._extract_month(merged) or 1
        day = Researcher._extract_day(str(item.get("date", ""))) or 1
        return (year * 10_000) + (month * 100) + day

    @staticmethod
    def _sort_by_recency(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(items or [], key=Researcher._published_sort_key, reverse=True)

    @staticmethod
    def _filter_recent_entity_results(
        items: List[Dict[str, Any]],
        entity_name: str = "",
        max_age_years: int = 2,
        strict_entity: bool = False
    ) -> List[Dict[str, Any]]:
        filtered = []
        for item in (items or []):
            if not Researcher._is_recent(item, max_age_years=max_age_years):
                continue
            if entity_name and not Researcher._is_entity_match(item, entity_name, strict=strict_entity):
                continue
            filtered.append(item)
        return Researcher._sort_by_recency(filtered)

    @staticmethod
    def _is_regulatory_domain(link: str) -> bool:
        domain = Researcher._source_name(link)
        trusted_suffixes = ("go.id", "iso.org", "ietf.org", "iec.ch", "nist.gov")
        return any(domain.endswith(sfx) for sfx in trusted_suffixes)
    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        if not text:
            return None
        years = re.findall(r'\b(20\d{2})\b', text)
        if not years:
            return None
        return max(int(y) for y in years)

    @staticmethod
    def _is_recent(item: Dict[str, Any], max_age_years: int = 2) -> bool:
        merged = " ".join([
            str(item.get('date', '')),
            str(item.get('snippet', '')),
            str(item.get('title', '')),
        ])
        year = Researcher._extract_year(merged)
        if year is None:
            return True
        return year >= (datetime.now().year - max_age_years)

    @staticmethod
    def _source_name(link: str) -> str:
        try:
            host = urlparse(link or "").netloc.lower().strip()
            host = host.replace("www.", "")
            return host or "sumber daring"
        except Exception:
            return "sumber daring"

    @staticmethod
    def _citation_year(item: Dict[str, Any]) -> str:
        merged = " ".join([
            str(item.get('date', '')),
            str(item.get('snippet', '')),
            str(item.get('title', '')),
        ])
        year = Researcher._extract_year(merged)
        return str(year) if year else "n.d."

    @staticmethod
    def _format_evidence(items: List[Dict[str, Any]], label: str, fallback: str) -> str:
        if not items:
            return f"{fallback} (sumber daring, n.d.)"
        lines = []
        for i, item in enumerate(items, start=1):
            title = item.get('title', 'Sumber tanpa judul')
            snippet = (item.get('snippet', '') or '').strip()
            link = item.get('link', '-')
            if not snippet:
                continue
            source_name = Researcher._source_name(link)
            citation = f"({source_name}, {Researcher._citation_year(item)})"
            lines.append(
                f"Sumber eksternal {i}: fakta={snippet} | sumber={title} | url={link} | sitasi_apa={citation}"
            )
        return "\n".join(lines) if lines else f"{fallback} (sumber daring, n.d.)"


    @staticmethod
    @lru_cache(maxsize=128)
    def get_entity_profile(entity_name: str) -> str:
        current_year = datetime.now().year
        strict_entity = Researcher._normalize_text(entity_name) == Researcher._normalize_text(WRITER_FIRM_NAME)
        query = (
            f'"{entity_name}" profil perusahaan OR "tentang kami" OR alamat OR kontak '
            f'{current_year} OR {current_year - 1}'
        )
        res = Researcher.search(query, limit=8, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(
            res,
            entity_name=entity_name,
            max_age_years=3,
            strict_entity=strict_entity
        )
        if not filtered and strict_entity:
            fallback_query = f'"{entity_name}" Yogyakarta pelatihan konsultasi kontak resmi'
            fallback_res = Researcher.search(fallback_query, limit=8)
            filtered = Researcher._filter_recent_entity_results(
                fallback_res,
                entity_name=entity_name,
                max_age_years=6,
                strict_entity=True
            )
        return Researcher._format_evidence(
            filtered[:4],
            label="OSINT_PROFILE",
            fallback=f"Data profil terbaru untuk {entity_name} terbatas; gunakan informasi umum yang terverifikasi saja."
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_latest_client_news(client_name: str) -> str:
        current_year = datetime.now().year
        prev_year = current_year - 1
        res = Researcher.search(
            f'"{client_name}" berita inovasi OR transformasi digital {current_year} OR {prev_year}',
            limit=8,
            recency_bucket="month"
        )
        filtered = Researcher._filter_recent_entity_results(
            res,
            entity_name=client_name,
            max_age_years=2,
            strict_entity=False
        )
        return Researcher._format_evidence(
            filtered[:4],
            label="OSINT_NEWS",
            fallback=f"Berita terbaru {client_name} tidak cukup kuat; jangan membuat klaim spesifik tanpa bukti."
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_client_track_record(client_name: str) -> str:
        current_year = datetime.now().year
        prev_year = current_year - 1
        res = Researcher.search(
            (
                f'"{client_name}" pencapaian OR kinerja OR ekspansi OR transformasi OR penghargaan '
                f'{current_year} OR {prev_year}'
            ),
            limit=10,
            recency_bucket="year"
        )
        filtered = Researcher._filter_recent_entity_results(
            res,
            entity_name=client_name,
            max_age_years=3,
            strict_entity=False
        )
        return Researcher._format_evidence(
            filtered[:5],
            label="OSINT_TRACK",
            fallback=f"Track record publik {client_name} terbatas; hindari klaim performa tanpa bukti."
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_client_ai_posture(client_name: str, ai_context: str = "") -> str:
        current_year = datetime.now().year
        prev_year = current_year - 1
        context_terms = " ".join(re.findall(r"[A-Za-z]{4,}", (ai_context or "").lower())[:8])
        query = (
            f'"{client_name}" AI OR artificial intelligence OR generative AI OR machine learning OR automation '
            f'OR data platform OR analytics {context_terms} {current_year} OR {prev_year}'
        )
        res = Researcher.search(query, limit=10, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(
            res,
            entity_name=client_name,
            max_age_years=3,
            strict_entity=False
        )
        return Researcher._format_evidence(
            filtered[:4],
            label="OSINT_AI",
            fallback=f"Data publik terkait AI posture {client_name} terbatas; perlakukan kesiapan adopsi sebagai area validasi awal, bukan fakta yang diasumsikan."
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_client_writer_collaboration(client_name: str, writer_firm_name: str = WRITER_FIRM_NAME) -> str:
        current_year = datetime.now().year
        strict_writer = Researcher._normalize_text(writer_firm_name) == Researcher._normalize_text(WRITER_FIRM_NAME)
        query = (
            f'"{writer_firm_name}" "{client_name}" kerja sama OR proyek OR pelatihan OR konsultasi '
            f'{current_year} OR {current_year - 1} OR {current_year - 2}'
        )
        res = Researcher.search(query, limit=10, recency_bucket="year")
        filtered: List[Dict[str, Any]] = []
        for item in Researcher._sort_by_recency(res):
            if not Researcher._is_recent(item, max_age_years=6):
                continue
            if not Researcher._is_entity_match(item, client_name, strict=False):
                continue
            if not Researcher._is_entity_match(item, writer_firm_name, strict=strict_writer):
                continue
            filtered.append(item)

        return Researcher._format_evidence(
            filtered[:4],
            label="OSINT_COLLAB",
            fallback=(
                f"Belum ditemukan bukti publik yang cukup kuat terkait kolaborasi {writer_firm_name} "
                f"dengan {client_name}; jangan menyatakan pernah bekerja sama tanpa verifikasi."
            )
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_regulatory_data(regulations_string: str) -> str:
        if not regulations_string:
            return "Tidak ada regulasi spesifik dari input user."

        query = f'Ringkasan implementasi standar {regulations_string.replace(",", " OR ")} site:.go.id OR site:iso.org'
        res = Researcher.search(query, limit=8, recency_bucket="year")
        recent = [i for i in Researcher._sort_by_recency(res) if Researcher._is_recent(i, max_age_years=5)]
        trusted = [i for i in recent if Researcher._is_regulatory_domain(str(i.get("link", "")))]
        filtered = trusted if trusted else recent
        return Researcher._format_evidence(
            filtered[:5],
            label="OSINT_REG",
            fallback=f"Data regulasi untuk {regulations_string} terbatas; nyatakan asumsi dan batasan data secara eksplisit."
        )


# Budget estimator from public financial context.
class FinancialAnalyzer:
    def __init__(self, ollama_client: Client):
        self.ollama = ollama_client

    DEFAULT_DURATION_MONTHS = {
        "diagnostic": 2.0,
        "strategic": 3.0,
        "transformation": 6.0,
        "implementation": 5.0,
    }

    BASE_MONTHLY_DELIVERY_RATE = {
        "diagnostic": 45_000_000,
        "strategic": 60_000_000,
        "transformation": 85_000_000,
        "implementation": 95_000_000,
    }

    @staticmethod
    def _has_signal(text: str, candidate: str) -> bool:
        value = re.sub(r"\s+", " ", str(candidate or "").strip().lower())
        if not value:
            return False
        pattern = re.escape(value)
        if " " not in value and value.isalpha():
            pattern = rf"\b{pattern}\b"
        return bool(re.search(pattern, (text or "").lower()))

    @classmethod
    def _ai_scope_summary(cls, *values: Any) -> Dict[str, Any]:
        combined = re.sub(r"\s+", " ", " ".join(str(value or "") for value in values)).strip().lower()
        strong_hits = [
            token for token in (SPIRIT_OF_AI_RULES.get("strong_trigger_keywords") or [])
            if cls._has_signal(combined, token)
        ]
        support_hits = [
            token for token in (SPIRIT_OF_AI_RULES.get("supporting_signals") or [])
            if token not in strong_hits and cls._has_signal(combined, token)
        ]
        enabled = bool(strong_hits) or len(support_hits) >= 2
        return {
            "enabled": enabled,
            "strong_hits": strong_hits[:8],
            "support_hits": support_hits[:10],
        }

    @classmethod
    def _ai_pricing_profile(
        cls,
        project_goal: str,
        objective: str,
        notes: str,
        frameworks: str,
    ) -> Dict[str, Any]:
        combined = " ".join([project_goal or "", objective or "", notes or "", frameworks or ""]).lower()
        ai_scope = cls._ai_scope_summary(project_goal, objective, notes, frameworks)
        if not ai_scope["enabled"]:
            return {
                "enabled": False,
                "level": "terkendali",
                "multiplier": 1.0,
                "drivers": [],
                "driver_labels": [],
            }

        driver_config = SPIRIT_OF_AI_RULES.get("pricing_driver_terms") or {}
        driver_labels = {
            "data_readiness": "kesiapan data/model",
            "model_uncertainty": "validasi solusi/model",
            "architecture_constraints": "kendala arsitektur dan keamanan",
            "governance_overhead": "governance dan compliance",
            "change_enablement": "enablement dan adopsi organisasi",
        }
        driver_weights = {
            "data_readiness": 0.10,
            "model_uncertainty": 0.10,
            "architecture_constraints": 0.08,
            "governance_overhead": 0.10,
            "change_enablement": 0.09,
        }

        multiplier = 1.0
        active_drivers: List[str] = []
        for driver, terms in driver_config.items():
            if any(cls._has_signal(combined, term) for term in (terms or [])):
                multiplier += driver_weights.get(driver, 0.06)
                active_drivers.append(driver)

        if not active_drivers:
            active_drivers = ["data_readiness", "governance_overhead", "change_enablement"]
            multiplier += 0.16

        multiplier = max(1.0, min(1.8, multiplier))
        if multiplier >= 1.45:
            level = "tinggi"
        elif multiplier >= 1.18:
            level = "menengah"
        else:
            level = "terkendali"

        return {
            "enabled": True,
            "level": level,
            "multiplier": multiplier,
            "drivers": active_drivers,
            "driver_labels": [driver_labels.get(driver, driver) for driver in active_drivers],
        }

    @staticmethod
    def _format_idr(amount: int) -> str:
        amount = max(0, int(amount))
        return "Rp " + f"{amount:,}".replace(",", ".")

    @staticmethod
    def _parse_number(raw: str) -> Optional[float]:
        value = (raw or "").strip()
        if not value:
            return None
        if "." in value and "," in value:
            value = value.replace(".", "").replace(",", ".")
        elif "," in value and "." not in value:
            value = value.replace(",", ".")
        elif "." in value and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
            value = value.replace(".", "")
        try:
            return float(value)
        except ValueError:
            return None

    @classmethod
    def _extract_financial_values(cls, text: str) -> List[int]:
        if not text:
            return []
        pattern = re.compile(
            r"(?i)(?P<rp>rp\.?\s*)?(?P<num>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?P<unit>triliun|miliar|juta|ribu)?"
        )
        multiplier = {
            "triliun": 1_000_000_000_000,
            "miliar": 1_000_000_000,
            "juta": 1_000_000,
            "ribu": 1_000,
        }
        values: List[int] = []
        for match in pattern.finditer(text):
            has_rp = bool(match.group("rp"))
            unit = (match.group("unit") or "").lower()
            base = cls._parse_number(match.group("num") or "")
            if base is None:
                continue
            if not has_rp and not unit:
                continue
            amount = int(base * multiplier.get(unit, 1))
            if amount <= 0:
                continue
            if amount < 1_000_000 and not unit:
                continue
            values.append(amount)
        return values

    @classmethod
    def _duration_to_months(cls, timeline: str) -> Optional[float]:
        text = (timeline or "").lower().strip()
        if not text:
            return None

        patterns = [
            (r"(\d+(?:[.,]\d+)?)\s*(tahun|thn|year|years)", 12.0),
            (r"(\d+(?:[.,]\d+)?)\s*(bulan|bln|month|months)", 1.0),
            (r"(\d+(?:[.,]\d+)?)\s*(minggu|week|weeks)", 1.0 / 4.345),
            (r"(\d+(?:[.,]\d+)?)\s*(hari|day|days)", 1.0 / 30.0),
        ]

        months = 0.0
        found = False
        for pattern, factor in patterns:
            for match in re.finditer(pattern, text):
                value = cls._parse_number(match.group(1))
                if value is None:
                    continue
                months += value * factor
                found = True

        if found:
            return max(months, 0.5)

        single_number = re.search(r"(\d+(?:[.,]\d+)?)", text)
        if single_number:
            value = cls._parse_number(single_number.group(1))
            if value is not None:
                return max(value, 0.5)
        return None

    @classmethod
    def _duration_multiplier(cls, timeline: str) -> float:
        months = cls._duration_to_months(timeline)
        if months is None:
            return 1.0
        raw = (months / 6.0) ** 0.45
        return max(0.75, min(2.4, raw))

    @classmethod
    def _default_duration_months(cls, project_type: str) -> float:
        return cls.DEFAULT_DURATION_MONTHS.get((project_type or "").strip().lower(), 4.0)

    @classmethod
    def _duration_months_or_default(cls, timeline: str, project_type: str) -> float:
        return cls._duration_to_months(timeline) or cls._default_duration_months(project_type)

    @classmethod
    def _complexity_profile(
        cls,
        project_goal: str,
        objective: str,
        notes: str,
        frameworks: str
    ) -> Dict[str, Any]:
        combined = " ".join([project_goal or "", objective or "", notes or "", frameworks or ""]).lower()
        if not combined.strip():
            return {"level": "moderat", "multiplier": 1.0, "signal_count": 0}

        high_complexity = {
            "multi": 0.08,
            "integrasi": 0.10,
            "integration": 0.10,
            "core banking": 0.14,
            "migrasi": 0.10,
            "migration": 0.10,
            "nasional": 0.06,
            "enterprise": 0.06,
            "regulasi": 0.08,
            "regulatory": 0.08,
            "compliance": 0.08,
            "24/7": 0.06,
            "high availability": 0.08,
            "multi-site": 0.06,
            "multisite": 0.06,
            "security": 0.06,
            "cyber": 0.06,
            "data governance": 0.06,
        }
        medium_complexity = {
            "dashboard": 0.03,
            "governance": 0.04,
            "kpi": 0.03,
            "workflow": 0.03,
            "automation": 0.04,
            "cloud": 0.04,
            "api": 0.04,
            "audit": 0.04,
            "change management": 0.05,
            "training": 0.03,
            "adoption": 0.03,
            "rollout": 0.04,
            "hypercare": 0.04,
        }

        multiplier = 1.0
        signal_count = 0
        for token, boost in high_complexity.items():
            if token in combined:
                multiplier += boost
                signal_count += 1
        for token, boost in medium_complexity.items():
            if token in combined:
                multiplier += boost
                signal_count += 1

        framework_tokens = [part.strip() for part in re.split(r"[,;/]| dan ", frameworks or "") if part.strip()]
        if framework_tokens:
            multiplier += min(0.12, max(0, len(framework_tokens) - 1) * 0.03)

        ai_profile = cls._ai_pricing_profile(project_goal, objective, notes, frameworks)
        if ai_profile["enabled"]:
            multiplier *= ai_profile["multiplier"]
            signal_count += len(ai_profile["drivers"])

        multiplier = max(0.9, min(1.9, multiplier))
        if multiplier >= 1.45:
            level = "tinggi"
        elif multiplier >= 1.15:
            level = "menengah"
        else:
            level = "terkendali"
        return {"level": level, "multiplier": multiplier, "signal_count": signal_count}

    @classmethod
    def _scope_multiplier(
        cls,
        project_type: str,
        service_type: str,
        project_goal: str,
        objective: str,
        notes: str,
        frameworks: str
    ) -> float:
        project_weight = {
            "diagnostic": 0.90,
            "strategic": 1.00,
            "transformation": 1.22,
            "implementation": 1.35,
        }
        service_weight = {
            "training": 0.85,
            "konsultan": 1.10,
            "training dan konsultan": 1.20,
        }

        base = project_weight.get((project_type or "").strip().lower(), 1.0)
        base *= service_weight.get((service_type or "").strip().lower(), 1.0)

        complexity_profile = cls._complexity_profile(project_goal, objective, notes, frameworks)
        complexity_boost = max(0.0, complexity_profile["multiplier"] - 1.0)

        need_items = [part.strip() for part in re.split(r"[,+;/]| dan ", (project_goal or "").lower()) if part.strip()]
        breadth_boost = min(0.25, max(0, len(need_items) - 1) * 0.05)

        scope_multiplier = base * (1.0 + complexity_boost + breadth_boost)
        return max(0.75, min(2.4, scope_multiplier))

    @classmethod
    def _project_effort_baseline(
        cls,
        timeline: str,
        project_type: str,
        service_type: str,
        project_goal: str,
        objective: str,
        notes: str,
        frameworks: str,
    ) -> Dict[str, Any]:
        project_key = (project_type or "").strip().lower()
        service_key = (service_type or "").strip().lower()
        months = cls._duration_months_or_default(timeline, project_type)
        monthly_rate = cls.BASE_MONTHLY_DELIVERY_RATE.get(project_key, 70_000_000)
        service_multiplier = {
            "training": 0.80,
            "konsultan": 1.00,
            "training dan konsultan": 1.12,
        }.get(service_key, 1.0)
        complexity_profile = cls._complexity_profile(project_goal, objective, notes, frameworks)
        ai_pricing = cls._ai_pricing_profile(project_goal, objective, notes, frameworks)

        breadth_inputs = [objective, notes, project_goal, frameworks]
        active_dimensions = sum(1 for item in breadth_inputs if str(item or "").strip())
        breadth_multiplier = 1.0 + min(0.18, max(0, active_dimensions - 2) * 0.04)

        effort_base = monthly_rate * months * service_multiplier * complexity_profile["multiplier"] * breadth_multiplier
        if ai_pricing["enabled"]:
            effort_base *= max(1.0, ai_pricing["multiplier"] * 0.92)
        effort_base = int(max(120_000_000, min(9_000_000_000, effort_base)))
        return {
            "months": months,
            "monthly_rate": int(monthly_rate),
            "complexity": complexity_profile,
            "ai_pricing": ai_pricing,
            "breadth_multiplier": breadth_multiplier,
            "effort_base": effort_base,
        }

    @staticmethod
    def _bounded_calibration(value: int, anchor: int, lower_ratio: float, upper_ratio: float) -> int:
        if value <= 0 or anchor <= 0:
            return max(value, anchor)
        lower = int(anchor * lower_ratio)
        upper = int(anchor * upper_ratio)
        return max(lower, min(upper, value))

    @staticmethod
    def _compact_keywords(text: str, max_terms: int = 7) -> str:
        stopwords = {
            "dan", "yang", "untuk", "dengan", "dari", "pada", "agar", "atau",
            "the", "and", "for", "with", "from", "into", "this", "that",
        }
        tokens = re.findall(r"[A-Za-z0-9]{4,}", (text or "").lower())
        seen = set()
        out = []
        for token in tokens:
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= max_terms:
                break
        return " ".join(out)

    @classmethod
    def _dynamic_budget_from_osint(
        cls,
        client_name: str,
        finance_snippets: List[Dict[str, Any]],
        benchmark_snippets: List[Dict[str, Any]],
        timeline: str = "",
        project_type: str = "",
        service_type: str = "",
        project_goal: str = "",
        objective: str = "",
        notes: str = "",
        frameworks: str = "",
    ) -> Dict[str, Any]:
        finance_text = " ".join(
            [str(item.get("title", "")) + " " + str(item.get("snippet", "")) for item in (finance_snippets or [])]
        )
        benchmark_text = " ".join(
            [str(item.get("title", "")) + " " + str(item.get("snippet", "")) for item in (benchmark_snippets or [])]
        )

        finance_values = sorted(cls._extract_financial_values(finance_text))
        benchmark_values = sorted(cls._extract_financial_values(benchmark_text))
        effort_profile = cls._project_effort_baseline(
            timeline=timeline,
            project_type=project_type,
            service_type=service_type,
            project_goal=project_goal,
            objective=objective,
            notes=notes,
            frameworks=frameworks,
        )
        effort_base = effort_profile["effort_base"]

        if finance_values:
            finance_median = finance_values[len(finance_values) // 2]
            financial_base = int(max(120_000_000, min(6_000_000_000, finance_median * 0.0015)))
            financial_base = cls._bounded_calibration(financial_base, effort_base, 0.70, 1.80)
        else:
            finance_median = None
            financial_base = effort_base

        if benchmark_values:
            benchmark_median = benchmark_values[len(benchmark_values) // 2]
            market_base = int(max(80_000_000, min(6_000_000_000, benchmark_median)))
            market_base = cls._bounded_calibration(market_base, effort_base, 0.75, 1.60)
        else:
            benchmark_median = None
            market_base = effort_base

        if finance_median and benchmark_median:
            base_price = int((effort_base * 0.65) + (market_base * 0.25) + (financial_base * 0.10))
        elif benchmark_median:
            base_price = int((effort_base * 0.78) + (market_base * 0.22))
        elif finance_median:
            base_price = int((effort_base * 0.88) + (financial_base * 0.12))
        else:
            base_price = effort_base

        calibration_cap = max(effort_base, int(financial_base * 1.75)) if finance_median else effort_base
        adjusted_base = int(max(120_000_000, min(9_000_000_000, min(base_price, calibration_cap))))

        basic = int(adjusted_base * 0.72)
        standard = max(basic + 40_000_000, adjusted_base)
        enterprise = max(standard + 80_000_000, int(adjusted_base * 1.65))

        months = effort_profile["months"]
        duration_note = f"{months:.1f} bulan" if months else "durasi belum spesifik"
        complexity_level = effort_profile["complexity"]["level"]
        ai_pricing = effort_profile.get("ai_pricing", {}) or {}
        if ai_pricing.get("enabled"):
            driver_labels = list(ai_pricing.get("driver_labels", []) or [
                "kesiapan data/model",
                "governance",
                "adopsi organisasi",
            ])
            aliases = {
                "governance dan compliance": "governance",
                "governance": "governance",
                "kesiapan data/model": "kesiapan data/model",
                "adopsi organisasi": "adopsi organisasi",
            }
            normalized_labels = {aliases.get(label, label) for label in driver_labels}
            for default_label in ["kesiapan data/model", "governance", "adopsi organisasi"]:
                if len(driver_labels) >= 3:
                    break
                if default_label not in normalized_labels:
                    driver_labels.append(default_label)
                    normalized_labels.add(default_label)
            analysis = (
                f"Estimasi untuk {client_name} terutama dihitung dari effort delivery proyek "
                f"({duration_note}, kompleksitas {complexity_level}) dengan penekanan pada "
                f"{', '.join(driver_labels[:3])}, lalu hanya dikalibrasi ringan menggunakan benchmark dan sinyal publik yang tersedia."
            )
        elif finance_median:
            analysis = (
                f"Estimasi untuk {client_name} terutama dihitung dari effort delivery proyek "
                f"({duration_note}, kompleksitas {complexity_level}) dan hanya dikalibrasi ringan "
                f"dengan sinyal finansial publik ({cls._format_idr(finance_median)})."
            )
        else:
            analysis = (
                f"Data finansial publik {client_name} terbatas; estimasi terutama dihitung dari effort delivery proyek "
                f"({duration_note}, kompleksitas {complexity_level}) lalu dicek dengan benchmark OSINT yang tersedia."
            )

        return {
            "analysis": analysis,
            "options": [
                {"tier": "Basic", "price": cls._format_idr(basic)},
                {"tier": "Standard", "price": cls._format_idr(standard)},
                {"tier": "Enterprise", "price": cls._format_idr(enterprise)},
            ],
        }

    @staticmethod
    def _is_valid_budget_payload(payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        options = payload.get("options")
        analysis = payload.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            return False
        if not isinstance(options, list) or len(options) < 3:
            return False
        for item in options[:3]:
            if not isinstance(item, dict):
                return False
            tier = str(item.get("tier", "")).strip()
            price = str(item.get("price", "")).strip()
            if not tier or not price:
                return False
            if "<" in price or ">" in price:
                return False
            if not re.search(r"\d", price):
                return False
        return True

    @staticmethod
    def _merge_with_context_adjustment(
        model_payload: Dict[str, Any],
        context_payload: Dict[str, Any],
        context_sensitive: bool,
        prefer_context_options: bool = True
    ) -> Dict[str, Any]:
        if not context_sensitive:
            return model_payload
        analysis = str(context_payload.get("analysis", "")).strip() or str(model_payload.get("analysis", "")).strip()
        if not prefer_context_options:
            merged = dict(model_payload)
            merged["analysis"] = analysis
            return merged
        return {
            "analysis": analysis,
            "options": context_payload.get("options", []),
        }

    def suggest_budget(
        self,
        client_name: str,
        timeline: str = "",
        project_type: str = "",
        service_type: str = "",
        project_goal: str = "",
        objective: str = "",
        notes: str = "",
        frameworks: str = "",
        commercial_context: str = "",
        pricing_mode: str = "demo",
    ) -> Dict[str, Any]:
        year = datetime.now().year
        ai_scope = self._ai_scope_summary(project_goal, objective, notes, frameworks, project_type, service_type)
        finance_results = Researcher.search(
            f'"{client_name}" laporan keuangan OR pendapatan OR pendanaan OR aset {year-2} OR {year-1} OR {year}',
            limit=10,
            recency_bucket="year"
        )
        finance_snippets = Researcher._filter_recent_entity_results(
            finance_results,
            entity_name=client_name,
            max_age_years=3,
            strict_entity=False
        )
        keyword_context = self._compact_keywords(
            f"{objective} {notes} {project_goal} {frameworks} {project_type} {service_type}"
        )
        ai_benchmark_hint = "adopsi AI governance pilot rollout readiness" if ai_scope["enabled"] else ""
        benchmark_query = (
            f'estimasi biaya proyek {project_type or "IT"} {service_type} {timeline} '
            f'Indonesia {keyword_context} {ai_benchmark_hint}'
        )
        benchmark_results = Researcher.search(benchmark_query, limit=10, recency_bucket="year")
        benchmark_snippets = [
            item for item in Researcher._sort_by_recency(benchmark_results)
            if Researcher._is_recent(item, max_age_years=3)
        ][:8]

        context_finance = "\n".join([item.get('snippet', '') for item in finance_snippets]) if finance_snippets else "-"
        context_benchmark = "\n".join([item.get('snippet', '') for item in benchmark_snippets]) if benchmark_snippets else "-"
        dynamic_estimate = self._dynamic_budget_from_osint(
            client_name=client_name,
            finance_snippets=finance_snippets,
            benchmark_snippets=benchmark_snippets,
            timeline=timeline,
            project_type=project_type,
            service_type=service_type,
            project_goal=project_goal,
            objective=objective,
            notes=notes,
            frameworks=frameworks,
        )
        staged_pricing = pricing_mode == "staged"
        commercial_note = (
            f"Baseline commercial rules internal:\n{commercial_context}\n"
            "Jadikan baseline internal ini sebagai acuan utama penentuan harga. "
            "Data OSINT hanya dipakai sebagai sinyal pendukung.\n"
            if staged_pricing and commercial_context.strip()
            else ""
        )

        prompt = f"""
        Menganalisa kekuatan finansial perusahaan: {client_name}.
        Data finansial OSINT:
        {context_finance}

        Data benchmark biaya OSINT:
        {context_benchmark}

        Konteks proyek:
        - Durasi: {timeline or '-'}
        - Jenis Proyek: {project_type or '-'}
        - Jenis Proposal/Layanan: {service_type or '-'}
        - Klasifikasi Kebutuhan: {project_goal or '-'}
        - Objective: {objective or '-'}
        - Pain Points: {notes or '-'}
        - Framework: {frameworks or '-'}
        {commercial_note}

        Berdasarkan data di atas, estimasikan kapasitas finansial mereka dan berikan 3 opsi estimasi budget proyek TI/Konsultasi.
        PRIORITAS PENENTUAN HARGA:
        1. Durasi/length proyek
        2. Tingkat kesulitan, kompleksitas integrasi, regulasi, dan perubahan
        2a. Jika konteks proyek terkait AI/adopsi AI, perhitungkan pula kesiapan data/model, governance, arsitektur, dan change enablement
        3. Jenis proyek dan jenis layanan
        4. Benchmark OSINT
        5. Sinyal finansial publik klien hanya sebagai kalibrasi, bukan faktor dominan

        FORMAT WAJIB JSON murni tanpa markdown, tanpa teks tambahan:
        {{
            "analysis": "Ringkasan 1 kalimat kekuatan finansial berdasarkan data (atau sebutkan estimasi jika data terbatas).",
            "options": [
                {{"tier": "Basic", "price": "Rp <angka>"}},
                {{"tier": "Standard", "price": "Rp <angka>"}},
                {{"tier": "Enterprise", "price": "Rp <angka>"}}
            ]
        }}
        Pastikan angka terutama mempertimbangkan durasi, tingkat kesulitan, dan scope proyek; jangan bertumpu hanya pada pendapatan tahunan klien.
        """
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'system', 'content': 'You output strictly valid JSON.'}, {'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            raw_text = res['message']['content']
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            parsed = {}
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(raw_text)
            if self._is_valid_budget_payload(parsed):
                context_sensitive = any(
                    [timeline.strip(), project_type.strip(), service_type.strip(), project_goal.strip(), objective.strip(), notes.strip(), frameworks.strip()]
                )
                return self._merge_with_context_adjustment(
                    parsed,
                    dynamic_estimate,
                    context_sensitive=context_sensitive,
                    prefer_context_options=not staged_pricing
                )
            return dynamic_estimate
        except Exception as e:
            logger.error(f"Financial Analyzer Error: {e}")
            return dynamic_estimate

class LogoManager:
    @staticmethod
    def _create_fallback_logo(client_name: str) -> io.BytesIO:
        initials = "".join([w[0] for w in re.findall(r"[A-Za-z0-9]+", client_name)[:3]]).upper() or "CL"
        canvas = Image.new('RGB', (320, 320), color=(236, 242, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((16, 16, 304, 304), radius=36, outline=(37, 99, 235), width=8)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 110)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), initials, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(initials, font=font)
        draw.text(((320 - w) / 2, (320 - h) / 2 - 6), initials, fill=(15, 23, 42), font=font)
        out = io.BytesIO()
        canvas.save(out, format='PNG')
        out.seek(0)
        return out

    @staticmethod
    def get_logo_and_color(client_name: str) -> Tuple[Optional[io.BytesIO], Tuple[int, int, int]]:
        if not Researcher._has_serper_key():
            return LogoManager._create_fallback_logo(client_name), DEFAULT_COLOR
        try:
            url = "https://google.serper.dev/images"
            payload = json.dumps({"q": f"{client_name} corporate logo png transparent", "num": 3})
            headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
            res = requests.post(url, headers=headers, data=payload, timeout=8).json()
            
            if 'images' in res and res['images']:
                for item in res['images']:
                    try:
                        img_resp = requests.get(item['imageUrl'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                        if img_resp.status_code == 200:
                            stream = io.BytesIO(img_resp.content)
                            img = Image.open(stream)
                            
                            # Normalize to PNG so python-docx can embed it reliably.
                            if img.mode in ("RGBA", "LA", "P"):
                                normalized = Image.new("RGBA", img.size, (255, 255, 255, 0))
                                normalized.paste(img, (0, 0), img if img.mode in ("RGBA", "LA") else None)
                                img = normalized.convert('RGB')
                            else:
                                img = img.convert('RGB')

                            img.thumbnail((150, 150))
                            dom_color = list(map(int, ImageStat.Stat(img).mean[:3]))

                            luminance = 0.299 * dom_color[0] + 0.587 * dom_color[1] + 0.114 * dom_color[2]
                            if luminance > 120:
                                factor = 120 / luminance
                                dom_color = [max(0, min(255, int(c * factor))) for c in dom_color]

                            # Use normalized bytes, not source bytes, to avoid format errors.
                            png_stream = io.BytesIO()
                            img.save(png_stream, format='PNG')
                            png_stream.seek(0)
                            return png_stream, tuple(dom_color)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Logo Retrieval Error: {e}")
        return LogoManager._create_fallback_logo(client_name), DEFAULT_COLOR


# Document rendering utilities.
class StyleEngine:
    @staticmethod
    def apply_document_styles(doc: Document) -> None:
        style = doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(11)
        pf = style.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.15
        pf.space_after = Pt(8) 
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for section in doc.sections:
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)

class ChartEngine:
    @staticmethod
    def _to_matplotlib_rgb(theme_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return tuple(c/255 for c in theme_color)

    @staticmethod
    def create_gantt_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = data_str.split('|')
            if len(parts) == 3:
                title_str, unit_str, raw_data = parts[0].strip(), parts[1].strip(), parts[2].strip()
            else:
                title_str, unit_str, raw_data = "Timeline", "Waktu", data_str
            tasks = []
            for p in raw_data.split(';'):
                t_parts = p.split(',')
                if len(t_parts) >= 3:
                    tasks.append({"task": t_parts[0].strip(), "start": float(re.sub(r'[^\d.]', '', t_parts[1])), "dur": float(re.sub(r'[^\d.]', '', t_parts[2]))})
            if not tasks: return None
            tasks = tasks[::-1] 
            
            fig, ax = plt.subplots(figsize=(8.5, max(4, len(tasks)*0.8)))
            for i, task in enumerate(tasks):
                rect = patches.FancyBboxPatch((task['start'], i-0.3), task['dur'], 0.6, boxstyle="round,pad=0.02", ec="#ffffff", fc=ChartEngine._to_matplotlib_rgb(theme_color), alpha=0.9, lw=1.5)
                ax.add_patch(rect)
                ax.text(task['start'] + (task['dur'] / 2), i, f"{task['dur']:g} {unit_str}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
            names = [t['task'] for t in tasks]
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_title(title_str, fontsize=13, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            max_x = max([t['start'] + t['dur'] for t in tasks])
            ax.set_xlim(0, max_x + (max_x * 0.1))
            ax.set_ylim(-0.6, len(names)-0.4)
            
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

class DocumentBuilder:
    @staticmethod
    def _append_text_run(paragraph, text: str, bold: bool = False, italic: bool = False) -> None:
        cleaned = re.sub(r'\s+', ' ', text or '').strip()
        if not cleaned:
            return

        # Keep spacing between text runs stable so inline formatting stays readable.
        if paragraph.runs:
            last_text = paragraph.runs[-1].text or ""
            if last_text and not last_text.endswith((" ", "\t", "\n", "(", "[", "/")) and not cleaned.startswith((".", ",", ";", ":", ")", "]", "%")):
                cleaned = " " + cleaned

        run = paragraph.add_run(cleaned)
        run.bold = bold
        run.italic = italic

    @staticmethod
    def _style_num_id(doc: Document, style_name: str) -> Optional[int]:
        try:
            style = doc.styles[style_name]
        except KeyError:
            return None
        ppr = style.element.pPr
        if ppr is None or ppr.numPr is None or ppr.numPr.numId is None:
            return None
        try:
            return int(ppr.numPr.numId.val)
        except Exception:
            return None

    @staticmethod
    def _create_list_num_id(doc: Document, style_name: str) -> Optional[int]:
        style_num_id = DocumentBuilder._style_num_id(doc, style_name)
        if style_num_id is None:
            return None

        numbering = doc.part.numbering_part.numbering_definitions._numbering
        abstract_num_id = None
        for num in numbering.findall(qn('w:num')):
            raw_num_id = num.get(qn('w:numId'))
            try:
                current_num_id = int(raw_num_id) if raw_num_id is not None else None
            except (TypeError, ValueError):
                current_num_id = None
            if current_num_id != style_num_id:
                continue
            abstract = num.find(qn('w:abstractNumId'))
            if abstract is None:
                continue
            raw_abstract = abstract.get(qn('w:val'))
            try:
                abstract_num_id = int(raw_abstract) if raw_abstract is not None else None
            except (TypeError, ValueError):
                abstract_num_id = None
            break

        if abstract_num_id is None:
            return None

        existing_num_ids: List[int] = []
        for num in numbering.findall(qn('w:num')):
            raw_num_id = num.get(qn('w:numId'))
            try:
                if raw_num_id is not None:
                    existing_num_ids.append(int(raw_num_id))
            except (TypeError, ValueError):
                continue
        next_num_id = (max(existing_num_ids) + 1) if existing_num_ids else (style_num_id + 1)

        num = OxmlElement('w:num')
        num.set(qn('w:numId'), str(next_num_id))
        abstract_ref = OxmlElement('w:abstractNumId')
        abstract_ref.set(qn('w:val'), str(abstract_num_id))
        num.append(abstract_ref)
        numbering.append(num)
        return next_num_id

    @staticmethod
    def _apply_list_num_id(paragraph, num_id: int, level: int = 0) -> None:
        p = paragraph._p
        ppr = p.get_or_add_pPr()
        num_pr = ppr.find(qn('w:numPr'))
        if num_pr is None:
            num_pr = OxmlElement('w:numPr')
            ppr.append(num_pr)

        ilvl = num_pr.find(qn('w:ilvl'))
        if ilvl is None:
            ilvl = OxmlElement('w:ilvl')
            num_pr.append(ilvl)
        ilvl.set(qn('w:val'), str(level))

        num_id_el = num_pr.find(qn('w:numId'))
        if num_id_el is None:
            num_id_el = OxmlElement('w:numId')
            num_pr.append(num_id_el)
        num_id_el.set(qn('w:val'), str(num_id))

    @staticmethod
    def parse_html_to_docx(doc: Document, html_content: str, theme_color: Tuple[int, int, int]) -> None:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup.children:
            if element.name is None: continue
            if element.name in ['h1', 'h2', 'h3']:
                level = int(element.name[1])
                p = doc.add_heading(element.get_text().strip(), level=level)
                if level == 1:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.color.rgb = RGBColor(*theme_color)
                    run.font.name = 'Arial'
                    run.bold = True
            elif element.name == 'p':
                p = doc.add_paragraph()
                DocumentBuilder._process_inline_html(p, element)
            elif element.name in ['ul', 'ol']:
                # Use native Word list formatting so alignment and wrapping stay consistent.
                direct_items = element.find_all('li', recursive=False)
                style_name = "List Number" if element.name == 'ol' else "List Bullet"
                list_num_id = DocumentBuilder._create_list_num_id(doc, style_name)
                for idx, li in enumerate(direct_items, start=1):
                    # Avoid orphan markers like "3." with no text.
                    if not li.get_text(" ", strip=True):
                        continue
                    use_manual_fallback = False
                    try:
                        p = doc.add_paragraph(style=style_name)
                    except KeyError:
                        p = doc.add_paragraph()
                        use_manual_fallback = True

                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(4)
                    if list_num_id is not None:
                        DocumentBuilder._apply_list_num_id(p, list_num_id, level=0)
                    elif use_manual_fallback:
                        marker = f"{idx}. " if element.name == 'ol' else "• "
                        p.add_run(marker).bold = True
                    DocumentBuilder._process_inline_html(p, li)
            elif element.name == 'table':
                rows = element.find_all('tr')
                if not rows: continue
                max_cols = max([len(r.find_all(['td', 'th'])) for r in rows])
                table = doc.add_table(rows=len(rows), cols=max_cols)
                table.style = 'Table Grid'
                for i, row in enumerate(rows):
                    cols = row.find_all(['td', 'th'])
                    for j, col in enumerate(cols):
                        if j < max_cols:
                            cell = table.cell(i, j)
                            cell._element.clear_content()
                            p = cell.add_paragraph()
                            DocumentBuilder._process_inline_html(p, col)

    @staticmethod
    def _process_inline_html(paragraph, element):
        for child in element.children:
            if child.name in ['strong', 'b']:
                DocumentBuilder._append_text_run(paragraph, child.get_text(" ", strip=True), bold=True)
            elif child.name in ['em', 'i']:
                DocumentBuilder._append_text_run(paragraph, child.get_text(" ", strip=True), italic=True)
            elif child.name == 'br':
                paragraph.add_run("\n")
            elif child.name is None:
                DocumentBuilder._append_text_run(paragraph, str(child))
            else:
                DocumentBuilder._process_inline_html(paragraph, child)

    @staticmethod
    def process_content(doc: Document, raw_text: str, theme_color: Tuple[int, int, int], chapter_title: str) -> None:
        clean_lines = []
        in_table = False
        for line in raw_text.split('\n'):
            line = line.strip()
            if line.startswith('[[GANTT:') and line.endswith(']]'):
                data = line.replace('[[GANTT:', '').replace(']]', '').strip()
                img = ChartEngine.create_gantt_chart(data, theme_color)
                if img: doc.add_paragraph().add_run().add_picture(img, width=Inches(6))
                continue
            if line.startswith('|'):
                if not in_table and clean_lines and clean_lines[-1] != "":
                    clean_lines.append("")
                in_table = True
            else:
                in_table = False
            clean_lines.append(line)
            
        html = markdown.markdown("\n".join(clean_lines), extensions=['tables', 'sane_lists'])
        DocumentBuilder.parse_html_to_docx(doc, html, theme_color)
