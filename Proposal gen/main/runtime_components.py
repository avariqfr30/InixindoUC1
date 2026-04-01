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
        base_profile["whatsapp"] = base_profile.get("whatsapp") or WRITER_FIRM_WHATSAPP
        base_profile["website"] = base_profile.get("website") or WRITER_FIRM_WEBSITE
        base_profile["legal_name"] = base_profile.get("legal_name") or WRITER_FIRM_LEGAL_NAME
        base_profile["operating_hours"] = base_profile.get("operating_hours") or WRITER_FIRM_OPERATING_HOURS
        base_profile["profile_summary"] = base_profile.get("profile_summary") or WRITER_FIRM_PROFILE_SUMMARY
        base_profile["credential_highlights"] = (
            base_profile.get("credential_highlights") or WRITER_FIRM_CREDENTIAL_HIGHLIGHTS
        )
        base_profile["official_source_urls"] = (
            base_profile.get("official_source_urls") or WRITER_FIRM_SOURCE_URLS
        )
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
        whatsapp_match = re.search(
            r"(?:whatsapp|wa)\s*[:\-]?\s*((?:\+62|62|0)\d[\d\-\s()]{7,}\d)",
            text or "",
            flags=re.IGNORECASE
        )
        return {
            "office_address": office_candidates[0] if office_candidates else "",
            "email": cls._extract_first(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text),
            "phone": cls._extract_first(r"(?:\+62|62|0)\d[\d\-\s()]{7,}\d", text),
            "whatsapp": whatsapp_match.group(1).strip() if whatsapp_match else "",
            "website": website.rstrip(".,;)"),
        }

    @classmethod
    def build_contact_lines(cls, firm_profile: Optional[Dict[str, Any]]) -> List[str]:
        profile = firm_profile or {}
        lines = []
        office_address = cls._clean_contact_value(str(profile.get("office_address") or ""))
        email = cls._clean_contact_value(str(profile.get("email") or ""))
        phone = cls._clean_contact_value(str(profile.get("phone") or ""))
        whatsapp = cls._clean_contact_value(str(profile.get("whatsapp") or ""))
        website = cls._clean_contact_value(str(profile.get("website") or ""))
        operating_hours = cls._clean_contact_value(str(profile.get("operating_hours") or ""))

        if office_address:
            lines.append(f"Alamat kantor: {office_address}")
        if email:
            lines.append(f"Email: {email}")
        if phone:
            lines.append(f"Telp: {phone}")
        if whatsapp and whatsapp != phone:
            lines.append(f"WhatsApp: {whatsapp}")
        if website:
            if not re.match(r"^https?://", website, flags=re.IGNORECASE):
                website = f"https://{website.lstrip('/')}"
            lines.append(f"Website: {website}")
        if operating_hours:
            lines.append(f"Jam operasional: {operating_hours}")
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
        whatsapp = cls._clean_contact_value(str(
            source.get("whatsapp")
            or source.get("wa")
            or parsed.get("whatsapp")
            or ""
        ))
        website = cls._clean_contact_value(str(
            source.get("website")
            or source.get("url")
            or parsed.get("website")
            or ""
        ))
        portfolio = cls._clean_contact_value(str(source.get("portfolio_highlights") or ""))
        legal_name = cls._clean_contact_value(str(
            source.get("legal_name")
            or source.get("company_legal_name")
            or WRITER_FIRM_LEGAL_NAME
        ))
        operating_hours = cls._clean_contact_value(str(
            source.get("operating_hours")
            or source.get("business_hours")
            or WRITER_FIRM_OPERATING_HOURS
        ))
        profile_summary = cls._clean_contact_value(str(
            source.get("profile_summary")
            or source.get("company_summary")
            or source.get("summary")
            or WRITER_FIRM_PROFILE_SUMMARY
        ))
        credential_highlights = cls._clean_contact_value(str(
            source.get("credential_highlights")
            or source.get("credentials")
            or source.get("capabilities")
            or WRITER_FIRM_CREDENTIAL_HIGHLIGHTS
        ))
        official_source_urls = source.get("official_source_urls") or WRITER_FIRM_SOURCE_URLS
        if isinstance(official_source_urls, str):
            official_source_urls = [item.strip() for item in official_source_urls.split(",") if item.strip()]
        elif isinstance(official_source_urls, list):
            official_source_urls = [str(item).strip() for item in official_source_urls if str(item).strip()]
        else:
            official_source_urls = list(WRITER_FIRM_SOURCE_URLS)
        contact_lines = cls.build_contact_lines(
            {
                "office_address": office_address,
                "email": email,
                "phone": phone,
                "whatsapp": whatsapp,
                "website": website,
                "operating_hours": operating_hours,
            }
        )

        return {
            "office_address": office_address,
            "email": email,
            "phone": phone,
            "whatsapp": whatsapp,
            "website": website,
            "legal_name": legal_name,
            "operating_hours": operating_hours,
            "profile_summary": profile_summary,
            "credential_highlights": credential_highlights,
            "official_source_urls": official_source_urls,
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
                "office_address": parsed.get("office_address", "") or WRITER_FIRM_OFFICE_ADDRESS,
                "email": parsed.get("email", "") or WRITER_FIRM_EMAIL,
                "phone": parsed.get("phone", "") or WRITER_FIRM_PHONE,
                "whatsapp": parsed.get("whatsapp", "") or WRITER_FIRM_WHATSAPP,
                "website": parsed.get("website", "") or WRITER_FIRM_WEBSITE,
                "legal_name": WRITER_FIRM_LEGAL_NAME,
                "operating_hours": WRITER_FIRM_OPERATING_HOURS,
                "profile_summary": WRITER_FIRM_PROFILE_SUMMARY,
                "credential_highlights": WRITER_FIRM_CREDENTIAL_HIGHLIGHTS,
                "official_source_urls": WRITER_FIRM_SOURCE_URLS,
                "portfolio_highlights": WRITER_FIRM_PORTFOLIO,
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
        self.last_refresh_error = ""
        try:
            self.refresh_data()
        except Exception as exc:
            self.last_refresh_error = str(exc)
            logger.warning(
                "Knowledge base startup refresh skipped because dependencies are not ready yet: %s",
                exc,
            )

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
            self.last_refresh_error = "Project data schema is missing required fields."
            return False

        try:
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
        except Exception as exc:
            self.last_refresh_error = str(exc)
            logger.warning(
                "Knowledge base vector sync is not ready yet. App will continue to boot, but semantic retrieval is degraded: %s",
                exc,
            )
            return False

        self.last_refresh_error = ""
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
            logger.debug(f"Serper unavailable | Query not executed: '{query[:50]}...'")
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
            results = response.json().get('organic', [])
            result_count = len(results)
            logger.debug(f"Serper search successful | query='{query[:60]}...' | results={result_count} | status_code={response.status_code}")
            return results
        except requests.RequestException as e:
            logger.warning(f"Serper API Error | query='{query[:60]}...' | error={str(e)[:100]}")
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
        # --------------------------------------------------------------
        # Filter out entries that are merely a date/timestamp followed by a
        # very short phrase (e.g., "02/7/2025 23:28. PT Aneka Tambang …").
        # These add noise without providing useful content.
        # --------------------------------------------------------------
        def _is_noise(item: str) -> bool:
            """Return True if *item* looks like a date‑only news snippet.

            The heuristic matches a leading date (dd/mm/yyyy, dd-mm-yy, or
            ISO‑style) optionally followed by a time.  If the remaining text
            after the date contains five words or fewer, the line is considered
            noise and dropped.
            """
            import re

            m = re.match(r"^\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})(?:\s+\d{1,2}:\d{2})?", item)
            if not m:
                return False
            remainder = item[m.end():].strip()
            if not remainder:
                return True
            return len(re.findall(r"\S+", remainder)) <= 5

        cleaned = [i for i in filtered if not _is_noise(i)]
        # Keep at least one entry so the fallback logic still works.
        final_items = cleaned if cleaned else filtered
        return Researcher._format_evidence(
            final_items[:4],
            label="OSINT_NEWS",
            # Suppress fallback text so missing news does not appear in the proposal.
            fallback=""
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

    # ========== FIRM OSINT METHODS ==========
    # Enhanced firm information gathering via OSINT for professional proposal closings
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_firm_certifications(firm_name: str) -> str:
        """Search for ISO, professional certifications, and credentials via Serper."""
        try:
            query = (
                f'"{firm_name}" ISO OR sertifikasi OR akreditasi OR certification '
                'OR qualified OR licensed OR registered'
            )
            res = Researcher.search(query, limit=10, recency_bucket="year")
            logger.debug(f"Serper | Firm certifications search | firm={firm_name} | results_found={len(res)}")
            filtered = Researcher._filter_recent_entity_results(
                res,
                entity_name=firm_name,
                max_age_years=5,
                strict_entity=False
            )
            
            if not filtered:
                return f"Kredensial perusahaan {firm_name} belum terverifikasi via sumber publik."
            
            certifications = []
            for item in filtered[:4]:
                title = item.get('title', '')
                snippet = (item.get('snippet', '') or '').strip()
                
                # Extract common certification patterns
                certs = re.findall(r'ISO[\s\-]?\d{4,5}|[A-Z]{2,4}\s+\d{3,5}', title + ' ' + snippet)
                for cert in set(certs):
                    if cert not in certifications:
                        certifications.append(cert)
            
            if certifications:
                return f"Sertifikasi: {', '.join(certifications[:6])}. Lihat website resmi untuk daftar lengkap kredensial."
            
            return f"Kredensial dan sertifikasi {firm_name} tersedia di saluran publik resmi perusahaan."
        except Exception as e:
            logger.warning(f"Serper | Error fetching firm certifications | firm={firm_name} | error={str(e)[:80]}")
            return f"Kredensial perusahaan {firm_name} belum terverifikasi via sumber publik."

    @staticmethod
    @lru_cache(maxsize=128)
    def get_firm_team_expertise(firm_name: str) -> str:
        """Search for team expertise, thought leaders, and key personnel via Serper."""
        try:
            query = (
                f'"{firm_name}" kepemimpinan OR tim OR expert OR consultant OR principal '
                'OR director OR founder OR expertise'
            )
            res = Researcher.search(query, limit=12, recency_bucket="year")
            logger.debug(f"Serper | Firm team expertise search | firm={firm_name} | results_found={len(res)}")
            filtered = Researcher._filter_recent_entity_results(
                res,
                entity_name=firm_name,
                max_age_years=4,
                strict_entity=False
            )
            
            if not filtered:
                return f"Tim ahli profesional {firm_name} berfokus pada delivery berkualitas tinggi dan inovasi berkelanjutan."
            
            expertise_areas = []
            snippets = []
            for item in filtered[:5]:
                snippet = (item.get('snippet', '') or '').strip()
                if snippet and len(snippet) > 50:
                    snippets.append(snippet[:120])
                    # Extract domain keywords
                    domains = re.findall(r'(transformasi|digital|konsultasi|teknologi|strategi|audit|keamanan|sistem)', 
                                       snippet, re.IGNORECASE)
                    expertise_areas.extend(set(domains))
            
            result = f"Tim profesional {firm_name} memiliki pengalaman terakumulasi di berbagai domain strategis."
            if expertise_areas:
                unique_areas = list(set(expertise_areas))[:6]
                result += f" Area keahlian utama: {', '.join(unique_areas).lower()}."
            
            return result
        except Exception as e:
            logger.warning(f"Serper | Error fetching firm team expertise | firm={firm_name} | error={str(e)[:80]}")
            return f"Tim ahli profesional {firm_name} berfokus pada delivery berkualitas tinggi dan inovasi berkelanjutan."

    @staticmethod
    @lru_cache(maxsize=128)
    def get_firm_portfolio_scale(firm_name: str) -> str:
        """Search for company scale, client portfolio, and project volume via Serper."""
        try:
            query = (
                f'"{firm_name}" klien OR portfolio OR proyek OR kontrak OR revenue '
                'OR skala OR ukuran OR enterprise OR tier-1'
            )
            res = Researcher.search(query, limit=12, recency_bucket="year")
            logger.debug(f"Serper | Firm portfolio/scale search | firm={firm_name} | results_found={len(res)}")
            filtered = Researcher._filter_recent_entity_results(
                res,
                entity_name=firm_name,
                max_age_years=3,
                strict_entity=False
            )
            
            if not filtered:
                return f"Portofolio {firm_name} mencakup berbagai klien enterprise dan organisasi mid-market."
            
            # Look for scale indicators
            scale_indicators = []
            for item in filtered[:6]:
                snippet = (item.get('snippet', '') or '').strip()
                
                # Search for numbers and scale terms
                numbers = re.findall(r'\d+\s*(?:klien|client|proyek|project|program|tahun|year|bulan|month)', snippet, re.IGNORECASE)
                scale_terms = re.findall(r'(enterprise|tier-?1|institutional|multinational|global|nasional|indonesia)', snippet, re.IGNORECASE)
                
                if numbers:
                    scale_indicators.extend(numbers)
                if scale_terms:
                    scale_indicators.extend(scale_terms)
            
            result = f"Pengalaman {firm_name} meliputi multipel proyek strategis di sektor kunci ekonomi nasional."
            if scale_indicators:
                unique_indicators = list(set(scale_indicators))[:4]
                result += f" Jangkauan: {', '.join(unique_indicators).lower()}."
            
            return result
        except Exception as e:
            logger.warning(f"Serper | Error fetching firm portfolio/scale | firm={firm_name} | error={str(e)[:80]}")
            return f"Portofolio {firm_name} mencakup berbagai klien enterprise dan organisasi mid-market."

    @staticmethod
    @lru_cache(maxsize=128)
    def get_firm_values_approach(firm_name: str) -> str:
        """Search for company mission, values, and working approach via Serper."""
        try:
            query = (
                f'"{firm_name}" misi OR visi OR nilai OR pendekatan OR metodologi '
                'OR komitmen OR prinsip OR filosofi'
            )
            res = Researcher.search(query, limit=10, recency_bucket="year")
            logger.debug(f"Serper | Firm values/approach search | firm={firm_name} | results_found={len(res)}")
            filtered = Researcher._filter_recent_entity_results(
                res,
                entity_name=firm_name,
                max_age_years=5,
                strict_entity=False
            )
            
            values_found = []
            for item in filtered[:4]:
                title = item.get('title', '')
                snippet = (item.get('snippet', '') or '').strip()
                
                # Extract value keywords
                values = re.findall(
                    r'(integritas|excellence|innovation|kolaborasi|partnership|'
                    r'kepercayaan|transparansi|keberlanjutan|quality|delivery)',
                    title + ' ' + snippet,
                    re.IGNORECASE
                )
                values_found.extend(values)
            
            core_approach = (
                f"Pendekatan {firm_name} menekankan delivery berkualitas tinggi, "
                "kolaborasi erat dengan klien, dan hasil bisnis yang terukur."
            )
            
            if values_found:
                unique_values = list(set(v.lower() for v in values_found))[:4]
                core_approach += f" Nilai inti: {', '.join(unique_values)}."
            
            return core_approach
        except Exception as e:
            logger.warning(f"Serper | Error fetching firm values/approach | firm={firm_name} | error={str(e)[:80]}")
            return f"Pendekatan {firm_name} menekankan delivery berkualitas tinggi, kolaborasi erat dengan klien, dan hasil bisnis yang terukur."

    @staticmethod
    @lru_cache(maxsize=128)
    def get_firm_key_contacts(firm_name: str, office_location: str = "") -> str:
        """Search for office locations and contact information via Serper."""
        try:
            location_context = f" {office_location}" if office_location else " Yogyakarta"
            query = (
                f'"{firm_name}" kantor{location_context} OR alamat OR kontak '
                'OR telephone OR email OR website resmi'
            )
            res = Researcher.search(query, limit=10, recency_bucket="year")
            logger.debug(f"Serper | Firm contacts search | firm={firm_name} | location={office_location or 'default'} | results_found={len(res)}")
            filtered = Researcher._filter_recent_entity_results(
                res,
                entity_name=firm_name,
                max_age_years=5,
                strict_entity=False
            )
            
            contact_items = []
            websites = set()
            for item in filtered[:5]:
                snippet = (item.get('snippet', '') or '').strip()
                link = item.get('link', '')
                
                # Extract domain from link
                domain_match = re.search(r'https?://([^/]+)', link)
                if domain_match:
                    domain = domain_match.group(1).replace('www.', '')
                    websites.add(domain)
                
                # Look for location indicators
                locations = re.findall(r'(Yogyakarta|Jakarta|Bandung|Surabaya|Indonesia)', snippet)
                if locations:
                    contact_items.append(f"Lokasi: {locations[0]}")
            
            result = f"Hubungi {firm_name} melalui saluran resmi untuk koordinasi detail implementasi."
            
            if websites:
                unique_websites = list(websites)[:2]
                result += f" Website resmi: {', '.join(unique_websites)}."
            
            if contact_items:
                result += f" {' | '.join(set(contact_items[:2]))}."
            
            return result
        except Exception as e:
            logger.warning(f"Serper | Error fetching firm contacts | firm={firm_name} | error={str(e)[:80]}")
            return f"Hubungi {firm_name} melalui saluran resmi untuk koordinasi detail implementasi."

    @staticmethod
    @lru_cache(maxsize=128)
    def get_firm_accolades_recognition(firm_name: str) -> str:
        """Search for awards, recognitions, and industry accolades via Serper."""
        try:
            query = (
                f'"{firm_name}" penghargaan OR award OR recognition OR terbaik '
                'OR excellence OR leader OR top OR finalist'
            )
            res = Researcher.search(query, limit=10, recency_bucket="year")
            logger.debug(f"Serper | Firm accolades/awards search | firm={firm_name} | results_found={len(res)}")
            filtered = Researcher._filter_recent_entity_results(
                res,
                entity_name=firm_name,
                max_age_years=4,
                strict_entity=False
            )
            
            accolades = []
            for item in filtered[:5]:
                title = item.get('title', '')
                snippet = (item.get('snippet', '') or '').strip()
                
                # Look for award indicators
                award_patterns = re.findall(
                    r'(penghargaan|award|finalist|top \d+|best|terbaik|leader)',
                    title + ' ' + snippet,
                    re.IGNORECASE
                )
                
                if award_patterns:
                    # Extract year if present
                    year = Researcher._extract_year(snippet)
                    year_str = f" ({year})" if year else ""
                    accolades.append(f"{title}{year_str}")
            
            if accolades:
                unique_accolades = list(set(accolades))[:3]
                return (
                    f"Pengakuan industri terhadap {firm_name} mencerminkan komitmen terhadap keunggulan: "
                    f"{' | '.join(unique_accolades)}. Lihat website resmi untuk daftar penghargaan lengkap."
                )
            
            return (
                f"{firm_name} terus membangun reputasi melalui proyek-proyek tepat guna "
                "dan kepuasan klien berkelanjutan."
            )
        except Exception as e:
            logger.warning(f"Serper | Error fetching firm accolades | firm={firm_name} | error={str(e)[:80]}")
            return f"{firm_name} terus membangun reputasi melalui proyek-proyek tepat guna dan kepuasan klien berkelanjutan."

    @staticmethod
    def build_comprehensive_firm_profile(firm_name: str, office_location: str = "") -> Dict[str, str]:
        """
        Build a comprehensive firm profile with all OSINT-gathered information.
        Returns a dictionary with multiple firm profile sections.
        """
        return {
            "values_approach": Researcher.get_firm_values_approach(firm_name),
            "team_expertise": Researcher.get_firm_team_expertise(firm_name),
            "certifications": Researcher.get_firm_certifications(firm_name),
            "portfolio_scale": Researcher.get_firm_portfolio_scale(firm_name),
            "key_contacts": Researcher.get_firm_key_contacts(firm_name, office_location),
            "accolades": Researcher.get_firm_accolades_recognition(firm_name),
        }


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


class AppStateStore:
    def __init__(self, db_path: Optional[Path] = None, asset_root: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or APP_STATE_DB_PATH)
        self.asset_root = Path(asset_root or APP_ASSET_ROOT)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.templates_dir = self.asset_root / "templates"
        self.supporting_docs_dir = self.asset_root / "supporting_documents"
        self.portfolio_docs_dir = self.supporting_docs_dir / "portfolio"
        self.credentials_docs_dir = self.supporting_docs_dir / "credentials"
        self.generated_dir = Path(GENERATED_OUTPUT_DIR)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_docs_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_docs_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._bootstrap_generated_history()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposal_history (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    finished_at REAL NOT NULL,
                    client TEXT NOT NULL,
                    project TEXT NOT NULL,
                    proposal_mode TEXT NOT NULL,
                    service_type TEXT NOT NULL,
                    project_type TEXT NOT NULL,
                    timeline TEXT NOT NULL,
                    budget TEXT NOT NULL,
                    acceptance_score INTEGER NOT NULL DEFAULT 0,
                    acceptance_passes INTEGER NOT NULL DEFAULT 0,
                    processing_seconds REAL NOT NULL DEFAULT 0,
                    acceptance_json TEXT NOT NULL DEFAULT '{}',
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supporting_documents (
                    id TEXT PRIMARY KEY,
                    document_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    extracted_text TEXT NOT NULL DEFAULT '',
                    byte_size INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(proposal_history)").fetchall()
            }
            column_backfills = {
                "acceptance_score": "INTEGER NOT NULL DEFAULT 0",
                "acceptance_passes": "INTEGER NOT NULL DEFAULT 0",
                "processing_seconds": "REAL NOT NULL DEFAULT 0",
                "acceptance_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column_name, ddl in column_backfills.items():
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE proposal_history ADD COLUMN {column_name} {ddl}")
            conn.commit()

    def _bootstrap_generated_history(self) -> None:
        existing_files = [
            path for path in sorted(self.generated_dir.glob("*.docx"))
            if not path.name.startswith("~$")
        ]
        if not existing_files:
            return

        with self._connect() as conn:
            conn.execute("DELETE FROM proposal_history WHERE filename LIKE '~$%' OR filepath LIKE '%/~$%'")
            known_paths = {
                str(row["filepath"])
                for row in conn.execute("SELECT filepath FROM proposal_history").fetchall()
            }
            for path in existing_files:
                if str(path) in known_paths:
                    continue
                stat = path.stat()
                inferred_client = path.stem.replace("Proposal_", "").replace("_", " ").strip()
                conn.execute(
                    """
                    INSERT INTO proposal_history(
                        id, created_at, finished_at, client, project, proposal_mode,
                        service_type, project_type, timeline, budget, acceptance_score,
                        acceptance_passes, processing_seconds, acceptance_json, filename, filepath, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        float(stat.st_mtime),
                        float(stat.st_mtime),
                        inferred_client or "Dokumen historis",
                        "Dokumen historis dari folder generated",
                        "historis",
                        "",
                        "",
                        "",
                        "",
                        0,
                        0,
                        0.0,
                        "{}",
                        path.name,
                        str(path),
                        "{}",
                    ),
                )
            conn.commit()

    @staticmethod
    def _sanitize_filename(value: str, fallback: str = "proposal", max_length: int = 140) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
        cleaned = re.sub(r"_+", "_", cleaned)
        if not cleaned:
            cleaned = fallback
        if len(cleaned) > max_length:
            stem, dot, suffix = cleaned.rpartition(".")
            if dot:
                head = stem[: max_length - len(suffix) - 1]
                cleaned = f"{head}.{suffix}"
            else:
                cleaned = cleaned[:max_length]
        return cleaned

    def _get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def _set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value or "")),
            )
            conn.commit()

    @staticmethod
    def _normalize_document_type(value: str) -> str:
        normalized = SchemaMapper.normalize_key(value)
        aliases = {
            "portfolio": "portfolio",
            "portofolio": "portfolio",
            "credential": "credentials",
            "credentials": "credentials",
            "sertifikasi": "credentials",
            "kapabilitas": "credentials",
        }
        if normalized not in aliases:
            raise ValueError("Jenis dokumen pendukung tidak dikenal.")
        return aliases[normalized]

    def _supporting_dir_for_type(self, document_type: str) -> Path:
        normalized = self._normalize_document_type(document_type)
        return self.portfolio_docs_dir if normalized == "portfolio" else self.credentials_docs_dir

    @staticmethod
    def _read_text_bytes(raw_bytes: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except Exception:
                continue
        return raw_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def _extract_docx_text(raw_bytes: bytes) -> str:
        try:
            doc = Document(io.BytesIO(raw_bytes))
        except Exception:
            return ""

        blocks: List[str] = []
        for paragraph in doc.paragraphs:
            text = re.sub(r"\s+", " ", str(paragraph.text or "")).strip()
            if text:
                blocks.append(text)
        for table in doc.tables:
            for row in table.rows:
                cells = [
                    re.sub(r"\s+", " ", str(cell.text or "")).strip()
                    for cell in row.cells
                ]
                cells = [cell for cell in cells if cell]
                if cells:
                    blocks.append(" | ".join(cells))
        return "\n".join(blocks).strip()

    @staticmethod
    def _normalize_extracted_text(text: str) -> str:
        lines: List[str] = []
        for raw_line in str(text or "").replace("\r\n", "\n").split("\n"):
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @classmethod
    def _extract_supporting_document_text(cls, filename: str, raw_bytes: bytes) -> str:
        suffix = Path(str(filename or "")).suffix.lower()
        if suffix == ".docx":
            return cls._normalize_extracted_text(cls._extract_docx_text(raw_bytes))
        if suffix in {".txt", ".md"}:
            return cls._normalize_extracted_text(cls._read_text_bytes(raw_bytes))
        if suffix == ".pdf" and PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(raw_bytes))
                extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
                return cls._normalize_extracted_text(extracted)
            except Exception:
                return ""
        return ""

    @staticmethod
    def _trim_supporting_text(text: str, max_words: int = 220) -> str:
        words = str(text or "").split()
        if len(words) <= max_words:
            return str(text or "").strip()
        return " ".join(words[:max_words]).strip()

    def _serialize_document_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        filepath = str(row["filepath"] or "")
        extracted_text = str(row["extracted_text"] or "")
        return {
            "id": row["id"],
            "document_type": row["document_type"],
            "original_name": row["original_name"],
            "stored_name": row["stored_name"],
            "filepath": filepath,
            "uploaded_at": float(row["created_at"] or 0.0),
            "uploaded_at_label": self._format_timestamp(float(row["created_at"] or 0.0)),
            "byte_size": int(row["byte_size"] or 0),
            "has_text": bool(extracted_text.strip()),
            "exists": bool(filepath and Path(filepath).exists()),
        }

    def list_supporting_documents(self, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT id, document_type, created_at, original_name, stored_name, filepath, extracted_text, byte_size
            FROM supporting_documents
        """
        params: Tuple[Any, ...] = ()
        if document_type:
            normalized = self._normalize_document_type(document_type)
            query += " WHERE document_type = ?"
            params = (normalized,)
        query += " ORDER BY created_at DESC, original_name ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._serialize_document_row(row) for row in rows]

    def save_supporting_documents(self, document_type: str, uploads: List[Tuple[str, bytes]]) -> Dict[str, Any]:
        normalized = self._normalize_document_type(document_type)
        target_dir = self._supporting_dir_for_type(normalized)
        saved_any = False
        with self._connect() as conn:
            for filename, raw_bytes in uploads:
                if not filename or not raw_bytes:
                    continue
                safe_name = self._sanitize_filename(filename, fallback=f"{normalized}_supporting_document")
                suffix = Path(safe_name).suffix.lower()
                if suffix not in {".docx", ".pdf", ".txt", ".md"}:
                    continue
                stored_name = f"{uuid.uuid4().hex}_{safe_name}"
                target_path = target_dir / stored_name
                target_path.write_bytes(raw_bytes)
                extracted_text = self._extract_supporting_document_text(filename, raw_bytes)
                conn.execute(
                    """
                    INSERT INTO supporting_documents(
                        id, document_type, created_at, original_name, stored_name, filepath, extracted_text, byte_size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        normalized,
                        time.time(),
                        str(filename),
                        stored_name,
                        str(target_path),
                        extracted_text,
                        int(len(raw_bytes)),
                    ),
                )
                saved_any = True
            conn.commit()
        if not saved_any:
            raise ValueError("Belum ada file pendukung yang valid untuk diunggah.")
        return self.get_settings()

    def delete_supporting_document(self, document_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT filepath
                FROM supporting_documents
                WHERE id = ?
                """,
                (str(document_id or ""),),
            ).fetchone()
            if not row:
                raise KeyError("Dokumen pendukung tidak ditemukan.")
            filepath = Path(str(row["filepath"] or ""))
            conn.execute("DELETE FROM supporting_documents WHERE id = ?", (str(document_id or ""),))
            conn.commit()
        if filepath.exists():
            try:
                filepath.unlink()
            except OSError:
                pass
        return self.get_settings()

    def _collect_supporting_document_text(self, document_type: str, max_documents: int = 4, max_words: int = 220) -> str:
        normalized = self._normalize_document_type(document_type)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT extracted_text
                FROM supporting_documents
                WHERE document_type = ? AND extracted_text <> ''
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (normalized, int(max_documents)),
            ).fetchall()
        chunks = [self._trim_supporting_text(str(row["extracted_text"] or ""), max_words=max_words) for row in rows]
        chunks = [chunk for chunk in chunks if chunk]
        return "\n".join(chunks).strip()

    @classmethod
    def _fallback_portfolio_rows_from_text(cls, text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for raw_line in str(text or "").splitlines():
            line = re.sub(r"^\s*[-*•\d.]+\s*", "", raw_line).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line.split()) < 4:
                continue
            normalized = SchemaMapper.normalize_key(line)
            if not normalized or normalized in seen:
                continue
            if normalized in {
                "profil_perusahaan", "profile_perusahaan", "kapabilitas", "sertifikasi",
                "pengalaman", "project_experience", "portofolio", "portfolio"
            }:
                continue
            seen.add(normalized)
            rows.append(
                {
                    "area": line[:160],
                    "relevansi": "relevan untuk menunjukkan pengalaman serupa dan kesiapan pelaksanaan pekerjaan",
                    "bukti": "dokumen portofolio internal perusahaan penyusun",
                    "nilai_tambah": "memperkuat kredibilitas proposal dan membantu klien melihat bukti kemampuan secara lebih konkret",
                }
            )
            if len(rows) >= 4:
                break
        return rows

    def get_settings(self) -> Dict[str, Any]:
        template_path = self.get_template_path()
        return {
            "internal_portfolio": self._get_setting("internal_portfolio"),
            "internal_credentials": self._get_setting("internal_credentials"),
            "active_template_name": self._get_setting("active_template_name"),
            "has_active_template": bool(template_path and Path(template_path).exists()),
            "portfolio_documents": self.list_supporting_documents("portfolio"),
            "credential_documents": self.list_supporting_documents("credentials"),
        }

    def save_settings(self, internal_portfolio: str = "", internal_credentials: str = "") -> Dict[str, Any]:
        self._set_setting("internal_portfolio", internal_portfolio or "")
        self._set_setting("internal_credentials", internal_credentials or "")
        return self.get_settings()

    def save_template(self, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        safe_name = self._sanitize_filename(filename or "template_blanko.docx", fallback="template_blanko.docx")
        if not safe_name.lower().endswith(".docx"):
            safe_name = f"{safe_name}.docx"
        target_path = self.templates_dir / f"active_{safe_name}"
        for existing in self.templates_dir.glob("active_*"):
            try:
                existing.unlink()
            except OSError:
                continue
        target_path.write_bytes(raw_bytes)
        self._set_setting("active_template_path", str(target_path))
        self._set_setting("active_template_name", safe_name)
        return self.get_settings()

    def clear_template(self) -> Dict[str, Any]:
        template_path = self.get_template_path()
        if template_path and Path(template_path).exists():
            try:
                Path(template_path).unlink()
            except OSError:
                pass
        self._set_setting("active_template_path", "")
        self._set_setting("active_template_name", "")
        return self.get_settings()

    def get_template_path(self) -> str:
        return self._get_setting("active_template_path", "")

    @staticmethod
    def _parse_structured_portfolio_rows(text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for raw_line in str(text or "").splitlines():
            line = re.sub(r"^\s*[-*•\d.]+\s*", "", raw_line).strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 4:
                rows.append(
                    {
                        "area": parts[0],
                        "relevansi": parts[1],
                        "bukti": parts[2],
                        "nilai_tambah": parts[3],
                    }
                )
            elif len(parts) == 3:
                rows.append(
                    {
                        "area": parts[0],
                        "relevansi": parts[1],
                        "bukti": parts[2],
                        "nilai_tambah": "menambah keyakinan klien terhadap kesiapan delivery dan kualitas hasil kerja",
                    }
                )
            else:
                rows.append(
                    {
                        "area": line,
                        "relevansi": "relevan untuk inisiatif yang membutuhkan pengalaman delivery dan advisory yang serupa",
                        "bukti": line,
                        "nilai_tambah": "membantu proposal terasa lebih konkret dan kredibel",
                    }
                )
        return rows[:6]

    def enrich_firm_profile(self, firm_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        profile = dict(firm_profile or {})
        internal_portfolio = self._get_setting("internal_portfolio", "")
        internal_credentials = self._get_setting("internal_credentials", "")
        portfolio_docs_text = self._collect_supporting_document_text("portfolio", max_documents=4, max_words=180)
        credential_docs_text = self._collect_supporting_document_text("credentials", max_documents=4, max_words=160)
        portfolio_bits = [
            str(profile.get("portfolio_highlights") or "").strip(),
            internal_portfolio.strip(),
            portfolio_docs_text.strip(),
        ]
        portfolio_text = " ; ".join([item for item in portfolio_bits if item])
        if portfolio_text:
            profile["portfolio_highlights"] = portfolio_text
        credential_bits = [
            str(profile.get("credential_highlights") or "").strip(),
            internal_credentials.strip(),
            credential_docs_text.strip(),
        ]
        credential_text = " ; ".join([item for item in credential_bits if item])
        if credential_text:
            profile["credential_highlights"] = credential_text
        structured_rows = self._parse_structured_portfolio_rows(internal_portfolio)
        if not structured_rows and "|" in portfolio_docs_text:
            structured_rows = self._parse_structured_portfolio_rows(portfolio_docs_text)
        if not structured_rows:
            structured_rows = self._fallback_portfolio_rows_from_text(portfolio_docs_text)
        profile["internal_portfolio_rows"] = structured_rows
        profile["portfolio_document_names"] = [
            item.get("original_name", "")
            for item in self.list_supporting_documents("portfolio")
            if str(item.get("original_name") or "").strip()
        ]
        profile["credential_document_names"] = [
            item.get("original_name", "")
            for item in self.list_supporting_documents("credentials")
            if str(item.get("original_name") or "").strip()
        ]
        return profile

    def persist_generated_file(self, suggested_name: str, content: bytes) -> Path:
        safe_name = self._sanitize_filename(suggested_name or "proposal.docx", fallback="proposal.docx")
        if not safe_name.lower().endswith(".docx"):
            safe_name = f"{safe_name}.docx"
        target = self.generated_dir / safe_name
        counter = 2
        while target.exists():
            stem = target.stem
            stem = re.sub(r"_\d+$", "", stem)
            target = self.generated_dir / f"{stem}_{counter}{target.suffix}"
            counter += 1
        target.write_bytes(content)
        return target

    def add_history_entry(
        self,
        payload: Dict[str, Any],
        filename: str,
        filepath: str,
        created_at: float,
        finished_at: float,
        acceptance_report: Optional[Dict[str, Any]] = None,
        processing_seconds: float = 0.0,
    ) -> str:
        entry_id = uuid.uuid4().hex
        acceptance = dict(acceptance_report or {})
        acceptance_score = int(acceptance.get("score") or 0)
        acceptance_passes = 1 if acceptance.get("passes") else 0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO proposal_history(
                    id, created_at, finished_at, client, project, proposal_mode,
                    service_type, project_type, timeline, budget, acceptance_score,
                    acceptance_passes, processing_seconds, acceptance_json, filename, filepath, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    float(created_at or time.time()),
                    float(finished_at or time.time()),
                    str(payload.get("nama_perusahaan") or ""),
                    str(payload.get("konteks_organisasi") or ""),
                    str(payload.get("mode_proposal") or "canvassing"),
                    str(payload.get("jenis_proposal") or ""),
                    str(payload.get("jenis_proyek") or ""),
                    str(payload.get("estimasi_waktu") or ""),
                    str(payload.get("estimasi_biaya") or ""),
                    acceptance_score,
                    acceptance_passes,
                    float(processing_seconds or 0.0),
                    json.dumps(acceptance, ensure_ascii=False),
                    str(filename or ""),
                    str(filepath or ""),
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
        return entry_id

    @staticmethod
    def _format_timestamp(epoch_value: float) -> str:
        try:
            return datetime.fromtimestamp(float(epoch_value)).strftime("%d %b %Y %H:%M")
        except Exception:
            return "-"

    def list_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, finished_at, client, project, proposal_mode,
                       service_type, project_type, timeline, budget, acceptance_score,
                       acceptance_passes, processing_seconds, acceptance_json, filename, filepath, payload_json
                FROM proposal_history
                ORDER BY finished_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        items: List[Dict[str, Any]] = []
        for row in rows:
            filepath = str(row["filepath"] or "")
            items.append(
                {
                    "id": row["id"],
                    "client": row["client"],
                    "project": row["project"],
                    "proposal_mode": row["proposal_mode"],
                    "service_type": row["service_type"],
                    "project_type": row["project_type"],
                    "timeline": row["timeline"],
                    "budget": row["budget"],
                    "acceptance_score": int(row["acceptance_score"] or 0),
                    "acceptance_passes": bool(row["acceptance_passes"]),
                    "processing_seconds": float(row["processing_seconds"] or 0.0),
                    "filename": row["filename"],
                    "filepath": filepath,
                    "exists": bool(filepath and Path(filepath).exists()),
                    "can_reuse": bool(str(row["payload_json"] or "").strip() not in {"", "{}"}),
                    "finished_at": float(row["finished_at"]),
                    "finished_at_label": self._format_timestamp(float(row["finished_at"])),
                }
            )
        return items

    def get_history_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM proposal_history
                WHERE id = ?
                """,
                (entry_id,),
            ).fetchone()
        if not row:
            return None
        payload = {}
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = {}
        acceptance_report = {}
        try:
            acceptance_report = json.loads(row["acceptance_json"] or "{}")
        except Exception:
            acceptance_report = {}
        filepath = str(row["filepath"] or "")
        return {
            "id": row["id"],
            "client": row["client"],
            "project": row["project"],
            "proposal_mode": row["proposal_mode"],
            "service_type": row["service_type"],
            "project_type": row["project_type"],
            "timeline": row["timeline"],
            "budget": row["budget"],
            "acceptance_score": int(row["acceptance_score"] or 0),
            "acceptance_passes": bool(row["acceptance_passes"]),
            "processing_seconds": float(row["processing_seconds"] or 0.0),
            "acceptance_report": acceptance_report,
            "filename": row["filename"],
            "filepath": filepath,
            "exists": bool(filepath and Path(filepath).exists()),
            "created_at": float(row["created_at"]),
            "finished_at": float(row["finished_at"]),
            "payload": payload,
            "can_reuse": bool(payload),
        }


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
    PROFESSIONAL_FONT = "Times New Roman"
    BODY_FONT_SIZE = 12
    BODY_LINE_SPACING = 1.15
    BODY_SPACE_AFTER = 6
    HEADING_1_SIZE = 14
    HEADING_2_SIZE = 12
    HEADING_3_SIZE = 11
    TABLE_FONT_SIZE = 10.5
    TEXT_COLOR = (0, 0, 0)
    SUBTLE_TEXT_COLOR = (89, 89, 89)
    TABLE_HEADER_FILL = "E7E6E6"

    @staticmethod
    def apply_document_styles(doc: Document, preserve_existing: bool = False) -> None:
        style = doc.styles['Normal']
        if not preserve_existing:
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.BODY_FONT_SIZE)
            pf = style.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = StyleEngine.BODY_LINE_SPACING
            pf.space_after = Pt(StyleEngine.BODY_SPACE_AFTER)
            pf.space_before = Pt(0)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            StyleEngine._apply_enhanced_heading_styles(doc, StyleEngine.TEXT_COLOR)
            StyleEngine._apply_enhanced_list_styles(doc)
            StyleEngine._apply_enhanced_table_styles(doc, StyleEngine.TEXT_COLOR)
        for section in doc.sections:
            if preserve_existing:
                continue
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)

    # ========== ENHANCED STYLING METHODS ==========
    # Enhanced document styling and formatting for professional proposals
    
    @staticmethod
    def apply_enhanced_styles(doc: Document, theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        """Apply enhanced document styles with improved typography and spacing."""
        StyleEngine._apply_base_enhanced_styles(doc)
        StyleEngine._apply_enhanced_heading_styles(doc, theme_color)
        StyleEngine._apply_enhanced_list_styles(doc)
        StyleEngine._apply_enhanced_table_styles(doc, theme_color)
    
    @staticmethod
    def _apply_base_enhanced_styles(doc: Document) -> None:
        """Apply base document styles."""
        try:
            style = doc.styles['Normal']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.BODY_FONT_SIZE)
            
            pf = style.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = StyleEngine.BODY_LINE_SPACING
            pf.space_after = Pt(StyleEngine.BODY_SPACE_AFTER)
            pf.space_before = Pt(0)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        except Exception as e:
            logger.warning(f"Could not apply base styles: {e}")
        
        # Set margins
        for section in doc.sections:
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)
    
    @staticmethod
    def _apply_enhanced_heading_styles(doc: Document, theme_color: Tuple[int, int, int]) -> None:
        """Apply formal heading styles with neutral color and spacing."""
        try:
            style = doc.styles['Heading 1']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.HEADING_1_SIZE)
            style.font.bold = True
            style.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            
            pf = style.paragraph_format
            pf.space_before = Pt(12)
            pf.space_after = Pt(6)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.0
            pf.keep_with_next = True
        except Exception as e:
            logger.warning(f"Could not apply Heading 1 style: {e}")
        
        try:
            style = doc.styles['Heading 2']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.HEADING_2_SIZE)
            style.font.bold = True
            style.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            
            pf = style.paragraph_format
            pf.space_before = Pt(10)
            pf.space_after = Pt(4)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.0
            pf.keep_with_next = True
        except Exception as e:
            logger.warning(f"Could not apply Heading 2 style: {e}")
        
        try:
            style = doc.styles['Heading 3']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.HEADING_3_SIZE)
            style.font.bold = True
            style.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            
            pf = style.paragraph_format
            pf.space_before = Pt(8)
            pf.space_after = Pt(3)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.0
            pf.keep_with_next = True
        except Exception as e:
            logger.warning(f"Could not apply Heading 3 style: {e}")
    
    @staticmethod
    def _apply_enhanced_list_styles(doc: Document) -> None:
        """Apply enhanced list item styles."""
        try:
            # List Bullet
            style = doc.styles['List Bullet']
            pf = style.paragraph_format
            pf.space_after = Pt(4)
            pf.space_before = Pt(0)
            pf.left_indent = Inches(0.25)
            pf.first_line_indent = Inches(-0.25)
        except Exception as e:
            logger.warning(f"Could not apply List Bullet style: {e}")
        
        try:
            # List Number
            style = doc.styles['List Number']
            pf = style.paragraph_format
            pf.space_after = Pt(4)
            pf.space_before = Pt(0)
            pf.left_indent = Inches(0.25)
            pf.first_line_indent = Inches(-0.25)
        except Exception as e:
            logger.warning(f"Could not apply List Number style: {e}")
    
    @staticmethod
    def _apply_enhanced_table_styles(doc: Document, theme_color: Tuple[int, int, int]) -> None:
        """Apply enhanced table styles."""
        try:
            style = doc.styles['Table Grid']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.TABLE_FONT_SIZE)
        except Exception as e:
            logger.warning(f"Could not apply Table Grid style: {e}")
    
    @staticmethod
    def add_colored_heading(doc: Document, text: str, level: int = 1, 
                           theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        """Add a formal heading with neutral styling."""
        heading = doc.add_heading(text, level=level)
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        for run in heading.runs:
            run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            run.font.name = StyleEngine.PROFESSIONAL_FONT
            if level == 1:
                run.font.size = Pt(StyleEngine.HEADING_1_SIZE)
                run.font.bold = True
            elif level == 2:
                run.font.size = Pt(StyleEngine.HEADING_2_SIZE)
                run.font.bold = True
            else:
                run.font.size = Pt(StyleEngine.HEADING_3_SIZE)
                run.font.bold = True
    
    @staticmethod
    def add_horizontal_line(doc: Document, color: Tuple[int, int, int] = (200, 200, 200)) -> None:
        """Add a horizontal line (visual separator) to the document."""
        try:
            paragraph = doc.add_paragraph()
            pPr = paragraph._element.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '12')
            bottom.set(qn('w:space'), '1')
            bottom.set(qn('w:color'), '%02x%02x%02x' % color)
            
            pBdr.append(bottom)
            pPr.append(pBdr)
            
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.space_before = Pt(6)
        except Exception as e:
            logger.warning(f"Could not add horizontal line: {e}")
    
    @staticmethod
    def add_info_box(doc: Document, title: str, content: str, 
                    theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        """Add a styled information box with title and content."""
        # Title
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.name = StyleEngine.PROFESSIONAL_FONT
        run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
        
        # Content
        p = doc.add_paragraph(content)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(6)
    
    @staticmethod
    def format_contact_block(doc: Document, contact_lines: list, 
                           theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        """Format a professional contact information block."""
        # Header
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(4)
        
        run = p.add_run("Kontak Resmi")
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.name = StyleEngine.PROFESSIONAL_FONT
        run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
        
        # Contact details
        for line in contact_lines:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(line)
            run.font.name = StyleEngine.PROFESSIONAL_FONT

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

    @staticmethod
    def _professional_palette(theme_color: Tuple[int, int, int]) -> List[str]:
        base = ChartEngine._to_matplotlib_rgb(theme_color)
        accent = '#5B9BD5'
        if sum(theme_color) < 120:
            accent = '#7F7F7F'
        return [accent, '#A5A5A5', '#D9D9D9', '#BFBFBF', '#7F7F7F', '#C9DAF8', '#9EADBA', '#EDEDED']

    @staticmethod
    def _parse_chart_items(raw_data: str) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for part in raw_data.split(';'):
            tokens = [token.strip() for token in part.split(',')]
            if len(tokens) < 2:
                continue
            label = tokens[0]
            value = re.sub(r'[^\d.\-]', '', tokens[1])
            if not label or not value:
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            items.append((label, numeric))
        return items

    @staticmethod
    def create_bar_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = [part.strip() for part in data_str.split('|')]
            if len(parts) >= 3:
                title_str, unit_str, raw_data = parts[0], parts[1], "|".join(parts[2:])
            elif len(parts) == 2:
                title_str, unit_str, raw_data = parts[0], "Nilai", parts[1]
            else:
                title_str, unit_str, raw_data = "Ringkasan", "Nilai", data_str
            items = ChartEngine._parse_chart_items(raw_data)
            if not items:
                return None

            labels = [label for label, _ in items]
            values = [value for _, value in items]
            palette = ChartEngine._professional_palette(theme_color)
            bar_colors = [palette[idx % len(palette)] for idx in range(len(values))]

            fig_height = max(3.6, len(items) * 0.7)
            fig, ax = plt.subplots(figsize=(8.4, fig_height))
            y_positions = list(range(len(items)))
            ax.barh(y_positions, values, color=bar_colors, edgecolor='#666666', linewidth=0.8)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=10)
            ax.invert_yaxis()
            ax.set_title(title_str, fontsize=12, fontweight='bold', pad=14)
            ax.set_xlabel(unit_str, fontsize=10)
            ax.grid(axis='x', linestyle='--', alpha=0.35)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            max_value = max(values) or 1.0
            ax.set_xlim(0, max_value * 1.18)

            for idx, value in enumerate(values):
                ax.text(value + (max_value * 0.02), idx, f"{value:g}", va='center', fontsize=9, color='#404040')

            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

    @staticmethod
    def create_donut_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = [part.strip() for part in data_str.split('|')]
            if len(parts) >= 2:
                title_str, raw_data = parts[0], "|".join(parts[1:])
            else:
                title_str, raw_data = "Komposisi", data_str
            items = ChartEngine._parse_chart_items(raw_data)
            if not items:
                return None

            labels = [label for label, _ in items]
            values = [value for _, value in items]
            palette = ChartEngine._professional_palette(theme_color)
            colors = [palette[idx % len(palette)] for idx in range(len(values))]

            fig, ax = plt.subplots(figsize=(6.6, 4.8))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                colors=colors,
                startangle=90,
                wedgeprops={'width': 0.45, 'edgecolor': 'white'},
                autopct=lambda pct: f"{pct:.0f}%" if pct >= 8 else '',
                pctdistance=0.8,
                labeldistance=1.05,
            )
            for text in texts:
                text.set_fontsize(9)
            for text in autotexts:
                text.set_fontsize(8.5)
                text.set_color('#404040')
            ax.text(0, 0, "Focus", ha='center', va='center', fontsize=11, fontweight='bold', color='#404040')
            ax.set_title(title_str, fontsize=12, fontweight='bold', pad=12)
            ax.axis('equal')

            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

class DocumentBuilder:
    @staticmethod
    def create_base_document(template_path: str = "") -> Tuple[Document, bool]:
        active_template = str(template_path or "").strip()
        if not active_template or not Path(active_template).exists():
            return Document(), False

        doc = Document(active_template)
        body = doc._element.body
        sect_pr = body.sectPr
        for child in list(body):
            if child is sect_pr:
                continue
            body.remove(child)
        return doc, True

    @staticmethod
    def _coerce_theme_color(theme_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if not isinstance(theme_color, tuple) or len(theme_color) != 3:
            return DEFAULT_COLOR
        channels = []
        for channel in theme_color:
            try:
                value = int(channel)
            except Exception:
                value = 0
            channels.append(max(0, min(255, value)))
        return tuple(channels)

    @staticmethod
    def _muted_theme_color(theme_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        base = DocumentBuilder._coerce_theme_color(theme_color)
        return tuple(max(25, min(235, int((channel * 0.72) + 24))) for channel in base)

    @staticmethod
    def _set_run_format(
        run,
        size: Optional[float] = None,
        bold: Optional[bool] = None,
        italic: Optional[bool] = None,
        color: Tuple[int, int, int] = StyleEngine.TEXT_COLOR,
        font_name: str = StyleEngine.PROFESSIONAL_FONT,
    ) -> None:
        run.font.name = font_name
        r_pr = run._element.get_or_add_rPr()
        r_fonts = r_pr.find(qn('w:rFonts'))
        if r_fonts is None:
            r_fonts = OxmlElement('w:rFonts')
            r_pr.append(r_fonts)
        r_fonts.set(qn('w:eastAsia'), font_name)
        r_fonts.set(qn('w:ascii'), font_name)
        r_fonts.set(qn('w:hAnsi'), font_name)
        if size is not None:
            run.font.size = Pt(size)
        if bold is not None:
            run.bold = bold
        if italic is not None:
            run.italic = italic
        if color:
            run.font.color.rgb = RGBColor(*color)

    @staticmethod
    def _apply_cell_shading(cell, fill_hex: str) -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn('w:shd'))
        if shd is None:
            shd = OxmlElement('w:shd')
            tc_pr.append(shd)
        shd.set(qn('w:fill'), fill_hex)

    @staticmethod
    def _format_table(table) -> None:
        table.style = 'Table Grid'
        table.autofit = True
        if not table.rows:
            return
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(2)
                    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        DocumentBuilder._set_run_format(
                            run,
                            size=StyleEngine.TABLE_FONT_SIZE,
                            bold=run.bold if row_index != 0 else True,
                        )
                if row_index == 0:
                    DocumentBuilder._apply_cell_shading(cell, StyleEngine.TABLE_HEADER_FILL)

    @staticmethod
    def _set_cell_text(cell, text: str, bold: bool = False) -> None:
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(str(text or "").strip())
        run.bold = bold
        DocumentBuilder._set_run_format(
            run,
            size=StyleEngine.TABLE_FONT_SIZE,
            bold=bold,
            color=StyleEngine.TEXT_COLOR,
        )

    @staticmethod
    def add_reference_cover_page(
        doc: Document,
        client: str,
        project: str,
        service_type: str,
        project_type: str,
        timeline: str,
        budget: str,
        firm_profile: Optional[Dict[str, Any]],
        theme_color: Tuple[int, int, int],
        logo_stream: Optional[io.BytesIO] = None,
    ) -> None:
        if logo_stream:
            try:
                logo_stream.seek(0)
                cover_logo = doc.add_paragraph()
                cover_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cover_logo.paragraph_format.space_before = Pt(18)
                cover_logo.paragraph_format.space_after = Pt(18)
                cover_logo.add_run().add_picture(logo_stream, width=Inches(3.0))
            except (UnrecognizedImageError, OSError, ValueError) as exc:
                logger.warning("Logo skipped due to unsupported image format: %s", exc)

        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_before = Pt(54 if not logo_stream else 0)
        title.paragraph_format.space_after = Pt(6)
        title_run = title.add_run("PROPOSAL STRATEGIS")
        DocumentBuilder._set_run_format(title_run, size=16, bold=True)

        client_name = doc.add_paragraph()
        client_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        client_name.paragraph_format.space_after = Pt(12)
        client_run = client_name.add_run((client or "Klien").upper())
        DocumentBuilder._set_run_format(client_run, size=24, bold=True)

        initiative = doc.add_paragraph()
        initiative.alignment = WD_ALIGN_PARAGRAPH.CENTER
        initiative.paragraph_format.space_after = Pt(8)
        initiative_run = initiative.add_run(project or f"{service_type} - {project_type}")
        DocumentBuilder._set_run_format(initiative_run, size=13, italic=True)

        meta_bits = [bit for bit in [service_type, project_type] if str(bit or "").strip()]
        if timeline:
            meta_bits.append(f"Durasi {timeline}")
        if meta_bits:
            meta_line = doc.add_paragraph(" | ".join(meta_bits))
            meta_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
            meta_line.paragraph_format.space_after = Pt(2)
            for run in meta_line.runs:
                DocumentBuilder._set_run_format(run, size=10.5, color=StyleEngine.SUBTLE_TEXT_COLOR)

        if budget:
            budget_line = doc.add_paragraph(f"Estimasi investasi: {budget}")
            budget_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
            budget_line.paragraph_format.space_after = Pt(0)
            for run in budget_line.runs:
                DocumentBuilder._set_run_format(run, size=10.5, color=StyleEngine.SUBTLE_TEXT_COLOR)

        signature = doc.add_paragraph()
        signature.alignment = WD_ALIGN_PARAGRAPH.CENTER
        signature.paragraph_format.space_before = Pt(92)
        signature_run = signature.add_run("Disusun Oleh:")
        DocumentBuilder._set_run_format(signature_run, size=11)
        signature.add_run().add_break()

        firm_run = signature.add_run(WRITER_FIRM_NAME)
        DocumentBuilder._set_run_format(firm_run, size=13, bold=True)

        legal_name = str((firm_profile or {}).get("legal_name") or "").strip()
        if legal_name and legal_name.lower() != WRITER_FIRM_NAME.lower():
            signature.add_run().add_break()
            legal_run = signature.add_run(legal_name)
            DocumentBuilder._set_run_format(legal_run, size=10, color=StyleEngine.SUBTLE_TEXT_COLOR)

        doc.add_page_break()

    @staticmethod
    def add_reference_chapter_heading(
        doc: Document,
        chapter_title: str,
        theme_color: Tuple[int, int, int],
    ) -> None:
        try:
            heading = doc.add_paragraph(style="Heading 1")
        except KeyError:
            heading = doc.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading.paragraph_format.space_before = Pt(0)
        heading.paragraph_format.space_after = Pt(12)
        run = heading.add_run(chapter_title)
        DocumentBuilder._set_run_format(run, size=14, bold=True)

    @staticmethod
    def add_writer_firm_profile_section(
        doc: Document,
        firm_profile: Optional[Dict[str, Any]],
        theme_color: Tuple[int, int, int],
    ) -> None:
        profile = firm_profile or {}
        summary = str(profile.get("profile_summary") or "").strip()
        portfolio = str(profile.get("portfolio_highlights") or "").strip()
        credentials = str(profile.get("credential_highlights") or "").strip()
        legal_name = str(profile.get("legal_name") or "").strip()

        contact_rows = [
            ("Alamat kantor", profile.get("office_address", "")),
            ("Email", profile.get("email", "")),
            ("Telp", profile.get("phone", "")),
            ("WhatsApp", profile.get("whatsapp", "")),
            ("Website", profile.get("website", "")),
            ("Jam operasional", profile.get("operating_hours", "")),
        ]
        detail_rows = [
            ("Entitas hukum", legal_name),
            ("Fokus layanan", "Pelatihan IT, sertifikasi, dan konsultasi IT."),
            ("Portofolio", portfolio),
            ("Kapabilitas", credentials),
        ]
        visible_detail_rows = [(label, str(value or "").strip()) for label, value in detail_rows if str(value or "").strip()]
        visible_contact_rows = [(label, str(value or "").strip()) for label, value in contact_rows if str(value or "").strip()]

        if not summary and not visible_detail_rows and not visible_contact_rows:
            return

        StyleEngine.add_horizontal_line(doc, color=(214, 220, 228))

        try:
            heading = doc.add_paragraph(style="Heading 2")
        except KeyError:
            heading = doc.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        heading.paragraph_format.space_before = Pt(8)
        heading.paragraph_format.space_after = Pt(8)
        heading_run = heading.add_run("Profil Penulis Proposal")
        DocumentBuilder._set_run_format(heading_run, size=12, bold=True)

        if summary:
            summary_paragraph = doc.add_paragraph(summary)
            summary_paragraph.paragraph_format.space_after = Pt(8)

        if visible_detail_rows:
            details_table = doc.add_table(rows=1, cols=2)
            DocumentBuilder._set_cell_text(details_table.rows[0].cells[0], "Aspek", bold=True)
            DocumentBuilder._set_cell_text(details_table.rows[0].cells[1], "Keterangan", bold=True)
            for label, value in visible_detail_rows:
                row = details_table.add_row().cells
                DocumentBuilder._set_cell_text(row[0], label, bold=True)
                DocumentBuilder._set_cell_text(row[1], value)
            DocumentBuilder._format_table(details_table)

        if visible_contact_rows:
            contact_title = doc.add_paragraph()
            contact_title.paragraph_format.space_before = Pt(8)
            contact_title.paragraph_format.space_after = Pt(6)
            run = contact_title.add_run("Kontak Resmi")
            DocumentBuilder._set_run_format(run, size=11, bold=True)

            contact_table = doc.add_table(rows=1, cols=2)
            DocumentBuilder._set_cell_text(contact_table.rows[0].cells[0], "Kanal", bold=True)
            DocumentBuilder._set_cell_text(contact_table.rows[0].cells[1], "Detail", bold=True)
            for label, value in visible_contact_rows:
                row = contact_table.add_row().cells
                DocumentBuilder._set_cell_text(row[0], label, bold=True)
                DocumentBuilder._set_cell_text(row[1], value)
            DocumentBuilder._format_table(contact_table)

        source_urls = profile.get("official_source_urls") or []
        if isinstance(source_urls, str):
            source_urls = [item.strip() for item in source_urls.split(",") if item.strip()]
        domains = []
        seen_domains = set()
        for url in source_urls:
            host = urlparse(str(url or "")).netloc.replace("www.", "").strip()
            if not host or host in seen_domains:
                continue
            seen_domains.add(host)
            domains.append(host)
        if domains:
            note = doc.add_paragraph()
            note.paragraph_format.space_before = Pt(6)
            note.paragraph_format.space_after = Pt(0)
            note_run = note.add_run(
                "Profil dan kontak pada bagian ini dirangkum dari kanal resmi yang terverifikasi: "
                + ", ".join(domains)
                + "."
            )
            DocumentBuilder._set_run_format(note_run, size=9, italic=True, color=StyleEngine.SUBTLE_TEXT_COLOR)

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
        DocumentBuilder._set_run_format(run, size=StyleEngine.BODY_FONT_SIZE, bold=bold, italic=italic)

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
                    DocumentBuilder._set_run_format(
                        run,
                        size=StyleEngine.HEADING_1_SIZE if level == 1 else StyleEngine.HEADING_2_SIZE if level == 2 else StyleEngine.HEADING_3_SIZE,
                        bold=True,
                    )
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
                for i, row in enumerate(rows):
                    cols = row.find_all(['td', 'th'])
                    for j, col in enumerate(cols):
                        if j < max_cols:
                            cell = table.cell(i, j)
                            cell._element.clear_content()
                            p = cell.add_paragraph()
                            DocumentBuilder._process_inline_html(p, col)
                            if col.name == 'th' or i == 0:
                                for run in p.runs:
                                    run.bold = True
                DocumentBuilder._format_table(table)

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
    def _normalize_markdown_blocks(raw_text: str) -> str:
        normalized: List[str] = []
        ordered_pattern = re.compile(r'^\d+\.\s+')
        bullet_pattern = re.compile(r'^[-*]\s+')

        for raw_line in (raw_text or "").split('\n'):
            stripped = raw_line.strip()
            previous = normalized[-1].strip() if normalized else ""
            is_ordered = bool(ordered_pattern.match(stripped))
            is_bullet = bool(bullet_pattern.match(stripped))
            is_list = is_ordered or is_bullet
            previous_is_ordered = bool(ordered_pattern.match(previous))
            previous_is_bullet = bool(bullet_pattern.match(previous))
            previous_is_list = previous_is_ordered or previous_is_bullet
            is_table = stripped.startswith('|')
            is_heading = stripped.startswith('#')
            is_visual = stripped.startswith('[[') and stripped.endswith(']]')

            if is_list and previous and (
                (
                    not previous_is_list
                    and not previous.startswith('|')
                    and not previous.startswith('#')
                    and not previous.startswith('[[')
                )
                or (previous_is_list and (is_ordered != previous_is_ordered or is_bullet != previous_is_bullet))
            ):
                normalized.append("")
            if stripped and previous_is_list and not is_list and not is_table and not is_heading and not is_visual:
                normalized.append("")
            normalized.append(stripped)

        compacted: List[str] = []
        blank_streak = 0
        for line in normalized:
            if line:
                blank_streak = 0
                compacted.append(line)
                continue
            blank_streak += 1
            if blank_streak <= 1:
                compacted.append("")
        return "\n".join(compacted).strip()

    @staticmethod
    def process_content(doc: Document, raw_text: str, theme_color: Tuple[int, int, int], chapter_title: str) -> None:
        clean_lines = []
        in_table = False
        normalized_text = DocumentBuilder._normalize_markdown_blocks(raw_text)
        for line in normalized_text.split('\n'):
            line = line.strip()
            if line.startswith('[[GANTT:') and line.endswith(']]'):
                data = line.replace('[[GANTT:', '').replace(']]', '').strip()
                img = ChartEngine.create_gantt_chart(data, theme_color)
                if img:
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.paragraph_format.space_after = Pt(8)
                    paragraph.add_run().add_picture(img, width=Inches(6.1))
                continue
            if line.startswith('[[BAR:') and line.endswith(']]'):
                data = line.replace('[[BAR:', '').replace(']]', '').strip()
                img = ChartEngine.create_bar_chart(data, theme_color)
                if img:
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.paragraph_format.space_after = Pt(8)
                    paragraph.add_run().add_picture(img, width=Inches(6.1))
                continue
            if line.startswith('[[DONUT:') and line.endswith(']]'):
                data = line.replace('[[DONUT:', '').replace(']]', '').strip()
                img = ChartEngine.create_donut_chart(data, theme_color)
                if img:
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.paragraph_format.space_after = Pt(8)
                    paragraph.add_run().add_picture(img, width=Inches(5.6))
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

    @staticmethod
    def _paragraph_is_blank(paragraph_el) -> bool:
        if paragraph_el.find('.//' + qn('w:drawing')) is not None:
            return False
        if paragraph_el.find('.//' + qn('w:pBdr')) is not None:
            return False
        for br in paragraph_el.findall('.//' + qn('w:br')):
            if br.get(qn('w:type')) == 'page':
                return False
        texts = [
            node.text or ""
            for node in paragraph_el.findall('.//' + qn('w:t'))
        ]
        return not "".join(texts).strip()

    @staticmethod
    def compact_layout(doc: Document) -> None:
        body = doc._element.body
        for child in list(body):
            if child.tag == qn('w:p') and DocumentBuilder._paragraph_is_blank(child):
                body.remove(child)
