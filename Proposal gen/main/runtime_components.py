"""Runtime components for data access, research, pricing, and document rendering."""

import concurrent.futures
import diskcache as dc
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple, Set

from .proposal_shared import *
from ollama import Client

# Initialize ultra-fast disk caching for OSINT (survives server restarts)
osint_cache_dir = Path(APP_STATE_DB_PATH).parent / '.osint_cache'
osint_cache = dc.Cache(str(osint_cache_dir))

# ==========================================
# PYDANTIC SCHEMAS FOR BULLETPROOF LLM DATA
# ==========================================
class InsightSchema(BaseModel):
    insight: str = Field(description="The extracted insight in Indonesian. 'NOT_FOUND' if missing.")

class FinancialSchema(BaseModel):
    revenue_idr: Optional[int] = Field(None, description="Total revenue in IDR")
    profit_idr: Optional[int] = Field(None, description="Total profit in IDR")
    project_budget_idr: Optional[int] = Field(None, description="Project budget in IDR")
    source_quote: str = Field("", description="Exact quote from text")

class ContactSchema(BaseModel):
    office_address: str = Field("", description="The primary or head office address. Empty if not found.")
    email: str = Field("", description="The official contact email. Empty if not found.")
    phone: str = Field("", description="The official phone or WhatsApp number. Empty if not found.")


class GenericProjectStandardsSchema(BaseModel):
    methodology: str = Field("", description="Delivery methodology or working approach.")
    team: str = Field("", description="Team composition, staffing plan, or experts involved.")
    commercial: str = Field("", description="Commercial terms, pricing model, or payment terms.")


class GenericClientRelationshipSchema(BaseModel):
    summary: str = Field("", description="Client relationship or engagement summary.")
    status: str = Field("", description="Relationship status such as existing, new, active, or renewal.")


class GenericFirmProfileSchema(BaseModel):
    office_address: str = Field("", description="Primary office address.")
    email: str = Field("", description="Official contact email.")
    phone: str = Field("", description="Official phone number.")
    whatsapp: str = Field("", description="Official WhatsApp number.")
    website: str = Field("", description="Official website.")
    legal_name: str = Field("", description="Legal entity name.")
    operating_hours: str = Field("", description="Operating or business hours.")
    profile_summary: str = Field("", description="Short company summary.")
    credential_highlights: str = Field("", description="Capability or credential highlights.")
    portfolio_highlights: str = Field("", description="Portfolio or capability highlights.")
    official_source_urls: List[str] = Field(default_factory=list, description="Official source URLs.")

# ==========================================

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
        self.base_url = FIRM_API_URL.rstrip("/")
        self.timeout_seconds = FIRM_API_TIMEOUT_SECONDS
        self.integration_mode = FIRM_API_INTEGRATION_MODE
        self.auth_mode = (FIRM_API_AUTH_MODE or "bearer").strip().lower()
        self.endpoint_config = dict(FIRM_API_ENDPOINT_CONFIG or {})
        self.dataset_config = dict(FIRM_API_DATASET_CONFIG or {})
        self.resource_config = dict(FIRM_API_RESOURCE_CONFIG or {})
        self.headers = {"Accept": "application/json"}
        self.auth: Optional[Tuple[str, str]] = None
        self._dataset_response_cache: Optional[Any] = None

        if self.auth_mode == "basic":
            if FIRM_API_USERNAME and FIRM_API_PASSWORD:
                self.auth = (FIRM_API_USERNAME, FIRM_API_PASSWORD)
        elif self.auth_mode == "bearer":
            token = str(API_AUTH_TOKEN or "").strip()
            if token and token != "isi_token_disini_nanti":
                self.headers["Authorization"] = f"Bearer {token}"
        elif self.auth_mode == "none":
            pass
        else:
            logger.warning("Unknown FIRM_API_AUTH_MODE=%s. Falling back to unauthenticated requests.", self.auth_mode)

    @staticmethod
    def _render_template_value(value: Any, context: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            try:
                return value.format(**context)
            except Exception:
                return value
        return value

    @classmethod
    def _render_template_payload(cls, payload: Any, context: Dict[str, Any]) -> Any:
        if isinstance(payload, dict):
            return {key: cls._render_template_payload(value, context) for key, value in payload.items()}
        if isinstance(payload, list):
            return [cls._render_template_payload(value, context) for value in payload]
        return cls._render_template_value(payload, context)

    @staticmethod
    def _extract_json_path(payload: Any, path: str) -> Any:
        current = payload
        for raw_part in filter(None, str(path or "").split(".")):
            if isinstance(current, list):
                if not raw_part.isdigit():
                    return None
                index = int(raw_part)
                if index < 0 or index >= len(current):
                    return None
                current = current[index]
                continue
            if isinstance(current, dict):
                if raw_part not in current:
                    return None
                current = current[raw_part]
                continue
            return None
        return current

    @staticmethod
    def _lookup_record_value(record: Dict[str, Any], field_name: str) -> Any:
        if field_name in record:
            return record[field_name]
        normalized_key = SchemaMapper.normalize_key(field_name)
        for key, value in record.items():
            if SchemaMapper.normalize_key(key) == normalized_key:
                return value
        return None

    @classmethod
    def _record_matches(cls, record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, expected in (filters or {}).items():
            if expected in (None, ""):
                continue
            actual = cls._lookup_record_value(record, str(key))
            if actual is None:
                return False
            if str(actual).strip() == str(expected).strip():
                continue
            if SchemaMapper.normalize_key(actual) == SchemaMapper.normalize_key(expected):
                continue
            return False
        return True

    def _request(
        self,
        path: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        body_encoding: str = "json",
    ) -> requests.Response:
        url = str(path or "").strip()
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            url = f"{self.base_url}/{url.lstrip('/')}"
        merged_headers = dict(self.headers)
        for key, value in (headers or {}).items():
            if value not in (None, ""):
                merged_headers[str(key)] = str(value)
        request_kwargs: Dict[str, Any] = {
            "params": params,
            "headers": merged_headers,
            "auth": self.auth,
            "timeout": self.timeout_seconds,
        }
        method_upper = str(method or "GET").strip().upper() or "GET"
        if method_upper != "GET":
            encoding = str(body_encoding or "json").strip().lower() or "json"
            if encoding == "form":
                request_kwargs["data"] = body or {}
            else:
                request_kwargs["json"] = body or {}
        return requests.request(
            method_upper,
            url,
            **request_kwargs,
        )

    def _request_named_endpoint(self, endpoint_name: str, **context: Any) -> requests.Response:
        spec = dict((self.endpoint_config or {}).get(endpoint_name) or {})
        return self._request_from_spec(spec, **context)

    def _request_from_spec(self, spec: Dict[str, Any], **context: Any) -> requests.Response:
        request_spec = dict(spec or {})
        path = self._render_template_value(request_spec.get("url") or request_spec.get("path") or "", context)
        method = str(request_spec.get("method") or "GET").strip().upper() or "GET"
        params = self._render_template_payload(request_spec.get("params") or {}, context)
        body = self._render_template_payload(request_spec.get("body") or {}, context)
        headers = self._render_template_payload(request_spec.get("headers") or {}, context)
        body_encoding = self._render_template_value(request_spec.get("body_encoding") or "json", context)
        return self._request(
            path,
            method=method,
            params=params,
            body=body,
            headers=headers,
            body_encoding=str(body_encoding or "json"),
        )

    def _get_dataset_response(self) -> Any:
        if self._dataset_response_cache is not None:
            return self._dataset_response_cache
        request_spec = dict((self.dataset_config or {}).get("request") or {})
        path = str(request_spec.get("path") or "").strip() or "/api/Resource/dataset"
        method = str(request_spec.get("method") or "POST").strip().upper() or "POST"
        params = request_spec.get("params") or {}
        body = request_spec.get("body") or {}
        body_encoding = str(request_spec.get("body_encoding") or "json").strip().lower() or "json"
        response = self._request(path, method=method, params=params, body=body, body_encoding=body_encoding)
        response.raise_for_status()
        self._dataset_response_cache = response.json()
        return self._dataset_response_cache

    def _select_dataset_payload(self, resource_name: str, **context: Any) -> Any:
        response_payload = self._get_dataset_response()
        payload_paths = dict((self.dataset_config or {}).get("payload_paths") or {})
        payload_path = self._render_template_value(payload_paths.get(resource_name) or "", context)
        if payload_path:
            selected = self._extract_json_path(response_payload, str(payload_path))
            if selected not in (None, ""):
                return selected

        records_path = self._render_template_value((self.dataset_config or {}).get("response_items_path") or "", context)
        records_payload = self._extract_json_path(response_payload, str(records_path)) if records_path else response_payload
        if isinstance(records_payload, dict):
            direct_match = records_payload.get(resource_name)
            if direct_match not in (None, ""):
                return direct_match
            return records_payload

        if not isinstance(records_payload, list):
            return {}

        resource_field = str((self.dataset_config or {}).get("resource_field") or "").strip()
        resource_values = dict((self.dataset_config or {}).get("resource_values") or {})
        expected_resource = self._render_template_value(resource_values.get(resource_name) or resource_name, context)
        filters = self._render_template_payload(
            dict(((self.dataset_config or {}).get("record_filters") or {}).get(resource_name) or {}),
            context,
        )

        matches: List[Dict[str, Any]] = []
        for item in records_payload:
            if not isinstance(item, dict):
                continue
            if resource_field:
                actual_resource = self._lookup_record_value(item, resource_field)
                if actual_resource is None:
                    continue
                if SchemaMapper.normalize_key(actual_resource) != SchemaMapper.normalize_key(expected_resource):
                    continue
            if not self._record_matches(item, filters):
                continue
            matches.append(item)

        if not matches:
            return {}
        return matches[0] if len(matches) == 1 else matches

    @staticmethod
    def _coerce_payload_to_mapping(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            dict_items = [item for item in payload if isinstance(item, dict)]
            if len(dict_items) == 1:
                return dict_items[0]
        return {}

    def _resolve_generic_payload(self, resource_name: str, **context: Any) -> Any:
        resource_spec = dict((self.resource_config or {}).get(resource_name) or {})
        request_spec = dict(resource_spec.get("request") or {})
        if not request_spec:
            return {}
        response = self._request_from_spec(request_spec, **context)
        response.raise_for_status()
        payload = response.json()
        response_path = self._render_template_value(resource_spec.get("response_path") or "", context)
        if response_path:
            payload = self._extract_json_path(payload, str(response_path))
        filters = self._render_template_payload(resource_spec.get("record_filters") or {}, context)
        if isinstance(payload, list):
            matches = [item for item in payload if isinstance(item, dict) and self._record_matches(item, filters)]
            if matches:
                return matches[0] if len(matches) == 1 else matches
        return payload

    def _resolve_resource_payload(self, resource_name: str, **context: Any) -> Any:
        if self.integration_mode == "dataset":
            return self._select_dataset_payload(resource_name, **context)
        if self.integration_mode == "generic":
            return self._resolve_generic_payload(resource_name, **context)
        response = self._request_named_endpoint(resource_name, **context)
        response.raise_for_status()
        return response.json()

    def _resource_allows_llm_extract(self, resource_name: str) -> bool:
        spec = dict((self.resource_config or {}).get(resource_name) or {})
        return bool(spec.get("allow_llm_extract", True))

    @staticmethod
    def _compact_json_for_llm(payload: Any) -> str:
        try:
            serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            serialized = str(payload)
        serialized = serialized.strip()
        max_chars = 18000
        return serialized[:max_chars]

    def _llm_extract_generic_resource(
        self,
        resource_name: str,
        payload: Any,
        schema_model: Any,
        context_note: str = "",
    ) -> Dict[str, Any]:
        prompt_map = {
            "firm_profile": (
                "Extract the writer firm's official identity and contact details from this JSON payload. "
                "Map only fields clearly supported by the payload."
            ),
            "project_standards": (
                "Extract the delivery methodology, team composition, and commercial terms from this JSON payload "
                "for proposal generation."
            ),
            "client_relationship": (
                "Extract the client relationship summary and relationship status from this JSON payload."
            ),
        }
        instruction = prompt_map.get(resource_name, "Extract the most relevant structured fields from this JSON payload.")
        payload_text = self._compact_json_for_llm(payload)
        prompt = f"""
You are an internal API schema adapter for a proposal generator.
{instruction}
{f"Context: {context_note}" if context_note else ""}

Rules:
- Read the payload as arbitrary JSON from an unknown internal API.
- Infer only what is reasonably supported by the payload.
- If a field is missing, return an empty string or empty list.
- Respond ONLY with valid JSON matching the requested schema.

JSON PAYLOAD:
{payload_text}
        """.strip()
        try:
            response = Client(host=OLLAMA_HOST).chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw_text = str(((response or {}).get("message") or {}).get("content") or "").strip()
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            parsed = json.loads(match.group(0) if match else raw_text)
            validated = schema_model.model_validate(parsed)
            return validated.model_dump()
        except Exception as exc:
            logger.warning("Generic internal API extraction failed for %s: %s", resource_name, exc)
            return {}

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
    def _is_weak_project_standards(payload: Dict[str, str]) -> bool:
        return all(str(payload.get(key) or "").strip() in {"", "TBD"} for key in ("methodology", "team", "commercial"))

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
    def _is_weak_client_relationship(payload: Dict[str, Any]) -> bool:
        return not str(payload.get("summary") or "").strip() and not bool(payload.get("verified"))

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
            payload = self._resolve_resource_payload("project_standards", project_type=project_type)
            normalized = self._normalize_project_standards(self._coerce_payload_to_mapping(payload))
            if self.integration_mode == "generic" and self._is_weak_project_standards(normalized) and self._resource_allows_llm_extract("project_standards"):
                extracted = self._llm_extract_generic_resource(
                    "project_standards",
                    payload,
                    GenericProjectStandardsSchema,
                    context_note=f"project_type={project_type}",
                )
                if extracted:
                    normalized = self._normalize_project_standards(extracted)
            return normalized
        except (requests.RequestException, ValueError) as e:
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

    @staticmethod
    def _looks_like_missing_profile(profile: Dict[str, str]) -> bool:
        if not isinstance(profile, dict):
            return True
        meaningful_keys = ("office_address", "email", "phone", "website", "contact_info")
        return not any(str(profile.get(key) or "").strip() for key in meaningful_keys)

    def get_firm_profile(self) -> Dict[str, str]:
        if self.demo_mode:
            return self._default_firm_profile()
        try:
            payload = self._resolve_resource_payload("firm_profile")
            normalized = self._normalize_firm_profile(self._coerce_payload_to_mapping(payload))
            if self.integration_mode == "generic" and self._looks_like_missing_profile(normalized) and self._resource_allows_llm_extract("firm_profile"):
                extracted = self._llm_extract_generic_resource(
                    "firm_profile",
                    payload,
                    GenericFirmProfileSchema,
                )
                if extracted:
                    normalized = self._normalize_firm_profile(extracted)
            return normalized
        except (requests.RequestException, ValueError) as e:
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
            payload = self._resolve_resource_payload("client_relationship", client_name=client_name)
            normalized = self._normalize_client_relationship(self._coerce_payload_to_mapping(payload))
            if self.integration_mode == "generic" and self._is_weak_client_relationship(normalized) and self._resource_allows_llm_extract("client_relationship"):
                extracted = self._llm_extract_generic_resource(
                    "client_relationship",
                    payload,
                    GenericClientRelationshipSchema,
                    context_note=f"client_name={client_name}",
                )
                if extracted:
                    normalized = self._normalize_client_relationship(extracted)
            return normalized
        except (requests.RequestException, ValueError) as e:
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
        """Upgraded OSINT builder using Deep Scrape and Pydantic."""
        current_year = datetime.now().year
        query = f'"{WRITER_FIRM_NAME}" kontak OR "hubungi kami" OR alamat kantor resmi {current_year}'
        
        # 1. Get the top search results
        raw_hits = Researcher.search(query, limit=5, recency_bucket="year")
        hits = Researcher._filter_recent_entity_results(raw_hits, entity_name=WRITER_FIRM_NAME, max_age_years=4)
        
        extracted_data = ContactSchema() # Default empty schema
        
        # 2. Deep scrape the #1 result (Usually the official website's Contact page)
        if hits and hits[0].get("link"):
            top_link = hits[0]["link"]
            markdown = Researcher.fetch_full_markdown(top_link)
            
            if markdown:
                prompt = f"""
                You are a data extractor. Read the following text about {WRITER_FIRM_NAME}.
                Extract their primary head office address, official email, and phone number.
                
                TEXT:
                {markdown}
                
                Respond ONLY with a valid JSON object matching the requested schema.
                """
                try:
                    res = Client(host=OLLAMA_HOST).chat(
                        model=LLM_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'temperature': 0.0}
                    )
                    raw_text = res['message']['content']
                    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                    parsed_dict = json.loads(match.group(0)) if match else json.loads(raw_text)
                    
                    # Force strict Pydantic validation
                    extracted_data = ContactSchema.model_validate(parsed_dict)
                except Exception as e:
                    logger.warning(f"Contact Deep Scrape failed: {e}")

        # 3. Pass the LLM-extracted data directly into your normalizer
        return self._normalize_firm_profile(
            {
                "office_address": extracted_data.office_address or WRITER_FIRM_OFFICE_ADDRESS,
                "email": extracted_data.email or WRITER_FIRM_EMAIL,
                "phone": extracted_data.phone or WRITER_FIRM_PHONE,
                # Keep your standard defaults for the rest
                "whatsapp": WRITER_FIRM_WHATSAPP,
                "website": WRITER_FIRM_WEBSITE,
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
        self.vector_store_dir = VECTOR_STORE_DIR
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.sync_state_path = KB_SYNC_STATE_PATH
        self.sync_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=str(self.vector_store_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings",
            model_name=EMBED_MODEL,
            timeout=KB_EMBED_TIMEOUT_SECONDS,
        )
        self.collection = self.chroma.get_or_create_collection(
            name="projects_db", embedding_function=self.embed_fn
        )
        self.df: Optional[pd.DataFrame] = None
        self.last_refresh_error = ""
        self.vector_ready = False
        self.sync_in_progress = False
        self._sync_lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None
        self._load_project_data()
        self.refresh_data(background=True)

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

    def _load_project_data(self) -> bool:
        try:
            self.df = self._normalize_projects_df(pd.read_sql("SELECT * FROM projects", self.engine))
        except Exception:
            if not PROJECT_CSV_PATH.exists():
                self.df = None
                self.vector_ready = False
                self.last_refresh_error = f"Project source file not found: {PROJECT_CSV_PATH}"
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
            self.vector_ready = False
            self.last_refresh_error = "Project data schema is missing required fields."
            return False
        return True

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        clean_row = row.fillna("").astype(str)
        return " | ".join(f"{col}: {val}" for col, val in clean_row.items())

    @staticmethod
    def _row_to_meta(row: pd.Series) -> Dict[str, str]:
        return row.fillna("").astype(str).to_dict()

    def _compute_vector_signature(self) -> str:
        if self.df is None or self.df.empty:
            return ""
        parts: List[str] = []
        for idx, row in self.df.iterrows():
            payload = {"_row_id": str(idx), **self._row_to_meta(row)}
            parts.append(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def _read_sync_state(self) -> Dict[str, Any]:
        if not self.sync_state_path.exists():
            return {}
        try:
            payload = json.loads(self.sync_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_sync_state(self, signature: str, row_count: int) -> None:
        payload = {
            "signature": signature,
            "row_count": int(row_count),
            "synced_at": time.time(),
            "embed_model": EMBED_MODEL,
        }
        self.sync_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _vector_store_current(self, signature: str) -> bool:
        if not signature:
            return False
        state = self._read_sync_state()
        if state.get("signature") != signature:
            return False
        if state.get("embed_model") != EMBED_MODEL:
            return False
        try:
            row_count = len(self.df) if self.df is not None else 0
            return int(self.collection.count()) == int(row_count)
        except Exception:
            return False

    def _sync_vector_store(self, force: bool = False) -> bool:
        if self.df is None or self.df.empty or not self._has_required_project_fields(self.df):
            self.vector_ready = False
            self.last_refresh_error = "Project data is not loaded."
            return False

        signature = self._compute_vector_signature()
        if not force and self._vector_store_current(signature):
            self.vector_ready = True
            self.last_refresh_error = ""
            return True

        try:
            existing_ids = set(self.collection.get(include=[])["ids"])
            new_ids_map = {str(idx): row for idx, row in self.df.iterrows()}
            new_ids_set = set(new_ids_map.keys())

            ids_to_delete = list(existing_ids - new_ids_set)
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)

            all_ids = list(new_ids_set)
            if all_ids:
                batch_size = KB_UPSERT_BATCH_SIZE
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    docs = [self._row_to_text(new_ids_map[b]) for b in batch_ids]
                    metas = [self._row_to_meta(new_ids_map[b]) for b in batch_ids]
                    self.collection.upsert(documents=docs, metadatas=metas, ids=batch_ids)
            self._write_sync_state(signature, len(all_ids))
        except Exception as exc:
            self.vector_ready = False
            self.last_refresh_error = str(exc)
            logger.warning(
                "Knowledge base vector sync is not ready yet. App will continue to serve with degraded semantic retrieval: %s",
                exc,
            )
            return False

        self.vector_ready = True
        self.last_refresh_error = ""
        return True

    def _refresh_worker(self, force: bool) -> None:
        try:
            for attempt in range(1, KB_STARTUP_MAX_RETRIES + 1):
                if not self._load_project_data():
                    success = False
                else:
                    with self._sync_lock:
                        success = self._sync_vector_store(force=force)
                if success:
                    return
                if attempt >= KB_STARTUP_MAX_RETRIES:
                    return
                delay = KB_STARTUP_RETRY_DELAY_SECONDS * attempt
                logger.info(
                    "Knowledge base retry scheduled in %ss (attempt %s/%s).",
                    delay,
                    attempt + 1,
                    KB_STARTUP_MAX_RETRIES,
                )
                time.sleep(delay)
        finally:
            with self._sync_lock:
                self.sync_in_progress = False
                self._sync_thread = None

    def _start_background_refresh(self, force: bool = False) -> bool:
        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                return False
            self.sync_in_progress = True
            self.vector_ready = False
            if not self.last_refresh_error:
                self.last_refresh_error = "Knowledge base semantic index is syncing in the background."
            self._sync_thread = threading.Thread(
                target=self._refresh_worker,
                args=(force,),
                name="knowledge-base-sync",
                daemon=True,
            )
            self._sync_thread.start()
            return True

    def refresh_data(self, force: bool = False, background: bool = False) -> bool:
        if not self._load_project_data():
            logger.warning(
                "Knowledge base startup refresh skipped because dependencies are not ready yet: %s",
                self.last_refresh_error,
            )
            return False

        signature = self._compute_vector_signature()
        if not force and self._vector_store_current(signature):
            self.vector_ready = True
            self.last_refresh_error = ""
            return True

        if background:
            self._start_background_refresh(force=force)
            return True

        with self._sync_lock:
            self.sync_in_progress = True
            try:
                return self._sync_vector_store(force=force)
            finally:
                self.sync_in_progress = False

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


# Public research helper (Serper & LLM Extractors).
class Researcher:
    CONTEXT_STOPWORDS = {
        "yang", "dengan", "untuk", "dari", "pada", "dan", "atau", "agar", "sebagai",
        "dalam", "akan", "lebih", "oleh", "serta", "suatu", "para", "bagi", "atas",
        "this", "that", "from", "with", "into", "their", "client", "proposal",
        "project", "service", "mode", "response", "canvassing", "kak", "kerangka",
        "acuan", "kerja", "jenis", "proyek", "proposal", "perusahaan", "klien",
        "pekerjaan", "kebutuhan", "inisiatif", "prioritas", "terukur",
    }

    @staticmethod
    def _has_serper_key() -> bool:
        key = (SERPER_API_KEY or "").strip()
        if not key:
            return False
        placeholder_keys = {"YOUR_SERPER_API_KEY", "YOUR_SERPER", "SERPER_API"}
        return key not in placeholder_keys

    @staticmethod
    @osint_cache.memoize(expire=43200)
    def fetch_full_markdown(url: str) -> str:
        """Fetches the clean markdown text of any URL using Jina Reader."""
        if not url: return ""
        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(jina_url, headers=headers, timeout=12)
            if response.status_code == 200:
                return response.text[:6000] 
            return ""
        except Exception as e:
            logger.warning(f"Failed to fetch full markdown for {url}: {e}")
            return ""

    @staticmethod
    @osint_cache.memoize(expire=43200)
    def extract_insight_with_llm(url: str, extraction_goal: str) -> str:
        """Universal Deep Scraper: Reads a URL and extracts a specific qualitative insight via Pydantic/LLM."""
        markdown_text = Researcher.fetch_full_markdown(url)
        if not markdown_text:
            return ""
            
        prompt = f"""
        You are an expert business researcher. Read the following source text.
        Your goal is to extract: {extraction_goal}
        
        SOURCE TEXT:
        {markdown_text}
        
        Respond ONLY with a valid JSON object using this schema. If the information is not present, use "NOT_FOUND".
        {{
            "insight": "<concise professional summary in Indonesian>"
        }}
        """
        
        try:
            client = Client(host=OLLAMA_HOST)
            res = client.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            raw_text = res['message']['content']
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            parsed_dict = json.loads(match.group(0)) if match else json.loads(raw_text)
            
            # Use Pydantic to strictly validate and coerce the dictionary
            data = InsightSchema.model_validate(parsed_dict)
            
            if "NOT_FOUND" in data.insight.upper() or not data.insight:
                return ""
            return Researcher._clean_osint_fact(data.insight)
        except Exception as e:
            logger.warning(f"Insight extraction failed for {url}: {e}")
            return ""

    @staticmethod
    def _clean_osint_fact(text: str) -> str:
        cleaned = str(text or "").replace("\xa0", " ").replace("…", " ")
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\.{3,}", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|,;:")
        return cleaned

    @staticmethod
    def _format_source_evidence_item(item: Dict[str, Any], fact_override: str = "", index: int = 1) -> str:
        title = str(item.get("title", "") or "Sumber tanpa judul").strip()
        link = str(item.get("link", "") or "-").strip()
        fact = Researcher._clean_osint_fact(str(fact_override or item.get("snippet", "") or ""))
        if not fact:
            return ""
        source_name = Researcher._source_name(link)
        citation = f"({source_name}, {Researcher._citation_year(item)})"
        return f"Sumber eksternal {max(1, int(index or 1))}: fakta={fact} | sumber={title} | url={link} | sitasi_apa={citation}"

    @staticmethod
    def _combine_summary_and_evidence(summary: str, evidence_lines: List[str], fallback: str = "") -> str:
        parts: List[str] = []
        cleaned_summary = str(summary or "").strip()
        if cleaned_summary:
            parts.append(cleaned_summary)

        deduped_lines: List[str] = []
        seen: Set[str] = set()
        for raw_line in evidence_lines or []:
            line = str(raw_line or "").strip()
            if not line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_lines.append(line)

        if deduped_lines:
            parts.append("\n".join(deduped_lines))

        if not parts:
            return fallback
        return "\n".join(parts)

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def search(query: str, limit: int = 5, recency_bucket: str = "") -> List[Dict[str, Any]]:
        """General web search using Serper.dev"""
        if not Researcher._has_serper_key():
            return []
        
        url = "https://google.serper.dev/search"
        payload_data = {"q": query, "gl": "id", "num": limit}
        recency_map = {"week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}
        if recency_bucket in recency_map:
            payload_data["tbs"] = recency_map[recency_bucket]
        
        try:
            response = requests.post(
                url, 
                headers={'X-API-KEY': (SERPER_API_KEY or "").strip(), 'Content-Type': 'application/json'}, 
                data=json.dumps(payload_data), 
                timeout=8
            )
            response.raise_for_status()
            return response.json().get('organic', [])
        except requests.RequestException as e:
            logger.warning(f"Serper API Error | query='{query[:60]}...' | error={str(e)[:100]}")
            return []

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _infer_industry_from_context(cls, client_name: str, ai_context: str = "", regulations: str = "") -> str:
        combined = " ".join([client_name or "", ai_context or "", regulations or ""]).lower()
        rules = [
            ("Perbankan", ["bank", "bri", "bca", "mandiri", "bni", "btn", "fintech", "kredit", "asset management"]),
            ("Telekomunikasi", ["telkom", "telkomsel", "indosat", "xl", "axiata", "operator"]),
            ("Energi & Utilitas", ["pertamina", "pln", "energi", "listrik", "oil", "gas"]),
            ("Ritel & E-Commerce", ["tokopedia", "e-commerce", "retail", "marketplace", "goto", "alfamart", "alfamidi"]),
            ("Transportasi & Aviasi", ["garuda", "aviation", "airline", "logistik", "transport", "bluebird"]),
            ("Pemerintah & BUMN", ["kementerian", "dinas", "pemprov", "pemkab", "bumn", "spbe"]),
            ("Manufaktur & Tambang", ["manufaktur", "adaro", "antam", "vale", "smelter", "mining"]),
        ]
        for label, tokens in rules:
            if any(token in combined for token in tokens):
                return label
        return "Lintas Industri"

    @staticmethod
    def _industry_query_terms(industry: str) -> List[str]:
        terms_map = {
            "Perbankan": ["manajemen investasi", "operasional investasi", "kepatuhan POJK", "service reliability"],
            "Telekomunikasi": ["service assurance", "network reliability", "customer experience"],
            "Energi & Utilitas": ["operational reliability", "governance", "asset integrity"],
            "Ritel & E-Commerce": ["omnichannel", "operational governance", "customer experience"],
            "Transportasi & Aviasi": ["operational control", "service reliability", "customer experience"],
            "Pemerintah & BUMN": ["tata kelola", "akuntabilitas", "audit trail", "layanan publik"],
            "Manufaktur & Tambang": ["production governance", "operational reliability", "safety compliance"],
        }
        return terms_map.get(industry, ["tata kelola", "operational excellence", "business value"])

    @classmethod
    def _context_terms(cls, text: str, excluded_terms: Optional[List[str]] = None, max_terms: int = 8) -> List[str]:
        excluded: Set[str] = set()
        for item in (excluded_terms or []):
            normalized = cls._normalize_text(item)
            if not normalized:
                continue
            excluded.add(normalized)
            excluded.update(normalized.split())
        tokens: List[str] = []
        seen: Set[str] = set()
        for token in re.findall(r"[A-Za-z]{4,}", (text or "").lower()):
            normalized = cls._normalize_text(token)
            if (
                not normalized
                or normalized in seen
                or normalized in cls.CONTEXT_STOPWORDS
                or normalized in excluded
            ):
                continue
            seen.add(normalized)
            tokens.append(token)
            if len(tokens) >= max_terms:
                break
        return tokens

    @staticmethod
    def _framework_terms(regulations: str, max_items: int = 4) -> List[str]:
        cleaned: List[str] = []
        seen: Set[str] = set()
        for part in re.split(r"[\n,;|/]+", str(regulations or "")):
            value = re.sub(r"\s+", " ", part).strip(" -.:")
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _focus_phrase(ai_context: str, max_words: int = 10) -> str:
        source = str(ai_context or "").replace("\r", "\n")
        line = re.split(r"[\n.;]+", source, maxsplit=1)[0]
        line = re.sub(r"\s+", " ", line).strip(" -|,:")
        if not line:
            return ""
        words = line.split()
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words).strip(" ,;:-")

    @staticmethod
    def _query_or_clause(terms: List[str], max_items: int = 4) -> str:
        cleaned: List[str] = []
        seen: Set[str] = set()
        for term in terms or []:
            value = re.sub(r"\s+", " ", str(term or "").strip())
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(f'"{value}"' if " " in value else value)
            if len(cleaned) >= max_items:
                break
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        return "(" + " OR ".join(cleaned) + ")"

    @classmethod
    def _build_query_plan(cls, client_name: str, ai_context: str = "", regulations: str = "") -> Dict[str, Any]:
        entity_terms = cls._entity_tokens(client_name)
        focus_phrase = cls._focus_phrase(ai_context)
        framework_terms = cls._framework_terms(regulations)
        industry = cls._infer_industry_from_context(client_name, ai_context, regulations)
        industry_terms = cls._industry_query_terms(industry)
        focus_terms = cls._context_terms(
            ai_context,
            excluded_terms=[client_name, focus_phrase, *framework_terms, *entity_terms],
            max_terms=8,
        )
        return {
            "focus_phrase": focus_phrase,
            "focus_terms": focus_terms,
            "framework_terms": framework_terms,
            "industry": industry,
            "industry_terms": industry_terms,
        }

    @staticmethod
    def _entity_tokens(entity_name: str) -> List[str]:
        legal_tokens = {"pt", "cv", "tbk", "inc", "ltd", "co", "corp", "persero", "company"}
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", (entity_name or "").lower())
            if len(token) >= 3 and token not in legal_tokens
        ]
        ordered = []
        seen = set()
        for token in tokens:
            if token in seen: continue
            ordered.append(token)
            seen.add(token)
        return ordered

    @staticmethod
    def _is_entity_match(item: Dict[str, Any], entity_name: str, strict: bool = False) -> bool:
        if not entity_name: return True
        merged = " ".join([str(item.get("title", "")), str(item.get("snippet", "")), str(item.get("link", ""))])
        normalized_merged = Researcher._normalize_text(merged)
        phrase = Researcher._normalize_text(entity_name)
        tokens = Researcher._entity_tokens(entity_name)

        if phrase and phrase in normalized_merged: return True
        if not tokens: return False

        merged_tokens = set(normalized_merged.split())
        hits = sum(1 for token in tokens if token in merged_tokens)
        if strict: return hits == len(tokens)
        if len(tokens) == 1: return hits == 1
        if len(tokens) == 2: return hits == 2
        return hits >= (len(tokens) - 1)

    @staticmethod
    def _extract_month(text: str) -> Optional[int]:
        if not text: return None
        month_map = {
            "jan": 1, "januari": 1, "feb": 2, "februari": 2, "mar": 3, "maret": 3,
            "apr": 4, "april": 4, "mei": 5, "may": 5, "jun": 6, "juni": 6,
            "jul": 7, "juli": 7, "agu": 8, "agustus": 8, "aug": 8, "sep": 9,
            "sept": 9, "september": 9, "okt": 10, "oct": 10, "oktober": 10,
            "nov": 11, "november": 11, "des": 12, "dec": 12, "desember": 12,
        }
        lowered = (text or "").lower()
        for key, month in month_map.items():
            if re.search(rf"\b{re.escape(key)}\b", lowered):
                return month
        return None

    @staticmethod
    def _extract_day(text: str) -> Optional[int]:
        if not text: return None
        match = re.search(r"\b([0-2]?\d|3[01])\b", text)
        if not match: return None
        day = int(match.group(1))
        if 1 <= day <= 31: return day
        return None

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        if not text: return None
        years = re.findall(r'\b(20\d{2})\b', text)
        if not years: return None
        return max(int(y) for y in years)

    @staticmethod
    def _published_sort_key(item: Dict[str, Any]) -> int:
        merged = " ".join([str(item.get("date", "")), str(item.get("title", "")), str(item.get("snippet", ""))])
        year = Researcher._extract_year(merged) or 0
        month = Researcher._extract_month(merged) or 1
        day = Researcher._extract_day(str(item.get("date", ""))) or 1
        return (year * 10_000) + (month * 100) + day

    @staticmethod
    def _sort_by_recency(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(items or [], key=Researcher._published_sort_key, reverse=True)

    @staticmethod
    def _is_recent(item: Dict[str, Any], max_age_years: int = 2) -> bool:
        merged = " ".join([str(item.get('date', '')), str(item.get('snippet', '')), str(item.get('title', ''))])
        year = Researcher._extract_year(merged)
        if year is None: return True
        return year >= (datetime.now().year - max_age_years)

    @staticmethod
    def _filter_recent_entity_results(items: List[Dict[str, Any]], entity_name: str = "", max_age_years: int = 2, strict_entity: bool = False) -> List[Dict[str, Any]]:
        filtered = []
        for item in (items or []):
            if not Researcher._is_recent(item, max_age_years=max_age_years): continue
            if entity_name and not Researcher._is_entity_match(item, entity_name, strict=strict_entity): continue
            filtered.append(item)
        return Researcher._sort_by_recency(filtered)

    @staticmethod
    def _source_name(link: str) -> str:
        try:
            host = urlparse(link or "").netloc.lower().strip().replace("www.", "")
            return host or "sumber daring"
        except Exception:
            return "sumber daring"

    @staticmethod
    def _citation_year(item: Dict[str, Any]) -> str:
        merged = " ".join([str(item.get('date', '')), str(item.get('snippet', '')), str(item.get('title', ''))])
        year = Researcher._extract_year(merged)
        return str(year) if year else "n.d."

    @staticmethod
    def _format_evidence(items: List[Dict[str, Any]], label: str, fallback: str) -> str:
        if not items: return f"{fallback} (sumber daring, n.d.)"
        lines = []
        for i, item in enumerate(items, start=1):
            title = item.get('title', 'Sumber tanpa judul')
            snippet = Researcher._clean_osint_fact(str(item.get('snippet', '') or ''))
            link = item.get('link', '-')
            if not snippet: continue
            source_name = Researcher._source_name(link)
            citation = f"({source_name}, {Researcher._citation_year(item)})"
            lines.append(f"Sumber eksternal {i}: fakta={snippet} | sumber={title} | url={link} | sitasi_apa={citation}")
        return "\n".join(lines) if lines else f"{fallback} (sumber daring, n.d.)"

    # =========================================================
    # UPGRADED OSINT METHODS: USING DEEP SCRAPING + PYDANTIC
    # =========================================================

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_latest_client_news(client_name: str, ai_context: str = "", regulations: str = "") -> str:
        current_year = datetime.now().year
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("focus_terms", [])[:3], *plan.get("framework_terms", [])[:2], *plan.get("industry_terms", [])[:2]],
            max_items=5,
        )
        query = f'"{client_name}" berita inovasi OR transformasi digital OR inisiatif strategis'
        if context_clause:
            query = f'"{client_name}" {context_clause} berita inovasi OR transformasi digital OR inisiatif strategis'
        query = f"{query} {current_year}"
        res = Researcher.search(query, limit=5, recency_bucket="month")
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=2)
        
        # Deep Scrape the #1 top news hit
        if filtered and filtered[0].get("link"):
            top_link = filtered[0]["link"]
            focus_hint = plan.get("focus_phrase") or Researcher._query_or_clause(plan.get("framework_terms", []), max_items=2)
            goal = (
                f"What is the company's latest major strategic initiative, digital transformation, or business innovation"
                f"{f' related to {focus_hint}' if focus_hint else ''}?"
            )
            insight = Researcher.extract_insight_with_llm(top_link, goal)
            if insight:
                source = Researcher._source_name(top_link)
                summary = f"Berdasarkan inisiatif strategis terbaru dari liputan {source}: {insight}"
                evidence_lines = [Researcher._format_source_evidence_item(filtered[0], fact_override=insight, index=1)]
                corroborating = [
                    line for line in (
                        Researcher._format_source_evidence_item(item, index=idx)
                        for idx, item in enumerate(filtered[1:3], start=2)
                    )
                    if line
                ]
                evidence_lines.extend(corroborating)
                return Researcher._combine_summary_and_evidence(summary, evidence_lines, fallback="")
                
        # Fallback to standard snippets
        cleaned = [i for i in filtered if len(re.findall(r"\S+", i.get('snippet',''))) > 5]
        return Researcher._format_evidence(cleaned[:3] if cleaned else filtered[:3], label="OSINT_NEWS", fallback="")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_client_ai_posture(client_name: str, ai_context: str = "", regulations: str = "") -> str:
        current_year = datetime.now().year
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [*plan.get("focus_terms", [])[:3], *plan.get("framework_terms", [])[:2], *plan.get("industry_terms", [])[:2]],
            max_items=5,
        )
        query = f'"{client_name}" AI OR artificial intelligence OR generative AI OR machine learning OR data platform'
        if context_clause:
            query = f'{query} {context_clause}'
        query = f"{query} {current_year}"
        res = Researcher.search(query, limit=5, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=3)
        
        # Deep Scrape the #1 AI posture link
        if filtered and filtered[0].get("link"):
            top_link = filtered[0]["link"]
            focus_hint = plan.get("focus_phrase") or Researcher._query_or_clause(plan.get("focus_terms", []), max_items=3)
            goal = (
                f"What is {client_name}'s current maturity level or stated plans regarding Artificial Intelligence, Data, or Machine Learning"
                f"{f' for initiatives related to {focus_hint}' if focus_hint else ''}?"
            )
            insight = Researcher.extract_insight_with_llm(top_link, goal)
            if insight:
                source = Researcher._source_name(top_link)
                summary = f"Kesiapan AI & Data {client_name} (via {source}): {insight}"
                evidence_lines = [Researcher._format_source_evidence_item(filtered[0], fact_override=insight, index=1)]
                corroborating = [
                    line for line in (
                        Researcher._format_source_evidence_item(item, index=idx)
                        for idx, item in enumerate(filtered[1:3], start=2)
                    )
                    if line
                ]
                evidence_lines.extend(corroborating)
                return Researcher._combine_summary_and_evidence(
                    summary,
                    evidence_lines,
                    fallback="Data publik terkait AI posture klien terbatas.",
                )
                
        return Researcher._format_evidence(filtered[:3], label="OSINT_AI", fallback="Data publik terkait AI posture klien terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_firm_values_approach(firm_name: str) -> str:
        query = f'"{firm_name}" misi OR visi OR nilai OR pendekatan OR metodologi OR prinsip'
        res = Researcher.search(query, limit=5, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, firm_name, max_age_years=5)
        
        # Deep Scrape the firm's approach from their top hit
        if filtered and filtered[0].get("link"):
            top_link = filtered[0]["link"]
            goal = f"What are the core professional values, mission statement, or unique working methodologies of {firm_name}?"
            insight = Researcher.extract_insight_with_llm(top_link, goal)
            if insight:
                summary = f"Pendekatan dan Nilai Inti {firm_name}: {insight}"
                evidence_line = Researcher._format_source_evidence_item(filtered[0], fact_override=insight, index=1)
                return Researcher._combine_summary_and_evidence(summary, [evidence_line], fallback=summary)
        
        return "Pendekatan menekankan delivery berkualitas tinggi, kolaborasi erat, dan hasil bisnis terukur."

    # Standard Snippet Methods (Kept for speed)
    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_entity_profile(entity_name: str, ai_context: str = "", regulations: str = "") -> str:
        current_year = datetime.now().year
        strict_entity = Researcher._normalize_text(entity_name) == Researcher._normalize_text(WRITER_FIRM_NAME)
        plan = Researcher._build_query_plan(entity_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [*plan.get("industry_terms", [])[:2], *plan.get("framework_terms", [])[:2], *plan.get("focus_terms", [])[:2]],
            max_items=4,
        )
        query = f'"{entity_name}" profil perusahaan OR "tentang kami" OR layanan utama'
        if context_clause:
            query = f'"{entity_name}" {context_clause} profil perusahaan OR "tentang kami" OR layanan utama'
        query = f"{query} {current_year}"
        res = Researcher.search(query, limit=6, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, entity_name, max_age_years=3, strict_entity=strict_entity)
        return Researcher._format_evidence(filtered[:3], label="OSINT_PROFILE", fallback="Data profil terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_client_track_record(client_name: str, ai_context: str = "", regulations: str = "") -> str:
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("focus_terms", [])[:3], *plan.get("industry_terms", [])[:2]],
            max_items=5,
        )
        query = f'"{client_name}" pencapaian OR kinerja OR penghargaan OR implementasi OR transformasi'
        if context_clause:
            query = f'"{client_name}" {context_clause} pencapaian OR kinerja OR penghargaan OR implementasi OR transformasi'
        res = Researcher.search(query, limit=6, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=3)
        return Researcher._format_evidence(filtered[:3], label="OSINT_TRACK", fallback="Track record terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_client_writer_collaboration(client_name: str, writer_firm_name: str = WRITER_FIRM_NAME, ai_context: str = "", regulations: str = "") -> str:
        plan = Researcher._build_query_plan(client_name, ai_context, regulations)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("framework_terms", [])[:2], *plan.get("focus_terms", [])[:2]],
            max_items=4,
        )
        query = f'"{writer_firm_name}" "{client_name}" kerja sama OR proyek OR konsultasi'
        if context_clause:
            query = f'"{writer_firm_name}" "{client_name}" {context_clause} kerja sama OR proyek OR konsultasi'
        res = Researcher.search(query, limit=6, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, client_name, max_age_years=6)
        return Researcher._format_evidence(filtered[:2], label="OSINT_COLLAB", fallback="Belum ditemukan bukti publik kolaborasi.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_regulatory_data(regulations_string: str, ai_context: str = "", client_name: str = "") -> str:
        if not regulations_string: return "Tidak ada regulasi spesifik dari input user."
        plan = Researcher._build_query_plan(client_name, ai_context, regulations_string)
        framework_clause = Researcher._query_or_clause(plan.get("framework_terms", []), max_items=4)
        context_clause = Researcher._query_or_clause(
            [plan.get("focus_phrase", ""), *plan.get("focus_terms", [])[:2], *plan.get("industry_terms", [])[:2]],
            max_items=4,
        )
        query = f'Ringkasan implementasi standar {framework_clause or regulations_string.replace(",", " OR ")}'
        if context_clause:
            query = f"{query} {context_clause}"
        query = f"{query} site:.go.id OR site:iso.org"
        res = Researcher.search(query, limit=5, recency_bucket="year")
        recent = [i for i in Researcher._sort_by_recency(res) if Researcher._is_recent(i, max_age_years=5)]
        trusted = [i for i in recent if any(str(i.get("link","")).endswith(sfx) for sfx in ("go.id", "iso.org", "ietf.org"))]
        return Researcher._format_evidence((trusted or recent)[:3], label="OSINT_REG", fallback="Data regulasi terbatas.")

    @staticmethod
    @osint_cache.memoize(expire=86400)
    def get_firm_certifications(firm_name: str) -> str:
        query = f'"{firm_name}" ISO OR sertifikasi OR akreditasi OR certification'
        res = Researcher.search(query, limit=5, recency_bucket="year")
        filtered = Researcher._filter_recent_entity_results(res, firm_name, max_age_years=5)
        certs = list(set(re.findall(r'ISO[\s\-]?\d{4,5}|[A-Z]{2,4}\s+\d{3,5}', " ".join(i.get('snippet','') for i in filtered))))
        if certs: return f"Sertifikasi: {', '.join(certs[:4])}. Lihat website resmi untuk daftar lengkap."
        return f"Kredensial dan sertifikasi {firm_name} tersedia di saluran publik resmi."

    @staticmethod
    def build_comprehensive_firm_profile(firm_name: str, office_location: str = "") -> Dict[str, str]:
        # Using concurrent.futures to fetch all firm OSINT at the EXACT same time
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_values = executor.submit(Researcher.get_firm_values_approach, firm_name)
            future_certs = executor.submit(Researcher.get_firm_certifications, firm_name)
            
        return {
            "values_approach": future_values.result(),
            "certifications": future_certs.result(),
            "team_expertise": f"Tim profesional {firm_name} memiliki pengalaman di berbagai domain strategis.",
            "portfolio_scale": f"Portofolio {firm_name} mencakup berbagai klien enterprise.",
            "key_contacts": f"Hubungi {firm_name} melalui saluran resmi untuk koordinasi detail.",
            "accolades": f"{firm_name} terus membangun reputasi melalui proyek-proyek tepat guna.",
        }


# Budget estimator from public financial context.
class FinancialAnalyzer:
    def __init__(self, ollama_client: Client):
        self.ollama = ollama_client

    DEFAULT_DURATION_MONTHS = {
        "diagnostic": 2.0, "strategic": 3.0, "transformation": 6.0, "implementation": 5.0,
    }

    BASE_MONTHLY_DELIVERY_RATE = {
        "diagnostic": 45_000_000, "strategic": 60_000_000, "transformation": 85_000_000, "implementation": 95_000_000,
    }

    def _extract_financials_with_llm(self, company_name: str, markdown_text: str) -> Dict[str, Any]:
        """Uses Ollama and Pydantic to strictly extract financial data from scraped markdown text."""
        if not markdown_text.strip():
            return {"revenue_idr": None, "profit_idr": None, "project_budget_idr": None, "source_quote": ""}
            
        prompt = f"""
        You are a financial analyst. Read the following text about {company_name}.
        Extract the most recent financial figures available. 
        
        TEXT:
        {markdown_text}
        
        Respond ONLY with a valid JSON object using this exact schema. If a value is not mentioned, use null.
        {{
            "revenue_idr": <number or null>,
            "profit_idr": <number or null>,
            "project_budget_idr": <number or null>,
            "source_quote": "<copy the exact sentence where you found the numbers>"
        }}
        """
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            raw_text = res['message']['content']
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            parsed_dict = json.loads(match.group(0)) if match else json.loads(raw_text)
            
            # Pydantic strictly validates the types and structure
            valid_data = FinancialSchema.model_validate(parsed_dict)
            return valid_data.model_dump()
            
        except Exception as e:
            logger.error(f"Pydantic/LLM Extraction failed: {e}")
            return {"revenue_idr": None, "profit_idr": None, "project_budget_idr": None, "source_quote": ""}

    @staticmethod
    def _has_signal(text: str, candidate: str) -> bool:
        value = re.sub(r"\s+", " ", str(candidate or "").strip().lower())
        if not value: return False
        pattern = re.escape(value)
        if " " not in value and value.isalpha(): pattern = rf"\b{pattern}\b"
        return bool(re.search(pattern, (text or "").lower()))

    @classmethod
    def _ai_scope_summary(cls, *values: Any) -> Dict[str, Any]:
        combined = re.sub(r"\s+", " ", " ".join(str(value or "") for value in values)).strip().lower()
        strong_hits = [token for token in (SPIRIT_OF_AI_RULES.get("strong_trigger_keywords") or []) if cls._has_signal(combined, token)]
        support_hits = [token for token in (SPIRIT_OF_AI_RULES.get("supporting_signals") or []) if token not in strong_hits and cls._has_signal(combined, token)]
        return {
            "enabled": bool(strong_hits) or len(support_hits) >= 2,
            "strong_hits": strong_hits[:8],
            "support_hits": support_hits[:10],
        }

    @classmethod
    def _ai_pricing_profile(cls, project_goal: str, objective: str, notes: str, frameworks: str) -> Dict[str, Any]:
        combined = " ".join([project_goal or "", objective or "", notes or "", frameworks or ""]).lower()
        ai_scope = cls._ai_scope_summary(project_goal, objective, notes, frameworks)
        if not ai_scope["enabled"]:
            return {"enabled": False, "level": "terkendali", "multiplier": 1.0, "drivers": [], "driver_labels": []}

        driver_config = SPIRIT_OF_AI_RULES.get("pricing_driver_terms") or {}
        driver_labels = {
            "data_readiness": "kesiapan data/model", "model_uncertainty": "validasi solusi/model",
            "architecture_constraints": "kendala arsitektur dan keamanan", "governance_overhead": "governance dan compliance",
            "change_enablement": "enablement dan adopsi organisasi",
        }
        driver_weights = {"data_readiness": 0.10, "model_uncertainty": 0.10, "architecture_constraints": 0.08, "governance_overhead": 0.10, "change_enablement": 0.09}

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
        level = "tinggi" if multiplier >= 1.45 else "menengah" if multiplier >= 1.18 else "terkendali"
        return {"enabled": True, "level": level, "multiplier": multiplier, "drivers": active_drivers, "driver_labels": [driver_labels.get(d, d) for d in active_drivers]}

    @staticmethod
    def _format_idr(amount: int) -> str:
        return "Rp " + f"{max(0, int(amount)):,}".replace(",", ".")

    @staticmethod
    def _parse_number(raw: str) -> Optional[float]:
        value = (raw or "").strip()
        if not value: return None
        if "." in value and "," in value: value = value.replace(".", "").replace(",", ".")
        elif "," in value and "." not in value: value = value.replace(",", ".")
        elif "." in value and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value): value = value.replace(".", "")
        try: return float(value)
        except ValueError: return None

    @classmethod
    def _extract_financial_values(cls, text: str) -> List[int]:
        if not text: return []
        pattern = re.compile(r"(?i)(?P<rp>rp\.?\s*)?(?P<num>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?P<unit>triliun|miliar|juta|ribu)?")
        multiplier = {"triliun": 1_000_000_000_000, "miliar": 1_000_000_000, "juta": 1_000_000, "ribu": 1_000}
        values: List[int] = []
        for match in pattern.finditer(text):
            has_rp = bool(match.group("rp"))
            unit = (match.group("unit") or "").lower()
            base = cls._parse_number(match.group("num") or "")
            if base is None or (not has_rp and not unit): continue
            amount = int(base * multiplier.get(unit, 1))
            if amount > 0 and (amount >= 1_000_000 or unit): values.append(amount)
        return values

    @classmethod
    def _duration_to_months(cls, timeline: str) -> Optional[float]:
        text = (timeline or "").lower().strip()
        if not text: return None
        patterns = [
            (r"(\d+(?:[.,]\d+)?)\s*(tahun|thn|year|years)", 12.0),
            (r"(\d+(?:[.,]\d+)?)\s*(bulan|bln|month|months)", 1.0),
            (r"(\d+(?:[.,]\d+)?)\s*(minggu|week|weeks)", 1.0 / 4.345),
        ]
        months = 0.0
        found = False
        for pattern, factor in patterns:
            for match in re.finditer(pattern, text):
                val = cls._parse_number(match.group(1))
                if val is not None:
                    months += val * factor
                    found = True
        if found: return max(months, 0.5)
        single_number = re.search(r"(\d+(?:[.,]\d+)?)", text)
        if single_number:
            val = cls._parse_number(single_number.group(1))
            if val is not None: return max(val, 0.5)
        return None

    @classmethod
    def _duration_months_or_default(cls, timeline: str, project_type: str) -> float:
        return cls._duration_to_months(timeline) or cls.DEFAULT_DURATION_MONTHS.get((project_type or "").strip().lower(), 4.0)

    @classmethod
    def _complexity_profile(cls, project_goal: str, objective: str, notes: str, frameworks: str) -> Dict[str, Any]:
        combined = " ".join([project_goal or "", objective or "", notes or "", frameworks or ""]).lower()
        if not combined.strip(): return {"level": "moderat", "multiplier": 1.0, "signal_count": 0}

        high_complexity = {"integrasi": 0.10, "core banking": 0.14, "migrasi": 0.10, "nasional": 0.06, "enterprise": 0.06, "regulasi": 0.08, "security": 0.06}
        medium_complexity = {"dashboard": 0.03, "governance": 0.04, "cloud": 0.04, "api": 0.04, "audit": 0.04, "change management": 0.05}

        multiplier = 1.0
        signal_count = 0
        for token, boost in {**high_complexity, **medium_complexity}.items():
            if token in combined:
                multiplier += boost
                signal_count += 1

        framework_tokens = [p.strip() for p in re.split(r"[,;/]| dan ", frameworks or "") if p.strip()]
        if framework_tokens: multiplier += min(0.12, max(0, len(framework_tokens) - 1) * 0.03)

        ai_profile = cls._ai_pricing_profile(project_goal, objective, notes, frameworks)
        if ai_profile["enabled"]:
            multiplier *= ai_profile["multiplier"]
            signal_count += len(ai_profile["drivers"])

        multiplier = max(0.9, min(1.9, multiplier))
        level = "tinggi" if multiplier >= 1.45 else "menengah" if multiplier >= 1.15 else "terkendali"
        return {"level": level, "multiplier": multiplier, "signal_count": signal_count}

    @classmethod
    def _project_effort_baseline(cls, timeline: str, project_type: str, service_type: str, project_goal: str, objective: str, notes: str, frameworks: str) -> Dict[str, Any]:
        months = cls._duration_months_or_default(timeline, project_type)
        monthly_rate = cls.BASE_MONTHLY_DELIVERY_RATE.get((project_type or "").strip().lower(), 70_000_000)
        service_multiplier = {"training": 0.80, "konsultan": 1.00, "training dan konsultan": 1.12}.get((service_type or "").strip().lower(), 1.0)
        complexity = cls._complexity_profile(project_goal, objective, notes, frameworks)
        ai_pricing = cls._ai_pricing_profile(project_goal, objective, notes, frameworks)

        active_dims = sum(1 for item in [objective, notes, project_goal, frameworks] if str(item or "").strip())
        breadth_mult = 1.0 + min(0.18, max(0, active_dims - 2) * 0.04)

        effort_base = monthly_rate * months * service_multiplier * complexity["multiplier"] * breadth_mult
        if ai_pricing["enabled"]: effort_base *= max(1.0, ai_pricing["multiplier"] * 0.92)
        return {
            "months": months, "monthly_rate": int(monthly_rate), "complexity": complexity,
            "ai_pricing": ai_pricing, "effort_base": int(max(120_000_000, min(9_000_000_000, effort_base)))
        }

    @staticmethod
    def _bounded_calibration(value: int, anchor: int, lower_ratio: float, upper_ratio: float) -> int:
        return max(int(anchor * lower_ratio), min(int(anchor * upper_ratio), max(value, anchor)))

    @classmethod
    def _dynamic_budget_from_osint(cls, client_name: str, finance_snippets: List[Dict[str, Any]], benchmark_snippets: List[Dict[str, Any]], timeline: str = "", project_type: str = "", service_type: str = "", project_goal: str = "", objective: str = "", notes: str = "", frameworks: str = "", llm_financial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        finance_text = " ".join([str(i.get("title", "")) + " " + str(i.get("snippet", "")) for i in (finance_snippets or [])])
        benchmark_text = " ".join([str(i.get("title", "")) + " " + str(i.get("snippet", "")) for i in (benchmark_snippets or [])])

        finance_values = sorted(cls._extract_financial_values(finance_text))
        benchmark_values = sorted(cls._extract_financial_values(benchmark_text))
        effort_profile = cls._project_effort_baseline(timeline, project_type, service_type, project_goal, objective, notes, frameworks)
        effort_base = effort_profile["effort_base"]

        llm_rev = (llm_financial_data or {}).get("revenue_idr")
        if llm_rev and isinstance(llm_rev, (int, float)) and llm_rev > 0:
            finance_median = int(llm_rev)
            financial_base = cls._bounded_calibration(int(max(120_000_000, min(6_000_000_000, finance_median * 0.0015))), effort_base, 0.70, 1.80)
        elif finance_values:
            finance_median = finance_values[len(finance_values) // 2]
            financial_base = cls._bounded_calibration(int(max(120_000_000, min(6_000_000_000, finance_median * 0.0015))), effort_base, 0.70, 1.80)
        else:
            finance_median, financial_base = None, effort_base

        benchmark_median = benchmark_values[len(benchmark_values) // 2] if benchmark_values else None
        market_base = cls._bounded_calibration(int(max(80_000_000, min(6_000_000_000, benchmark_median))), effort_base, 0.75, 1.60) if benchmark_median else effort_base

        if finance_median and benchmark_median: base_price = int((effort_base * 0.65) + (market_base * 0.25) + (financial_base * 0.10))
        elif benchmark_median: base_price = int((effort_base * 0.78) + (market_base * 0.22))
        elif finance_median: base_price = int((effort_base * 0.88) + (financial_base * 0.12))
        else: base_price = effort_base

        calibration_cap = max(effort_base, int(financial_base * 1.75)) if finance_median else effort_base
        adjusted_base = int(max(120_000_000, min(9_000_000_000, min(base_price, calibration_cap))))

        basic = int(adjusted_base * 0.72)
        return {
            "analysis": f"Estimasi untuk {client_name} dihitung berdasarkan durasi {effort_profile['months']} bulan dengan tingkat kompleksitas {effort_profile['complexity']['level']}.",
            "options": [
                {"tier": "Basic", "price": cls._format_idr(basic)},
                {"tier": "Standard", "price": cls._format_idr(max(basic + 40_000_000, adjusted_base))},
                {"tier": "Enterprise", "price": cls._format_idr(max(adjusted_base + 80_000_000, int(adjusted_base * 1.65)))},
            ],
        }

    def suggest_budget(self, client_name: str, timeline: str = "", project_type: str = "", service_type: str = "", project_goal: str = "", objective: str = "", notes: str = "", frameworks: str = "", commercial_context: str = "", pricing_mode: str = "demo") -> Dict[str, Any]:
        year = datetime.now().year
        
        # Parallel OSINT Fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_finance = executor.submit(Researcher.search, f'"{client_name}" laporan keuangan OR pendapatan {year-1} OR {year}', 6, "year")
            future_bench = executor.submit(Researcher.search, f'estimasi biaya proyek {project_type} {service_type} {timeline} Indonesia', 6, "year")
        
        finance_snippets = Researcher._filter_recent_entity_results(future_finance.result(), client_name, max_age_years=3)
        benchmark_snippets = Researcher._sort_by_recency(future_bench.result())[:5]
        
        # Deep Scrape Financials
        llm_financial_data = None
        if finance_snippets and finance_snippets[0].get("link"):
            markdown = Researcher.fetch_full_markdown(finance_snippets[0]["link"])
            if markdown: llm_financial_data = self._extract_financials_with_llm(client_name, markdown)
        
        return self._dynamic_budget_from_osint(client_name, finance_snippets, benchmark_snippets, timeline, project_type, service_type, project_goal, objective, notes, frameworks, llm_financial_data)


class AppStateStore:
    def __init__(self, db_path: Optional[Path] = None, asset_root: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or APP_STATE_DB_PATH)
        self.asset_root = Path(asset_root or APP_ASSET_ROOT)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.templates_dir = self.asset_root / "templates"
        self.supporting_docs_dir = self.asset_root / "supporting_documents"
        self.portfolio_docs_dir = self.supporting_docs_dir / "portfolio"
        self.credentials_docs_dir = self.supporting_docs_dir / "credentials"
        self.kak_docs_dir = self.supporting_docs_dir / "kak"
        self.generated_dir = Path(GENERATED_OUTPUT_DIR)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_docs_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_docs_dir.mkdir(parents=True, exist_ok=True)
        self.kak_docs_dir.mkdir(parents=True, exist_ok=True)
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    username_key TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
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
    def _normalize_username(username: str) -> str:
        normalized = re.sub(r"\s+", " ", str(username or "").strip())
        return normalized

    @classmethod
    def _username_key(cls, username: str) -> str:
        return SchemaMapper.normalize_key(cls._normalize_username(username))

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        username_key = self._username_key(username)
        if not username_key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, created_at
                FROM app_users
                WHERE username_key = ?
                """,
                (username_key,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "password_hash": row["password_hash"],
            "created_at": float(row["created_at"] or 0.0),
        }

    def create_user(self, username: str, password_hash: str) -> bool:
        normalized_username = self._normalize_username(username)
        username_key = self._username_key(normalized_username)
        if not normalized_username or not username_key or not password_hash:
            return False
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO app_users(id, username, username_key, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        normalized_username,
                        username_key,
                        str(password_hash),
                        time.time(),
                    ),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

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
            "kak": "kak",
            "tor": "kak",
            "kerangka_acuan_kerja": "kak",
            "kerangka_acuan": "kak",
        }
        if normalized not in aliases:
            raise ValueError("Jenis dokumen pendukung tidak dikenal.")
        return aliases[normalized]

    def _supporting_dir_for_type(self, document_type: str) -> Path:
        normalized = self._normalize_document_type(document_type)
        if normalized == "portfolio":
            return self.portfolio_docs_dir
        if normalized == "credentials":
            return self.credentials_docs_dir
        return self.kak_docs_dir

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

    @staticmethod
    def _first_non_empty_match(text: str, patterns: List[str], flags: int = re.IGNORECASE) -> str:
        source = str(text or "")
        for pattern in patterns:
            match = re.search(pattern, source, flags)
            if match:
                value = re.sub(r"\s+", " ", str(match.group(1) or "")).strip(" -:;,.")
                if value:
                    return value
        return ""

    @staticmethod
    def _best_money_signal(text: str) -> str:
        matches = re.findall(
            r"(Rp\.?\s?[\d.,]+(?:\s?(?:miliar|juta|triliun))?)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        cleaned = []
        for match in matches:
            value = re.sub(r"\s+", " ", match).strip()
            if value and value.lower() not in {item.lower() for item in cleaned}:
                cleaned.append(value)
        return cleaned[0] if cleaned else ""

    @staticmethod
    def _best_duration_signal(text: str) -> str:
        patterns = [
            r"(?:jangka waktu(?:\s+pelaksanaan)?|durasi|masa pelaksanaan|waktu pelaksanaan)\s*[:\-]?\s*([^\n.;]{3,60})",
            r"(?:selama|dalam kurun waktu)\s*[:\-]?\s*(\d+\s*(?:hari|minggu|bulan|tahun)(?:\s*(?:kalender|kerja))?)",
            r"(\d+\s*(?:hari|minggu|bulan|tahun)(?:\s*(?:kalender|kerja))?)",
        ]
        value = AppStateStore._first_non_empty_match(text, patterns)
        value = re.sub(r"^(?:pelaksanaan|jangka waktu|durasi)\s*[:\-]?\s*", "", value, flags=re.IGNORECASE)
        return value.strip(" :;-.,")

    @staticmethod
    def _extract_first_section(text: str, labels: List[str], max_lines: int = 3) -> str:
        source = str(text or "")
        label_blob = "|".join(re.escape(item) for item in labels if item)
        if not label_blob:
            return ""
        pattern = re.compile(
            rf"(?:^|\n)\s*(?:{label_blob})\s*[:\-]?\s*(.+?)(?=\n\s*[A-Z][^\n]{{0,80}}[:\-]|\n\s*\d+[\).\s]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(source)
        if not match:
            return ""
        raw = re.sub(r"\n{2,}", "\n", str(match.group(1) or "")).strip()
        lines = [re.sub(r"\s+", " ", line).strip(" -•\t") for line in raw.splitlines() if re.sub(r"\s+", " ", line).strip()]
        if not lines:
            return ""
        return " ".join(lines[:max_lines]).strip()

    @staticmethod
    def _extract_kak_frameworks(text: str) -> List[str]:
        source = str(text or "")
        known = [
            ("ITIL", r"\bitil\b"),
            ("TOGAF", r"\btogaf\b"),
            ("COBIT", r"\bcobit\b"),
            ("ISO 27001", r"\biso\s*27001\b"),
            ("ISO 20000", r"\biso\s*20000\b"),
            ("ISO 9001", r"\biso\s*9001\b"),
            ("NIST", r"\bnist\b"),
            ("POJK", r"\bpojk\b"),
            ("OJK", r"\bojk\b"),
            ("UU PDP", r"\bpdp\b"),
            ("DAMA", r"\bdama\b"),
            ("TM Forum", r"\btm\s*forum\b"),
            ("Responsible AI", r"\bresponsible ai\b|\bai governance\b|\bai rmf\b"),
            ("Regulasi", r"\bregulasi\b|\bkepatuhan\b"),
        ]
        hits: List[str] = []
        for label, pattern in known:
            if re.search(pattern, source, re.IGNORECASE):
                hits.append(label)
        return hits[:6]

    @classmethod
    def _detect_service_type(cls, text: str) -> str:
        source = str(text or "").lower()
        has_explicit_training = any(
            token in source for token in ["pelatihan", "training", "bimbingan teknis", "bootcamp", "kelas", "kurikulum"]
        )
        has_workshop = "workshop" in source
        has_consulting = any(
            token in source
            for token in [
                "konsultan", "konsultansi", "pendampingan", "assessment",
                "kajian", "review", "penyusunan", "roadmap", "tata kelola",
            ]
        )
        if has_consulting and has_explicit_training:
            return "Training dan Konsultan"
        if has_consulting:
            return "Konsultan"
        if has_explicit_training or has_workshop:
            return "Training"
        return "Konsultan"

    @classmethod
    def _detect_project_type(cls, text: str) -> str:
        source = str(text or "").lower()
        scores = {
            "Diagnostic": 0,
            "Strategic": 0,
            "Transformation": 0,
            "Implementation": 0,
        }
        keyword_map = {
            "Diagnostic": ["assessment", "as is", "gap analysis", "diagnostic", "evaluasi", "kajian awal", "baseline"],
            "Strategic": ["roadmap", "strategi", "blueprint", "target operating model", "arah kebijakan", "rencana strategis", "tata kelola", "governance", "operating cadence", "decision forum"],
            "Transformation": ["transformasi", "redesign", "perubahan", "operating model", "change management"],
            "Implementation": ["implementasi", "deployment", "rollout", "go-live", "uat", "konfigurasi", "pendampingan pelaksanaan"],
        }
        for project_type, keywords in keyword_map.items():
            scores[project_type] = sum(1 for keyword in keywords if keyword in source)
        priority = {"Strategic": 4, "Transformation": 3, "Implementation": 2, "Diagnostic": 1}
        best_type = max(scores.items(), key=lambda item: (item[1], priority.get(item[0], 0)))[0]
        return best_type if scores[best_type] > 0 else "Implementation"

    @classmethod
    def _detect_need_classification(cls, text: str) -> str:
        source = str(text or "").lower()
        needs: List[str] = []
        if any(token in source for token in ["masalah", "kendala", "hambatan", "gap", "isu", "belum", "tidak sinkron"]):
            needs.append("Problem")
        if any(token in source for token in ["optimalisasi", "peningkatan", "peluang", "opportunity", "improvement"]):
            needs.append("Opportunity")
        if any(token in source for token in ["wajib", "ketentuan", "kepatuhan", "regulasi", "mandat", "pojk", "ojk", "peraturan"]):
            needs.append("Directive")
        if not needs:
            needs.append("Problem")
        order = ["Problem", "Opportunity", "Directive"]
        return ", ".join(item for item in order if item in needs)

    @classmethod
    def _infer_company_from_kak(cls, text: str, company_candidates: Optional[List[str]] = None) -> str:
        source = str(text or "")
        candidates = [str(item).strip() for item in (company_candidates or []) if str(item).strip()]
        for candidate in sorted(candidates, key=len, reverse=True):
            if re.search(re.escape(candidate), source, re.IGNORECASE):
                return candidate
        guessed = cls._first_non_empty_match(
            source,
            [
                r"(?:nama perusahaan|instansi|satuan kerja|klien|pemberi kerja)\s*[:\-]\s*([^\n]{3,120})",
                r"\b(PT\.?\s+[A-Z][^\n]{3,80})",
            ],
        )
        return guessed

    @classmethod
    def _summarize_kak_analysis(cls, suggestions: Dict[str, str], source_name: str) -> str:
        bits = []
        if suggestions.get("konteks_organisasi"):
            bits.append(f"inisiatif {suggestions['konteks_organisasi']}")
        if suggestions.get("jenis_proposal"):
            bits.append(f"layanan {suggestions['jenis_proposal']}")
        if suggestions.get("jenis_proyek"):
            bits.append(f"tipe proyek {suggestions['jenis_proyek']}")
        if suggestions.get("estimasi_waktu"):
            bits.append(f"durasi {suggestions['estimasi_waktu']}")
        if suggestions.get("estimasi_biaya"):
            bits.append(f"anggaran {suggestions['estimasi_biaya']}")
        if suggestions.get("potensi_framework"):
            bits.append(f"framework {suggestions['potensi_framework']}")
        joined = "; ".join(bits) if bits else "belum ada field utama yang berhasil dibaca"
        return f"Pembacaan KAK dari {source_name} menangkap {joined}."

    def get_latest_kak_context(
        self,
        company_candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        documents = self.list_supporting_documents("kak")
        result: Dict[str, Any] = {
            "documents": documents,
            "analysis": {
                "available": False,
                "source_document": "",
                "summary": "Belum ada dokumen KAK/TOR yang dibaca.",
                "suggestions": {},
                "warnings": [],
            },
        }
        if not documents:
            return result

        latest = documents[0]
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT extracted_text
                FROM supporting_documents
                WHERE id = ?
                """,
                (str(latest.get("id") or ""),),
            ).fetchone()
        text = str((row["extracted_text"] if row else "") or "").strip()
        if not text:
            result["analysis"] = {
                "available": False,
                "source_document": latest.get("original_name", ""),
                "summary": "Dokumen KAK/TOR sudah diunggah, tetapi teksnya belum berhasil dibaca.",
                "suggestions": {},
                "warnings": ["Teks KAK belum bisa diekstrak dari file yang diunggah."],
            }
            return result

        objective = self._extract_first_section(
            text,
            ["maksud dan tujuan", "tujuan", "objective", "sasaran"],
            max_lines=3,
        )
        initiative = self._first_non_empty_match(
            text,
            [
                r"(?:nama|judul|paket|objek)\s*(?:pekerjaan|pengadaan|jasa)?\s*[:\-]\s*([^\n]{6,180})",
                r"(?:pekerjaan|kegiatan)\s*[:\-]\s*([^\n]{6,180})",
            ],
        )
        if not initiative:
            lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if re.sub(r"\s+", " ", line).strip()]
            for line in lines[:12]:
                if 5 <= len(line.split()) <= 22 and not re.search(r"^(bab|pasal|latar belakang|maksud dan tujuan|ruang lingkup)\b", line, re.IGNORECASE):
                    initiative = line
                    break
        context_text = " ".join(item for item in [initiative, objective] if item).strip() or initiative or objective
        problem_excerpt = self._extract_first_section(
            text,
            ["latar belakang", "permasalahan", "isu utama", "tantangan", "kondisi eksisting"],
            max_lines=4,
        )
        budget = self._best_money_signal(text)
        duration = self._best_duration_signal(text)
        frameworks = self._extract_kak_frameworks(text)
        service_type = self._detect_service_type(text)
        project_type = self._detect_project_type(text)
        need_classification = self._detect_need_classification(text)
        company = self._infer_company_from_kak(text, company_candidates=company_candidates)
        suggestion_warnings: List[str] = []
        if not budget:
            suggestion_warnings.append("Nilai anggaran tidak terbaca jelas dari KAK.")
        if not duration:
            suggestion_warnings.append("Durasi pelaksanaan tidak terbaca jelas dari KAK.")
        if not frameworks:
            suggestion_warnings.append("Framework/acuan belum terbaca eksplisit; user mungkin masih perlu mengisi manual.")

        suggestions = {
            "nama_perusahaan": company,
            "jenis_proposal": service_type,
            "jenis_proyek": project_type,
            "konteks_organisasi": context_text,
            "permasalahan": problem_excerpt,
            "klasifikasi_kebutuhan": need_classification,
            "estimasi_waktu": duration,
            "estimasi_biaya": budget,
            "potensi_framework": ", ".join(frameworks),
        }
        suggestions = {key: value for key, value in suggestions.items() if str(value or "").strip()}

        result["analysis"] = {
            "available": True,
            "source_document": latest.get("original_name", ""),
            "summary": self._summarize_kak_analysis(suggestions, latest.get("original_name", "dokumen KAK")),
            "suggestions": suggestions,
            "warnings": suggestion_warnings,
            "initiative_title": initiative,
            "objective_excerpt": objective,
            "problem_excerpt": problem_excerpt,
        }
        return result

    @classmethod
    def _fallback_portfolio_rows_from_text(cls, text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for raw_line in str(text or "").splitlines():
            line = re.sub(r"^\s*[-*•\d.]+\s*", "", raw_line).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line.split()) < 4:
                continue
            if re.search(r"\b(belum pernah|tidak pernah|n/a|none)\b", line, re.IGNORECASE):
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
                    "bukti": "ringkasan portofolio internal dan pengalaman sejenis perusahaan penyusun",
                    "nilai_tambah": "memperkuat kredibilitas proposal dan membantu klien melihat bukti kemampuan secara lebih konkret",
                }
            )
            if len(rows) >= 4:
                break
        return rows

    @staticmethod
    def _dedupe_phrases(items: List[str], limit: int = 6) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for item in items:
            clean = re.sub(r"\s+", " ", str(item or "")).strip(" -;|,.:")
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(clean)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _summarize_profile_blob(text: str, fallback: str, max_items: int = 4) -> str:
        source = str(text or "")
        if not source.strip():
            return fallback
        normalized = source.replace("|", "\n").replace(";", "\n")
        candidates: List[str] = []
        for raw_line in normalized.splitlines():
            line = re.sub(r"^\s*[-*•\d.]+\s*", "", raw_line).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line.split()) < 4:
                continue
            if re.search(
                r"\b(no|nama tenaga|posisi diusulkan|tingkat pendidikan|lama pengalaman|peran dalam penugasan|tahun)\b",
                line,
                re.IGNORECASE,
            ):
                continue
            if re.search(r"\b(belum pernah|tidak pernah|n/a|none)\b", line, re.IGNORECASE):
                continue
            candidates.append(line[:180])
        picked = AppStateStore._dedupe_phrases(candidates, limit=max_items)
        if not picked:
            return fallback
        return "; ".join(picked)

    @classmethod
    def _summarize_credential_blob(cls, text: str, fallback: str) -> str:
        source = str(text or "")
        if not source.strip():
            return fallback
        certifications = re.findall(
            r"\b(?:TOGAF|COBIT(?:\s*\d+)?|ITIL(?:\s*[A-Za-z0-9.+-]+)?|ISO\s*\/?\s*IEC?\s*\d+|ISO\s*\d+|CEH|CISA|CHFI|CCNA|CAPM|Project\+|Lead Auditor ISO 27001|Microsoft Certified Database Administrator)\b",
            source,
            flags=re.IGNORECASE,
        )
        certification_list = cls._dedupe_phrases(certifications, limit=6)

        role_signals: List[str] = []
        role_map = [
            (r"\bproject manager\b|\bpmo\b", "project management"),
            (r"\bsecurity\b|\bethical hacker\b|\bincident handler\b|\bforensic\b", "cyber security"),
            (r"\bgovernance\b|\bcobit\b|\biso\b", "IT governance"),
            (r"\bnetwork\b|\bccna\b", "network & infrastructure"),
            (r"\bprivacy\b|\bpdp\b|\bropa\b|\bdpia\b", "data privacy & compliance"),
            (r"\btechnical writer\b|\bdokumentasi\b", "documentation support"),
        ]
        for pattern, label in role_map:
            if re.search(pattern, source, re.IGNORECASE):
                role_signals.append(label)
        role_list = cls._dedupe_phrases(role_signals, limit=4)

        parts: List[str] = []
        if role_list:
            parts.append(f"Kapabilitas tim mencakup {', '.join(role_list)}")
        if certification_list:
            parts.append(f"Sertifikasi inti meliputi {', '.join(certification_list)}")
        if parts:
            return ". ".join(parts) + "."
        return cls._summarize_profile_blob(source, fallback, max_items=3)

    @classmethod
    def _summarize_portfolio_blob(cls, text: str, fallback: str) -> str:
        source = str(text or "")
        if not source.strip():
            return fallback
        if "|" in source:
            rows = cls._parse_structured_portfolio_rows(source)
            areas = cls._dedupe_phrases([row.get("area", "") for row in rows], limit=4)
            if areas:
                return f"Pengalaman perusahaan mencakup {', '.join(areas)}."
        return cls._summarize_profile_blob(source, fallback, max_items=3)

    def get_settings(self) -> Dict[str, Any]:
        template_path = self.get_template_path()
        return {
            "internal_portfolio": self._get_setting("internal_portfolio"),
            "internal_credentials": self._get_setting("internal_credentials"),
            "active_template_name": self._get_setting("active_template_name"),
            "has_active_template": bool(template_path and Path(template_path).exists()),
            "portfolio_documents": self.list_supporting_documents("portfolio"),
            "credential_documents": self.list_supporting_documents("credentials"),
            "kak_documents": self.list_supporting_documents("kak"),
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
            profile["portfolio_highlights"] = self._summarize_portfolio_blob(
                portfolio_text,
                str(profile.get("portfolio_highlights") or "").strip() or "Pengalaman perusahaan disesuaikan dengan kebutuhan proyek klien.",
            )
        credential_bits = [
            str(profile.get("credential_highlights") or "").strip(),
            internal_credentials.strip(),
            credential_docs_text.strip(),
        ]
        credential_text = " ; ".join([item for item in credential_bits if item])
        if credential_text:
            profile["credential_highlights"] = self._summarize_credential_blob(
                credential_text,
                str(profile.get("credential_highlights") or "").strip() or "Kapabilitas inti dan sertifikasi relevan perusahaan penyusun.",
            )
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
        canvas = Image.new('RGBA', (420, 180), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 88)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), initials, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(initials, font=font)
        draw.text(((420 - w) / 2, (180 - h) / 2 - 4), initials, fill=(15, 23, 42, 255), font=font)
        out = io.BytesIO()
        canvas.save(out, format='PNG')
        out.seek(0)
        return out

    @staticmethod
    def _crop_transparent_bounds(img: Image.Image, padding: int = 4) -> Image.Image:
        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        if not bbox:
            return img
        left, top, right, bottom = bbox
        return img.crop((
            max(0, left - padding),
            max(0, top - padding),
            min(img.width, right + padding),
            min(img.height, bottom + padding),
        ))

    @staticmethod
    def _normalize_logo_image(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int]]:
        rgba = img.convert("RGBA")
        alpha_min, alpha_max = rgba.getchannel("A").getextrema()

        if alpha_min < 250:
            normalized = LogoManager._crop_transparent_bounds(rgba)
        else:
            width, height = rgba.size
            border_pixels: List[Tuple[int, int, int]] = []
            step_x = max(1, width // 24)
            step_y = max(1, height // 24)
            for x in range(0, width, step_x):
                border_pixels.append(rgba.getpixel((x, 0))[:3])
                border_pixels.append(rgba.getpixel((x, height - 1))[:3])
            for y in range(0, height, step_y):
                border_pixels.append(rgba.getpixel((0, y))[:3])
                border_pixels.append(rgba.getpixel((width - 1, y))[:3])

            if border_pixels:
                background = tuple(
                    int(sum(pixel[idx] for pixel in border_pixels) / len(border_pixels))
                    for idx in range(3)
                )
                max_border_delta = max(
                    max(abs(pixel[idx] - background[idx]) for idx in range(3))
                    for pixel in border_pixels
                )
            else:
                background = (255, 255, 255)
                max_border_delta = 255

            normalized = rgba
            if max_border_delta <= 28:
                tolerance = 34
                transparent_data = []
                for r, g, b, a in rgba.getdata():
                    if a <= 8:
                        transparent_data.append((r, g, b, 0))
                        continue
                    if max(abs(r - background[0]), abs(g - background[1]), abs(b - background[2])) <= tolerance:
                        transparent_data.append((r, g, b, 0))
                    else:
                        transparent_data.append((r, g, b, a))
                candidate = Image.new("RGBA", rgba.size)
                candidate.putdata(transparent_data)
                cropped = LogoManager._crop_transparent_bounds(candidate)
                if cropped.getchannel("A").getbbox():
                    normalized = cropped

        visible_pixels = [pixel[:3] for pixel in normalized.getdata() if pixel[3] > 32]
        if visible_pixels:
            dominant = tuple(
                int(sum(pixel[idx] for pixel in visible_pixels) / len(visible_pixels))
                for idx in range(3)
            )
        else:
            dominant = tuple(int(value) for value in ImageStat.Stat(normalized.convert("RGB")).mean[:3])

        return normalized, dominant

    @staticmethod
    def _border_opacity_ratio(img: Image.Image) -> float:
        rgba = img.convert("RGBA")
        width, height = rgba.size
        if width <= 1 or height <= 1:
            return 1.0

        border_coords = set()
        for x in range(width):
            border_coords.add((x, 0))
            border_coords.add((x, height - 1))
        for y in range(height):
            border_coords.add((0, y))
            border_coords.add((width - 1, y))

        opaque_count = 0
        for x, y in border_coords:
            if rgba.getpixel((x, y))[3] > 32:
                opaque_count += 1
        return opaque_count / max(1, len(border_coords))

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
                best_candidate: Optional[Tuple[float, io.BytesIO, Tuple[int, int, int]]] = None
                for item in res['images']:
                    try:
                        img_resp = requests.get(item['imageUrl'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                        if img_resp.status_code == 200:
                            stream = io.BytesIO(img_resp.content)
                            img = Image.open(stream)
                            img, dom_color = LogoManager._normalize_logo_image(img)
                            img.thumbnail((600, 240))
                            border_ratio = LogoManager._border_opacity_ratio(img)
                            if border_ratio > 0.18:
                                continue

                            luminance = 0.299 * dom_color[0] + 0.587 * dom_color[1] + 0.114 * dom_color[2]
                            if luminance > 120:
                                factor = 120 / luminance
                                dom_color = [max(0, min(255, int(c * factor))) for c in dom_color]

                            png_stream = io.BytesIO()
                            img.save(png_stream, format='PNG')
                            png_stream.seek(0)
                            score = border_ratio + (0.01 * abs((img.width / max(1, img.height)) - 3.0))
                            candidate = (score, png_stream, tuple(dom_color))
                            if best_candidate is None or candidate[0] < best_candidate[0]:
                                best_candidate = candidate
                    except Exception:
                        continue
                if best_candidate is not None:
                    return best_candidate[1], best_candidate[2]
        except Exception as e:
            logger.warning(f"Logo Retrieval Error: {e}")
        return LogoManager._create_fallback_logo(client_name), DEFAULT_COLOR


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
    
    @staticmethod
    def apply_enhanced_styles(doc: Document, theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        StyleEngine._apply_base_enhanced_styles(doc)
        StyleEngine._apply_enhanced_heading_styles(doc, theme_color)
        StyleEngine._apply_enhanced_list_styles(doc)
        StyleEngine._apply_enhanced_table_styles(doc, theme_color)
    
    @staticmethod
    def _apply_base_enhanced_styles(doc: Document) -> None:
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
        
        for section in doc.sections:
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)
    
    @staticmethod
    def _apply_enhanced_heading_styles(doc: Document, theme_color: Tuple[int, int, int]) -> None:
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
        try:
            style = doc.styles['List Bullet']
            pf = style.paragraph_format
            pf.space_after = Pt(4)
            pf.space_before = Pt(0)
            pf.left_indent = Inches(0.25)
            pf.first_line_indent = Inches(-0.25)
        except Exception as e:
            logger.warning(f"Could not apply List Bullet style: {e}")
        
        try:
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
        try:
            style = doc.styles['Table Grid']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.TABLE_FONT_SIZE)
        except Exception as e:
            logger.warning(f"Could not apply Table Grid style: {e}")
    
    @staticmethod
    def add_colored_heading(doc: Document, text: str, level: int = 1, 
                           theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
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
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.name = StyleEngine.PROFESSIONAL_FONT
        run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
        
        p = doc.add_paragraph(content)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(6)
    
    @staticmethod
    def format_contact_block(doc: Document, contact_lines: list, 
                           theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(4)
        
        run = p.add_run("Kontak Resmi")
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.name = StyleEngine.PROFESSIONAL_FONT
        run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
        
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
        is_first_chapter: bool = False,
    ) -> None:
        try:
            heading = doc.add_paragraph(style="Heading 1")
        except KeyError:
            heading = doc.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading.paragraph_format.space_before = Pt(0)
        heading.paragraph_format.space_after = Pt(10)
        heading.paragraph_format.keep_with_next = True
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
                direct_items = element.find_all('li', recursive=False)
                style_name = "List Number" if element.name == 'ol' else "List Bullet"
                list_num_id = DocumentBuilder._create_list_num_id(doc, style_name)
                for idx, li in enumerate(direct_items, start=1):
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
    def _paragraph_is_page_break_only(paragraph_el) -> bool:
        if paragraph_el.find('.//' + qn('w:drawing')) is not None:
            return False
        texts = [node.text or "" for node in paragraph_el.findall('.//' + qn('w:t'))]
        if "".join(texts).strip():
            return False
        breaks = paragraph_el.findall('.//' + qn('w:br'))
        return bool(breaks) and all(br.get(qn('w:type')) == 'page' for br in breaks)

    @staticmethod
    def compact_layout(doc: Document) -> None:
        body = doc._element.body
        for child in list(body):
            if child.tag == qn('w:p') and DocumentBuilder._paragraph_is_blank(child):
                body.remove(child)

        children = list(body)
        for idx, child in enumerate(children):
            if child.tag != qn('w:p') or not DocumentBuilder._paragraph_is_page_break_only(child):
                continue

            previous_substantive = None
            for prev in reversed(children[:idx]):
                if prev.tag == qn('w:p') and DocumentBuilder._paragraph_is_blank(prev):
                    continue
                previous_substantive = prev
                break

            next_substantive = None
            for nxt in children[idx + 1:]:
                if nxt.tag == qn('w:p') and DocumentBuilder._paragraph_is_blank(nxt):
                    continue
                next_substantive = nxt
                break

            previous_is_page_break = bool(
                previous_substantive is not None
                and previous_substantive.tag == qn('w:p')
                and DocumentBuilder._paragraph_is_page_break_only(previous_substantive)
            )
            next_is_page_break = bool(
                next_substantive is not None
                and next_substantive.tag == qn('w:p')
                and DocumentBuilder._paragraph_is_page_break_only(next_substantive)
            )
            next_is_real_content = bool(
                next_substantive is not None
                and not next_is_page_break
                and (
                    next_substantive.tag != qn('w:p')
                    or not DocumentBuilder._paragraph_is_blank(next_substantive)
                )
            )

            if previous_is_page_break or next_is_page_break or not next_is_real_content:
                body.remove(child)
