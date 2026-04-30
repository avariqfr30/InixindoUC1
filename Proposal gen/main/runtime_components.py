"""Runtime components for internal data access and knowledge-base retrieval."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple

from .proposal_shared import *
from .schema_mapping import SchemaMapper
from .research import Researcher
from ollama import Client

# ==========================================
# PYDANTIC SCHEMAS FOR BULLETPROOF LLM DATA
# ==========================================
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
    mode: str = Field("", description="Normalized relationship mode such as existing or new.")


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

# Internal API adapter.
class FirmAPIClient:
    REQUIRED_RESOURCE_FIELDS: Dict[str, Tuple[str, ...]] = {
        "firm_profile": ("office_address", "email", "phone", "website"),
        "project_standards": ("methodology", "team", "commercial"),
        "client_relationship": ("summary",),
        "project_records": ("entity", "topic"),
    }

    def __init__(self, force_source: str = "") -> None:
        normalized_force_source = str(force_source or "").strip().lower()
        if normalized_force_source not in {"demo", "api"}:
            normalized_force_source = INTERNAL_DATA_SOURCE
        self.app_profile = APP_PROFILE
        self.internal_data_source = normalized_force_source
        self.demo_mode = normalized_force_source == "demo"
        self.data_acquisition_mode = "demo" if self.demo_mode else "staged"
        self.base_url = FIRM_API_URL.rstrip("/")
        self.timeout_seconds = FIRM_API_TIMEOUT_SECONDS
        self.config_file = self._resolve_config_file()
        runtime_config = self._load_json_config(self.config_file)
        self.integration_mode = str(runtime_config.get("mode") or FIRM_API_INTEGRATION_MODE or "rest").strip().lower()
        if self.integration_mode not in {"rest", "dataset", "generic"}:
            self.integration_mode = "rest"
        self.auth_mode = str(runtime_config.get("auth_mode") or FIRM_API_AUTH_MODE or "bearer").strip().lower()
        self.request_defaults = self._merge_spec(
            dict(FIRM_API_REQUEST_DEFAULTS or {}),
            runtime_config.get("request_defaults") if isinstance(runtime_config.get("request_defaults"), dict) else {},
        )
        self.endpoint_config = dict(FIRM_API_ENDPOINT_CONFIG or {})
        self.dataset_config = dict(FIRM_API_DATASET_CONFIG or {})
        self.resource_config = self._merge_runtime_resource_config(
            dict(FIRM_API_RESOURCE_CONFIG or {}),
            runtime_config.get("resources") if isinstance(runtime_config.get("resources"), dict) else {},
        )
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        if self.auth_mode == "basic":
            if FIRM_API_USERNAME and FIRM_API_PASSWORD:
                self.session.auth = (FIRM_API_USERNAME, FIRM_API_PASSWORD)
        elif self.auth_mode == "bearer":
            token = str(API_AUTH_TOKEN or "").strip()
            if token and token != "isi_token_disini_nanti":
                self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif self.auth_mode == "none":
            pass
        else:
            logger.warning("Unknown FIRM_API_AUTH_MODE=%s. Falling back to unauthenticated requests.", self.auth_mode)

    @staticmethod
    def _load_json_config(path_value: str) -> Dict[str, Any]:
        path = Path(str(path_value or "").strip()).expanduser()
        if not path or not path.exists() or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _resolve_config_file() -> str:
        explicit = str(FIRM_API_CONFIG_FILE or "").strip()
        if explicit:
            return explicit
        if MANAGED_INTERNAL_API_CONFIG_PATH.exists():
            return str(MANAGED_INTERNAL_API_CONFIG_PATH)
        return ""

    @classmethod
    def _merge_runtime_resource_config(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base or {})
        neutral_resource = {
            "request": {},
            "response_path": "",
            "record_filters": {},
            "allow_llm_extract": True,
        }
        for resource_name, resource_spec in (override or {}).items():
            if not isinstance(resource_spec, dict):
                continue
            merged[resource_name] = cls._merge_spec(neutral_resource, resource_spec)
        return merged

    @staticmethod
    def _extract_with_jmespath(payload: Any, path: str) -> Any:
        try:
            import jmespath
            return jmespath.search(path, payload)
        except ImportError:
            current = payload
            for part in str(path or "").split("."):
                if not part: continue
                if isinstance(current, list):
                    if not part.isdigit(): return None
                    idx = int(part)
                    if idx < 0 or idx >= len(current): return None
                    current = current[idx]
                elif isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            return current

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
            return {k: cls._render_template_payload(v, context) for k, v in payload.items()}
        if isinstance(payload, list):
            return [cls._render_template_payload(v, context) for v in payload]
        return cls._render_template_value(payload, context)

    @classmethod
    def _merge_spec(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(base or {})
        for key, value in (override or {}).items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = cls._merge_spec(current, value)
            else:
                merged[key] = value
        return merged

    def _request_from_spec(self, spec: Dict[str, Any], **context: Any) -> Any:
        request_spec = self._merge_spec(self.request_defaults, dict(spec.get("request") or {}))
        path = self._render_template_value(request_spec.get("url") or request_spec.get("path") or "", context)
        method = str(request_spec.get("method") or "GET").strip().upper() or "GET"
        params = self._render_template_payload(request_spec.get("params") or {}, context)
        body = self._render_template_payload(request_spec.get("body") or {}, context)
        headers = self._render_template_payload(request_spec.get("headers") or {}, context)

        url = str(path or "").strip()
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            url = f"{self.base_url}/{url.lstrip('/')}"

        kwargs = {"params": params, "timeout": self.timeout_seconds}
        if isinstance(headers, dict) and headers:
            kwargs["headers"] = headers
        if method != "GET":
            encoding = str(request_spec.get("body_encoding") or "json").strip().lower()
            if encoding == "form":
                kwargs["data"] = body or {}
            else:
                kwargs["json"] = body or {}

        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _flatten_json_paths(payload: Any, prefix: str = "", limit: int = 400) -> List[str]:
        paths: List[str] = []

        def walk(value: Any, current: str) -> None:
            if len(paths) >= limit:
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    key_text = str(key)
                    next_path = f"{current}.{key_text}" if current else key_text
                    paths.append(next_path)
                    walk(child, next_path)
            elif isinstance(value, list):
                if not value:
                    return
                next_path = f"{current}.0" if current else "0"
                paths.append(next_path)
                walk(value[0], next_path)

        walk(payload, prefix)
        return paths[:limit]

    @classmethod
    def _missing_required_mapping_fields(cls, resource_name: str, field_mapping: Dict[str, Any]) -> List[str]:
        required = cls.REQUIRED_RESOURCE_FIELDS.get(resource_name, ())
        return [
            field
            for field in required
            if not str((field_mapping or {}).get(field) or "").strip()
        ]

    def validate_config(self, sample_payload: Optional[Any] = None) -> Dict[str, Any]:
        resources: Dict[str, Any] = {}
        resource_names = ["firm_profile", "project_standards", "client_relationship"]
        if PROJECT_DATA_SOURCE == "api" or "project_records" in (self.resource_config or {}):
            resource_names.append("project_records")
        for resource_name in resource_names:
            resource_spec = dict((self.resource_config or {}).get(resource_name) or {})
            request_spec = self._merge_spec(self.request_defaults, dict(resource_spec.get("request") or {}))
            field_mapping = resource_spec.get("field_mapping") or {}
            path = str(request_spec.get("url") or request_spec.get("path") or "").strip()
            method = str(request_spec.get("method") or "GET").strip().upper() or "GET"
            missing_mapping = self._missing_required_mapping_fields(resource_name, field_mapping)
            resources[resource_name] = {
                "ok": bool(path) and not missing_mapping,
                "method": method,
                "path": path,
                "response_path": str(resource_spec.get("response_path") or "").strip(),
                "mapped_fields": sorted([str(key) for key in field_mapping.keys()]),
                "missing_required_mapping": missing_mapping,
                "allow_llm_extract": bool(resource_spec.get("allow_llm_extract", True)),
            }

        result: Dict[str, Any] = {
            "ok": all(bool(item.get("ok")) for item in resources.values()),
            "mode": self.integration_mode,
            "base_url": self.base_url,
            "auth_mode": self.auth_mode,
            "resources": resources,
        }
        if sample_payload is not None:
            result["sample_paths"] = self._flatten_json_paths(sample_payload)
        return result

    def _resolve_resource_payload(self, resource_name: str, **context: Any) -> Any:
        resource_spec = dict((self.resource_config or {}).get(resource_name) or {})
        if not resource_spec:
            return {}

        payload = self._request_from_spec(resource_spec, **context)

        response_path = self._render_template_value(resource_spec.get("response_path") or "", context)
        if response_path:
            payload = self._extract_with_jmespath(payload, response_path)

        filters = self._render_template_payload(resource_spec.get("record_filters") or {}, context)
        if isinstance(payload, list) and filters:
            matches = []
            for item in payload:
                if isinstance(item, dict):
                    match = True
                    for k, expected in filters.items():
                        field_name = str(k)
                        operator = "eq"
                        if "__" in field_name:
                            field_name, operator = field_name.rsplit("__", 1)
                        actual = item.get(field_name)
                        actual_text = str(actual or "").strip().lower()
                        expected_text = str(expected or "").strip().lower()
                        if operator in {"icontains", "contains"}:
                            if expected_text not in actual_text:
                                match = False
                                break
                            continue
                        if actual_text != expected_text:
                            match = False
                            break
                    if match:
                        matches.append(item)
            payload = matches[0] if len(matches) == 1 else matches

        field_mapping = resource_spec.get("field_mapping")
        if not field_mapping:
            return payload

        extracted = {}
        target_payload = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(target_payload, dict):
            return {}

        for target_field, json_path in field_mapping.items():
            extracted[target_field] = self._extract_with_jmespath(target_payload, str(json_path))

        return extracted

    @classmethod
    def _coerce_record_list(cls, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("dataset_result", "records", "items", "results", "data"):
                value = payload.get(key)
                records = cls._coerce_record_list(value)
                if records:
                    return records
            return [payload]
        return []

    def get_project_records(self) -> List[Dict[str, Any]]:
        if self.demo_mode:
            return []
        resource_spec = dict((self.resource_config or {}).get("project_records") or {})
        if not resource_spec:
            raise ValueError("project_records resource is not configured for API-backed project data.")

        payload = self._request_from_spec(resource_spec)
        response_path = str(resource_spec.get("response_path") or "").strip()
        if response_path:
            payload = self._extract_with_jmespath(payload, response_path)

        records = self._coerce_record_list(payload)
        field_mapping = resource_spec.get("field_mapping") or {}
        if field_mapping:
            mapped_records: List[Dict[str, Any]] = []
            for record in records:
                mapped_record = {
                    target_field: self._extract_with_jmespath(record, str(json_path))
                    for target_field, json_path in field_mapping.items()
                }
                mapped_records.append(mapped_record)
            records = mapped_records

        return [SchemaMapper.normalize_record(record, PROJECT_DATA_FIELD_ALIASES) for record in records]

    def uses_demo_logic(self) -> bool:
        return self.demo_mode or self.data_acquisition_mode == "demo"

    def describe_runtime(self) -> Dict[str, Any]:
        return {
            "provider": "api",
            "app_profile": self.app_profile,
            "internal_data_source": self.internal_data_source,
            "integration_mode": self.integration_mode,
            "auth_mode": self.auth_mode,
            "base_url": self.base_url,
            "config_file": self.config_file,
            "configured_resources": sorted((self.resource_config or {}).keys()),
        }

    @staticmethod
    def _normalize_project_standards(payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
        raw = payload or {}
        return {
            "methodology": str(raw.get("methodology") or "").strip() or "TBD",
            "team": str(raw.get("team") or "").strip() or "TBD",
            "commercial": str(raw.get("commercial") or "").strip() or "TBD",
        }

    @staticmethod
    def _is_weak_project_standards(payload: Dict[str, str]) -> bool:
        return all(str(payload.get(key) or "").strip() in {"", "TBD"} for key in ("methodology", "team", "commercial"))

    @staticmethod
    def _empty_relationship_context(source: str = "internal_api") -> Dict[str, Any]:
        return {"summary": "", "mode": "new", "source": source, "verified": False}

    @staticmethod
    def _project_history_relationship_summary(payload: Dict[str, Any]) -> str:
        project_name = str(payload.get("project_name") or payload.get("summary") or "").strip()
        product_name = str(payload.get("product_name") or "").strip()
        expert_name = str(payload.get("expert_name") or "").strip()
        position_name = str(payload.get("position_name") or "").strip()
        if not project_name:
            return ""

        parts = [f"Data internal mencatat riwayat proyek: {project_name}."]
        if product_name:
            parts.append(f"Lingkup/produk terkait: {product_name}.")
        if expert_name:
            role = f" sebagai {position_name}" if position_name else ""
            parts.append(f"Tenaga ahli tercatat: {expert_name}{role}.")
        return " ".join(parts)

    @staticmethod
    def _account_reference_relationship_summary(payload: Dict[str, Any]) -> str:
        company_name = str(payload.get("company_name") or "").strip()
        if not company_name:
            return ""
        region = str(payload.get("company_region_name") or "").strip()
        province = str(payload.get("company_province_name") or "").strip()
        segment = str(payload.get("company_segment") or "").strip()
        sub_segment = str(payload.get("company_sub_segment") or "").strip()
        location = ", ".join(part for part in (region, province) if part)
        classification = " / ".join(part for part in (segment, sub_segment) if part)
        details = []
        if location:
            details.append(f"lokasi {location}")
        if classification:
            details.append(f"segmentasi {classification}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Data internal ReferenceAccount mencatat {company_name}{suffix}."

    @staticmethod
    def _normalized_relationship_mode(raw_mode: str, summary: str) -> str:
        mode = str(raw_mode or "").strip().lower()
        if mode in {"existing", "new"}:
            return mode
        return "existing" if str(summary or "").strip() else "new"

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
            return GenericProjectStandardsSchema.model_validate(payload).model_dump()
        except Exception as e:
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
        base_profile["credential_highlights"] = base_profile.get("credential_highlights") or WRITER_FIRM_CREDENTIAL_HIGHLIGHTS
        base_profile["official_source_urls"] = base_profile.get("official_source_urls") or WRITER_FIRM_SOURCE_URLS
        return base_profile

    @staticmethod
    def _looks_like_missing_profile(profile: Dict[str, str]) -> bool:
        if not isinstance(profile, dict): return True
        meaningful_keys = ("office_address", "email", "phone", "website", "contact_info")
        return not any(str(profile.get(key) or "").strip() for key in meaningful_keys)

    def get_firm_profile(self) -> Dict[str, str]:
        if self.demo_mode: return self._default_firm_profile()
        try:
            payload = self._resolve_resource_payload("firm_profile")
            return GenericFirmProfileSchema.model_validate(payload).model_dump()
        except Exception as e:
            logger.error(f"Internal API Error: {e}")
            return self._default_firm_profile()

    def get_client_relationship(self, client_name: str) -> Dict[str, Any]:
        if self.uses_demo_logic():
            return {"summary": "", "mode": "new", "source": "osint", "verified": False}
        try:
            payload = self._resolve_resource_payload("client_relationship", client_name=client_name)
            raw_payload = payload if isinstance(payload, dict) else {}
            data = GenericClientRelationshipSchema.model_validate(raw_payload).model_dump()
            summary = self._project_history_relationship_summary(raw_payload) or str(data.get("summary") or "").strip()
            data["summary"] = summary
            data["mode"] = self._normalized_relationship_mode(data.get("mode") or data.get("status"), summary)
            data["source"] = "internal_api"
            data["verified"] = bool(data.get("summary"))
            if data["verified"]:
                return data
        except Exception as e:
            logger.error(f"Internal API Error: {e}")

        try:
            account_payload = self._resolve_resource_payload("account_records", client_name=client_name)
            account_summary = self._account_reference_relationship_summary(
                account_payload if isinstance(account_payload, dict) else {}
            )
            if account_summary:
                return {
                    "summary": account_summary,
                    "mode": "new",
                    "source": "internal_api",
                    "verified": True,
                }
        except Exception as e:
            logger.error(f"Internal API Account Reference Error: {e}")

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


class DemoDataProvider:
    def __init__(self) -> None:
        self.app_profile = APP_PROFILE
        self.internal_data_source = "demo"
        self.demo_mode = True
        self.data_acquisition_mode = "demo"

    def uses_demo_logic(self) -> bool:
        return True

    def describe_runtime(self) -> Dict[str, Any]:
        return {
            "provider": "demo",
            "app_profile": self.app_profile,
            "internal_data_source": self.internal_data_source,
            "integration_mode": "demo",
            "auth_mode": "none",
            "base_url": "",
            "config_file": FIRM_API_CONFIG_FILE,
        }

    def get_project_standards(self, project_type: str) -> Dict[str, str]:
        logger.info("Using demo standards for project type: %s", project_type)
        demo_standards = MOCK_FIRM_STANDARDS.get(project_type, MOCK_FIRM_STANDARDS.get("Implementation"))
        return FirmAPIClient._normalize_project_standards(demo_standards)

    def get_firm_profile(self) -> Dict[str, str]:
        return FirmAPIClient._default_firm_profile()

    def get_client_relationship(self, client_name: str) -> Dict[str, Any]:
        summary = Researcher.get_client_writer_collaboration(client_name, WRITER_FIRM_NAME)
        has_evidence = FirmAPIClient._has_osint_evidence(summary)
        return {
            "summary": summary,
            "mode": "existing" if has_evidence else "new",
            "source": "osint",
            "verified": has_evidence,
        }


class InternalDataClient:
    def __init__(self, force_source: str = "") -> None:
        self.app_profile = APP_PROFILE
        normalized_force_source = str(force_source or "").strip().lower()
        if normalized_force_source not in {"demo", "api"}:
            normalized_force_source = INTERNAL_DATA_SOURCE
        self.internal_data_source = normalized_force_source
        self.internal_data_fallback = INTERNAL_DATA_FALLBACK
        self.api_provider = FirmAPIClient(force_source="api")
        self.demo_provider = DemoDataProvider()
        self.demo_mode = self.internal_data_source == "demo"
        self.data_acquisition_mode = "demo" if self.demo_mode else "staged"

    def uses_demo_logic(self) -> bool:
        return self.internal_data_source == "demo"

    def describe_runtime(self) -> Dict[str, Any]:
        runtime = (
            self.demo_provider.describe_runtime()
            if self.internal_data_source == "demo"
            else self.api_provider.describe_runtime()
        )
        runtime["fallback"] = self.internal_data_fallback
        runtime["operator_mode"] = f"{self.internal_data_source}:{self.internal_data_fallback}"
        return runtime

    def _use_demo_fallback(self) -> bool:
        return self.internal_data_fallback == "demo"

    def get_project_standards(self, project_type: str) -> Dict[str, str]:
        if self.internal_data_source == "demo":
            return self.demo_provider.get_project_standards(project_type)
        normalized = self.api_provider.get_project_standards(project_type)
        if self._use_demo_fallback() and FirmAPIClient._is_weak_project_standards(normalized):
            logger.warning("Internal data fallback triggered for project_standards(%s)", project_type)
            return self.demo_provider.get_project_standards(project_type)
        return normalized

    def get_firm_profile(self) -> Dict[str, str]:
        if self.internal_data_source == "demo":
            return self.demo_provider.get_firm_profile()
        normalized = self.api_provider.get_firm_profile()
        if self._use_demo_fallback() and FirmAPIClient._looks_like_missing_profile(normalized):
            logger.warning("Internal data fallback triggered for firm_profile")
            return self.demo_provider.get_firm_profile()
        return normalized

    def get_client_relationship(self, client_name: str) -> Dict[str, Any]:
        if self.internal_data_source == "demo":
            return self.demo_provider.get_client_relationship(client_name)
        normalized = self.api_provider.get_client_relationship(client_name)
        if self._use_demo_fallback() and FirmAPIClient._is_weak_client_relationship(normalized):
            logger.warning("Internal data fallback triggered for client_relationship(%s)", client_name)
            return self.demo_provider.get_client_relationship(client_name)
        return normalized

    def doctor_snapshot(self, project_type: str = "Implementation", client_name: str = "PT Contoh Klien") -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "runtime": self.describe_runtime(),
            "resources": {},
        }
        if self.internal_data_source == "api" or PROJECT_DATA_SOURCE == "api":
            snapshot["api_config"] = self.api_provider.validate_config()
        try:
            firm_profile = self.get_firm_profile()
            snapshot["resources"]["firm_profile"] = {
                "ok": not FirmAPIClient._looks_like_missing_profile(firm_profile),
                "fields": {
                    "office_address": bool(str(firm_profile.get("office_address") or "").strip()),
                    "email": bool(str(firm_profile.get("email") or "").strip()),
                    "phone": bool(str(firm_profile.get("phone") or "").strip()),
                    "website": bool(str(firm_profile.get("website") or "").strip()),
                },
            }
        except Exception as exc:
            snapshot["resources"]["firm_profile"] = {"ok": False, "error": str(exc)}

        try:
            standards = self.get_project_standards(project_type)
            snapshot["resources"]["project_standards"] = {
                "ok": not FirmAPIClient._is_weak_project_standards(standards),
                "fields": {
                    "methodology": bool(str(standards.get("methodology") or "").strip()),
                    "team": bool(str(standards.get("team") or "").strip()),
                    "commercial": bool(str(standards.get("commercial") or "").strip()),
                },
                "project_type": project_type,
            }
        except Exception as exc:
            snapshot["resources"]["project_standards"] = {"ok": False, "project_type": project_type, "error": str(exc)}

        try:
            relationship = self.get_client_relationship(client_name)
            snapshot["resources"]["client_relationship"] = {
                "ok": bool(str(relationship.get("summary") or "").strip()) or bool(relationship.get("verified")),
                "fields": {
                    "summary": bool(str(relationship.get("summary") or "").strip()),
                    "mode": bool(str(relationship.get("mode") or "").strip()),
                    "verified": bool(relationship.get("verified")),
                },
                "client_name": client_name,
            }
        except Exception as exc:
            snapshot["resources"]["client_relationship"] = {"ok": False, "client_name": client_name, "error": str(exc)}

        snapshot["ok"] = all(bool(item.get("ok")) for item in snapshot["resources"].values())
        return snapshot


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
        self.project_data_source = PROJECT_DATA_SOURCE
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

    def set_project_data_source(self, source: str) -> None:
        normalized = str(source or "").strip().lower()
        self.project_data_source = "api" if normalized in {"api", "internal_api", "live"} else "local"

    def _load_project_data(self) -> bool:
        if self.project_data_source == "api":
            if self._load_project_data_from_api():
                return True
            if INTERNAL_DATA_FALLBACK == "demo":
                logger.warning(
                    "API-backed project data failed. Falling back to local project data because INTERNAL_DATA_FALLBACK=demo."
                )
                return self._load_project_data_from_local()
            return False
        return self._load_project_data_from_local()

    def _load_project_data_from_api(self) -> bool:
        try:
            records = FirmAPIClient(force_source="api").get_project_records()
            if not records:
                raise ValueError("project_records API returned no records.")
            normalized_df = self._normalize_projects_df(pd.DataFrame(records))
            normalized_df.to_sql("projects", self.engine, index=False, if_exists="replace")
            self.df = normalized_df
        except Exception as exc:
            self.vector_ready = False
            self.last_refresh_error = f"API project data load failed: {exc}"
            logger.warning("API project data load failed: %s", exc)
            return False

        if not self._has_required_project_fields(self.df):
            logger.warning(
                "API project data schema is missing required fields. Expected aliases for: %s. Available columns: %s",
                ", ".join(self._required_project_fields()),
                ", ".join(self.df.columns.astype(str).tolist()) if self.df is not None else "-",
            )
            self.vector_ready = False
            self.last_refresh_error = "API project data schema is missing required fields."
            return False
        return True

    def _load_project_data_from_local(self) -> bool:
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


# Budget estimator from public financial context.
