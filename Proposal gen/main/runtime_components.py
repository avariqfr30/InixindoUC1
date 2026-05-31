"""Runtime components for internal data access and knowledge-base retrieval."""

from pydantic import BaseModel, Field
from collections import Counter
from typing import Optional, List, Dict, Any, Tuple

from .proposal_shared import *
from .schema_mapping import SchemaMapper
from .research import Researcher
from .capability_intelligence import build_capability_intelligence
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
        "framework_catalog": ("value", "label"),
        "employee_expertise": ("employee_name",),
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
        self._resource_record_cache: Dict[str, List[Dict[str, Any]]] = {}

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
        configured_resources = self.resource_config or {}
        resource_names = ["firm_profile", "project_standards", "client_relationship"]
        if "account_records" in configured_resources:
            resource_names.append("account_records")
        if PROJECT_DATA_SOURCE == "api" or "project_records" in configured_resources:
            resource_names.append("project_records")
        if "framework_catalog" in configured_resources:
            resource_names.append("framework_catalog")
        if "employee_expertise" in configured_resources:
            resource_names.append("employee_expertise")
        for resource_name in resource_names:
            resource_spec = dict((self.resource_config or {}).get(resource_name) or {})
            if resource_spec.get("optional") and not str((((resource_spec.get("request") or {}).get("body") or {}).get("dataset") or "")).strip():
                continue
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
        if getattr(self, "demo_mode", False):
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

    def _get_mapped_resource_records(
        self,
        resource_name: str,
        apply_filters: bool = True,
        **context: Any,
    ) -> List[Dict[str, Any]]:
        if getattr(self, "demo_mode", False):
            return []
        resource_spec = dict((self.resource_config or {}).get(resource_name) or {})
        if not resource_spec:
            return []

        cache_context = {
            "resource_name": resource_name,
            "apply_filters": apply_filters,
            "context": context,
            "request": resource_spec.get("request") or {},
            "response_path": resource_spec.get("response_path") or "",
            "record_filters": resource_spec.get("record_filters") or {},
            "field_mapping": resource_spec.get("field_mapping") or {},
        }
        try:
            cache_key = json.dumps(cache_context, sort_keys=True, default=str)
        except TypeError:
            cache_key = str(cache_context)
        cache = getattr(self, "_resource_record_cache", None)
        if cache is None:
            cache = {}
            self._resource_record_cache = cache
        if cache_key in cache:
            return [dict(record) for record in cache[cache_key]]

        payload = self._request_from_spec(resource_spec, **context)
        response_path = self._render_template_value(resource_spec.get("response_path") or "", context)
        if response_path:
            payload = self._extract_with_jmespath(payload, response_path)

        filters = self._render_template_payload(resource_spec.get("record_filters") or {}, context) if apply_filters else {}
        records = self._coerce_record_list(payload)
        if filters:
            filtered_records: List[Dict[str, Any]] = []
            for item in records:
                match = True
                for key, expected in filters.items():
                    field_name = str(key)
                    operator = "eq"
                    if "__" in field_name:
                        field_name, operator = field_name.rsplit("__", 1)
                    actual_text = str(item.get(field_name) or "").strip().lower()
                    expected_text = str(expected or "").strip().lower()
                    if operator in {"icontains", "contains"}:
                        if expected_text not in actual_text:
                            match = False
                            break
                    elif actual_text != expected_text:
                        match = False
                        break
                if match:
                    filtered_records.append(item)
            records = filtered_records

        field_mapping = resource_spec.get("field_mapping") or {}
        if not field_mapping:
            cache[cache_key] = [dict(record) for record in records]
            return records
        mapped_records = [
            {
                target_field: self._extract_with_jmespath(record, str(json_path))
                for target_field, json_path in field_mapping.items()
            }
            for record in records
        ]
        cache[cache_key] = [dict(record) for record in mapped_records]
        return mapped_records

    def get_account_records(self) -> List[Dict[str, Any]]:
        return self._get_mapped_resource_records("account_records", apply_filters=False)

    def get_employee_expertise_records(self) -> List[Dict[str, Any]]:
        return self._get_mapped_resource_records("employee_expertise", apply_filters=False)

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
    def _normalize_client_match_key(value: str) -> str:
        cleaned = re.sub(r"\b(Cabang|Branch|Tbk)\b.*$", "", str(value or ""), flags=re.IGNORECASE)
        cleaned = re.sub(r"^(PT\.?|CV\.?)\s+", "", cleaned.strip(), flags=re.IGNORECASE)
        return "".join(ch for ch in cleaned.lower() if ch.isalnum())

    @staticmethod
    def _looks_like_project_title(value: str) -> bool:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return True
        if re.match(r"^[A-Z]{1,4}\d{1,5}\s*[-–]\s+", text):
            return True
        project_terms = (
            "penyusunan", "roadmap", "asesmen", "assessment", "audit", "implementasi",
            "pengembangan", "konsultansi", "pelatihan", "training", "pendampingan",
            "soc", "noc", "governance", "tata kelola"
        )
        return " - " in text and any(term in text.lower() for term in project_terms)

    @classmethod
    def _dedupe_records(cls, records: List[Dict[str, Any]], keys: Tuple[str, ...], limit: int = 12) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for record in records:
            marker = tuple(str(record.get(key) or "").strip().lower() for key in keys)
            if not any(marker) or marker in seen:
                continue
            seen.add(marker)
            deduped.append(record)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _dedupe_phrases(items: List[str], limit: int = 6) -> List[str]:
        values: List[str] = []
        seen = set()
        for item in items:
            cleaned = FirmAPIClient._naturalize_internal_text(item)
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            values.append(cleaned)
            if len(values) >= limit:
                break
        return values

    @staticmethod
    def _naturalize_internal_text(value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            return ""
        text = re.sub(r"(?i)\bData internal\b", "", text)
        text = re.sub(r"(?i)\bReferenceAccount\b", "", text)
        text = re.sub(r"(?i)\bConsultantProjectExpertHistory\b", "", text)
        text = re.sub(r"^[A-Z]{1,4}\d{1,5}\s*[-–]\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip(" .:-")
        if text.isupper() and len(text) > 3:
            preserved = {"AI", "API", "BPR", "BPRS", "BUMN", "ISO", "IT", "NOC", "NTB", "PT", "SOC", "SPBE"}
            words = []
            for word in text.split():
                stripped = re.sub(r"[^A-Za-z0-9/]+", "", word)
                if stripped in preserved:
                    words.append(word)
                else:
                    words.append(word.lower().capitalize())
            text = " ".join(words)
        return text

    @staticmethod
    def _project_history_relationship_summary(payload: Dict[str, Any]) -> str:
        project_name = FirmAPIClient._naturalize_internal_text(payload.get("project_name") or payload.get("summary"))
        product_name = FirmAPIClient._naturalize_internal_text(payload.get("product_name"))
        position_name = FirmAPIClient._naturalize_internal_text(payload.get("position_name"))
        if not project_name:
            return ""

        parts = [f"Riwayat internal menunjukkan pengalaman proyek pada {project_name}."]
        if product_name:
            parts.append(f"Lingkup/produk terkait: {product_name}.")
        if position_name:
            parts.append(f"Peran tenaga ahli yang tercatat: {position_name}.")
        return " ".join(parts)

    @classmethod
    def _format_project_expert_history(
        cls,
        records: List[Dict[str, Any]],
        limit_products: int = 5,
        limit_positions: int = 4,
        limit_experts: int = 4,
        limit_projects: int = 3,
        allow_named_experts: bool = False,
    ) -> Dict[str, Any]:
        grouped: Dict[str, Dict[str, Any]] = {}
        product_order: List[str] = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            product_name = cls._naturalize_internal_text(record.get("product_name") or record.get("topic"))
            project_name = cls._naturalize_internal_text(record.get("project_name") or record.get("entity"))
            expert_name = cls._naturalize_internal_text(record.get("expert_name"))
            position_name = cls._naturalize_internal_text(record.get("position_name")) or "Tenaga Ahli"
            if not product_name:
                product_name = "Produk atau lingkup tidak tercatat"
            if product_name not in grouped:
                grouped[product_name] = {
                    "product_name": product_name,
                    "positions": {},
                    "project_examples": [],
                }
                product_order.append(product_name)
            bucket = grouped[product_name]
            if project_name and project_name not in bucket["project_examples"]:
                bucket["project_examples"].append(project_name)
            positions = bucket["positions"]
            if position_name not in positions:
                positions[position_name] = []
            if expert_name and expert_name not in positions[position_name]:
                positions[position_name].append(expert_name)
            elif not expert_name:
                positions[position_name].append(f"record-{len(positions[position_name]) + 1}")

        matrix: List[Dict[str, Any]] = []
        for product_name in product_order[: max(1, int(limit_products or 5))]:
            bucket = grouped[product_name]
            positions_payload: List[Dict[str, Any]] = []
            for position_name, experts in list(bucket["positions"].items())[: max(1, int(limit_positions or 4))]:
                positions_payload.append(
                    {
                        "position_name": position_name,
                        "expert_count": len(experts),
                        "experts": experts[: max(1, int(limit_experts or 4))] if allow_named_experts else [],
                    }
                )
            matrix.append(
                {
                    "product_name": product_name,
                    "positions": positions_payload,
                    "project_examples": bucket["project_examples"][: max(1, int(limit_projects or 3))],
                }
            )

        if not matrix:
            return {
                "available": False,
                "summary": "",
                "formatted_summary": "",
                "expert_guidance": "",
                "product_expert_matrix": [],
            }

        product_names = [item["product_name"] for item in matrix if item.get("product_name")]
        summary = (
            "Riwayat proyek internal menunjukkan kapabilitas pada "
            + ", ".join(product_names[:4])
            + "."
        )
        guidance_lines: List[str] = []
        formatted_lines: List[str] = [summary]
        for item in matrix:
            product_name = item["product_name"]
            role_bits: List[str] = []
            for position in item.get("positions", []):
                expert_count = int(position.get("expert_count") or len(position.get("experts", []) or []))
                if expert_count <= 0:
                    continue
                role_bits.append(f"{position.get('position_name') or 'Tenaga Ahli'} ({expert_count} tenaga/riwayat)")
            if role_bits:
                line = f"{product_name}: " + "; ".join(role_bits)
                guidance_lines.append(line)
                formatted_lines.append(f"- {line}.")
            examples = item.get("project_examples", [])
            if examples:
                formatted_lines.append(f"  Contoh riwayat: {', '.join(examples[:2])}.")

        return {
            "available": True,
            "summary": summary,
            "formatted_summary": "\n".join(formatted_lines),
            "expert_guidance": "; ".join(guidance_lines),
            "product_expert_matrix": matrix,
        }

    @staticmethod
    def _account_reference_relationship_summary(payload: Dict[str, Any]) -> str:
        company_name = FirmAPIClient._naturalize_internal_text(payload.get("company_name"))
        if not company_name:
            return ""
        region = FirmAPIClient._naturalize_internal_text(payload.get("company_region_name"))
        province = FirmAPIClient._naturalize_internal_text(payload.get("company_province_name"))
        segment = FirmAPIClient._naturalize_internal_text(payload.get("company_segment"))
        sub_segment = FirmAPIClient._naturalize_internal_text(payload.get("company_sub_segment"))
        location = ", ".join(part for part in (region, province) if part)
        classification = " / ".join(part for part in (segment, sub_segment) if part)
        details = []
        if location:
            details.append(f"berlokasi di {location}")
        if classification:
            details.append(f"segmen {classification.lower()}")
        suffix = f" dengan {'; '.join(details)}" if details else ""
        return f"Konteks akun internal menempatkan {company_name}{suffix}. Gunakan informasi ini sebagai latar segmentasi dan lokasi, bukan sebagai rumusan tujuan proyek."

    @staticmethod
    def _flatten_text_items(value: Any, limit: int = 6) -> List[str]:
        items: List[str] = []

        def collect(node: Any) -> None:
            if len(items) >= max(1, int(limit or 6)):
                return
            if isinstance(node, dict):
                for child in node.values():
                    collect(child)
            elif isinstance(node, list):
                for child in node:
                    collect(child)
            else:
                text = FirmAPIClient._naturalize_internal_text(node)
                if text and text not in items:
                    items.append(text)

        collect(value)
        return items[: max(1, int(limit or 6))]

    @classmethod
    def _format_employee_expertise(cls, records: List[Dict[str, Any]], limit: int = 5) -> Dict[str, Any]:
        normalized: List[Dict[str, Any]] = []
        certification_terms: List[str] = []
        project_terms: List[str] = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            employee_name = cls._naturalize_internal_text(record.get("employee_name"))
            certifications = cls._flatten_text_items(record.get("certifications"), limit=8)
            projects = cls._flatten_text_items(record.get("projects"), limit=6)
            if not employee_name and not certifications and not projects:
                continue
            for cert in certifications:
                if cert not in certification_terms:
                    certification_terms.append(cert)
            for project in projects:
                if project not in project_terms:
                    project_terms.append(project)
            normalized.append(
                {
                    "employee_name": employee_name,
                    "certifications": certifications,
                    "projects": projects,
                }
            )

        if not normalized:
            return {"available": False, "record_count": len(records or []), "summary": "", "rows": []}

        cert_line = ", ".join(certification_terms[:8])
        project_line = ", ".join(project_terms[:5])
        parts = [f"Data sertifikasi internal mencakup {len(normalized)} tenaga ahli."]
        if cert_line:
            parts.append(f"Area sertifikasi yang dapat ditonjolkan: {cert_line}.")
        if project_line:
            parts.append(f"Pengalaman terkait yang tercatat mencakup {project_line}.")
        return {
            "available": True,
            "record_count": len(records or []),
            "usable_record_count": len(normalized),
            "summary": " ".join(parts),
            "rows": normalized[: max(1, int(limit or 5))],
            "certifications": certification_terms[:12],
            "projects": project_terms[:10],
        }

    @staticmethod
    def _normalized_relationship_mode(raw_mode: str, summary: str) -> str:
        mode = str(raw_mode or "").strip().lower()
        if mode in {"existing", "new"}:
            return mode
        return "existing" if str(summary or "").strip() else "new"

    @staticmethod
    def _has_osint_evidence(summary: str) -> bool:
        return any(line.strip().startswith(("Sumber eksternal", "Bukti eksternal")) for line in (summary or "").splitlines())

    def get_client_options(self, limit: int = 500) -> List[str]:
        names: List[str] = []
        seen = set()
        for record in self.get_account_records():
            company_name = re.sub(r"\s+", " ", str(record.get("company_name") or "").strip())
            if not company_name or self._looks_like_project_title(company_name):
                continue
            key = self._normalize_client_match_key(company_name)
            if not key or key in seen:
                continue
            seen.add(key)
            names.append(company_name)
            if len(names) >= limit:
                break
        return sorted(names, key=lambda item: item.lower())

    def get_client_context(self, client_name: str) -> Dict[str, Any]:
        client_key = self._normalize_client_match_key(client_name)
        account_records = self.get_account_records()
        matched_account = next(
            (
                record for record in account_records
                if client_key
                and (
                    client_key in self._normalize_client_match_key(str(record.get("company_name") or ""))
                    or self._normalize_client_match_key(str(record.get("company_name") or "")) in client_key
                )
            ),
            {},
        )
        account_name = str(matched_account.get("company_name") or client_name or "").strip()
        account_key = self._normalize_client_match_key(account_name)
        match_keys = {key for key in (client_key, account_key) if key}

        project_matches: List[Dict[str, Any]] = []
        for record in self.get_project_records():
            project_name = str(record.get("project_name") or record.get("entity") or "").strip()
            project_key = self._normalize_client_match_key(project_name)
            if not project_key or not any(key in project_key or project_key in key for key in match_keys):
                continue
            project_matches.append(
                {
                    "project_name": project_name,
                    "product_name": str(record.get("product_name") or record.get("topic") or "").strip(),
                    "position_name": str(record.get("position_name") or "").strip(),
                }
            )

        use_cases = self._dedupe_records(
            project_matches,
            keys=("project_name", "product_name", "position_name"),
            limit=8,
        )
        history_format = self._format_project_expert_history(use_cases, limit_products=4)
        account_summary = self._account_reference_relationship_summary(matched_account if isinstance(matched_account, dict) else {})
        return {
            "available": bool(account_summary or use_cases),
            "client_name": account_name,
            "account_summary": account_summary,
            "use_case_summary": history_format.get("summary", ""),
            "use_cases": use_cases,
            "expert_guidance": history_format.get("expert_guidance", ""),
            "expert_history_summary": history_format.get("formatted_summary", ""),
            "product_expert_matrix": history_format.get("product_expert_matrix", []),
        }

    @staticmethod
    def _capability_terms(project_type: str, service_type: str, focus_terms: Optional[List[str]] = None) -> List[str]:
        raw_terms = [project_type, service_type, *(focus_terms or [])]
        expanded: List[str] = []
        for item in raw_terms:
            text = re.sub(r"[^A-Za-z0-9/+.\-\s]", " ", str(item or " "))
            for part in re.split(r"[,;|/\n]+|\s+dan\s+|\s+atau\s+", text, flags=re.IGNORECASE):
                term = re.sub(r"\s+", " ", part).strip(" .:-")
                if len(term) < 3:
                    continue
                expanded.append(term)
        lowered = " ".join(raw_terms).lower()
        if "spbe" in lowered:
            expanded.extend(["SPBE", "Arsitektur SPBE", "Pemerintahan Digital", "Tata Kelola"])
        if "iso" in lowered or "27001" in lowered:
            expanded.extend(["ISO 27001", "ISO/IEC 27001", "Pendampingan ISO"])
        if "regulasi" in lowered or "governance" in lowered or "tata kelola" in lowered:
            expanded.extend(["Tata Kelola", "Governance", "SOP", "Kajian"])
        if "strategic" in lowered or "strategi" in lowered:
            expanded.extend(["Roadmap", "Kajian", "Arsitektur"])
        return FirmAPIClient._dedupe_phrases(expanded, limit=12)

    def get_capability_context(
        self,
        project_type: str = "",
        service_type: str = "",
        focus_terms: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        terms = self._capability_terms(project_type, service_type, focus_terms)
        if not terms:
            return {"available": False, "summary": "", "matches": [], "expert_guidance": ""}
        all_records = self.get_project_records()
        mapped_all_records = [
            {
                "project_name": str(record.get("project_name") or record.get("entity") or "").strip(),
                "product_name": str(record.get("product_name") or record.get("topic") or "").strip(),
                "expert_name": str(record.get("expert_name") or "").strip(),
                "position_name": str(record.get("position_name") or "").strip(),
            }
            for record in all_records
        ]
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for record in mapped_all_records:
            project_name = str(record.get("project_name") or record.get("entity") or "").strip()
            product_name = str(record.get("product_name") or record.get("topic") or "").strip()
            haystack = f"{project_name} {product_name}".lower()
            score = sum(1 for term in terms if term.lower() in haystack)
            if score <= 0:
                continue
            scored.append((
                score,
                {
                    "project_name": project_name,
                    "product_name": product_name,
                    "position_name": str(record.get("position_name") or "").strip(),
                    "matched_terms": [term for term in terms if term.lower() in haystack],
                },
            ))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("project_name") or "").lower()))
        matches = self._dedupe_records(
            [item for _, item in scored],
            keys=("project_name", "product_name", "position_name"),
            limit=max(1, int(limit or 5)),
        )
        if not matches:
            return {"available": False, "summary": "", "matches": [], "expert_guidance": ""}
        history_format = self._format_project_expert_history(matches, limit_products=max(1, int(limit or 5)))
        intelligence = build_capability_intelligence(
            mapped_all_records,
            focus_terms=terms,
            naturalize=self._naturalize_internal_text,
            limit_cards=max(1, int(limit or 5)),
        )
        return {
            "available": True,
            "summary": history_format.get("summary", ""),
            "matches": matches,
            "expert_guidance": history_format.get("expert_guidance", ""),
            "expert_history_summary": history_format.get("formatted_summary", ""),
            "product_expert_matrix": history_format.get("product_expert_matrix", []),
            "aggregate_summary": intelligence.get("aggregate_summary", ""),
            "evidence_cards": intelligence.get("evidence_cards", []),
            "coverage_gaps": intelligence.get("coverage_gaps", []),
            "total_record_count": intelligence.get("total_record_count", len(all_records)),
            "usable_record_count": intelligence.get("usable_record_count", len(matches)),
            "strongest_roles": intelligence.get("strongest_roles", []),
        }

    def get_expert_bench_context(self, limit_products: int = 8) -> Dict[str, Any]:
        records = self.get_project_records()
        try:
            employee_records = self.get_employee_expertise_records()
        except AttributeError:
            employee_records = []
        except Exception:
            logger.exception("Internal employee expertise lookup failed")
            employee_records = []
        employee_expertise = self._format_employee_expertise(
            employee_records,
            limit=limit_products,
        )
        mapped: List[Dict[str, Any]] = []
        for record in records:
            mapped.append(
                {
                    "project_name": str(record.get("project_name") or record.get("entity") or "").strip(),
                    "product_name": str(record.get("product_name") or record.get("topic") or "").strip(),
                    "expert_name": str(record.get("expert_name") or "").strip(),
                    "position_name": str(record.get("position_name") or "").strip(),
                }
            )
        intelligence = build_capability_intelligence(
            mapped,
            naturalize=self._naturalize_internal_text,
            limit_cards=limit_products,
        )
        product_counts = Counter(str(record.get("product_name") or "").strip() for record in mapped)
        formatted = self._format_project_expert_history(
            self._dedupe_records(
                sorted(
                    mapped,
                    key=lambda item: (
                        -product_counts[str(item.get("product_name") or "").strip()],
                        str(item.get("product_name") or "").lower(),
                        str(item.get("project_name") or "").lower(),
                    ),
                ),
                keys=("project_name", "product_name", "position_name"),
                limit=max(1, len(mapped)),
            ),
            limit_products=limit_products,
            limit_positions=5,
            limit_experts=5,
            limit_projects=2,
            allow_named_experts=True,
        )
        if not formatted.get("available"):
            return {
                "available": bool(employee_expertise.get("available")),
                "record_count": len(records),
                "summary": employee_expertise.get("summary", ""),
                "product_expert_matrix": [],
                "employee_expertise_summary": employee_expertise.get("summary", ""),
                "employee_expertise_rows": employee_expertise.get("rows", []),
                "employee_expertise_record_count": employee_expertise.get("record_count", 0),
            }
        return {
            "available": True,
            "record_count": len(records),
            "summary": formatted.get("summary", ""),
            "expert_guidance": formatted.get("expert_guidance", ""),
            "expert_history_summary": formatted.get("formatted_summary", ""),
            "product_expert_matrix": formatted.get("product_expert_matrix", []),
            "aggregate_summary": intelligence.get("aggregate_summary", ""),
            "evidence_cards": intelligence.get("evidence_cards", []),
            "coverage_gaps": intelligence.get("coverage_gaps", []),
            "total_record_count": intelligence.get("total_record_count", len(records)),
            "usable_record_count": intelligence.get("usable_record_count", len(mapped)),
            "strongest_roles": intelligence.get("strongest_roles", []),
            "source": "internal_api",
            "employee_expertise_summary": employee_expertise.get("summary", ""),
            "employee_expertise_rows": employee_expertise.get("rows", []),
            "employee_expertise_record_count": employee_expertise.get("record_count", 0),
            "name_policy": {
                "allow_named_specialists": True,
                "allowed_use": "team_chapter_only",
                "source": "ConsultantProjectExpertHistory+EmployeeExpertise"
                if employee_expertise.get("available")
                else "ConsultantProjectExpertHistory",
            },
        }

    def get_framework_catalog(self) -> List[Dict[str, Any]]:
        if self.demo_mode:
            return []
        resource_spec = dict((self.resource_config or {}).get("framework_catalog") or {})
        if not resource_spec:
            return []
        try:
            payload = self._request_from_spec(resource_spec)
            response_path = str(resource_spec.get("response_path") or "").strip()
            if response_path:
                payload = self._extract_with_jmespath(payload, response_path)
            records = payload if isinstance(payload, list) else []
            field_mapping = resource_spec.get("field_mapping") or {}
            mapped: List[Dict[str, Any]] = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                if field_mapping:
                    item = {
                        target_field: self._extract_with_jmespath(record, str(json_path))
                        for target_field, json_path in field_mapping.items()
                    }
                else:
                    item = dict(record)
                if not str(item.get("value") or "").strip():
                    item["value"] = record.get("parent_code") or record.get("parent_short_name") or record.get("framework_code")
                if not str(item.get("label") or "").strip():
                    item["label"] = record.get("parent_short_name") or record.get("parent_name") or record.get("framework_name")
                if not str(item.get("description") or "").strip():
                    item["description"] = record.get("parent_description") or record.get("proposal_use_guidance")
                if not str(item.get("issuer") or "").strip():
                    item["issuer"] = record.get("parent_issuer")
                if not str(item.get("category") or "").strip():
                    item["category"] = record.get("parent_category")
                if "versions" not in item and "children" in record:
                    item["versions"] = record.get("children")
                if str(item.get("value") or item.get("label") or "").strip():
                    mapped.append(item)
            return mapped
        except Exception as exc:
            logger.warning("Internal framework catalog lookup failed: %s", exc)
            return []

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

    def get_client_options(self, limit: int = 500) -> List[str]:
        return []

    def get_client_context(self, client_name: str) -> Dict[str, Any]:
        return {
            "available": False,
            "client_name": client_name,
            "account_summary": "",
            "use_case_summary": "",
            "use_cases": [],
            "expert_guidance": "",
        }

    def get_capability_context(
        self,
        project_type: str = "",
        service_type: str = "",
        focus_terms: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        return {"available": False, "summary": "", "matches": [], "expert_guidance": ""}

    def get_expert_bench_context(self, limit_products: int = 8) -> Dict[str, Any]:
        return {"available": False, "record_count": 0, "summary": "", "product_expert_matrix": []}

    def get_framework_catalog(self) -> List[Dict[str, Any]]:
        return []


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

    def get_client_options(self, limit: int = 500) -> List[str]:
        if self.internal_data_source == "demo":
            return self.demo_provider.get_client_options(limit=limit)
        options = self.api_provider.get_client_options(limit=limit)
        if self._use_demo_fallback() and not options:
            logger.warning("Internal data fallback triggered for client options")
            return self.demo_provider.get_client_options(limit=limit)
        return options

    def get_client_context(self, client_name: str) -> Dict[str, Any]:
        if self.internal_data_source == "demo":
            return self.demo_provider.get_client_context(client_name)
        context = self.api_provider.get_client_context(client_name)
        if self._use_demo_fallback() and not context.get("available"):
            logger.warning("Internal data fallback triggered for client_context(%s)", client_name)
            return self.demo_provider.get_client_context(client_name)
        return context

    def get_capability_context(
        self,
        project_type: str = "",
        service_type: str = "",
        focus_terms: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        if self.internal_data_source == "demo":
            return self.demo_provider.get_capability_context(project_type, service_type, focus_terms, limit)
        context = self.api_provider.get_capability_context(project_type, service_type, focus_terms, limit)
        if self._use_demo_fallback() and not context.get("available"):
            logger.warning("Internal data fallback triggered for capability_context(%s)", project_type)
            return self.demo_provider.get_capability_context(project_type, service_type, focus_terms, limit)
        return context

    def get_expert_bench_context(self, limit_products: int = 8) -> Dict[str, Any]:
        if self.internal_data_source == "demo":
            return self.demo_provider.get_expert_bench_context(limit_products)
        context = self.api_provider.get_expert_bench_context(limit_products)
        if self._use_demo_fallback() and not context.get("available"):
            logger.warning("Internal data fallback triggered for expert_bench_context")
            return self.demo_provider.get_expert_bench_context(limit_products)
        return context

    def get_framework_catalog(self) -> List[Dict[str, Any]]:
        if self.internal_data_source == "demo":
            return []
        catalog = self.api_provider.get_framework_catalog()
        return catalog if isinstance(catalog, list) else []

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
from .project_knowledge_base import KnowledgeBase


# Budget estimator from public financial context.
