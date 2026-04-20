with open("main/runtime_components.py", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "class FirmAPIClient:" in line:
        start_idx = i - 1
        break

for i in range(start_idx + 1, len(lines)):
    if "def _extract_first(" in line:
        end_idx = i - 1
        break

print(f"Start: {start_idx}, End: {end_idx}")

new_class = """# Internal API adapter.
class FirmAPIClient:
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
        self.integration_mode = FIRM_API_INTEGRATION_MODE
        self.auth_mode = (FIRM_API_AUTH_MODE or "bearer").strip().lower()
        self.request_defaults = dict(FIRM_API_REQUEST_DEFAULTS or {})
        self.endpoint_config = dict(FIRM_API_ENDPOINT_CONFIG or {})
        self.dataset_config = dict(FIRM_API_DATASET_CONFIG or {})
        self.resource_config = dict(FIRM_API_RESOURCE_CONFIG or {})
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
        
        url = str(path or "").strip()
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            url = f"{self.base_url}/{url.lstrip('/')}"
            
        kwargs = {"params": params, "timeout": self.timeout_seconds}
        if method != "GET":
            encoding = str(request_spec.get("body_encoding") or "json").strip().lower()
            if encoding == "form":
                kwargs["data"] = body or {}
            else:
                kwargs["json"] = body or {}
                
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

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
                        actual = item.get(k)
                        if str(actual).strip().lower() != str(expected).strip().lower():
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
            "config_file": FIRM_API_CONFIG_FILE,
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
            if self.uses_demo_logic(): 
                # Fallback to empty context logic since osint builder is removed
                pass
            return self._default_firm_profile()

    def get_client_relationship(self, client_name: str) -> Dict[str, Any]:
        if self.uses_demo_logic():
            summary = "" 
            # In a real scenario Researcher.get_client_writer_collaboration would be called here
            return {"summary": summary, "mode": "new", "source": "osint", "verified": False}
        try:
            payload = self._resolve_resource_payload("client_relationship", client_name=client_name)
            data = GenericClientRelationshipSchema.model_validate(payload).model_dump()
            data["source"] = "internal_api"
            data["verified"] = bool(data.get("summary"))
            return data
        except Exception as e:
            logger.error(f"Internal API Error: {e}")
            return self._empty_relationship_context()

"""

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [new_class] + lines[end_idx:]
    with open("main/runtime_components.py", "w") as f:
        f.writelines(new_lines)
    print("Replaced successfully")
else:
    print("Could not find boundaries")

