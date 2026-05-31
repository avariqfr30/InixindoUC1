"""Focused services for internal-data runtime and client context use cases."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .data_sources import FirmAPIClient, InternalDataClient
from .internal_api_runtime import (
    internal_api_activation_payload,
    internal_api_has_runtime_resources,
    internal_api_refresh_payload,
    internal_api_setup_status_payload,
)
from .internal_api_setup import build_internal_api_config, write_json_config
from .framework_catalog import FrameworkCatalogService
from .research import Researcher
from .proposal_shared import MANAGED_INTERNAL_API_CONFIG_PATH, logger


def normalize_client_name(raw_name: str) -> str:
    return re.sub(r"\b(Cabang|Branch|Tbk)\b.*$|^(PT\.|CV\.)", "", raw_name or "", flags=re.IGNORECASE).strip()


class InternalApiRuntimeService:
    """Operator-facing Internal API setup and refresh boundary."""

    def __init__(
        self,
        knowledge_base: Any,
        proposal_generator: Any,
        generation_queue: Any,
        managed_config_path: Path = MANAGED_INTERNAL_API_CONFIG_PATH,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.proposal_generator = proposal_generator
        self.generation_queue = generation_queue
        self.managed_config_path = managed_config_path

    def config_target(self) -> Path:
        configured = str(FirmAPIClient._resolve_config_file() or "").strip()
        if configured:
            return Path(configured).expanduser()
        return self.managed_config_path

    def write_config(self, config_payload: Dict[str, Any]) -> Path:
        target = self.config_target()
        write_json_config(target, config_payload)
        return target

    def setup_status(self) -> Dict[str, Any]:
        api_client = FirmAPIClient(force_source="api")
        return internal_api_setup_status_payload(api_client, self.knowledge_base, self.managed_config_path)

    def activate_from_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        config_payload = build_internal_api_config(data)
        target_path = self.write_config(config_payload)
        api_client = FirmAPIClient(force_source="api")
        validation = api_client.validate_config()

        activated = bool(data.get("activate_now", True))
        refresh_started = False
        if activated:
            self.proposal_generator.firm_api = InternalDataClient(force_source="api")
            self.knowledge_base.set_project_data_source("api")
            if not self.generation_queue.has_live_jobs():
                refresh_started = self.knowledge_base.refresh_data(background=True)

        return internal_api_activation_payload(
            target_path,
            activated,
            refresh_started,
            self.knowledge_base,
            validation,
        )

    def refresh(self) -> tuple[Dict[str, Any], int]:
        if self.generation_queue.has_live_jobs():
            return {
                "status": "error",
                "error": "Dataset internal tidak bisa di-refresh saat masih ada generate job yang berjalan atau mengantre.",
            }, 409

        api_client = FirmAPIClient(force_source="api")
        config_file = api_client.config_file or str(self.managed_config_path)
        config_exists = bool(config_file and Path(config_file).exists())
        if not config_exists and not internal_api_has_runtime_resources(api_client):
            return {
                "status": "error",
                "error": "Config Internal API belum tersedia. Aktifkan endpoint Internal API terlebih dahulu.",
                "config_file": config_file,
            }, 400

        self.proposal_generator.firm_api = InternalDataClient(force_source="api")
        self.knowledge_base.set_project_data_source("api")
        refresh_started = self.knowledge_base.refresh_data(force=True, background=True)
        return (
            internal_api_refresh_payload(config_file, refresh_started, self.knowledge_base),
            202 if refresh_started else 200,
        )


class ClientContextService:
    """Client lookup and internal-context composition boundary."""

    def __init__(
        self,
        proposal_generator: Any,
        knowledge_base: Any,
        prefetch_research: Callable[[Dict[str, Any]], str],
    ) -> None:
        self.proposal_generator = proposal_generator
        self.knowledge_base = knowledge_base
        self.prefetch_research = prefetch_research

    def company_candidates(self) -> List[str]:
        try:
            api_companies = self.proposal_generator.firm_api.get_client_options()
            if api_companies:
                return api_companies
        except Exception as exc:
            logger.warning("Internal client candidate lookup failed; falling back to knowledge-base entities: %s", exc)
        df = getattr(self.knowledge_base, "df", None)
        if df is None or df.empty or "entity" not in df.columns:
            return []
        companies = df["entity"].dropna().astype(str).str.strip().unique().tolist()
        return sorted([item for item in companies if item.lower() != "nan" and item])

    def client_context_payload(self, client_name: str) -> Dict[str, Any]:
        normalized_client = normalize_client_name(client_name)
        if not normalized_client:
            return {
                "available": False,
                "client_name": "",
                "account_summary": "",
                "use_case_summary": "",
                "use_cases": [],
                "expert_guidance": "",
            }
        osint_prefetch_status = ""
        osint_context_note = (
            "OSINT client research is being prepared for generation: public profile, recent signals, "
            "track record, and persuasive context will be used as proposal helpers."
        )
        try:
            osint_prefetch_status = self.prefetch_research(
                {
                    "nama_perusahaan": normalized_client,
                    "potensi_framework": "",
                    "konteks_organisasi": normalized_client,
                }
            )
        except Exception:
            logger.exception("OSINT prefetch failed for selected client %s", normalized_client)
            osint_prefetch_status = "failed"
        try:
            context = self.proposal_generator.firm_api.get_client_context(normalized_client)
            context["osint_prefetch_status"] = osint_prefetch_status
            context["osint_context_note"] = osint_context_note
            return context
        except Exception as exc:
            logger.info("Internal client context lookup missed for %s: %s", normalized_client, exc)
            return {
                "available": False,
                "client_name": normalized_client,
                "account_summary": "",
                "use_case_summary": "",
                "use_cases": [],
                "expert_guidance": "",
                "osint_prefetch_status": osint_prefetch_status,
                "osint_context_note": osint_context_note,
            }

    def internal_context_text(self, client_name: str, payload: Optional[Dict[str, Any]] = None) -> str:
        payload = payload or {}
        lines: List[str] = []
        try:
            context = self.proposal_generator.firm_api.get_client_context(client_name)
        except Exception:
            logger.exception("Internal context lookup failed for %s", client_name)
            context = {}
        account_summary = str(context.get("account_summary") or "").strip()
        if account_summary:
            lines.append(account_summary)
        use_case_summary = str(context.get("use_case_summary") or "").strip()
        if use_case_summary:
            lines.append(use_case_summary)
        expert_guidance = str(context.get("expert_guidance") or "").strip()
        if expert_guidance:
            lines.append(f"Riwayat tenaga ahli internal yang relevan: {expert_guidance}.")
        for item in (context.get("use_cases") or [])[:4]:
            if not isinstance(item, dict):
                continue
            product_name = str(item.get("product_name") or "").strip()
            project_name = str(item.get("project_name") or "").strip()
            position_name = str(item.get("position_name") or "").strip()
            if product_name or project_name or position_name:
                lines.append(
                    " | ".join(part for part in [
                        f"Use case internal: {product_name}" if product_name else "",
                        f"Contoh riwayat: {project_name}" if project_name else "",
                        f"Peran tenaga ahli: {position_name}" if position_name else "",
                    ] if part)
                )

        project_type = str(payload.get("jenis_proyek") or "")
        service_type = str(payload.get("jenis_proposal") or "")
        focus_terms = [
            str(payload.get("konteks_organisasi") or ""),
            str(payload.get("permasalahan") or ""),
            str(payload.get("potensi_framework") or ""),
        ]
        try:
            capability = self.proposal_generator.firm_api.get_capability_context(
                project_type=project_type,
                service_type=service_type,
                focus_terms=focus_terms,
                limit=5,
            )
        except Exception:
            logger.exception("Internal capability context enrichment failed for %s", client_name)
            capability = {}
        if capability.get("available"):
            aggregate_summary = str(capability.get("aggregate_summary") or "").strip()
            if aggregate_summary:
                lines.append(aggregate_summary)
            lines.append(str(capability.get("summary") or "").strip())
            evidence_cards = [item for item in (capability.get("evidence_cards") or []) if isinstance(item, dict)]
            for card in evidence_cards[:3]:
                safe_sentence = str(card.get("safe_sentence") or "").strip()
                if safe_sentence:
                    lines.append(f"Bukti agregat kapabilitas: {safe_sentence}")
            capability_guidance = str(capability.get("expert_guidance") or "").strip()
            if capability_guidance:
                lines.append(f"Tenaga ahli relevan dari riwayat proyek internal: {capability_guidance}.")
            for item in (capability.get("matches") or [])[:4]:
                if not isinstance(item, dict):
                    continue
                product_name = str(item.get("product_name") or "").strip()
                project_name = str(item.get("project_name") or "").strip()
                position_name = str(item.get("position_name") or "").strip()
                if product_name or position_name:
                    lines.append(
                        " | ".join(part for part in [
                            f"Kapabilitas relevan: {product_name}" if product_name else "",
                            f"Contoh riwayat: {project_name}" if project_name else "",
                            f"Peran tenaga ahli: {position_name}" if position_name else "",
                        ] if part)
                    )
        return "\n".join(line for line in lines if line).strip()


class FrameworkOptionService:
    """Framework option and resolver boundary for UI and request normalization."""

    def __init__(self, provider_factory: Callable[[], Any]) -> None:
        self.provider_factory = provider_factory

    def _catalog(self) -> FrameworkCatalogService:
        try:
            provider = self.provider_factory()
        except Exception:
            provider = None
        return FrameworkCatalogService(provider, researcher=Researcher)

    def options_payload(self) -> Dict[str, Any]:
        return self._catalog().options()

    def resolve_selection(self, raw_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        return self._catalog().resolve(raw_value, context=context)

    def confirmation_payload(self, raw_value: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._catalog().confirmation_payload(raw_value, context=context)
