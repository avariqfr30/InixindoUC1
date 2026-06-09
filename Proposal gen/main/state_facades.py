"""Focused facades over AppStateStore's persistence responsibilities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class _StoreFacade:
    def __init__(self, store: Any) -> None:
        self._store = store


class AuthStateFacade(_StoreFacade):
    """Authentication and active-session persistence boundary."""

    def get_login_block_seconds(self, username: str, remote_ip: str) -> int:
        return self._store.get_login_block_seconds(username, remote_ip)

    def register_login_failure(
        self,
        username: str,
        remote_ip: str,
        window_seconds: int,
        max_attempts: int,
        block_seconds: int,
    ) -> int:
        return self._store.register_login_failure(
            username=username,
            remote_ip=remote_ip,
            window_seconds=window_seconds,
            max_attempts=max_attempts,
            block_seconds=block_seconds,
        )

    def clear_login_failures(self, username: str, remote_ip: str) -> None:
        self._store.clear_login_failures(username, remote_ip)

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        return self._store.get_user(username)

    def get_user_fullname(self, username: str) -> str:
        return self._store.get_user_fullname(username)

    def create_user(self, username: str, password_hash: str, status: str, approved_by: str, full_name: str = "") -> bool:
        return self._store.create_user(
            username=username,
            password_hash=password_hash,
            status=status,
            approved_by=approved_by,
            full_name=full_name,
        )

    def active_session_snapshot(self, username: str = "") -> Dict[str, Any]:
        return self._store.active_session_snapshot(username=username)

    def set_registration_otp(self, username: str, otp_code: str) -> None:
        self._store.set_registration_otp(username, otp_code)

    def verify_registration_otp(self, username: str, otp_code: str) -> bool:
        return self._store.verify_registration_otp(username, otp_code)

    def clear_registration_otp(self, username: str) -> None:
        self._store.clear_registration_otp(username)

    def set_login_otp(self, username: str, otp_code: str) -> None:
        self._store.set_login_otp(username, otp_code)

    def verify_login_otp(self, username: str, otp_code: str) -> bool:
        return self._store.verify_login_otp(username, otp_code)

    def clear_login_otp(self, username: str) -> None:
        self._store.clear_login_otp(username)

    def is_user_approved(self, username: str) -> bool:
        return self._store.is_user_approved(username)

    def verify_email_in_reference_internal_account(self, email: str) -> bool:
        return self._store.verify_email_in_reference_internal_account(email)

    def lookup_reference_internal_account(self, email: str) -> Optional[Dict[str, Any]]:
        return self._store.lookup_reference_internal_account(email)

    def approve_user(self, username: str, approved_by: str = "admin") -> bool:
        return self._store.approve_user(username, approved_by=approved_by)

    def delete_user(self, username: str) -> bool:
        return self._store.delete_user(username)


class SettingsStateFacade(_StoreFacade):
    """Proposal settings, templates, supporting documents, and KAK context boundary."""

    def get_settings(self) -> Dict[str, Any]:
        return self._store.get_settings()

    def save_settings(self) -> Dict[str, Any]:
        return self._store.save_settings()

    def save_template(self, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        return self._store.save_template(filename, raw_bytes)

    def clear_template(self) -> Dict[str, Any]:
        return self._store.clear_template()

    def get_template_path(self) -> str:
        return self._store.get_template_path()

    def save_supporting_documents(self, document_type: str, uploads: List[Tuple[str, bytes]]) -> Dict[str, Any]:
        return self._store.save_supporting_documents(document_type, uploads)

    def delete_supporting_document(self, document_id: str) -> Dict[str, Any]:
        return self._store.delete_supporting_document(document_id)

    def get_latest_kak_context(self, company_candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._store.get_latest_kak_context(company_candidates=company_candidates)

    def set_active_kak_document(self, document_id: str) -> Dict[str, Any]:
        return self._store.set_active_kak_document(document_id)

    def build_generation_context(self, company_candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._store.build_generation_context(company_candidates=company_candidates)

    def resolve_kak_references_in_payload(
        self,
        payload: Dict[str, Any],
        company_candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._store.resolve_kak_references_in_payload(payload, company_candidates=company_candidates)

    def enrich_firm_profile(self, firm_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return self._store.enrich_firm_profile(firm_profile)


class HistoryStateFacade(_StoreFacade):
    """Generated proposal artifact and history persistence boundary."""

    def persist_generated_file(self, suggested_name: str, content: bytes):
        return self._store.persist_generated_file(suggested_name, content)

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
        return self._store.add_history_entry(
            payload=payload,
            filename=filename,
            filepath=filepath,
            created_at=created_at,
            finished_at=finished_at,
            acceptance_report=acceptance_report,
            processing_seconds=processing_seconds,
        )

    def list_history(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return self._store.list_history(*args, **kwargs)

    def get_history_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get_history_entry(entry_id)
