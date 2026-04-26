"""Flask authentication/session flow helpers."""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Iterable

from flask import jsonify, redirect, request, session, url_for


class AuthFlow:
    def __init__(
        self,
        state_store: Any,
        public_endpoints: Iterable[str],
        no_store_paths: Iterable[str],
        idle_timeout_minutes: int,
        absolute_timeout_hours: int,
        touch_interval_seconds: int,
        force_single_session: bool,
        max_sessions_per_user: int,
        max_global_sessions: int,
    ) -> None:
        self.state_store = state_store
        self.public_endpoints = set(public_endpoints)
        self.no_store_paths = set(no_store_paths)
        self.idle_timeout_minutes = idle_timeout_minutes
        self.absolute_timeout_hours = absolute_timeout_hours
        self.touch_interval_seconds = touch_interval_seconds
        self.force_single_session = force_single_session
        self.max_sessions_per_user = max_sessions_per_user
        self.max_global_sessions = max_global_sessions

    def current_username(self) -> str:
        return str(session.get("auth_username") or "").strip()

    def current_session_id(self) -> str:
        return str(session.get("auth_session_id") or "").strip()

    @staticmethod
    def request_ip() -> str:
        forwarded = str(request.headers.get("X-Forwarded-For", "") or "").strip()
        if forwarded:
            return forwarded.split(",")[0].strip()[:96]
        return str(request.remote_addr or "").strip()[:96]

    @staticmethod
    def request_user_agent() -> str:
        return str(request.headers.get("User-Agent", "") or "").strip()[:256]

    def is_authenticated(self) -> bool:
        return bool(self.current_username() and self.current_session_id())

    @staticmethod
    def is_api_request() -> bool:
        return request.path.startswith("/api/") or request.path == "/generate"

    def requires_no_store_headers(self, path: str) -> bool:
        return path in self.no_store_paths or path.startswith("/api/")

    @staticmethod
    def safe_next_target(raw_target: str) -> str:
        next_target = str(raw_target or "").strip()
        if next_target.startswith("/") and not next_target.startswith("//"):
            return next_target
        return url_for("home")

    def login_redirect_target(self) -> str:
        return self.safe_next_target(request.args.get("next"))

    def set_authenticated_user(self, username: str) -> None:
        normalized_username = str(username or "").strip()
        if not normalized_username:
            raise ValueError("Username is required.")
        tracked_session_id = self.state_store.create_user_session(
            username=normalized_username,
            remote_ip=self.request_ip(),
            user_agent=self.request_user_agent(),
            idle_timeout_seconds=self.idle_timeout_minutes * 60,
            absolute_timeout_seconds=self.absolute_timeout_hours * 3600,
            force_single_session=self.force_single_session,
            max_sessions_per_user=self.max_sessions_per_user,
            max_global_sessions=self.max_global_sessions,
        )
        if not tracked_session_id:
            raise RuntimeError("Unable to create authenticated session.")
        session.clear()
        session["auth_username"] = normalized_username
        session["auth_session_id"] = tracked_session_id
        session.permanent = True

    def clear_authenticated_user(self, revoke: bool = True) -> None:
        tracked_session_id = self.current_session_id()
        if revoke and tracked_session_id:
            self.state_store.revoke_user_session(tracked_session_id)
        session.clear()

    def touch_active_auth_session(self) -> bool:
        username = self.current_username()
        tracked_session_id = self.current_session_id()
        if not username or not tracked_session_id:
            return False
        return self.state_store.touch_user_session(
            session_id=tracked_session_id,
            username=username,
            remote_ip=self.request_ip(),
            idle_timeout_seconds=self.idle_timeout_minutes * 60,
            touch_interval_seconds=self.touch_interval_seconds,
        )

    def login_required(self, view_func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(view_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.is_authenticated():
                return view_func(*args, **kwargs)
            if self.is_api_request():
                return jsonify({"error": "Authentication required."}), 401
            return redirect(url_for("auth_page", next=request.full_path if request.query_string else request.path))

        return wrapper

    def enforce_authentication(self) -> Any:
        endpoint = request.endpoint or ""
        if self.is_authenticated():
            if self.touch_active_auth_session():
                return None
            self.clear_authenticated_user(revoke=False)
        if endpoint in self.public_endpoints:
            return None
        if self.is_api_request():
            return jsonify({"error": "Authentication required."}), 401
        return redirect(url_for("auth_page", next=request.full_path if request.query_string else request.path))

    def apply_cache_headers(self, response: Any) -> Any:
        path = request.path or ""
        if self.requires_no_store_headers(path):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
