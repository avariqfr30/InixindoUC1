"""Flask entrypoint for the proposal generator app."""
import concurrent.futures
from datetime import timedelta
from functools import wraps
import hashlib
import io
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_cors import CORS
import requests
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from waitress import serve as waitress_serve
except Exception:  # pragma: no cover - optional local server dependency
    waitress_serve = None

from .config import (
    APP_HOST,
    APP_PORT,
    APP_SECRET_KEY,
    AUTH_FORCE_SINGLE_SESSION,
    AUTH_MAX_GLOBAL_ACTIVE_SESSIONS,
    AUTH_MAX_SESSIONS_PER_USER,
    DB_URI,
    GENERATION_PROFILE,
    JOB_POLL_INTERVAL_MS,
    JOB_RETENTION_SECONDS,
    LOGIN_RATE_LIMIT_BLOCK_SECONDS,
    LOGIN_RATE_LIMIT_MAX_ATTEMPTS,
    LOGIN_RATE_LIMIT_WINDOW_SECONDS,
    MANAGED_INTERNAL_API_CONFIG_PATH,
    MAX_ACTIVE_GENERATIONS,
    MAX_GENERATION_BACKLOG,
    OLLAMA_HOST,
    PROJECT_CSV_PATH,
    PROJECT_DB_PATH,
    PROPOSAL_MODES,
    SESSION_ABSOLUTE_TIMEOUT_HOURS,
    SESSION_COOKIE_SECURE,
    SESSION_IDLE_TIMEOUT_MINUTES,
    SESSION_TOUCH_INTERVAL_SECONDS,
    SMART_SUGGESTIONS,
)
from .core import FinancialAnalyzer, KnowledgeBase, ProposalGenerator
from .runtime_components import AppStateStore
from .runtime_components import InternalDataClient, FirmAPIClient

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from .config import SERPER_API_KEY
from .runtime_components import Researcher

# Initialize and log Serper availability
def _log_serper_status() -> None:
    """Log Serper API availability at app startup."""
    try:
        has_serper = Researcher._has_serper_key()
        if has_serper:
            logger.info("✓ Serper API Key loaded successfully | OSINT research features ENABLED")
        else:
            key_status = "not configured" if not SERPER_API_KEY else "placeholder key detected"
            logger.warning(f"⚠ Serper API Key {key_status} | OSINT research features DISABLED | Set SERPER_API_KEY environment variable to enable")
    except Exception as e:
        logger.error(f"✗ Error checking Serper status: {e}")

_log_serper_status()

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
CORS(app)
app.config.update(
    SECRET_KEY=APP_SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=SESSION_IDLE_TIMEOUT_MINUTES),
)

knowledge_base = KnowledgeBase(DB_URI)
proposal_generator = ProposalGenerator(knowledge_base)
app_state_store = AppStateStore()
proposal_generator.app_state_store = app_state_store


class GenerationQueue:
    """Small in-memory queue for low-level shared testing."""

    def __init__(
        self,
        generator: ProposalGenerator,
        state_store: AppStateStore,
        max_active: int,
        max_backlog: int,
        retention_seconds: int,
    ) -> None:
        self.generator = generator
        self.state_store = state_store
        self.max_active = max_active
        self.max_backlog = max_backlog
        self.retention_seconds = retention_seconds
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_active,
            thread_name_prefix="proposal-job"
        )
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._sequence = 0

    @staticmethod
    def _payload_fingerprint(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _cleanup_locked(self) -> None:
        now = time.time()
        stale_ids = []
        for job_id, job in self._jobs.items():
            if job["status"] in {"done", "failed"} and job.get("finished_at"):
                if now - float(job["finished_at"]) > self.retention_seconds:
                    stale_ids.append(job_id)
        for job_id in stale_ids:
            self._jobs.pop(job_id, None)

    def _queue_position_locked(self, sequence: int) -> int:
        ahead = sum(
            1
            for job in self._jobs.values()
            if job["status"] == "queued" and int(job["sequence"]) < sequence
        )
        return ahead + 1

    def _live_count_locked(self) -> int:
        return sum(1 for job in self._jobs.values() if job["status"] in {"queued", "running"})

    def _build_message_locked(self, job: Dict[str, Any]) -> str:
        status = job["status"]
        if status == "queued":
            position = self._queue_position_locked(int(job["sequence"]))
            return f"Permintaan tersimpan. Menunggu antrean, posisi {position}."
        if status == "running":
            return "Sedang menyusun proposal. Dokumen akan otomatis diunduh setelah selesai."
        if status == "done":
            return "Proposal selesai dibuat dan siap diunduh."
        return str(job.get("error") or "Proses generate proposal gagal.")

    def _snapshot_locked(self, job: Dict[str, Any]) -> Dict[str, Any]:
        started_at = job.get("started_at")
        finished_at = job.get("finished_at")
        processing_seconds = None
        if started_at:
            processing_seconds = round((finished_at or time.time()) - float(started_at), 1)
        snapshot = {
            "job_id": job["job_id"],
            "history_id": job.get("history_id"),
            "status": job["status"],
            "message": self._build_message_locked(job),
            "queue_position": self._queue_position_locked(int(job["sequence"])) if job["status"] == "queued" else None,
            "filename": job.get("filename"),
            "created_at": job.get("created_at"),
            "started_at": started_at,
            "finished_at": finished_at,
            "processing_seconds": processing_seconds,
            "download_ready": job["status"] == "done",
            "acceptance_score": job.get("acceptance_score"),
            "acceptance_passes": job.get("acceptance_passes"),
        }
        if job["status"] == "failed":
            snapshot["error"] = job.get("error") or "Proses generate proposal gagal."
        return snapshot

    def submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._cleanup_locked()
            fingerprint = self._payload_fingerprint(payload)
            for job in self._jobs.values():
                if job["status"] in {"queued", "running"} and job.get("fingerprint") == fingerprint:
                    snapshot = self._snapshot_locked(job)
                    snapshot["deduplicated"] = True
                    snapshot["message"] = "Permintaan yang sama sudah sedang diproses. Sistem melanjutkan job yang sudah ada."
                    return snapshot
            if self._live_count_locked() >= self.max_backlog:
                raise OverflowError(
                    "Server sedang padat. Coba lagi beberapa menit lagi setelah antrean berkurang."
                )
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "sequence": self._next_sequence(),
                "status": "queued",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "fingerprint": fingerprint,
                "payload": dict(payload),
                "filename": None,
                "result_bytes": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._executor.submit(self._run_job, job_id)
            return self._snapshot_locked(job)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["started_at"] = time.time()
            payload = dict(job.get("payload") or {})

        try:
            doc, filename, generation_meta = self.generator.generate_document(
                client=payload["nama_perusahaan"],
                project=payload["konteks_organisasi"],
                budget=payload["estimasi_biaya"],
                service_type=payload["jenis_proposal"],
                project_goal=payload["klasifikasi_kebutuhan"],
                project_type=payload["jenis_proyek"],
                timeline=payload["estimasi_waktu"],
                notes=payload["permasalahan"],
                regulations=payload["potensi_framework"],
                chapter_id=payload.get("chapter_id"),
                proposal_mode=payload.get("mode_proposal", "canvassing"),
            )
            output = io.BytesIO()
            doc.save(output)
            result_bytes = output.getvalue()
            finished_at = time.time()
            acceptance_report = dict((generation_meta or {}).get("acceptance_report") or {})
            processing_seconds = max(0.0, finished_at - float(job.get("started_at") or finished_at))

            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                stored_path = self.state_store.persist_generated_file(f"{filename}.docx", result_bytes)
                history_id = self.state_store.add_history_entry(
                    payload=payload,
                    filename=stored_path.name,
                    filepath=str(stored_path),
                    created_at=job.get("created_at") or time.time(),
                    finished_at=finished_at,
                    acceptance_report=acceptance_report,
                    processing_seconds=processing_seconds,
                )
                job["status"] = "done"
                job["finished_at"] = finished_at
                job["filename"] = stored_path.name
                job["result_bytes"] = result_bytes
                job["history_id"] = history_id
                job["acceptance_score"] = acceptance_report.get("score")
                job["acceptance_passes"] = acceptance_report.get("passes")
                job["payload"] = None
        except Exception as exc:  # keep the queue resilient during shared testing
            logger.exception("Generation job %s failed", job_id)
            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                job["status"] = "failed"
                job["finished_at"] = time.time()
                job["error"] = str(exc) or "Proses generate proposal gagal."
                job["payload"] = None

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                return None
            return self._snapshot_locked(job)

    def get_download(self, job_id: str) -> Optional[Tuple[str, bytes]]:
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job or job["status"] != "done" or not job.get("result_bytes"):
                return None
            return str(job.get("filename") or f"proposal_{job_id}.docx"), bytes(job["result_bytes"])

    def has_live_jobs(self) -> bool:
        with self._lock:
            self._cleanup_locked()
            return any(job["status"] in {"queued", "running"} for job in self._jobs.values())


generation_queue = GenerationQueue(
    generator=proposal_generator,
    state_store=app_state_store,
    max_active=MAX_ACTIVE_GENERATIONS,
    max_backlog=MAX_GENERATION_BACKLOG,
    retention_seconds=JOB_RETENTION_SECONDS,
)

PUBLIC_ENDPOINTS = {
    "auth_page",
    "login",
    "signup",
    "logout",
    "health",
    "ready",
    "static",
}
NO_STORE_EXACT_PATHS = {"/", "/auth", "/login", "/signup", "/logout", "/generate"}


def _current_username() -> str:
    return str(session.get("auth_username") or "").strip()


def _current_session_id() -> str:
    return str(session.get("auth_session_id") or "").strip()


def _request_ip() -> str:
    forwarded = str(request.headers.get("X-Forwarded-For", "") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()[:96]
    return str(request.remote_addr or "").strip()[:96]


def _request_user_agent() -> str:
    return str(request.headers.get("User-Agent", "") or "").strip()[:256]


def _is_authenticated() -> bool:
    return bool(_current_username() and _current_session_id())


def _is_api_request() -> bool:
    return request.path.startswith("/api/") or request.path == "/generate"


def _requires_no_store_headers(path: str) -> bool:
    return path in NO_STORE_EXACT_PATHS or path.startswith("/api/")


def _safe_next_target(raw_target: str) -> str:
    next_target = str(raw_target or "").strip()
    if next_target.startswith("/") and not next_target.startswith("//"):
        return next_target
    return url_for("home")


def _login_redirect_target() -> str:
    return _safe_next_target(request.args.get("next"))


def _set_authenticated_user(username: str) -> None:
    normalized_username = str(username or "").strip()
    if not normalized_username:
        raise ValueError("Username is required.")
    tracked_session_id = app_state_store.create_user_session(
        username=normalized_username,
        remote_ip=_request_ip(),
        user_agent=_request_user_agent(),
        idle_timeout_seconds=SESSION_IDLE_TIMEOUT_MINUTES * 60,
        absolute_timeout_seconds=SESSION_ABSOLUTE_TIMEOUT_HOURS * 3600,
        force_single_session=AUTH_FORCE_SINGLE_SESSION,
        max_sessions_per_user=AUTH_MAX_SESSIONS_PER_USER,
        max_global_sessions=AUTH_MAX_GLOBAL_ACTIVE_SESSIONS,
    )
    if not tracked_session_id:
        raise RuntimeError("Unable to create authenticated session.")
    session.clear()
    session["auth_username"] = normalized_username
    session["auth_session_id"] = tracked_session_id
    session.permanent = True


def _clear_authenticated_user(revoke: bool = True) -> None:
    tracked_session_id = _current_session_id()
    if revoke and tracked_session_id:
        app_state_store.revoke_user_session(tracked_session_id)
    session.clear()


def _touch_active_auth_session() -> bool:
    username = _current_username()
    tracked_session_id = _current_session_id()
    if not username or not tracked_session_id:
        return False
    return app_state_store.touch_user_session(
        session_id=tracked_session_id,
        username=username,
        remote_ip=_request_ip(),
        idle_timeout_seconds=SESSION_IDLE_TIMEOUT_MINUTES * 60,
        touch_interval_seconds=SESSION_TOUCH_INTERVAL_SECONDS,
    )


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if _is_authenticated():
            return view_func(*args, **kwargs)
        if _is_api_request():
            return jsonify({"error": "Authentication required."}), 401
        return redirect(url_for("auth_page", next=request.full_path if request.query_string else request.path))
    return wrapper


@app.before_request
def _enforce_authentication():
    endpoint = request.endpoint or ""
    if _is_authenticated():
        if _touch_active_auth_session():
            return None
        _clear_authenticated_user(revoke=False)
    if endpoint in PUBLIC_ENDPOINTS:
        return None
    if _is_api_request():
        return jsonify({"error": "Authentication required."}), 401
    return redirect(url_for("auth_page", next=request.full_path if request.query_string else request.path))


@app.after_request
def _apply_auth_cache_headers(response):
    path = request.path or ""
    if _requires_no_store_headers(path):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _normalize_client_name(raw_name: str) -> str:
    return re.sub(r'\b(Cabang|Branch|Tbk)\b.*$|^(PT\.|CV\.)', '', raw_name or '', flags=re.IGNORECASE).strip()


def _warm_request_context(data: Dict[str, Any]) -> str:
    client_name = _normalize_client_name(str(data.get("nama_perusahaan", "")).strip())
    regulations = str(data.get("potensi_framework", "")).strip()
    ai_context = " ".join([
        str(data.get("konteks_organisasi", "")).strip(),
        str(data.get("permasalahan", "")).strip(),
        str(data.get("klasifikasi_kebutuhan", "")).strip(),
        str(data.get("jenis_proyek", "")).strip(),
        str(data.get("jenis_proposal", "")).strip(),
        str(data.get("mode_proposal", "")).strip(),
    ]).strip()
    if not client_name:
        return "skipped"
    return proposal_generator.prefetch_research_bundle(
        base_client=client_name,
        regulations=regulations,
        include_collaboration=proposal_generator.firm_api.uses_demo_logic(),
        ai_context=ai_context,
    )


def _build_internal_api_config(data: Dict[str, Any]) -> Dict[str, Any]:
    endpoint_url = str(data.get("url") or "").strip()
    if not re.match(r"^https?://", endpoint_url, flags=re.IGNORECASE):
        raise ValueError("Endpoint API harus berupa URL HTTP/HTTPS penuh.")

    method = str(data.get("method") or "POST").strip().upper()
    if method not in {"GET", "POST", "PUT", "PATCH"}:
        method = "POST"
    body_encoding = str(data.get("body_encoding") or "form").strip().lower()
    if body_encoding not in {"form", "json"}:
        body_encoding = "form"
    auth_mode = str(data.get("auth_mode") or "basic").strip().lower()
    if auth_mode not in {"basic", "bearer", "none"}:
        auth_mode = "basic"

    datasets = data.get("datasets") if isinstance(data.get("datasets"), dict) else {}
    paths = data.get("response_paths") if isinstance(data.get("response_paths"), dict) else {}
    return {
        "mode": "generic",
        "auth_mode": auth_mode,
        "request_defaults": {
            "url": endpoint_url,
            "method": method,
            "body_encoding": body_encoding,
            "params": {},
            "headers": {},
        },
        "resources": {
            "firm_profile": {
                "request": {"body": {"dataset": str(datasets.get("firm_profile") or "ReferenceAccount")}},
                "response_path": str(paths.get("firm_profile") or "data.dataset_result.0"),
                "field_mapping": {
                    "office_address": "company_address",
                    "email": "official_email",
                    "phone": "telephone",
                    "whatsapp": "whatsapp",
                    "website": "website_url",
                    "legal_name": "legal_name",
                    "operating_hours": "operating_hours",
                    "profile_summary": "profile_summary",
                    "credential_highlights": "credential_highlights",
                    "portfolio_highlights": "portfolio_highlights",
                },
            },
            "project_standards": {
                "request": {"body": {"dataset": str(datasets.get("project_standards") or "ProjectStandards")}},
                "response_path": str(paths.get("project_standards") or "data.dataset_result"),
                "record_filters": {"project_type": "{project_type}"},
                "field_mapping": {
                    "methodology": "delivery_methodology",
                    "team": "team_composition",
                    "commercial": "commercial_terms",
                },
            },
            "client_relationship": {
                "request": {"body": {"dataset": str(datasets.get("client_relationship") or "ClientRelationship")}},
                "response_path": str(paths.get("client_relationship") or "data.dataset_result"),
                "record_filters": {"client_name": "{client_name}"},
                "field_mapping": {
                    "summary": "relationship_summary",
                    "mode": "relationship_status",
                },
            },
            "project_records": {
                "request": {"body": {"dataset": str(datasets.get("project_records") or "Projects")}},
                "response_path": str(paths.get("project_records") or "data.dataset_result"),
                "field_mapping": {
                    "entity": "client_entity",
                    "topic": "strategic_initiative",
                    "budget": "investment_estimation",
                },
            },
        },
    }


def _internal_api_config_target() -> Path:
    configured = str(FirmAPIClient._resolve_config_file() or "").strip()
    if configured:
        return Path(configured).expanduser()
    return MANAGED_INTERNAL_API_CONFIG_PATH


def _write_internal_api_config(config_payload: Dict[str, Any]) -> Path:
    target = _internal_api_config_target()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


@app.route('/')
@login_required
def home():
    return render_template('index.html', current_user=_current_username())


@app.route('/auth')
def auth_page():
    if _is_authenticated():
        if _touch_active_auth_session():
            return redirect(_login_redirect_target())
        _clear_authenticated_user(revoke=False)
    return render_template(
        'auth.html',
        login_error=request.args.get("login_error", "").strip(),
        signup_error=request.args.get("signup_error", "").strip(),
        signup_success=request.args.get("signup_success", "").strip(),
        next_target=_login_redirect_target(),
    )


@app.route('/login', methods=['POST'])
def login():
    username = str(request.form.get("username") or "").strip()
    password = str(request.form.get("password") or "")
    next_target = _safe_next_target(request.form.get("next"))
    remote_ip = _request_ip()
    if not username or not password:
        return redirect(url_for("auth_page", login_error="Username dan password wajib diisi.", next=next_target))
    blocked_seconds = app_state_store.get_login_block_seconds(username=username, remote_ip=remote_ip)
    if blocked_seconds > 0:
        return redirect(
            url_for(
                "auth_page",
                login_error=f"Terlalu banyak percobaan login. Coba lagi dalam {blocked_seconds} detik.",
                next=next_target,
            )
        )

    user = app_state_store.get_user(username)
    password_hash = str((user or {}).get("password_hash") or "")
    if not user or not password_hash or not check_password_hash(password_hash, password):
        blocked_after_fail = app_state_store.register_login_failure(
            username=username,
            remote_ip=remote_ip,
            window_seconds=LOGIN_RATE_LIMIT_WINDOW_SECONDS,
            max_attempts=LOGIN_RATE_LIMIT_MAX_ATTEMPTS,
            block_seconds=LOGIN_RATE_LIMIT_BLOCK_SECONDS,
        )
        if blocked_after_fail > 0:
            return redirect(
                url_for(
                    "auth_page",
                    login_error=f"Terlalu banyak percobaan login. Coba lagi dalam {blocked_after_fail} detik.",
                    next=next_target,
                )
            )
        return redirect(url_for("auth_page", login_error="Username atau password tidak cocok.", next=next_target))

    app_state_store.clear_login_failures(username=username, remote_ip=remote_ip)
    try:
        _set_authenticated_user(str(user.get("username") or username))
    except Exception:
        logger.exception("Failed creating authenticated session for user=%s", username)
        return redirect(
            url_for(
                "auth_page",
                login_error="Sesi login gagal dibuat. Coba lagi beberapa saat.",
                next=next_target,
            )
        )
    return redirect(next_target)


@app.route('/signup', methods=['POST'])
def signup():
    username = str(request.form.get("username") or "").strip()
    password = str(request.form.get("password") or "")
    confirm_password = str(request.form.get("confirm_password") or "")
    next_target = _safe_next_target(request.form.get("next"))

    if not username or not password:
        return redirect(url_for("auth_page", signup_error="Username dan password wajib diisi.", next=next_target))
    if len(username) < 3:
        return redirect(url_for("auth_page", signup_error="Username minimal 3 karakter.", next=next_target))
    if len(password) < 6:
        return redirect(url_for("auth_page", signup_error="Password minimal 6 karakter.", next=next_target))
    if password != confirm_password:
        return redirect(url_for("auth_page", signup_error="Konfirmasi password tidak cocok.", next=next_target))

    created = app_state_store.create_user(
        username=username,
        password_hash=generate_password_hash(password, method="pbkdf2:sha256"),
    )
    if not created:
        return redirect(url_for("auth_page", signup_error="Username sudah dipakai. Gunakan username lain.", next=next_target))

    return redirect(url_for("auth_page", signup_success="Akun berhasil dibuat. Silakan login.", next=next_target))


@app.route('/logout', methods=['POST'])
def logout():
    _clear_authenticated_user()
    return redirect(url_for("auth_page"))


@app.route('/api/auth/session-status')
@login_required
def auth_session_status():
    username = _current_username()
    snapshot = app_state_store.active_session_snapshot(username=username)
    return jsonify({
        "username": username,
        "session": {
            "user_active_sessions": snapshot.get("user_active_sessions", 0),
            "global_active_sessions": snapshot.get("global_active_sessions", 0),
            "force_single_session": AUTH_FORCE_SINGLE_SESSION,
            "max_per_user": AUTH_MAX_SESSIONS_PER_USER,
            "max_global": AUTH_MAX_GLOBAL_ACTIVE_SESSIONS,
            "idle_timeout_minutes": SESSION_IDLE_TIMEOUT_MINUTES,
            "absolute_timeout_hours": SESSION_ABSOLUTE_TIMEOUT_HOURS,
        },
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "service": "proposal-generator",
        "timestamp": time.time(),
    })


@app.route('/ready')
def ready():
    checks: Dict[str, Dict[str, Any]] = {}

    checks["project_db"] = {
        "ok": PROJECT_DB_PATH.exists(),
        "path": str(PROJECT_DB_PATH),
    }
    checks["project_csv"] = {
        "ok": PROJECT_CSV_PATH.exists(),
        "path": str(PROJECT_CSV_PATH),
    }
    checks["knowledge_base"] = {
        "ok": bool(
            knowledge_base.df is not None
            and not knowledge_base.df.empty
            and "entity" in knowledge_base.df.columns
            and "topic" in knowledge_base.df.columns
            and getattr(knowledge_base, "vector_ready", False)
            and not getattr(knowledge_base, "last_refresh_error", "")
        ),
        "error": getattr(knowledge_base, "last_refresh_error", ""),
        "sync_in_progress": getattr(knowledge_base, "sync_in_progress", False),
        "vector_store_dir": str(getattr(knowledge_base, "vector_store_dir", "")),
    }

    app_state_ok = False
    app_state_error = ""
    try:
        app_state_store.get_settings()
        app_state_ok = (
            app_state_store.db_path.exists()
            and app_state_store.generated_dir.exists()
            and app_state_store.templates_dir.exists()
            and app_state_store.supporting_docs_dir.exists()
        )
    except Exception as exc:
        app_state_error = str(exc)
    checks["app_state"] = {
        "ok": app_state_ok,
        "db_path": str(app_state_store.db_path),
        "generated_dir": str(app_state_store.generated_dir),
        "templates_dir": str(app_state_store.templates_dir),
        "supporting_docs_dir": str(app_state_store.supporting_docs_dir),
        "error": app_state_error,
    }

    ollama_ok = False
    ollama_error = ""
    try:
        resp = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/version", timeout=4)
        ollama_ok = resp.ok
        if not resp.ok:
            ollama_error = f"HTTP {resp.status_code}"
    except Exception as exc:
        ollama_error = str(exc)
    checks["ollama"] = {
        "ok": ollama_ok,
        "host": OLLAMA_HOST,
        "error": ollama_error,
    }

    ready_ok = all(item.get("ok") for item in checks.values())
    status_code = 200 if ready_ok else 503
    return jsonify({
        "status": "ready" if ready_ok else "degraded",
        "checks": checks,
        "timestamp": time.time(),
    }), status_code


@app.route('/api/config')
def get_base_config():
    return jsonify({
        "suggestions": SMART_SUGGESTIONS,
        "data_acquisition_mode": proposal_generator.firm_api.data_acquisition_mode,
        "demo_mode": proposal_generator.firm_api.demo_mode,
        "internal_data_runtime": proposal_generator.firm_api.describe_runtime(),
        "generation_profile": GENERATION_PROFILE,
        "job_poll_interval_ms": JOB_POLL_INTERVAL_MS,
        "max_active_generations": MAX_ACTIVE_GENERATIONS,
        "max_generation_backlog": MAX_GENERATION_BACKLOG,
        "proposal_modes": PROPOSAL_MODES,
    })


@app.route('/api/internal-api/setup', methods=['GET', 'POST'])
def internal_api_setup():
    if request.method == 'GET':
        api_client = FirmAPIClient(force_source="api")
        config_file = api_client.config_file or str(MANAGED_INTERNAL_API_CONFIG_PATH)
        return jsonify({
            "config_file": config_file,
            "config_exists": bool(config_file and Path(config_file).exists()),
            "project_data_source": getattr(knowledge_base, "project_data_source", "local"),
            "runtime": api_client.describe_runtime(),
            "validation": api_client.validate_config(),
        })

    data = request.json or {}
    try:
        config_payload = _build_internal_api_config(data)
        target_path = _write_internal_api_config(config_payload)
        api_client = FirmAPIClient(force_source="api")
        validation = api_client.validate_config()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Internal API setup failed")
        return jsonify({"error": str(exc)}), 500

    activated = bool(data.get("activate_now", True))
    refresh_started = False
    if activated:
        proposal_generator.firm_api = InternalDataClient(force_source="api")
        knowledge_base.set_project_data_source("api")
        if not generation_queue.has_live_jobs():
            refresh_started = knowledge_base.refresh_data(background=True)

    return jsonify({
        "status": "ok",
        "config_file": str(target_path),
        "activated": activated,
        "refresh_started": refresh_started,
        "project_data_source": getattr(knowledge_base, "project_data_source", "local"),
        "validation": validation,
        "notes": [
            "Kredensial tetap dibaca dari environment variable agar password tidak disimpan di UI.",
            "Agar aktif permanen setelah restart, set PROJECT_DATA_SOURCE=api dan FIRM_API_CONFIG_FILE ke path config ini.",
        ],
    })


@app.route('/api/companies')
def get_companies():
    if knowledge_base.df is None or knowledge_base.df.empty or 'entity' not in knowledge_base.df.columns:
        return jsonify([])
    companies = knowledge_base.df['entity'].dropna().astype(str).str.strip().unique().tolist()
    companies = [c for c in companies if c.lower() != 'nan' and c]
    return jsonify(sorted(companies))


def _company_candidates() -> list[str]:
    if knowledge_base.df is None or knowledge_base.df.empty or 'entity' not in knowledge_base.df.columns:
        return []
    companies = knowledge_base.df['entity'].dropna().astype(str).str.strip().unique().tolist()
    return sorted([c for c in companies if c.lower() != 'nan' and c])


@app.route('/api/suggest-budget', methods=['POST'])
def suggest_budget():
    """Estimate pricing tiers from public financial signals."""
    data = request.json or {}
    required_fields = [
        'nama_perusahaan',
        'mode_proposal',
        'jenis_proposal',
        'jenis_proyek',
        'konteks_organisasi',
        'permasalahan',
        'klasifikasi_kebutuhan',
        'estimasi_waktu',
        'potensi_framework',
    ]
    missing = [field for field in required_fields if not str(data.get(field, '')).strip()]
    if missing:
        return jsonify({"error": f"Lengkapi field berikut sebelum analisis finansial: {', '.join(missing)}"}), 400

    _warm_request_context(data)

    client_name = data.get('nama_perusahaan', '')
    project_type = data.get('jenis_proyek', '')
    acquisition_mode = proposal_generator.firm_api.data_acquisition_mode
    commercial_context = ""
    if not proposal_generator.firm_api.uses_demo_logic():
        commercial_context = proposal_generator.firm_api.get_project_standards(project_type).get('commercial', '')

    analyzer = FinancialAnalyzer(proposal_generator.ollama)
    result = analyzer.suggest_budget(
        client_name=client_name,
        timeline=data.get('estimasi_waktu', ''),
        project_type=project_type,
        service_type=data.get('jenis_proposal', ''),
        project_goal=data.get('klasifikasi_kebutuhan', ''),
        objective=data.get('konteks_organisasi', ''),
        notes=data.get('permasalahan', ''),
        frameworks=data.get('potensi_framework', ''),
        commercial_context=commercial_context,
        pricing_mode=acquisition_mode,
    )
    return jsonify(result)


@app.route('/api/preview-outline', methods=['POST'])
def preview_outline():
    data = request.json or {}
    _warm_request_context(data)
    outline = proposal_generator.build_preview_outline(data)
    return jsonify({"outline": outline})


@app.route('/api/prefetch-context', methods=['POST'])
def prefetch_context():
    data = request.json or {}
    status = _warm_request_context(data)
    return jsonify({"status": status})


@app.route('/api/kak-context')
def get_kak_context():
    return jsonify(app_state_store.get_latest_kak_context(company_candidates=_company_candidates()))


@app.route('/generate', methods=['POST'])
def generate_proposal():
    data = request.json or {}
    required_fields = [
        'nama_perusahaan',
        'konteks_organisasi',
        'estimasi_biaya',
        'mode_proposal',
        'jenis_proposal',
        'klasifikasi_kebutuhan',
        'jenis_proyek',
        'estimasi_waktu',
        'permasalahan',
        'potensi_framework',
    ]
    missing = [field for field in required_fields if not str(data.get(field, '')).strip()]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        ticket = generation_queue.submit(data)
    except OverflowError as exc:
        return jsonify({"error": str(exc)}), 429

    return jsonify(ticket), 202


@app.route('/api/jobs/<job_id>')
def get_job_status(job_id: str):
    job = generation_queue.get(job_id)
    if not job:
        return jsonify({"error": "Job tidak ditemukan atau hasilnya sudah dibersihkan dari antrean."}), 404
    return jsonify(job)


@app.route('/api/jobs/<job_id>/download')
def download_job(job_id: str):
    result = generation_queue.get_download(job_id)
    if not result:
        status = generation_queue.get(job_id)
        if not status:
            return jsonify({"error": "Job tidak ditemukan atau hasilnya sudah dibersihkan dari antrean."}), 404
        if status["status"] != "done":
            return jsonify({"error": "Dokumen belum siap diunduh."}), 409
        return jsonify({"error": "Dokumen selesai dibuat, tetapi file hasil tidak tersedia."}), 410

    filename, result_bytes = result
    return send_file(
        io.BytesIO(result_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


@app.route('/api/history')
def list_history():
    return jsonify({"items": app_state_store.list_history(limit=30)})


@app.route('/api/history/<entry_id>')
def get_history_entry(entry_id: str):
    entry = app_state_store.get_history_entry(entry_id)
    if not entry:
        return jsonify({"error": "Riwayat proposal tidak ditemukan."}), 404
    return jsonify(entry)


@app.route('/api/history/<entry_id>/reuse')
def reuse_history(entry_id: str):
    entry = app_state_store.get_history_entry(entry_id)
    if not entry:
        return jsonify({"error": "Riwayat proposal tidak ditemukan."}), 404
    if not entry.get("can_reuse"):
        return jsonify({"error": "Proposal historis ini hanya tersedia sebagai referensi unduhan, belum memiliki payload yang bisa dipakai ulang."}), 409
    return jsonify({"payload": entry.get("payload", {})})


@app.route('/api/history/<entry_id>/download')
def download_history(entry_id: str):
    entry = app_state_store.get_history_entry(entry_id)
    if not entry:
        return jsonify({"error": "Riwayat proposal tidak ditemukan."}), 404
    filepath = Path(str(entry.get("filepath") or ""))
    if not filepath.exists():
        return jsonify({"error": "File proposal pada riwayat ini sudah tidak tersedia."}), 410
    return send_file(
        str(filepath),
        as_attachment=True,
        download_name=str(entry.get("filename") or filepath.name),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


@app.route('/api/settings', methods=['GET', 'POST'])
def proposal_settings():
    if request.method == 'GET':
        return jsonify(app_state_store.get_settings())

    data = request.json or {}
    settings = app_state_store.save_settings(
        internal_portfolio=str(data.get("internal_portfolio") or ""),
        internal_credentials=str(data.get("internal_credentials") or ""),
    )
    return jsonify(settings)


@app.route('/api/settings/template', methods=['POST', 'DELETE'])
def proposal_template_settings():
    if request.method == 'DELETE':
        return jsonify(app_state_store.clear_template())

    uploaded = request.files.get('template')
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "File template .docx belum dipilih."}), 400
    if not uploaded.filename.lower().endswith('.docx'):
        return jsonify({"error": "Template harus berformat .docx."}), 400
    settings = app_state_store.save_template(uploaded.filename, uploaded.read())
    return jsonify(settings)


@app.route('/api/settings/documents', methods=['POST'])
def proposal_supporting_documents():
    document_type = str(request.form.get('document_type') or "").strip()
    uploaded_files = request.files.getlist('files')
    uploads: list[tuple[str, bytes]] = []
    for item in uploaded_files:
        if not item or not item.filename:
            continue
        uploads.append((str(item.filename), item.read()))
    if not uploads:
        return jsonify({"error": "Pilih minimal satu file pendukung untuk diunggah."}), 400
    try:
        settings = app_state_store.save_supporting_documents(document_type, uploads)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(settings)


@app.route('/api/settings/documents/<document_id>', methods=['DELETE'])
def delete_proposal_supporting_document(document_id: str):
    try:
        settings = app_state_store.delete_supporting_document(document_id)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(settings)


@app.route('/refresh-knowledge', methods=['POST'])
def refresh_knowledge():
    if generation_queue.has_live_jobs():
        return jsonify({"status": "error", "error": "Knowledge base tidak bisa di-refresh saat masih ada generate job yang berjalan atau mengantre."}), 409
    success = knowledge_base.refresh_data()
    return jsonify({"status": "success" if success else "error"})


if __name__ == '__main__':
    if waitress_serve is not None:
        waitress_threads = max(6, MAX_ACTIVE_GENERATIONS * 4)
        logger.info(
            "Starting Waitress on %s:%s with %s worker threads",
            APP_HOST,
            APP_PORT,
            waitress_threads,
        )
        waitress_serve(app, host=APP_HOST, port=APP_PORT, threads=waitress_threads)
    else:
        logger.warning("Waitress is not installed. Falling back to Flask dev server.")
        app.run(host=APP_HOST, port=APP_PORT, debug=False, threaded=True)
