"""Flask entrypoint for the proposal generator app."""
from datetime import timedelta
import io
import os
import secrets
import string
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from waitress import serve as waitress_serve
except Exception:  # pragma: no cover - optional local server dependency
    waitress_serve = None

from .config import (
    APP_HOST,
    APP_PORT,
    APP_SECRET_KEY,
    AUTH_ALLOW_SIGNUP,
    AUTH_FORCE_SINGLE_SESSION,
    AUTH_MAX_GLOBAL_ACTIVE_SESSIONS,
    AUTH_MAX_SESSIONS_PER_USER,
    AUTH_REQUIRE_SIGNUP_APPROVAL,
    AUTH_SIGNUP_VERIFICATION_DELIVERY_MODE,
    AUTH_SIGNUP_VERIFICATION_TIMEOUT_SECONDS,
    AUTH_SIGNUP_VERIFICATION_WEBHOOK_URL,
    AUTH_SIGNUP_EMAIL_DOMAIN,
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
    PROJECT_DATA_SOURCE,
    PROJECT_DB_PATH,
    PROPOSAL_MODES,
    SESSION_ABSOLUTE_TIMEOUT_HOURS,
    SESSION_COOKIE_SECURE,
    SESSION_IDLE_TIMEOUT_MINUTES,
    SESSION_TOUCH_INTERVAL_SECONDS,
    SMART_SUGGESTIONS,
    CORS_ALLOWED_ORIGINS,
)
from .auth_flow import AuthFlow
from .core import ProposalGenerator
from .finance import FinancialAnalyzer
from .internal_evidence_summary import build_internal_evidence_summary
from .job_queue import GenerationQueue
from .knowledge_store import KnowledgeBase
from .proposal_request_service import BUDGET_REQUIRED_FIELDS, ProposalRequestService
from .readiness import build_readiness_payload
from .runtime_services import ClientContextService, FrameworkOptionService, InternalApiRuntimeService, normalize_client_name
from .state_store import AppStateStore
from .text_hygiene import normalize_payload

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from .config import SERPER_API_KEY
from .research import Researcher

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

from flask_wtf.csrf import CSRFProtect, generate_csrf

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
CORS(app, origins=CORS_ALLOWED_ORIGINS, supports_credentials=True)
app.config.update(
    SECRET_KEY=APP_SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=SESSION_IDLE_TIMEOUT_MINUTES),
    WTF_CSRF_ENABLED=not (
        app.config.get("TESTING")
        or app.testing
        or os.getenv("WTF_CSRF_ENABLED") == "0"
        or os.getenv("DISABLE_CSRF_FOR_TESTING") == "1"
    )
)

csrf = CSRFProtect(app)

@app.after_request
def _inject_csrf_cookie(response):
    response.set_cookie("csrf_token", generate_csrf(), samesite="Lax", httponly=False)
    return response


@app.after_request
def _apply_security_headers(response):
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        ),
    )
    return response

knowledge_base = KnowledgeBase(DB_URI)
proposal_generator = ProposalGenerator(knowledge_base)
app_state_store = AppStateStore()
proposal_generator.app_state_store = app_state_store

# Seed test1234 account
if not app_state_store.auth.get_user("test1234"):
    app_state_store.auth.create_user(
        username="test1234",
        password_hash=generate_password_hash("1234", method="pbkdf2:sha256"),
        status="approved",
        approved_by="bootstrap",
    )


generation_queue = GenerationQueue(
    generator=proposal_generator,
    state_store=app_state_store,
    max_active=MAX_ACTIVE_GENERATIONS,
    max_backlog=MAX_GENERATION_BACKLOG,
    retention_seconds=JOB_RETENTION_SECONDS,
)
client_context_service = ClientContextService(
    proposal_generator=proposal_generator,
    knowledge_base=knowledge_base,
    prefetch_research=lambda data: proposal_request_service.warm_request_context(data),
)
framework_option_service = FrameworkOptionService(lambda: proposal_generator.firm_api)
proposal_request_service = ProposalRequestService(
    proposal_generator=proposal_generator,
    app_state_store=app_state_store,
    client_context_service=client_context_service,
    generation_queue=generation_queue,
    framework_option_service=framework_option_service,
)
internal_api_service = InternalApiRuntimeService(
    knowledge_base=knowledge_base,
    proposal_generator=proposal_generator,
    generation_queue=generation_queue,
    managed_config_path=MANAGED_INTERNAL_API_CONFIG_PATH,
)

PUBLIC_ENDPOINTS = {
    "auth_page",
    "login",
    "login_page",
    "signup",
    "signup_page",
    "logout",
    "health",
    "ready",
    "static",
    "verify_otp",
}
NO_STORE_EXACT_PATHS = {"/", "/auth", "/login", "/signup", "/logout", "/generate", "/verify-otp"}

auth_flow = AuthFlow(
    state_store=app_state_store,
    public_endpoints=PUBLIC_ENDPOINTS,
    no_store_paths=NO_STORE_EXACT_PATHS,
    idle_timeout_minutes=SESSION_IDLE_TIMEOUT_MINUTES,
    absolute_timeout_hours=SESSION_ABSOLUTE_TIMEOUT_HOURS,
    touch_interval_seconds=SESSION_TOUCH_INTERVAL_SECONDS,
    force_single_session=AUTH_FORCE_SINGLE_SESSION,
    max_sessions_per_user=AUTH_MAX_SESSIONS_PER_USER,
    max_global_sessions=AUTH_MAX_GLOBAL_ACTIVE_SESSIONS,
)
login_required = auth_flow.login_required


@app.before_request
def _enforce_authentication():
    if request.path == "/logout" and request.method != "POST":
        abort(405)
    return auth_flow.enforce_authentication()


@app.after_request
def _apply_auth_cache_headers(response):
    return auth_flow.apply_cache_headers(response)


def _normalize_client_name(raw_name: str) -> str:
    return normalize_client_name(raw_name)


def _warm_request_context(data: Dict[str, Any]) -> str:
    return proposal_request_service.warm_request_context(data)


def _normalize_signup_email(raw_value: str) -> str:
    return re.sub(r"\s+", "", str(raw_value or "").strip()).lower()


def _signup_email_error(email: str) -> str:
    if not re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}", email or ""):
        return "Gunakan format email internal yang valid."
    return ""


def send_signup_verification_webhook(email: str, full_name: str, verification_token: str, initial_password: str):
    import requests

    payload = {
        "user_email": str(email or "").strip().lower(),
        "user_fullname": str(full_name or "").strip(),
        "verification_token": str(verification_token or "").strip(),
        "initial_password": str(initial_password or "").strip(),
    }
    if not payload["user_email"] or not payload["verification_token"] or not payload["initial_password"]:
        return False

    delivery_mode = str(AUTH_SIGNUP_VERIFICATION_DELIVERY_MODE or "webhook").strip().lower()
    if delivery_mode == "capture":
        return payload
    if delivery_mode == "log":
        logger.info("Signup verification queued for %s", payload["user_email"])
        return True
    if delivery_mode != "webhook":
        return False

    webhook_url = str(AUTH_SIGNUP_VERIFICATION_WEBHOOK_URL or "").strip()
    if not webhook_url:
        logger.warning("Signup verification webhook URL is not configured.")
        return False

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=AUTH_SIGNUP_VERIFICATION_TIMEOUT_SECONDS,
        )
        return 200 <= int(getattr(response, "status_code", 0) or 0) < 300
    except Exception as exc:
        logger.error("Gagal mengirim webhook verifikasi ke %s: %s", payload["user_email"], str(exc))
        return False


@app.route('/')
@login_required
def home():
    current_user = auth_flow.current_username()
    current_user_fullname = app_state_store.auth.get_user_fullname(current_user)
    return render_template(
        'index.html',
        current_user=current_user,
        current_user_fullname=current_user_fullname,
    )


def generate_initial_password() -> str:
    upper = secrets.choice(string.ascii_uppercase)
    lower = secrets.choice(string.ascii_lowercase)
    digit = secrets.choice(string.digits)
    pool = string.ascii_letters + string.digits
    length = secrets.choice([8, 9, 10])
    remaining = "".join(secrets.choice(pool) for _ in range(max(0, length - 3)))
    chars = list(upper + lower + digit + remaining)
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def _render_auth(mode="login", error="", notice="", username="", next_target=None):
    if next_target is None:
        next_target = auth_flow.login_redirect_target()
    return render_template(
        'auth.html',
        mode=mode,
        error=error,
        notice=notice,
        username=username,
        next_target=next_target,
        allow_signup=AUTH_ALLOW_SIGNUP,
        signup_email_domain=AUTH_SIGNUP_EMAIL_DOMAIN,
    )


@app.route('/auth')
def auth_page():
    if auth_flow.is_authenticated():
        if auth_flow.touch_active_auth_session():
            return redirect(auth_flow.login_redirect_target())
        auth_flow.clear_authenticated_user(revoke=False)
    requested_mode = request.args.get("mode", "login").strip().lower()
    if requested_mode not in {"login", "signup"}:
        requested_mode = "login"
    # Check if there are query parameters redirecting with messages
    login_error = request.args.get("login_error", "").strip()
    signup_error = request.args.get("signup_error", "").strip()
    signup_success = request.args.get("signup_success", "").strip()

    if login_error:
        return _render_auth(mode="login", error=login_error)
    if signup_error:
        return _render_auth(mode="signup", error=signup_error)
    if signup_success:
        return _render_auth(mode="login", notice=signup_success)

    return _render_auth(mode=requested_mode)


@app.route('/login', methods=['GET'])
def login_page():
    return _render_auth(mode="login")


@app.route('/signup', methods=['GET'])
def signup_page():
    return _render_auth(mode="signup")


@app.route('/login', methods=['POST'])
def login():
    username = str(request.form.get("username") or "").strip().lower()
    password = str(request.form.get("password") or "")
    next_target = auth_flow.safe_next_target(request.form.get("next"))
    remote_ip = auth_flow.request_ip()
    if not username or not password:
        return _render_auth(mode="login", error="Username dan password wajib diisi.", next_target=next_target), 400
    blocked_seconds = app_state_store.auth.get_login_block_seconds(username=username, remote_ip=remote_ip)
    if blocked_seconds > 0:
        return _render_auth(
            mode="login",
            error=f"Terlalu banyak percobaan login. Coba lagi dalam {blocked_seconds} detik.",
            next_target=next_target,
            username=username
        ), 429

    user = app_state_store.auth.get_user(username)
    password_hash = str((user or {}).get("password_hash") or "")
    if not user or not password_hash or not check_password_hash(password_hash, password):
        blocked_after_fail = app_state_store.auth.register_login_failure(
            username=username,
            remote_ip=remote_ip,
            window_seconds=LOGIN_RATE_LIMIT_WINDOW_SECONDS,
            max_attempts=LOGIN_RATE_LIMIT_MAX_ATTEMPTS,
            block_seconds=LOGIN_RATE_LIMIT_BLOCK_SECONDS,
        )
        if blocked_after_fail > 0:
            return _render_auth(
                mode="login",
                error=f"Terlalu banyak percobaan login. Coba lagi dalam {blocked_after_fail} detik.",
                next_target=next_target,
                username=username
            ), 429
        return _render_auth(mode="login", error="Username atau password tidak cocok.", next_target=next_target, username=username), 401

    if str(user.get("status") or "approved") != "approved":
        app_state_store.auth.clear_login_failures(username=username, remote_ip=remote_ip)
        return _render_auth(
            mode="login",
            error="Akun belum dikonfirmasi admin. Hubungi admin internal untuk mengaktifkan akses.",
            next_target=next_target,
            username=username
        ), 403

    app_state_store.auth.clear_login_failures(username=username, remote_ip=remote_ip)
    auth_flow.set_authenticated_user(username)
    return redirect(next_target)


@app.route('/signup', methods=['POST'])
def signup():
    if not AUTH_ALLOW_SIGNUP:
        next_target = auth_flow.safe_next_target(request.form.get("next"))
        return _render_auth(
            mode="login",
            error="Pendaftaran mandiri dinonaktifkan. Hubungi admin internal untuk dibuatkan akun.",
            next_target=next_target
        ), 403

    username = _normalize_signup_email(request.form.get("username") or "")
    next_target = auth_flow.safe_next_target(request.form.get("next"))

    if not username:
        return _render_auth(mode="signup", error="Email internal wajib diisi.", next_target=next_target), 400
    email_error = _signup_email_error(username)
    if email_error:
        return _render_auth(mode="signup", error=email_error, next_target=next_target, username=username), 400

    reference_account = app_state_store.auth.lookup_reference_internal_account(username)
    if not reference_account:
        return _render_auth(
            mode="signup",
            error="Email tidak terdaftar di sistem internal.",
            next_target=next_target,
            username=username,
        ), 400

    # Check if user already exists
    user = app_state_store.auth.get_user(username)
    if user:
        return _render_auth(
            mode="signup",
            error="Username sudah dipakai. Gunakan username lain.",
            next_target=next_target,
            username=username,
        ), 400

    generated_pass = generate_initial_password()
    otp_code = "".join(secrets.choice(string.digits) for _ in range(6))
    full_name = str(reference_account.get("user_fullname") or "").strip()

    # Create the user as pending first
    created = app_state_store.auth.create_user(
        username=username,
        password_hash=generate_password_hash(generated_pass, method="pbkdf2:sha256"),
        status="pending",
        approved_by="signup-auto",
        full_name=full_name,
    )
    if not created:
        return _render_auth(
            mode="signup",
            error="Gagal membuat akun. Silakan coba lagi.",
            next_target=next_target,
            username=username,
        ), 500

    app_state_store.auth.set_registration_otp(username, otp_code)
    logger.info("Signup verification queued for %s", username)

    webhook_delivered = send_signup_verification_webhook(username, full_name, otp_code, generated_pass)
    if not webhook_delivered:
        app_state_store.auth.clear_registration_otp(username)
        app_state_store.auth.delete_user(username)
        return _render_auth(
            mode="signup",
            error="Gagal mengirim detail verifikasi melalui webhook.",
            next_target=next_target,
            username=username,
        ), 500

    notice = "Pendaftaran berhasil! Detail verifikasi telah dikirim melalui webhook."

    return _render_auth(
        mode="verify_signup",
        notice=notice,
        username=username,
        next_target=next_target,
    )


@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    username = str(request.form.get("username") or "").strip().lower()
    otp_code = str(request.form.get("verification_token") or request.form.get("otp_code") or "").strip()
    next_target = auth_flow.safe_next_target(request.form.get("next"))

    if app_state_store.auth.verify_registration_otp(username, otp_code):
        app_state_store.auth.clear_registration_otp(username)
        if AUTH_REQUIRE_SIGNUP_APPROVAL:
            notice = "Email berhasil diverifikasi. Akun Anda sekarang menunggu persetujuan admin sebelum bisa masuk."
        else:
            app_state_store.auth.approve_user(username, approved_by="otp_verified")
            notice = "Akun berhasil diverifikasi dan diaktifkan. Silakan masuk menggunakan kata sandi baru Anda."

        return _render_auth(
            mode="login",
            notice=notice,
            username=username,
            next_target=next_target,
        )
    else:
        notice = "Token verifikasi salah. Masukkan kembali token yang benar."
        return _render_auth(
            mode="verify_signup",
            error="Token verifikasi salah.",
            notice=notice,
            username=username,
            next_target=next_target,
        ), 400


@app.route('/logout', methods=['POST'])
def logout():
    auth_flow.clear_authenticated_user()
    return redirect(url_for("auth_page"))


@app.route('/api/auth/session-status')
@login_required
def auth_session_status():
    username = auth_flow.current_username()
    snapshot = app_state_store.auth.active_session_snapshot(username=username)
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
    payload, status_code = build_readiness_payload(
        project_db_path=PROJECT_DB_PATH,
        project_csv_path=PROJECT_CSV_PATH,
        knowledge_base=knowledge_base,
        app_state_store=app_state_store,
        ollama_host=OLLAMA_HOST,
        project_data_source=PROJECT_DATA_SOURCE,
    )
    return jsonify(payload), status_code


@app.route('/api/config')
def get_base_config():
    return jsonify({
        "suggestions": SMART_SUGGESTIONS,
        "data_acquisition_mode": proposal_generator.firm_api.data_acquisition_mode,
        "demo_mode": proposal_generator.firm_api.demo_mode,
        "internal_data_runtime": proposal_generator.firm_api.describe_runtime(),
        "data_source_active": proposal_generator.firm_api.data_acquisition_mode,
        "data_source_is_fallback": proposal_generator.firm_api.is_using_fallback(),
        "generation_profile": GENERATION_PROFILE,
        "job_poll_interval_ms": JOB_POLL_INTERVAL_MS,
        "max_active_generations": MAX_ACTIVE_GENERATIONS,
        "max_generation_backlog": MAX_GENERATION_BACKLOG,
        "proposal_modes": PROPOSAL_MODES,
        "framework_options": framework_option_service.options_payload(),
    })


@app.route('/api/internal-api/setup', methods=['GET', 'POST'])
def internal_api_setup():
    if request.method == 'GET':
        return jsonify(internal_api_service.setup_status())

    data = request.json or {}
    try:
        return jsonify(internal_api_service.activate_from_payload(data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Internal API setup failed")
        return jsonify({"error": str(exc)}), 500


@app.route('/api/internal-api/refresh', methods=['POST'])
def internal_api_refresh():
    payload, status_code = internal_api_service.refresh()
    return jsonify(payload), status_code


@app.route('/api/companies')
def get_companies():
    return jsonify(client_context_service.company_candidates())


def _company_candidates() -> list[str]:
    return client_context_service.company_candidates()


@app.route('/api/client-context')
def get_client_context():
    client_name = _normalize_client_name(str(request.args.get("client_name") or "").strip())
    payload = client_context_service.client_context_payload(client_name)
    return jsonify(payload), 502 if payload.get("error") else 200


def _request_payload_with_kak_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    return proposal_request_service.payload_with_kak_defaults(data)


def _client_internal_context_text(client_name: str, payload: Optional[Dict[str, Any]] = None) -> str:
    return client_context_service.internal_context_text(client_name, payload)


@app.route('/api/suggest-budget', methods=['POST'])
def suggest_budget():
    """Estimate pricing tiers from public financial signals."""
    data = _request_payload_with_kak_defaults(request.json or {})
    missing = proposal_request_service.missing_required_fields(data, BUDGET_REQUIRED_FIELDS)
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
    scope_context = FinancialAnalyzer.build_silent_scope_context(
        timeline=data.get('estimasi_waktu', ''),
        project_type=project_type,
        service_type=data.get('jenis_proposal', ''),
        project_goal=data.get('klasifikasi_kebutuhan', ''),
        objective=data.get('konteks_organisasi', ''),
        notes=data.get('permasalahan', ''),
        frameworks=data.get('potensi_framework', ''),
        commercial_context=commercial_context,
    )
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
        scope_context=scope_context,
    )
    return jsonify(result)


@app.route('/api/preview-outline', methods=['POST'])
def preview_outline():
    return jsonify(proposal_request_service.preview_outline(request.json or {}))


@app.route('/api/generation-precheck', methods=['POST'])
def generation_precheck():
    return jsonify(proposal_request_service.generation_precheck(request.json or {}))


@app.route('/api/framework-resolution', methods=['POST'])
def framework_resolution():
    data = _request_payload_with_kak_defaults(request.json or {})
    return jsonify(
        framework_option_service.confirmation_payload(
            str(data.get("potensi_framework") or ""),
            context=data,
        )
    )


@app.route('/api/prefetch-context', methods=['POST'])
def prefetch_context():
    return jsonify(proposal_request_service.prefetch_context(request.json or {}))


@app.route('/api/kak-context')
def get_kak_context():
    return jsonify(app_state_store.settings.get_latest_kak_context(company_candidates=_company_candidates()))


@app.route('/api/kak-context/active', methods=['POST'])
def set_active_kak_context():
    data = request.json or {}
    try:
        settings = app_state_store.settings.set_active_kak_document(str(data.get("document_id") or ""))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({
        "status": "ok",
        "settings": settings,
        "kak_context": app_state_store.settings.get_latest_kak_context(company_candidates=_company_candidates()),
    })


@app.route('/generate', methods=['POST'])
def generate_proposal():
    try:
        ticket = proposal_request_service.submit_generation(request.json or {})
        if ticket.get("status_code"):
            status_code = int(ticket.pop("status_code"))
            return jsonify(ticket), status_code
    except OverflowError as exc:
        return jsonify({"error": str(exc)}), 429

    return jsonify(ticket), 202


@app.route('/api/jobs/<job_id>')
def get_job_status(job_id: str):
    job = generation_queue.get(job_id)
    if not job:
        return jsonify({"error": "Job tidak ditemukan atau hasilnya sudah dibersihkan dari antrean."}), 404
    return jsonify(job)


@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
@login_required
def cancel_job(job_id: str):
    result = generation_queue.cancel(job_id)
    return jsonify(result), 200 if result.get("cancelled") else 409


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
@login_required
def list_history():
    q = str(request.args.get("q") or "").strip()
    try:
        limit = min(100, max(1, int(request.args.get("limit") or 30)))
    except ValueError:
        limit = 30
    try:
        offset = max(0, int(request.args.get("offset") or 0))
    except ValueError:
        offset = 0
    return jsonify({"items": app_state_store.history.list_history(q=q, limit=limit, offset=offset)})


@app.route('/api/history/<entry_id>')
def get_history_entry(entry_id: str):
    entry = app_state_store.history.get_history_entry(entry_id)
    if not entry:
        return jsonify({"error": "Riwayat proposal tidak ditemukan."}), 404
    return jsonify(entry)


@app.route('/api/history/<entry_id>/reuse')
def reuse_history(entry_id: str):
    entry = app_state_store.history.get_history_entry(entry_id)
    if not entry:
        return jsonify({"error": "Riwayat proposal tidak ditemukan."}), 404
    if not entry.get("can_reuse"):
        return jsonify({"error": "Proposal historis ini hanya tersedia sebagai referensi unduhan, belum memiliki payload yang bisa dipakai ulang."}), 409
    return jsonify({"payload": entry.get("payload", {})})


@app.route('/api/history/<entry_id>/download')
def download_history(entry_id: str):
    entry = app_state_store.history.get_history_entry(entry_id)
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
        return jsonify(app_state_store.settings.get_settings())

    return jsonify(app_state_store.settings.save_settings())


@app.route('/api/settings/internal-evidence')
def proposal_internal_evidence():
    try:
        bench_context = proposal_generator.firm_api.get_expert_bench_context(limit_products=8)
        return jsonify(build_internal_evidence_summary(bench_context))
    except Exception:
        logger.exception("Internal evidence summary failed")
        return jsonify(build_internal_evidence_summary({}))


@app.route('/api/settings/template', methods=['POST', 'DELETE'])
def proposal_template_settings():
    if request.method == 'DELETE':
        return jsonify(app_state_store.settings.clear_template())

    uploaded = request.files.get('template')
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "File template .docx belum dipilih."}), 400
    if not uploaded.filename.lower().endswith('.docx'):
        return jsonify({"error": "Template harus berformat .docx."}), 400
    settings = app_state_store.settings.save_template(uploaded.filename, uploaded.read())
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
        settings = app_state_store.settings.save_supporting_documents(document_type, uploads)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(settings)


@app.route('/api/settings/documents/<document_id>', methods=['DELETE'])
def delete_proposal_supporting_document(document_id: str):
    try:
        settings = app_state_store.settings.delete_supporting_document(document_id)
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
