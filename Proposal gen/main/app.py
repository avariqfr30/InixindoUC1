"""Flask entrypoint for the proposal generator app."""
from datetime import timedelta
import io
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
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
from .auth_flow import AuthFlow
from .core import ProposalGenerator
from .data_sources import FirmAPIClient, InternalDataClient
from .finance import FinancialAnalyzer
from .internal_api_setup import build_internal_api_config, write_json_config
from .job_queue import GenerationQueue
from .knowledge_store import KnowledgeBase
from .state_store import AppStateStore

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
    return auth_flow.enforce_authentication()


@app.after_request
def _apply_auth_cache_headers(response):
    return auth_flow.apply_cache_headers(response)


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


def _internal_api_config_target() -> Path:
    configured = str(FirmAPIClient._resolve_config_file() or "").strip()
    if configured:
        return Path(configured).expanduser()
    return MANAGED_INTERNAL_API_CONFIG_PATH


def _write_internal_api_config(config_payload: Dict[str, Any]) -> Path:
    target = _internal_api_config_target()
    write_json_config(target, config_payload)
    return target


@app.route('/')
@login_required
def home():
    return render_template('index.html', current_user=auth_flow.current_username())


@app.route('/auth')
def auth_page():
    if auth_flow.is_authenticated():
        if auth_flow.touch_active_auth_session():
            return redirect(auth_flow.login_redirect_target())
        auth_flow.clear_authenticated_user(revoke=False)
    return render_template(
        'auth.html',
        login_error=request.args.get("login_error", "").strip(),
        signup_error=request.args.get("signup_error", "").strip(),
        signup_success=request.args.get("signup_success", "").strip(),
        next_target=auth_flow.login_redirect_target(),
    )


@app.route('/login', methods=['POST'])
def login():
    username = str(request.form.get("username") or "").strip()
    password = str(request.form.get("password") or "")
    next_target = auth_flow.safe_next_target(request.form.get("next"))
    remote_ip = auth_flow.request_ip()
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
        auth_flow.set_authenticated_user(str(user.get("username") or username))
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
    next_target = auth_flow.safe_next_target(request.form.get("next"))

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
    auth_flow.clear_authenticated_user()
    return redirect(url_for("auth_page"))


@app.route('/api/auth/session-status')
@login_required
def auth_session_status():
    username = auth_flow.current_username()
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
        config_payload = build_internal_api_config(data)
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
