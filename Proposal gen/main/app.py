"""Flask entrypoint for the proposal generator app."""
import concurrent.futures
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

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

try:
    from waitress import serve as waitress_serve
except Exception:  # pragma: no cover - optional local server dependency
    waitress_serve = None

from .config import (
    APP_HOST,
    APP_PORT,
    DB_URI,
    GENERATION_PROFILE,
    JOB_POLL_INTERVAL_MS,
    JOB_RETENTION_SECONDS,
    MAX_ACTIVE_GENERATIONS,
    MAX_GENERATION_BACKLOG,
    SMART_SUGGESTIONS,
)
from .core import FinancialAnalyzer, KnowledgeBase, ProposalGenerator

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
CORS(app)

knowledge_base = KnowledgeBase(DB_URI)
proposal_generator = ProposalGenerator(knowledge_base)


class GenerationQueue:
    """Small in-memory queue for low-level shared testing."""

    def __init__(
        self,
        generator: ProposalGenerator,
        max_active: int,
        max_backlog: int,
        retention_seconds: int,
    ) -> None:
        self.generator = generator
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
            "status": job["status"],
            "message": self._build_message_locked(job),
            "queue_position": self._queue_position_locked(int(job["sequence"])) if job["status"] == "queued" else None,
            "filename": job.get("filename"),
            "created_at": job.get("created_at"),
            "started_at": started_at,
            "finished_at": finished_at,
            "processing_seconds": processing_seconds,
            "download_ready": job["status"] == "done",
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
            doc, filename = self.generator.generate_document(
                client=payload["nama_perusahaan"],
                project=payload["konteks_organisasi"],
                budget=payload["estimasi_biaya"],
                service_type=payload["jenis_proposal"],
                project_goal=payload["klasifikasi_kebutuhan"],
                project_type=payload["jenis_proyek"],
                timeline=payload["estimasi_waktu"],
                notes=payload["permasalahan"],
                regulations=payload["potensi_framework"],
                chapter_id=payload.get("chapter_id")
            )
            output = io.BytesIO()
            doc.save(output)
            result_bytes = output.getvalue()

            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                job["status"] = "done"
                job["finished_at"] = time.time()
                job["filename"] = f"{filename}.docx"
                job["result_bytes"] = result_bytes
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
    max_active=MAX_ACTIVE_GENERATIONS,
    max_backlog=MAX_GENERATION_BACKLOG,
    retention_seconds=JOB_RETENTION_SECONDS,
)


def _normalize_client_name(raw_name: str) -> str:
    return re.sub(r'\b(Cabang|Branch|Tbk)\b.*$|^(PT\.|CV\.)', '', raw_name or '', flags=re.IGNORECASE).strip()


def _warm_request_context(data: Dict[str, Any]) -> str:
    client_name = _normalize_client_name(str(data.get("nama_perusahaan", "")).strip())
    regulations = str(data.get("potensi_framework", "")).strip()
    if not client_name:
        return "skipped"
    return proposal_generator.prefetch_research_bundle(
        base_client=client_name,
        regulations=regulations,
        include_collaboration=proposal_generator.firm_api.uses_demo_logic()
    )


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/config')
def get_base_config():
    return jsonify({
        "suggestions": SMART_SUGGESTIONS,
        "data_acquisition_mode": proposal_generator.firm_api.data_acquisition_mode,
        "demo_mode": proposal_generator.firm_api.demo_mode,
        "generation_profile": GENERATION_PROFILE,
        "job_poll_interval_ms": JOB_POLL_INTERVAL_MS,
        "max_active_generations": MAX_ACTIVE_GENERATIONS,
        "max_generation_backlog": MAX_GENERATION_BACKLOG,
    })


@app.route('/api/companies')
def get_companies():
    if knowledge_base.df is None or knowledge_base.df.empty or 'entity' not in knowledge_base.df.columns:
        return jsonify([])
    companies = knowledge_base.df['entity'].dropna().astype(str).str.strip().unique().tolist()
    companies = [c for c in companies if c.lower() != 'nan' and c]
    return jsonify(sorted(companies))


@app.route('/api/suggest-budget', methods=['POST'])
def suggest_budget():
    """Estimate pricing tiers from public financial signals."""
    data = request.json or {}
    required_fields = [
        'nama_perusahaan',
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


@app.route('/generate', methods=['POST'])
def generate_proposal():
    data = request.json or {}
    required_fields = [
        'nama_perusahaan',
        'konteks_organisasi',
        'estimasi_biaya',
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
