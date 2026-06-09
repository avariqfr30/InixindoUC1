"""Background proposal generation queue."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


class GenerationQueue:
    """Small in-memory queue for low-level shared testing."""

    def __init__(
        self,
        generator: Any,
        state_store: Any,
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
            thread_name_prefix="proposal-job",
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
        if status == "cancelled":
            return "Proses dibatalkan oleh pengguna."
        return str(job.get("error") or "Proses generate proposal gagal.")

    def _snapshot_locked(self, job: Dict[str, Any]) -> Dict[str, Any]:
        started_at = job.get("started_at")
        finished_at = job.get("finished_at")
        processing_seconds = None
        if started_at:
            processing_seconds = round((finished_at or time.time()) - float(started_at), 1)
        score = job.get("research_signal_score")
        label = "Tinggi" if score is not None and score >= 10 else ("Sedang" if score is not None and score >= 5 else "Terbatas")
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
            "research_signal_score": score,
            "research_signal_label": label,
        }
        if job["status"] in {"failed", "cancelled"}:
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
                "research_signal_score": None,
            }
            self._jobs[job_id] = job
            self.state_store.upsert_generation_job(
                job_id=job_id,
                fingerprint=fingerprint,
                status="queued",
                created_at=job["created_at"]
            )
            self._executor.submit(self._run_job, job_id)
            return self._snapshot_locked(job)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if job["status"] == "cancelled":
                return
            job["status"] = "running"
            job["started_at"] = time.time()
            payload = dict(job.get("payload") or {})
            self.state_store.upsert_generation_job(
                job_id=job_id,
                status="running",
                started_at=job["started_at"]
            )

        try:
            def check_cancelled() -> None:
                with self._lock:
                    curr_job = self._jobs.get(job_id)
                    if curr_job and curr_job.get("cancellation_requested"):
                        raise RuntimeError("Proses dibatalkan oleh pengguna.")

            supporting_context = dict(payload.get("_supporting_context") or {})
            supporting_context["cancellation_checker"] = check_cancelled

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
                supporting_context=supporting_context,
            )
            output = io.BytesIO()
            doc.save(output)
            result_bytes = output.getvalue()
            finished_at = time.time()
            acceptance_report = dict((generation_meta or {}).get("acceptance_report") or {})
            processing_seconds = max(0.0, finished_at - float(job.get("started_at") or finished_at))
            research_signal_score = (generation_meta or {}).get("research_signal_score")

            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                if job.get("cancellation_requested") or job["status"] == "cancelled":
                    raise RuntimeError("Proses dibatalkan oleh pengguna.")
                stored_path = self.state_store.persist_generated_file(f"{filename}.docx", result_bytes)
                history_id = self.state_store.add_history_entry(
                    payload=payload,
                    filename=stored_path.name,
                    filepath=str(stored_path),
                    created_at=job.get("created_at") or time.time(),
                    finished_at=finished_at,
                    acceptance_report=acceptance_report,
                    processing_seconds=processing_seconds,
                    research_signal_score=research_signal_score,
                )
                job["status"] = "done"
                job["finished_at"] = finished_at
                job["filename"] = stored_path.name
                job["result_bytes"] = result_bytes
                job["history_id"] = history_id
                job["acceptance_score"] = acceptance_report.get("score")
                job["acceptance_passes"] = acceptance_report.get("passes")
                job["research_signal_score"] = research_signal_score
                job["payload"] = None
                self.state_store.upsert_generation_job(
                    job_id=job_id,
                    status="done",
                    finished_at=finished_at,
                    history_id=history_id,
                    filename=stored_path.name,
                    acceptance_score=acceptance_report.get("score"),
                    acceptance_passes=1 if acceptance_report.get("passes") else 0
                )
        except Exception as exc:  # keep the queue resilient during shared testing
            logger.exception("Generation job %s failed", job_id)
            with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return
                is_cancelled = job.get("cancellation_requested") or str(exc) == "Proses dibatalkan oleh pengguna." or job["status"] == "cancelled"
                job["status"] = "cancelled" if is_cancelled else "failed"
                job["finished_at"] = time.time()
                job["error"] = "Proses dibatalkan oleh pengguna." if is_cancelled else (str(exc) or "Proses generate proposal gagal.")
                job["payload"] = None
                self.state_store.upsert_generation_job(
                    job_id=job_id,
                    status=job["status"],
                    finished_at=job["finished_at"],
                    error=job["error"]
                )

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                db_job = self.state_store.get_generation_job(job_id)
                if db_job:
                    status = db_job["status"]
                    if status in {"queued", "running"}:
                        finished_at = time.time()
                        self.state_store.upsert_generation_job(
                            job_id=job_id,
                            status="cancelled",
                            finished_at=finished_at,
                            error="Proses dibatalkan oleh pengguna."
                        )
                        return {"cancelled": True}
                    return {"cancelled": False, "reason": f"job_is_{status}"}
                return {"cancelled": False, "reason": "not_found"}
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = time.time()
                job["error"] = "Proses dibatalkan oleh pengguna."
                self.state_store.upsert_generation_job(
                    job_id=job_id,
                    status="cancelled",
                    finished_at=job["finished_at"],
                    error=job["error"]
                )
                return {"cancelled": True}
            if job["status"] == "running":
                job["cancellation_requested"] = True
                return {"cancelled": False, "reason": "running_jobs_cannot_be_instantly_cancelled"}
            return {"cancelled": False, "reason": f"job_is_{job['status']}"}

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if not job:
                db_job = self.state_store.get_generation_job(job_id)
                if db_job:
                    status = db_job["status"]
                    if status in {"queued", "running"}:
                        return {
                            "job_id": job_id,
                            "status": "failed",
                            "message": "Proses terhenti karena server restart. Silakan ulangi generate proposal.",
                            "download_ready": False,
                            "created_at": db_job.get("created_at"),
                            "started_at": db_job.get("started_at"),
                            "finished_at": db_job.get("finished_at"),
                            "error": "Server restart",
                        }
                    processing_seconds = None
                    if db_job.get("started_at") and db_job.get("finished_at"):
                        processing_seconds = round(db_job["finished_at"] - db_job["started_at"], 1)

                    if status == "done":
                        message = "Proposal selesai dibuat dan siap diunduh."
                    elif status == "cancelled":
                        message = "Proses dibatalkan oleh pengguna."
                    else:
                        message = db_job.get("error") or "Proses generate proposal gagal."

                    score = db_job.get("research_signal_score")
                    label = "Tinggi" if score is not None and score >= 10 else ("Sedang" if score is not None and score >= 5 else "Terbatas")

                    return {
                        "job_id": job_id,
                        "history_id": db_job.get("history_id"),
                        "status": status,
                        "message": message,
                        "queue_position": None,
                        "filename": db_job.get("filename"),
                        "created_at": db_job.get("created_at"),
                        "started_at": db_job.get("started_at"),
                        "finished_at": db_job.get("finished_at"),
                        "processing_seconds": processing_seconds,
                        "download_ready": status == "done",
                        "acceptance_score": db_job.get("acceptance_score"),
                        "acceptance_passes": bool(db_job.get("acceptance_passes")),
                        "research_signal_score": score,
                        "research_signal_label": label,
                    }
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
