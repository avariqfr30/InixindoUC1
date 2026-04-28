"""Persistent app settings, history, documents, and auth-session storage."""
from __future__ import annotations

from .proposal_shared import *
from .schema_mapping import SchemaMapper
from .text_hygiene import compact_context_lines, formalize_caps_text, normalize_field_value, normalize_payload

class AppStateStore:
    def __init__(self, db_path: Optional[Path] = None, asset_root: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or APP_STATE_DB_PATH)
        self.asset_root = Path(asset_root or APP_ASSET_ROOT)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.templates_dir = self.asset_root / "templates"
        self.supporting_docs_dir = self.asset_root / "supporting_documents"
        self.portfolio_docs_dir = self.supporting_docs_dir / "portfolio"
        self.credentials_docs_dir = self.supporting_docs_dir / "credentials"
        self.kak_docs_dir = self.supporting_docs_dir / "kak"
        self.generated_dir = Path(GENERATED_OUTPUT_DIR)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_docs_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_docs_dir.mkdir(parents=True, exist_ok=True)
        self.kak_docs_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._bootstrap_generated_history()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposal_history (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    finished_at REAL NOT NULL,
                    client TEXT NOT NULL,
                    project TEXT NOT NULL,
                    proposal_mode TEXT NOT NULL,
                    service_type TEXT NOT NULL,
                    project_type TEXT NOT NULL,
                    timeline TEXT NOT NULL,
                    budget TEXT NOT NULL,
                    acceptance_score INTEGER NOT NULL DEFAULT 0,
                    acceptance_passes INTEGER NOT NULL DEFAULT 0,
                    processing_seconds REAL NOT NULL DEFAULT 0,
                    acceptance_json TEXT NOT NULL DEFAULT '{}',
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supporting_documents (
                    id TEXT PRIMARY KEY,
                    document_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    extracted_text TEXT NOT NULL DEFAULT '',
                    byte_size INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    username_key TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    username_key TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    absolute_expires_at REAL NOT NULL,
                    last_ip TEXT NOT NULL DEFAULT '',
                    user_agent TEXT NOT NULL DEFAULT '',
                    revoked_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_login_attempts (
                    key TEXT PRIMARY KEY,
                    username_key TEXT NOT NULL,
                    remote_ip TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    window_started_at REAL NOT NULL DEFAULT 0,
                    blocked_until REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(proposal_history)").fetchall()
            }
            column_backfills = {
                "acceptance_score": "INTEGER NOT NULL DEFAULT 0",
                "acceptance_passes": "INTEGER NOT NULL DEFAULT 0",
                "processing_seconds": "REAL NOT NULL DEFAULT 0",
                "acceptance_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column_name, ddl in column_backfills.items():
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE proposal_history ADD COLUMN {column_name} {ddl}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_app_sessions_username_active ON app_sessions(username_key, revoked_at, expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_app_sessions_active ON app_sessions(revoked_at, expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_user_ip ON app_login_attempts(username_key, remote_ip)")
            conn.commit()

    def _bootstrap_generated_history(self) -> None:
        existing_files = [
            path for path in sorted(self.generated_dir.glob("*.docx"))
            if not path.name.startswith("~$")
        ]
        if not existing_files:
            return

        with self._connect() as conn:
            conn.execute("DELETE FROM proposal_history WHERE filename LIKE '~$%' OR filepath LIKE '%/~$%'")
            known_paths = {
                str(row["filepath"])
                for row in conn.execute("SELECT filepath FROM proposal_history").fetchall()
            }
            for path in existing_files:
                if str(path) in known_paths:
                    continue
                stat = path.stat()
                inferred_client = path.stem.replace("Proposal_", "").replace("_", " ").strip()
                conn.execute(
                    """
                    INSERT INTO proposal_history(
                        id, created_at, finished_at, client, project, proposal_mode,
                        service_type, project_type, timeline, budget, acceptance_score,
                        acceptance_passes, processing_seconds, acceptance_json, filename, filepath, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        float(stat.st_mtime),
                        float(stat.st_mtime),
                        inferred_client or "Dokumen historis",
                        "Dokumen historis dari folder generated",
                        "historis",
                        "",
                        "",
                        "",
                        "",
                        0,
                        0,
                        0.0,
                        "{}",
                        path.name,
                        str(path),
                        "{}",
                    ),
                )
            conn.commit()

    @staticmethod
    def _sanitize_filename(value: str, fallback: str = "proposal", max_length: int = 140) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
        cleaned = re.sub(r"_+", "_", cleaned)
        if not cleaned:
            cleaned = fallback
        if len(cleaned) > max_length:
            stem, dot, suffix = cleaned.rpartition(".")
            if dot:
                head = stem[: max_length - len(suffix) - 1]
                cleaned = f"{head}.{suffix}"
            else:
                cleaned = cleaned[:max_length]
        return cleaned

    def _get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def _set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value or "")),
            )
            conn.commit()

    @staticmethod
    def _normalize_username(username: str) -> str:
        normalized = re.sub(r"\s+", " ", str(username or "").strip())
        return normalized

    @classmethod
    def _username_key(cls, username: str) -> str:
        return SchemaMapper.normalize_key(cls._normalize_username(username))

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        username_key = self._username_key(username)
        if not username_key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, username_key, password_hash, created_at
                FROM app_users
                WHERE username_key = ?
                """,
                (username_key,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "username_key": row["username_key"],
            "password_hash": row["password_hash"],
            "created_at": float(row["created_at"] or 0.0),
        }

    def create_user(self, username: str, password_hash: str) -> bool:
        normalized_username = self._normalize_username(username)
        username_key = self._username_key(normalized_username)
        if not normalized_username or not username_key or not password_hash:
            return False
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO app_users(id, username, username_key, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        normalized_username,
                        username_key,
                        str(password_hash),
                        time.time(),
                    ),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    @staticmethod
    def _login_attempt_key(username_key: str, remote_ip: str) -> str:
        return f"{username_key}|{remote_ip}"

    def _cleanup_auth_state(self, now: Optional[float] = None) -> None:
        current = float(now or time.time())
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM app_sessions WHERE revoked_at IS NOT NULL AND revoked_at <= ?",
                (current - 86400.0,),
            )
            conn.execute(
                "DELETE FROM app_sessions WHERE expires_at <= ? OR absolute_expires_at <= ?",
                (current, current),
            )
            conn.execute(
                "DELETE FROM app_login_attempts WHERE blocked_until <= ? AND (window_started_at <= ? OR attempt_count <= 0)",
                (current, current - 86400.0),
            )
            conn.commit()

    def count_active_sessions(self, username: str = "") -> int:
        self._cleanup_auth_state()
        now = time.time()
        with self._connect() as conn:
            if username:
                username_key = self._username_key(username)
                if not username_key:
                    return 0
                row = conn.execute(
                    """
                    SELECT COUNT(1) AS total
                    FROM app_sessions
                    WHERE username_key = ?
                      AND revoked_at IS NULL
                      AND expires_at > ?
                      AND absolute_expires_at > ?
                    """,
                    (username_key, now, now),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT COUNT(1) AS total
                    FROM app_sessions
                    WHERE revoked_at IS NULL
                      AND expires_at > ?
                      AND absolute_expires_at > ?
                    """,
                    (now, now),
                ).fetchone()
        return int(row["total"] if row else 0)

    def _enforce_session_limits(
        self,
        username_key: str,
        force_single_session: bool,
        max_per_user: int,
        max_global: int,
        now: float,
    ) -> None:
        with self._connect() as conn:
            if force_single_session:
                conn.execute(
                    """
                    UPDATE app_sessions
                    SET revoked_at = ?
                    WHERE username_key = ?
                      AND revoked_at IS NULL
                      AND expires_at > ?
                      AND absolute_expires_at > ?
                    """,
                    (now, username_key, now, now),
                )
            else:
                active_user_rows = conn.execute(
                    """
                    SELECT id
                    FROM app_sessions
                    WHERE username_key = ?
                      AND revoked_at IS NULL
                      AND expires_at > ?
                      AND absolute_expires_at > ?
                    ORDER BY last_seen_at ASC, created_at ASC
                    """,
                    (username_key, now, now),
                ).fetchall()
                overflow = max(0, len(active_user_rows) - max(0, int(max_per_user) - 1))
                for row in active_user_rows[:overflow]:
                    conn.execute("UPDATE app_sessions SET revoked_at = ? WHERE id = ?", (now, row["id"]))

            active_global_rows = conn.execute(
                """
                SELECT id
                FROM app_sessions
                WHERE revoked_at IS NULL
                  AND expires_at > ?
                  AND absolute_expires_at > ?
                ORDER BY last_seen_at ASC, created_at ASC
                """,
                (now, now),
            ).fetchall()
            overflow_global = max(0, len(active_global_rows) - max(0, int(max_global) - 1))
            for row in active_global_rows[:overflow_global]:
                conn.execute("UPDATE app_sessions SET revoked_at = ? WHERE id = ?", (now, row["id"]))
            conn.commit()

    def create_user_session(
        self,
        username: str,
        remote_ip: str = "",
        user_agent: str = "",
        idle_timeout_seconds: int = 2700,
        absolute_timeout_seconds: int = 43200,
        force_single_session: bool = True,
        max_sessions_per_user: int = 1,
        max_global_sessions: int = 30,
    ) -> str:
        user = self.get_user(username)
        if not user:
            return ""
        now = time.time()
        self._cleanup_auth_state(now)
        username_key = str(user.get("username_key") or "")
        if not username_key:
            return ""

        self._enforce_session_limits(
            username_key=username_key,
            force_single_session=bool(force_single_session),
            max_per_user=max(1, int(max_sessions_per_user or 1)),
            max_global=max(1, int(max_global_sessions or 1)),
            now=now,
        )

        session_id = uuid.uuid4().hex
        absolute_expires_at = now + max(60, int(absolute_timeout_seconds or 43200))
        expires_at = min(
            now + max(60, int(idle_timeout_seconds or 2700)),
            absolute_expires_at,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_sessions(
                    id, user_id, username_key, created_at, last_seen_at, expires_at, absolute_expires_at, last_ip, user_agent, revoked_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    session_id,
                    str(user.get("id") or ""),
                    username_key,
                    now,
                    now,
                    expires_at,
                    absolute_expires_at,
                    str(remote_ip or "")[:96],
                    str(user_agent or "")[:256],
                ),
            )
            conn.commit()
        return session_id

    def touch_user_session(
        self,
        session_id: str,
        username: str = "",
        remote_ip: str = "",
        idle_timeout_seconds: int = 2700,
        touch_interval_seconds: int = 20,
    ) -> bool:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return False
        username_key = self._username_key(username) if username else ""
        now = time.time()
        self._cleanup_auth_state(now)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username_key, last_seen_at, expires_at, absolute_expires_at, revoked_at
                FROM app_sessions
                WHERE id = ?
                """,
                (normalized_session_id,),
            ).fetchone()
            if not row:
                return False
            if row["revoked_at"] is not None:
                return False
            if username_key and str(row["username_key"] or "") != username_key:
                return False
            if float(row["expires_at"] or 0.0) <= now or float(row["absolute_expires_at"] or 0.0) <= now:
                conn.execute("UPDATE app_sessions SET revoked_at = ? WHERE id = ?", (now, normalized_session_id))
                conn.commit()
                return False

            last_seen = float(row["last_seen_at"] or 0.0)
            if (now - last_seen) >= max(1, int(touch_interval_seconds or 20)):
                absolute_expires_at = float(row["absolute_expires_at"] or now)
                next_expires = min(
                    now + max(60, int(idle_timeout_seconds or 2700)),
                    absolute_expires_at,
                )
                conn.execute(
                    """
                    UPDATE app_sessions
                    SET last_seen_at = ?, expires_at = ?, last_ip = ?
                    WHERE id = ?
                    """,
                    (now, next_expires, str(remote_ip or "")[:96], normalized_session_id),
                )
                conn.commit()
        return True

    def revoke_user_session(self, session_id: str) -> None:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE app_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (time.time(), normalized_session_id),
            )
            conn.commit()

    def active_session_snapshot(self, username: str = "") -> Dict[str, Any]:
        self._cleanup_auth_state()
        now = time.time()
        snapshot: Dict[str, Any] = {
            "global_active_sessions": 0,
            "user_active_sessions": 0,
        }
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS total
                FROM app_sessions
                WHERE revoked_at IS NULL
                  AND expires_at > ?
                  AND absolute_expires_at > ?
                """,
                (now, now),
            ).fetchone()
            snapshot["global_active_sessions"] = int(row["total"] if row else 0)
            if username:
                username_key = self._username_key(username)
                if username_key:
                    row_user = conn.execute(
                        """
                        SELECT COUNT(1) AS total
                        FROM app_sessions
                        WHERE username_key = ?
                          AND revoked_at IS NULL
                          AND expires_at > ?
                          AND absolute_expires_at > ?
                        """,
                        (username_key, now, now),
                    ).fetchone()
                    snapshot["user_active_sessions"] = int(row_user["total"] if row_user else 0)
        return snapshot

    def get_login_block_seconds(self, username: str, remote_ip: str) -> int:
        username_key = self._username_key(username)
        ip = str(remote_ip or "").strip()[:96]
        if not username_key:
            return 0
        key = self._login_attempt_key(username_key, ip)
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT blocked_until FROM app_login_attempts WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return 0
        blocked_until = float(row["blocked_until"] or 0.0)
        if blocked_until <= now:
            return 0
        return max(1, int(blocked_until - now))

    def register_login_failure(
        self,
        username: str,
        remote_ip: str,
        window_seconds: int = 300,
        max_attempts: int = 6,
        block_seconds: int = 600,
    ) -> int:
        username_key = self._username_key(username)
        ip = str(remote_ip or "").strip()[:96]
        if not username_key:
            return 0
        key = self._login_attempt_key(username_key, ip)
        now = time.time()
        self._cleanup_auth_state(now)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT attempt_count, window_started_at, blocked_until
                FROM app_login_attempts
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
            if row and float(row["blocked_until"] or 0.0) > now:
                return max(1, int(float(row["blocked_until"]) - now))

            if not row or (now - float(row["window_started_at"] or 0.0)) > max(30, int(window_seconds)):
                attempt_count = 1
                window_started_at = now
            else:
                attempt_count = int(row["attempt_count"] or 0) + 1
                window_started_at = float(row["window_started_at"] or now)

            blocked_until = 0.0
            if attempt_count >= max(1, int(max_attempts)):
                blocked_until = now + max(30, int(block_seconds))

            conn.execute(
                """
                INSERT INTO app_login_attempts(
                    key, username_key, remote_ip, attempt_count, window_started_at, blocked_until, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    attempt_count = excluded.attempt_count,
                    window_started_at = excluded.window_started_at,
                    blocked_until = excluded.blocked_until,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    username_key,
                    ip,
                    attempt_count,
                    window_started_at,
                    blocked_until,
                    now,
                ),
            )
            conn.commit()
        if blocked_until > now:
            return max(1, int(blocked_until - now))
        return 0

    def clear_login_failures(self, username: str, remote_ip: str) -> None:
        username_key = self._username_key(username)
        ip = str(remote_ip or "").strip()[:96]
        if not username_key:
            return
        key = self._login_attempt_key(username_key, ip)
        with self._connect() as conn:
            conn.execute("DELETE FROM app_login_attempts WHERE key = ?", (key,))
            conn.commit()

    @staticmethod
    def _normalize_document_type(value: str) -> str:
        normalized = SchemaMapper.normalize_key(value)
        aliases = {
            "portfolio": "portfolio",
            "portofolio": "portfolio",
            "credential": "credentials",
            "credentials": "credentials",
            "sertifikasi": "credentials",
            "kapabilitas": "credentials",
            "kak": "kak",
            "tor": "kak",
            "kerangka_acuan_kerja": "kak",
            "kerangka_acuan": "kak",
        }
        if normalized not in aliases:
            raise ValueError("Jenis dokumen pendukung tidak dikenal.")
        return aliases[normalized]

    def _supporting_dir_for_type(self, document_type: str) -> Path:
        normalized = self._normalize_document_type(document_type)
        if normalized == "portfolio":
            return self.portfolio_docs_dir
        if normalized == "credentials":
            return self.credentials_docs_dir
        return self.kak_docs_dir

    @staticmethod
    def _read_text_bytes(raw_bytes: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except Exception:
                continue
        return raw_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def _extract_docx_text(raw_bytes: bytes) -> str:
        try:
            doc = Document(io.BytesIO(raw_bytes))
        except Exception:
            return ""

        blocks: List[str] = []
        for paragraph in doc.paragraphs:
            text = re.sub(r"\s+", " ", str(paragraph.text or "")).strip()
            if text:
                blocks.append(text)
        for table in doc.tables:
            for row in table.rows:
                cells = [
                    re.sub(r"\s+", " ", str(cell.text or "")).strip()
                    for cell in row.cells
                ]
                cells = [cell for cell in cells if cell]
                if cells:
                    blocks.append(" | ".join(cells))
        return "\n".join(blocks).strip()

    @staticmethod
    def _normalize_extracted_text(text: str) -> str:
        lines: List[str] = []
        for raw_line in str(text or "").replace("\r\n", "\n").split("\n"):
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return formalize_caps_text(cleaned)

    @classmethod
    def _extract_supporting_document_text(cls, filename: str, raw_bytes: bytes) -> str:
        suffix = Path(str(filename or "")).suffix.lower()
        if suffix == ".docx":
            return cls._normalize_extracted_text(cls._extract_docx_text(raw_bytes))
        if suffix in {".txt", ".md"}:
            return cls._normalize_extracted_text(cls._read_text_bytes(raw_bytes))
        if suffix == ".pdf" and PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(raw_bytes))
                extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
                return cls._normalize_extracted_text(extracted)
            except Exception:
                return ""
        return ""

    @staticmethod
    def _trim_supporting_text(text: str, max_words: int = 220) -> str:
        words = str(text or "").split()
        if len(words) <= max_words:
            return str(text or "").strip()
        return " ".join(words[:max_words]).strip()

    def _serialize_document_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        filepath = str(row["filepath"] or "")
        extracted_text = str(row["extracted_text"] or "")
        return {
            "id": row["id"],
            "document_type": row["document_type"],
            "original_name": row["original_name"],
            "stored_name": row["stored_name"],
            "filepath": filepath,
            "uploaded_at": float(row["created_at"] or 0.0),
            "uploaded_at_label": self._format_timestamp(float(row["created_at"] or 0.0)),
            "byte_size": int(row["byte_size"] or 0),
            "has_text": bool(extracted_text.strip()),
            "exists": bool(filepath and Path(filepath).exists()),
        }

    def list_supporting_documents(self, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT id, document_type, created_at, original_name, stored_name, filepath, extracted_text, byte_size
            FROM supporting_documents
        """
        params: Tuple[Any, ...] = ()
        if document_type:
            normalized = self._normalize_document_type(document_type)
            query += " WHERE document_type = ?"
            params = (normalized,)
        query += " ORDER BY created_at DESC, original_name ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._serialize_document_row(row) for row in rows]

    def save_supporting_documents(self, document_type: str, uploads: List[Tuple[str, bytes]]) -> Dict[str, Any]:
        normalized = self._normalize_document_type(document_type)
        target_dir = self._supporting_dir_for_type(normalized)
        saved_any = False
        with self._connect() as conn:
            for filename, raw_bytes in uploads:
                if not filename or not raw_bytes:
                    continue
                safe_name = self._sanitize_filename(filename, fallback=f"{normalized}_supporting_document")
                suffix = Path(safe_name).suffix.lower()
                if suffix not in {".docx", ".pdf", ".txt", ".md"}:
                    continue
                stored_name = f"{uuid.uuid4().hex}_{safe_name}"
                target_path = target_dir / stored_name
                target_path.write_bytes(raw_bytes)
                extracted_text = self._extract_supporting_document_text(filename, raw_bytes)
                conn.execute(
                    """
                    INSERT INTO supporting_documents(
                        id, document_type, created_at, original_name, stored_name, filepath, extracted_text, byte_size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        normalized,
                        time.time(),
                        str(filename),
                        stored_name,
                        str(target_path),
                        extracted_text,
                        int(len(raw_bytes)),
                    ),
                )
                saved_any = True
            conn.commit()
        if not saved_any:
            raise ValueError("Belum ada file pendukung yang valid untuk diunggah.")
        return self.get_settings()

    def delete_supporting_document(self, document_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT filepath
                FROM supporting_documents
                WHERE id = ?
                """,
                (str(document_id or ""),),
            ).fetchone()
            if not row:
                raise KeyError("Dokumen pendukung tidak ditemukan.")
            filepath = Path(str(row["filepath"] or ""))
            conn.execute("DELETE FROM supporting_documents WHERE id = ?", (str(document_id or ""),))
            conn.commit()
        if filepath.exists():
            try:
                filepath.unlink()
            except OSError:
                pass
        return self.get_settings()

    def _collect_supporting_document_text(self, document_type: str, max_documents: int = 4, max_words: int = 220) -> str:
        normalized = self._normalize_document_type(document_type)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT extracted_text
                FROM supporting_documents
                WHERE document_type = ? AND extracted_text <> ''
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (normalized, int(max_documents)),
            ).fetchall()
        chunks = [self._trim_supporting_text(str(row["extracted_text"] or ""), max_words=max_words) for row in rows]
        chunks = [chunk for chunk in chunks if chunk]
        return "\n".join(chunks).strip()

    @staticmethod
    def _first_non_empty_match(text: str, patterns: List[str], flags: int = re.IGNORECASE) -> str:
        source = str(text or "")
        for pattern in patterns:
            match = re.search(pattern, source, flags)
            if match:
                value = re.sub(r"\s+", " ", str(match.group(1) or "")).strip(" -:;,.")
                if value:
                    return value
        return ""

    @staticmethod
    def _best_money_signal(text: str) -> str:
        matches = re.findall(
            r"(Rp\.?\s?[\d.,]+(?:\s?(?:miliar|juta|triliun))?)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        cleaned = []
        for match in matches:
            value = re.sub(r"\s+", " ", match).strip()
            if value and value.lower() not in {item.lower() for item in cleaned}:
                cleaned.append(value)
        return cleaned[0] if cleaned else ""

    @staticmethod
    def _best_duration_signal(text: str) -> str:
        timeline_items = AppStateStore._extract_kak_timeline_items(text)
        max_week = 0
        max_month = 0
        for item in timeline_items:
            period = str(item.get("period") or "")
            week_match = re.search(r"\bminggu\s+(\d+)(?:\s*[-–]\s*(\d+))?", period, flags=re.IGNORECASE)
            month_match = re.search(r"\bbulan\s+(\d+)(?:\s*[-–]\s*(\d+))?", period, flags=re.IGNORECASE)
            if week_match:
                max_week = max(max_week, int(week_match.group(2) or week_match.group(1)))
            if month_match:
                max_month = max(max_month, int(month_match.group(2) or month_match.group(1)))
        if max_week:
            return f"{max_week} Minggu"
        if max_month:
            return f"{max_month} Bulan"

        patterns = [
            r"(?:jangka waktu(?:\s+pelaksanaan)?|durasi|masa pelaksanaan|waktu pelaksanaan)\s*[:\-]?\s*([^\n.;]{3,60})",
            r"(?:selama|dalam kurun waktu)\s*[:\-]?\s*(\d+\s*(?:hari|minggu|bulan|tahun)(?:\s*(?:kalender|kerja))?)",
            r"(\d+\s*(?:hari|minggu|bulan|tahun)(?:\s*(?:kalender|kerja))?)",
        ]
        value = AppStateStore._first_non_empty_match(text, patterns)
        value = re.sub(r"^(?:pelaksanaan|jangka waktu|durasi)\s*[:\-]?\s*", "", value, flags=re.IGNORECASE)
        return value.strip(" :;-.,")

    @staticmethod
    def _extract_kak_timeline_items(text: str, limit: int = 8) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        seen: Set[str] = set()
        source = formalize_caps_text(text)
        for raw_line in source.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line or not re.search(r"\b(minggu|pekan|bulan|hari)\b", line, flags=re.IGNORECASE):
                continue
            if re.search(r"\b(no|periode|fase|aktivitas|deliverable)\b", line, flags=re.IGNORECASE) and "|" in line:
                continue

            cells = [cell.strip(" -") for cell in line.split("|") if cell.strip(" -")]
            period = ""
            phase = ""
            activity = ""
            deliverable = ""
            if len(cells) >= 2:
                for idx, cell in enumerate(cells):
                    if re.search(r"\b(minggu|pekan|bulan|hari)\b", cell, flags=re.IGNORECASE):
                        period = cell
                        phase = cells[idx + 1] if idx + 1 < len(cells) else ""
                        activity = cells[idx + 2] if idx + 2 < len(cells) else ""
                        deliverable = cells[idx + 3] if idx + 3 < len(cells) else ""
                        break
            else:
                match = re.search(
                    r"((?:minggu|pekan|bulan|hari)\s+\d+(?:\s*[-–]\s*\d+)?)\s*[:\-]?\s*(.+)$",
                    line,
                    flags=re.IGNORECASE,
                )
                if match:
                    period = match.group(1)
                    activity = match.group(2)

            if not period:
                continue
            key = SchemaMapper.normalize_key(f"{period}|{phase}|{activity}|{deliverable}")
            if not key or key in seen:
                continue
            seen.add(key)
            items.append({
                "period": normalize_field_value("estimasi_waktu", period),
                "phase": formalize_caps_text(phase)[:120],
                "activity": formalize_caps_text(activity)[:220],
                "deliverable": formalize_caps_text(deliverable)[:180],
            })
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _extract_first_section(text: str, labels: List[str], max_lines: int = 3) -> str:
        source = str(text or "")
        label_blob = "|".join(re.escape(item) for item in labels if item)
        if not label_blob:
            return ""
        pattern = re.compile(
            rf"(?:^|\n)\s*(?:{label_blob})\s*[:\-]?\s*(.+?)(?=\n\s*[A-Z][^\n]{{0,80}}[:\-]|\n\s*\d+[\).\s]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(source)
        if not match:
            return ""
        raw = re.sub(r"\n{2,}", "\n", str(match.group(1) or "")).strip()
        lines = [re.sub(r"\s+", " ", line).strip(" -•\t") for line in raw.splitlines() if re.sub(r"\s+", " ", line).strip()]
        if not lines:
            return ""
        return " ".join(lines[:max_lines]).strip()

    @staticmethod
    def _extract_kak_frameworks(text: str) -> List[str]:
        source = str(text or "")
        known = [
            ("ITIL", r"\bitil\b"),
            ("TOGAF", r"\btogaf\b"),
            ("COBIT", r"\bcobit\b"),
            ("ISO 27001", r"\biso\s*27001\b"),
            ("ISO 20000", r"\biso\s*20000\b"),
            ("ISO 9001", r"\biso\s*9001\b"),
            ("NIST", r"\bnist\b"),
            ("POJK", r"\bpojk\b"),
            ("OJK", r"\bojk\b"),
            ("UU PDP", r"\bpdp\b"),
            ("DAMA", r"\bdama\b"),
            ("TM Forum", r"\btm\s*forum\b"),
            ("Responsible AI", r"\bresponsible ai\b|\bai governance\b|\bai rmf\b"),
            ("Regulasi", r"\bregulasi\b|\bkepatuhan\b"),
        ]
        hits: List[str] = []
        for label, pattern in known:
            if re.search(pattern, source, re.IGNORECASE):
                hits.append(label)
        return hits[:6]

    @classmethod
    def _detect_service_type(cls, text: str) -> str:
        source = str(text or "").lower()
        has_explicit_training = any(
            token in source for token in ["pelatihan", "training", "bimbingan teknis", "bootcamp", "kelas", "kurikulum"]
        )
        has_workshop = "workshop" in source
        has_consulting = any(
            token in source
            for token in [
                "konsultan", "konsultansi", "pendampingan", "assessment",
                "kajian", "review", "penyusunan", "roadmap", "tata kelola",
            ]
        )
        if has_consulting and has_explicit_training:
            return "Training dan Konsultan"
        if has_consulting:
            return "Konsultan"
        if has_explicit_training or has_workshop:
            return "Training"
        return "Konsultan"

    @classmethod
    def _detect_project_type(cls, text: str) -> str:
        source = str(text or "").lower()
        scores = {
            "Diagnostic": 0,
            "Strategic": 0,
            "Transformation": 0,
            "Implementation": 0,
        }
        keyword_map = {
            "Diagnostic": ["assessment", "as is", "gap analysis", "diagnostic", "evaluasi", "kajian awal", "baseline"],
            "Strategic": ["roadmap", "strategi", "blueprint", "target operating model", "arah kebijakan", "rencana strategis", "tata kelola", "governance", "operating cadence", "decision forum"],
            "Transformation": ["transformasi", "redesign", "perubahan", "operating model", "change management"],
            "Implementation": ["implementasi", "deployment", "rollout", "go-live", "uat", "konfigurasi", "pendampingan pelaksanaan"],
        }
        for project_type, keywords in keyword_map.items():
            scores[project_type] = sum(1 for keyword in keywords if keyword in source)
        priority = {"Strategic": 4, "Transformation": 3, "Implementation": 2, "Diagnostic": 1}
        best_type = max(scores.items(), key=lambda item: (item[1], priority.get(item[0], 0)))[0]
        return best_type if scores[best_type] > 0 else "Implementation"

    @classmethod
    def _detect_need_classification(cls, text: str) -> str:
        source = str(text or "").lower()
        needs: List[str] = []
        if any(token in source for token in ["masalah", "kendala", "hambatan", "gap", "isu", "belum", "tidak sinkron"]):
            needs.append("Problem")
        if any(token in source for token in ["optimalisasi", "peningkatan", "peluang", "opportunity", "improvement"]):
            needs.append("Opportunity")
        if any(token in source for token in ["wajib", "ketentuan", "kepatuhan", "regulasi", "mandat", "pojk", "ojk", "peraturan"]):
            needs.append("Directive")
        if not needs:
            needs.append("Problem")
        order = ["Problem", "Opportunity", "Directive"]
        return ", ".join(item for item in order if item in needs)

    @classmethod
    def _infer_company_from_kak(cls, text: str, company_candidates: Optional[List[str]] = None) -> str:
        source = str(text or "")
        candidates = [str(item).strip() for item in (company_candidates or []) if str(item).strip()]
        for candidate in sorted(candidates, key=len, reverse=True):
            if re.search(re.escape(candidate), source, re.IGNORECASE):
                return candidate
        guessed = cls._first_non_empty_match(
            source,
            [
                r"(?:nama perusahaan|instansi|satuan kerja|klien|pemberi kerja)\s*[:\-]\s*([^\n]{3,120})",
                r"\b(PT\.?\s+[A-Z][^\n]{3,80})",
            ],
        )
        return guessed

    @classmethod
    def _summarize_kak_analysis(cls, suggestions: Dict[str, str], source_name: str) -> str:
        bits = []
        if suggestions.get("konteks_organisasi"):
            bits.append(f"inisiatif {suggestions['konteks_organisasi']}")
        if suggestions.get("jenis_proposal"):
            bits.append(f"layanan {suggestions['jenis_proposal']}")
        if suggestions.get("jenis_proyek"):
            bits.append(f"tipe proyek {suggestions['jenis_proyek']}")
        if suggestions.get("estimasi_waktu"):
            bits.append(f"durasi {suggestions['estimasi_waktu']}")
        if suggestions.get("estimasi_biaya"):
            bits.append(f"anggaran {suggestions['estimasi_biaya']}")
        if suggestions.get("potensi_framework"):
            bits.append(f"framework {suggestions['potensi_framework']}")
        joined = "; ".join(bits) if bits else "belum ada field utama yang berhasil dibaca"
        return f"Pembacaan KAK dari {source_name} menangkap {joined}."

    def set_active_kak_document(self, document_id: str) -> Dict[str, Any]:
        normalized_id = str(document_id or "").strip()
        if not normalized_id:
            self._set_setting("active_kak_document_id", "")
            return self.get_settings()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM supporting_documents WHERE id = ? AND document_type = 'kak'",
                (normalized_id,),
            ).fetchone()
        if not row:
            raise KeyError("Dokumen KAK/TOR tidak ditemukan.")
        self._set_setting("active_kak_document_id", normalized_id)
        return self.get_settings()

    def get_latest_kak_context(
        self,
        company_candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        documents = self.list_supporting_documents("kak")
        result: Dict[str, Any] = {
            "documents": documents,
            "analysis": {
                "available": False,
                "source_document": "",
                "summary": "Belum ada dokumen KAK/TOR yang dibaca.",
                "suggestions": {},
                "warnings": [],
            },
        }
        if not documents:
            return result

        active_id = self._get_setting("active_kak_document_id", "")
        latest = next((item for item in documents if str(item.get("id") or "") == active_id), None) or documents[0]
        active_id = str(latest.get("id") or "")
        for item in documents:
            item["is_active"] = str(item.get("id") or "") == active_id
        result["documents"] = documents
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT extracted_text
                FROM supporting_documents
                WHERE id = ?
                """,
                (str(latest.get("id") or ""),),
            ).fetchone()
        text = str((row["extracted_text"] if row else "") or "").strip()
        if not text:
            result["analysis"] = {
                "available": False,
                "source_document": latest.get("original_name", ""),
                "summary": "Dokumen KAK/TOR sudah diunggah, tetapi teksnya belum berhasil dibaca.",
                "suggestions": {},
                "warnings": ["Teks KAK belum bisa diekstrak dari file yang diunggah."],
            }
            return result

        objective = self._extract_first_section(
            text,
            ["maksud dan tujuan", "tujuan", "objective", "sasaran"],
            max_lines=3,
        )
        initiative = self._first_non_empty_match(
            text,
            [
                r"(?:nama|judul|paket|objek)\s*(?:pekerjaan|pengadaan|jasa)?\s*[:\-]\s*([^\n]{6,180})",
                r"(?:pekerjaan|kegiatan)\s*[:\-]\s*([^\n]{6,180})",
            ],
        )
        if not initiative:
            lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if re.sub(r"\s+", " ", line).strip()]
            for line in lines[:12]:
                if 5 <= len(line.split()) <= 22 and not re.search(r"^(bab|pasal|latar belakang|maksud dan tujuan|ruang lingkup)\b", line, re.IGNORECASE):
                    initiative = line
                    break
        context_text = " ".join(item for item in [initiative, objective] if item).strip() or initiative or objective
        problem_excerpt = self._extract_first_section(
            text,
            ["latar belakang", "permasalahan", "isu utama", "tantangan", "kondisi eksisting"],
            max_lines=4,
        )
        budget = self._best_money_signal(text)
        duration = self._best_duration_signal(text)
        timeline_items = self._extract_kak_timeline_items(text)
        frameworks = self._extract_kak_frameworks(text)
        service_type = self._detect_service_type(text)
        project_type = self._detect_project_type(text)
        need_classification = self._detect_need_classification(text)
        company = self._infer_company_from_kak(text, company_candidates=company_candidates)
        suggestion_warnings: List[str] = []
        if not budget:
            suggestion_warnings.append("Nilai anggaran tidak terbaca jelas dari KAK.")
        if not duration:
            suggestion_warnings.append("Durasi pelaksanaan tidak terbaca jelas dari KAK.")
        if not frameworks:
            suggestion_warnings.append("Framework/acuan belum terbaca eksplisit; user mungkin masih perlu mengisi manual.")

        suggestions = normalize_payload({
            "nama_perusahaan": company,
            "jenis_proposal": service_type,
            "jenis_proyek": project_type,
            "konteks_organisasi": context_text,
            "permasalahan": problem_excerpt,
            "klasifikasi_kebutuhan": need_classification,
            "estimasi_waktu": duration,
            "estimasi_biaya": budget,
            "potensi_framework": ", ".join(frameworks),
        })
        suggestions = {key: value for key, value in suggestions.items() if str(value or "").strip()}

        result["analysis"] = {
            "available": True,
            "source_document": latest.get("original_name", ""),
            "source_document_id": latest.get("id", ""),
            "summary": self._summarize_kak_analysis(suggestions, latest.get("original_name", "dokumen KAK")),
            "suggestions": suggestions,
            "warnings": suggestion_warnings,
            "initiative_title": initiative,
            "objective_excerpt": objective,
            "problem_excerpt": problem_excerpt,
            "timeline_items": timeline_items,
        }
        return result

    @classmethod
    def _fallback_portfolio_rows_from_text(cls, text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for raw_line in str(text or "").splitlines():
            line = re.sub(r"^\s*[-*•\d.]+\s*", "", raw_line).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line.split()) < 4:
                continue
            if re.search(r"\b(belum pernah|tidak pernah|n/a|none)\b", line, re.IGNORECASE):
                continue
            normalized = SchemaMapper.normalize_key(line)
            if not normalized or normalized in seen:
                continue
            if normalized in {
                "profil_perusahaan", "profile_perusahaan", "kapabilitas", "sertifikasi",
                "pengalaman", "project_experience", "portofolio", "portfolio"
            }:
                continue
            seen.add(normalized)
            rows.append(
                {
                    "area": line[:160],
                    "relevansi": "relevan untuk menunjukkan pengalaman serupa dan kesiapan pelaksanaan pekerjaan",
                    "bukti": "ringkasan portofolio internal dan pengalaman sejenis perusahaan penyusun",
                    "nilai_tambah": "memperkuat kredibilitas proposal dan membantu klien melihat bukti kemampuan secara lebih konkret",
                }
            )
            if len(rows) >= 4:
                break
        return rows

    @staticmethod
    def _dedupe_phrases(items: List[str], limit: int = 6) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for item in items:
            clean = re.sub(r"\s+", " ", str(item or "")).strip(" -;|,.:")
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(clean)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _summarize_profile_blob(text: str, fallback: str, max_items: int = 4) -> str:
        source = str(text or "")
        if not source.strip():
            return fallback
        normalized = source.replace("|", "\n").replace(";", "\n")
        candidates: List[str] = []
        for raw_line in normalized.splitlines():
            line = re.sub(r"^\s*[-*•\d.]+\s*", "", raw_line).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line.split()) < 4:
                continue
            if re.search(
                r"\b(no|nama tenaga|posisi diusulkan|tingkat pendidikan|lama pengalaman|peran dalam penugasan|tahun)\b",
                line,
                re.IGNORECASE,
            ):
                continue
            if re.search(r"\b(belum pernah|tidak pernah|n/a|none)\b", line, re.IGNORECASE):
                continue
            candidates.append(line[:180])
        picked = AppStateStore._dedupe_phrases(candidates, limit=max_items)
        if not picked:
            return fallback
        return "; ".join(picked)

    @classmethod
    def _summarize_credential_blob(cls, text: str, fallback: str) -> str:
        source = str(text or "")
        if not source.strip():
            return fallback
        certifications = re.findall(
            r"\b(?:TOGAF|COBIT(?:\s*\d+)?|ITIL(?:\s*[A-Za-z0-9.+-]+)?|ISO\s*\/?\s*IEC?\s*\d+|ISO\s*\d+|CEH|CISA|CHFI|CCNA|CAPM|Project\+|Lead Auditor ISO 27001|Microsoft Certified Database Administrator)\b",
            source,
            flags=re.IGNORECASE,
        )
        certification_list = cls._dedupe_phrases(certifications, limit=6)

        role_signals: List[str] = []
        role_map = [
            (r"\bproject manager\b|\bpmo\b", "project management"),
            (r"\bsecurity\b|\bethical hacker\b|\bincident handler\b|\bforensic\b", "cyber security"),
            (r"\bgovernance\b|\bcobit\b|\biso\b", "IT governance"),
            (r"\bnetwork\b|\bccna\b", "network & infrastructure"),
            (r"\bprivacy\b|\bpdp\b|\bropa\b|\bdpia\b", "data privacy & compliance"),
            (r"\btechnical writer\b|\bdokumentasi\b", "documentation support"),
        ]
        for pattern, label in role_map:
            if re.search(pattern, source, re.IGNORECASE):
                role_signals.append(label)
        role_list = cls._dedupe_phrases(role_signals, limit=4)

        parts: List[str] = []
        if role_list:
            parts.append(f"Kapabilitas tim mencakup {', '.join(role_list)}")
        if certification_list:
            parts.append(f"Sertifikasi inti meliputi {', '.join(certification_list)}")
        if parts:
            return ". ".join(parts) + "."
        return cls._summarize_profile_blob(source, fallback, max_items=3)

    @classmethod
    def _summarize_portfolio_blob(cls, text: str, fallback: str) -> str:
        source = str(text or "")
        if not source.strip():
            return fallback
        if "|" in source:
            rows = cls._parse_structured_portfolio_rows(source)
            areas = cls._dedupe_phrases([row.get("area", "") for row in rows], limit=4)
            if areas:
                return f"Pengalaman perusahaan mencakup {', '.join(areas)}."
        return cls._summarize_profile_blob(source, fallback, max_items=3)

    def get_settings(self) -> Dict[str, Any]:
        template_path = self.get_template_path()
        active_kak_document_id = self._get_setting("active_kak_document_id")
        kak_documents = self.list_supporting_documents("kak")
        if not active_kak_document_id and kak_documents:
            active_kak_document_id = str(kak_documents[0].get("id") or "")
        for item in kak_documents:
            item["is_active"] = bool(active_kak_document_id and str(item.get("id") or "") == active_kak_document_id)
        return {
            "internal_portfolio": self._get_setting("internal_portfolio"),
            "internal_credentials": self._get_setting("internal_credentials"),
            "active_template_name": self._get_setting("active_template_name"),
            "active_kak_document_id": active_kak_document_id,
            "has_active_template": bool(template_path and Path(template_path).exists()),
            "portfolio_documents": self.list_supporting_documents("portfolio"),
            "credential_documents": self.list_supporting_documents("credentials"),
            "kak_documents": kak_documents,
        }

    def save_settings(self, internal_portfolio: str = "", internal_credentials: str = "") -> Dict[str, Any]:
        self._set_setting("internal_portfolio", formalize_caps_text(internal_portfolio or ""))
        self._set_setting("internal_credentials", formalize_caps_text(internal_credentials or ""))
        return self.get_settings()

    def build_generation_context(self, company_candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        kak = self.get_latest_kak_context(company_candidates=company_candidates).get("analysis", {})
        settings = self.get_settings()
        portfolio_lines = compact_context_lines(
            [
                settings.get("internal_portfolio", ""),
                self._collect_supporting_document_text("portfolio", max_documents=3, max_words=120),
            ],
            limit=4,
        )
        credential_lines = compact_context_lines(
            [
                settings.get("internal_credentials", ""),
                self._collect_supporting_document_text("credentials", max_documents=3, max_words=100),
            ],
            limit=4,
        )
        suggestions = normalize_payload(kak.get("suggestions") or {}) if kak.get("available") else {}
        timeline_items = kak.get("timeline_items") if isinstance(kak.get("timeline_items"), list) else []
        kak_lines = compact_context_lines(
            [
                kak.get("summary", ""),
                kak.get("objective_excerpt", ""),
                kak.get("problem_excerpt", ""),
                *(f"{item.get('period', '')}: {item.get('phase', '')} {item.get('activity', '')} {item.get('deliverable', '')}" for item in timeline_items[:5]),
            ],
            limit=7,
        )
        return {
            "kak_available": bool(kak.get("available")),
            "kak_source_document": formalize_caps_text(kak.get("source_document", "")),
            "kak_suggestions": suggestions,
            "kak_timeline_items": timeline_items[:8],
            "kak_context": "\n".join(kak_lines),
            "settings_context": "\n".join(portfolio_lines + credential_lines),
            "portfolio_context": "\n".join(portfolio_lines),
            "credential_context": "\n".join(credential_lines),
        }

    def save_template(self, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        safe_name = self._sanitize_filename(filename or "template_blanko.docx", fallback="template_blanko.docx")
        if not safe_name.lower().endswith(".docx"):
            safe_name = f"{safe_name}.docx"
        target_path = self.templates_dir / f"active_{safe_name}"
        for existing in self.templates_dir.glob("active_*"):
            try:
                existing.unlink()
            except OSError:
                continue
        target_path.write_bytes(raw_bytes)
        self._set_setting("active_template_path", str(target_path))
        self._set_setting("active_template_name", safe_name)
        return self.get_settings()

    def clear_template(self) -> Dict[str, Any]:
        template_path = self.get_template_path()
        if template_path and Path(template_path).exists():
            try:
                Path(template_path).unlink()
            except OSError:
                pass
        self._set_setting("active_template_path", "")
        self._set_setting("active_template_name", "")
        return self.get_settings()

    def get_template_path(self) -> str:
        return self._get_setting("active_template_path", "")

    @staticmethod
    def _parse_structured_portfolio_rows(text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for raw_line in str(text or "").splitlines():
            line = re.sub(r"^\s*[-*•\d.]+\s*", "", raw_line).strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("|")]
            normalized_line = re.sub(r"\s+", " ", line).strip()
            if re.search(r"\b(belum pernah|tidak pernah|n/a|none|not found)\b", normalized_line, re.IGNORECASE):
                continue
            if re.search(
                r"\b(area pengalaman|relevansi|bukti kapabilitas|nilai tambah|portofolio|portfolio)\b",
                normalized_line,
                re.IGNORECASE,
            ):
                continue
            if len(parts) >= 4:
                area, relevansi, bukti, nilai_tambah = [re.sub(r"\s+", " ", part).strip(" -;:") for part in parts[:4]]
                if not area or len(re.findall(r"\w+", area)) < 3:
                    continue
                rows.append(
                    {
                        "area": area,
                        "relevansi": relevansi,
                        "bukti": bukti,
                        "nilai_tambah": nilai_tambah,
                    }
                )
            elif len(parts) == 3:
                area, relevansi, bukti = [re.sub(r"\s+", " ", part).strip(" -;:") for part in parts[:3]]
                if not area or len(re.findall(r"\w+", area)) < 3:
                    continue
                rows.append(
                    {
                        "area": area,
                        "relevansi": relevansi,
                        "bukti": bukti,
                        "nilai_tambah": "menambah keyakinan klien terhadap kesiapan delivery dan kualitas hasil kerja",
                    }
                )
            else:
                if len(re.findall(r"\w+", normalized_line)) < 5:
                    continue
                rows.append(
                    {
                        "area": normalized_line,
                        "relevansi": "relevan untuk inisiatif yang membutuhkan pengalaman delivery dan advisory yang serupa",
                        "bukti": normalized_line,
                        "nilai_tambah": "membantu proposal terasa lebih konkret dan kredibel",
                    }
                )
        return rows[:6]

    def enrich_firm_profile(self, firm_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        profile = dict(firm_profile or {})
        internal_portfolio = self._get_setting("internal_portfolio", "")
        internal_credentials = self._get_setting("internal_credentials", "")
        portfolio_docs_text = self._collect_supporting_document_text("portfolio", max_documents=4, max_words=180)
        credential_docs_text = self._collect_supporting_document_text("credentials", max_documents=4, max_words=160)
        portfolio_bits = [
            str(profile.get("portfolio_highlights") or "").strip(),
            internal_portfolio.strip(),
            portfolio_docs_text.strip(),
        ]
        portfolio_text = " ; ".join([item for item in portfolio_bits if item])
        if portfolio_text:
            profile["portfolio_highlights"] = self._summarize_portfolio_blob(
                portfolio_text,
                str(profile.get("portfolio_highlights") or "").strip() or "Pengalaman perusahaan disesuaikan dengan kebutuhan proyek klien.",
            )
        credential_bits = [
            str(profile.get("credential_highlights") or "").strip(),
            internal_credentials.strip(),
            credential_docs_text.strip(),
        ]
        credential_text = " ; ".join([item for item in credential_bits if item])
        if credential_text:
            profile["credential_highlights"] = self._summarize_credential_blob(
                credential_text,
                str(profile.get("credential_highlights") or "").strip() or "Kapabilitas inti dan sertifikasi relevan perusahaan penyusun.",
            )
        structured_rows = self._parse_structured_portfolio_rows(internal_portfolio)
        if not structured_rows and "|" in portfolio_docs_text:
            structured_rows = self._parse_structured_portfolio_rows(portfolio_docs_text)
        if not structured_rows:
            structured_rows = self._fallback_portfolio_rows_from_text(portfolio_docs_text)
        profile["internal_portfolio_rows"] = structured_rows
        profile["portfolio_document_names"] = [
            item.get("original_name", "")
            for item in self.list_supporting_documents("portfolio")
            if str(item.get("original_name") or "").strip()
        ]
        profile["credential_document_names"] = [
            item.get("original_name", "")
            for item in self.list_supporting_documents("credentials")
            if str(item.get("original_name") or "").strip()
        ]
        return profile

    def persist_generated_file(self, suggested_name: str, content: bytes) -> Path:
        safe_name = self._sanitize_filename(suggested_name or "proposal.docx", fallback="proposal.docx")
        if not safe_name.lower().endswith(".docx"):
            safe_name = f"{safe_name}.docx"
        target = self.generated_dir / safe_name
        counter = 2
        while target.exists():
            stem = target.stem
            stem = re.sub(r"_\d+$", "", stem)
            target = self.generated_dir / f"{stem}_{counter}{target.suffix}"
            counter += 1
        target.write_bytes(content)
        return target

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
        entry_id = uuid.uuid4().hex
        acceptance = dict(acceptance_report or {})
        acceptance_score = int(acceptance.get("score") or 0)
        acceptance_passes = 1 if acceptance.get("passes") else 0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO proposal_history(
                    id, created_at, finished_at, client, project, proposal_mode,
                    service_type, project_type, timeline, budget, acceptance_score,
                    acceptance_passes, processing_seconds, acceptance_json, filename, filepath, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    float(created_at or time.time()),
                    float(finished_at or time.time()),
                    str(payload.get("nama_perusahaan") or ""),
                    str(payload.get("konteks_organisasi") or ""),
                    str(payload.get("mode_proposal") or "canvassing"),
                    str(payload.get("jenis_proposal") or ""),
                    str(payload.get("jenis_proyek") or ""),
                    str(payload.get("estimasi_waktu") or ""),
                    str(payload.get("estimasi_biaya") or ""),
                    acceptance_score,
                    acceptance_passes,
                    float(processing_seconds or 0.0),
                    json.dumps(acceptance, ensure_ascii=False),
                    str(filename or ""),
                    str(filepath or ""),
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
        return entry_id

    @staticmethod
    def _format_timestamp(epoch_value: float) -> str:
        try:
            return datetime.fromtimestamp(float(epoch_value)).strftime("%d %b %Y %H:%M")
        except Exception:
            return "-"

    def list_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, finished_at, client, project, proposal_mode,
                       service_type, project_type, timeline, budget, acceptance_score,
                       acceptance_passes, processing_seconds, acceptance_json, filename, filepath, payload_json
                FROM proposal_history
                ORDER BY finished_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        items: List[Dict[str, Any]] = []
        for row in rows:
            filepath = str(row["filepath"] or "")
            items.append(
                {
                    "id": row["id"],
                    "client": row["client"],
                    "project": row["project"],
                    "proposal_mode": row["proposal_mode"],
                    "service_type": row["service_type"],
                    "project_type": row["project_type"],
                    "timeline": row["timeline"],
                    "budget": row["budget"],
                    "acceptance_score": int(row["acceptance_score"] or 0),
                    "acceptance_passes": bool(row["acceptance_passes"]),
                    "processing_seconds": float(row["processing_seconds"] or 0.0),
                    "filename": row["filename"],
                    "filepath": filepath,
                    "exists": bool(filepath and Path(filepath).exists()),
                    "can_reuse": bool(str(row["payload_json"] or "").strip() not in {"", "{}"}),
                    "finished_at": float(row["finished_at"]),
                    "finished_at_label": self._format_timestamp(float(row["finished_at"])),
                }
            )
        return items

    def get_history_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM proposal_history
                WHERE id = ?
                """,
                (entry_id,),
            ).fetchone()
        if not row:
            return None
        payload = {}
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = {}
        acceptance_report = {}
        try:
            acceptance_report = json.loads(row["acceptance_json"] or "{}")
        except Exception:
            acceptance_report = {}
        filepath = str(row["filepath"] or "")
        return {
            "id": row["id"],
            "client": row["client"],
            "project": row["project"],
            "proposal_mode": row["proposal_mode"],
            "service_type": row["service_type"],
            "project_type": row["project_type"],
            "timeline": row["timeline"],
            "budget": row["budget"],
            "acceptance_score": int(row["acceptance_score"] or 0),
            "acceptance_passes": bool(row["acceptance_passes"]),
            "processing_seconds": float(row["processing_seconds"] or 0.0),
            "acceptance_report": acceptance_report,
            "filename": row["filename"],
            "filepath": filepath,
            "exists": bool(filepath and Path(filepath).exists()),
            "created_at": float(row["created_at"]),
            "finished_at": float(row["finished_at"]),
            "payload": payload,
            "can_reuse": bool(payload),
        }
