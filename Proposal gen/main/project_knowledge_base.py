"""Knowledge-base and semantic retrieval runtime."""
from __future__ import annotations

from .proposal_shared import *
from .schema_mapping import SchemaMapper


class KnowledgeBase:
    def __init__(self, db_uri: str) -> None:
        self.engine = create_engine(db_uri)
        self.vector_store_dir = VECTOR_STORE_DIR
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.sync_state_path = KB_SYNC_STATE_PATH
        self.sync_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=str(self.vector_store_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embed_fn = embedding_functions.OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings",
            model_name=EMBED_MODEL,
            timeout=KB_EMBED_TIMEOUT_SECONDS,
        )
        self.collection = self.chroma.get_or_create_collection(
            name="projects_db", embedding_function=self.embed_fn
        )
        self.df: Optional[pd.DataFrame] = None
        self.last_refresh_error = ""
        self.vector_ready = False
        self.sync_in_progress = False
        self.project_data_source = PROJECT_DATA_SOURCE
        self._sync_lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None
        self._load_project_data()
        self.refresh_data(background=True)

    @staticmethod
    def _required_project_fields() -> Tuple[str, ...]:
        return ("entity", "topic")

    @classmethod
    def _normalize_projects_df(cls, raw_df: pd.DataFrame) -> pd.DataFrame:
        normalized = raw_df.copy()
        normalized.columns = [str(col).strip() for col in normalized.columns]
        normalized = SchemaMapper.remap_dataframe(normalized, PROJECT_DATA_FIELD_ALIASES)
        return normalized

    @classmethod
    def _has_required_project_fields(cls, df: Optional[pd.DataFrame]) -> bool:
        if df is None:
            return False
        return all(field in df.columns for field in cls._required_project_fields())

    def set_project_data_source(self, source: str) -> None:
        normalized = str(source or "").strip().lower()
        self.project_data_source = "api" if normalized in {"api", "internal_api", "live"} else "local"

    def _load_project_data(self) -> bool:
        if self.project_data_source == "api":
            if self._load_project_data_from_api():
                return True
            if INTERNAL_DATA_FALLBACK == "demo":
                logger.warning(
                    "API-backed project data failed. Falling back to local project data because INTERNAL_DATA_FALLBACK=demo."
                )
                return self._load_project_data_from_local()
            return False
        return self._load_project_data_from_local()

    def _load_project_data_from_api(self) -> bool:
        try:
            from .runtime_components import FirmAPIClient
            records = FirmAPIClient(force_source="api").get_project_records()
            if not records:
                raise ValueError("project_records API returned no records.")
            normalized_df = self._normalize_projects_df(pd.DataFrame(records))
            normalized_df.to_sql("projects", self.engine, index=False, if_exists="replace")
            self.df = normalized_df
        except Exception as exc:
            self.vector_ready = False
            self.last_refresh_error = f"API project data load failed: {exc}"
            logger.warning("API project data load failed: %s", exc)
            return False

        if not self._has_required_project_fields(self.df):
            logger.warning(
                "API project data schema is missing required fields. Expected aliases for: %s. Available columns: %s",
                ", ".join(self._required_project_fields()),
                ", ".join(self.df.columns.astype(str).tolist()) if self.df is not None else "-",
            )
            self.vector_ready = False
            self.last_refresh_error = "API project data schema is missing required fields."
            return False
        return True

    def _load_project_data_from_local(self) -> bool:
        try:
            self.df = self._normalize_projects_df(pd.read_sql("SELECT * FROM projects", self.engine))
        except Exception:
            if not PROJECT_CSV_PATH.exists():
                self.df = None
                self.vector_ready = False
                self.last_refresh_error = f"Project source file not found: {PROJECT_CSV_PATH}"
                return False
            raw_df = pd.read_csv(PROJECT_CSV_PATH)
            normalized_df = self._normalize_projects_df(raw_df)
            normalized_df.to_sql("projects", self.engine, index=False, if_exists='replace')
            self.df = normalized_df

        if not self._has_required_project_fields(self.df):
            logger.warning(
                "Project data schema is missing required fields. Expected aliases for: %s. Available columns: %s",
                ", ".join(self._required_project_fields()),
                ", ".join(self.df.columns.astype(str).tolist()) if self.df is not None else "-",
            )
            self.vector_ready = False
            self.last_refresh_error = "Project data schema is missing required fields."
            return False
        return True

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        clean_row = row.fillna("").astype(str)
        return " | ".join(f"{col}: {val}" for col, val in clean_row.items())

    @staticmethod
    def _row_to_meta(row: pd.Series) -> Dict[str, str]:
        return row.fillna("").astype(str).to_dict()

    def _compute_vector_signature(self) -> str:
        if self.df is None or self.df.empty:
            return ""
        parts: List[str] = []
        for idx, row in self.df.iterrows():
            payload = {"_row_id": str(idx), **self._row_to_meta(row)}
            parts.append(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def _read_sync_state(self) -> Dict[str, Any]:
        if not self.sync_state_path.exists():
            return {}
        try:
            payload = json.loads(self.sync_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_sync_state(self, signature: str, row_count: int) -> None:
        payload = {
            "signature": signature,
            "row_count": int(row_count),
            "synced_at": time.time(),
            "embed_model": EMBED_MODEL,
        }
        self.sync_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _vector_store_current(self, signature: str) -> bool:
        if not signature:
            return False
        state = self._read_sync_state()
        if state.get("signature") != signature:
            return False
        if state.get("embed_model") != EMBED_MODEL:
            return False
        try:
            row_count = len(self.df) if self.df is not None else 0
            return int(self.collection.count()) == int(row_count)
        except Exception:
            return False

    def _sync_vector_store(self, force: bool = False) -> bool:
        if self.df is None or self.df.empty or not self._has_required_project_fields(self.df):
            self.vector_ready = False
            self.last_refresh_error = "Project data is not loaded."
            return False

        signature = self._compute_vector_signature()
        if not force and self._vector_store_current(signature):
            self.vector_ready = True
            self.last_refresh_error = ""
            return True

        try:
            existing_payload = self.collection.get(include=["metadatas"])
            existing_ids_list = existing_payload.get("ids") or []
            existing_metas_list = existing_payload.get("metadatas") or []
            existing_ids = set(existing_ids_list)
            existing_meta_by_id = {
                str(item_id): dict(existing_metas_list[index] or {})
                for index, item_id in enumerate(existing_ids_list)
                if index < len(existing_metas_list)
            }
            new_ids_map = {str(idx): row for idx, row in self.df.iterrows()}
            new_ids_set = set(new_ids_map.keys())

            ids_to_delete = list(existing_ids - new_ids_set)
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)

            ids_to_upsert = [
                item_id
                for item_id in sorted(new_ids_set, key=lambda value: int(value) if str(value).isdigit() else str(value))
                if existing_meta_by_id.get(item_id) != self._row_to_meta(new_ids_map[item_id])
            ]
            if ids_to_upsert:
                batch_size = KB_UPSERT_BATCH_SIZE
                for i in range(0, len(ids_to_upsert), batch_size):
                    batch_ids = ids_to_upsert[i:i + batch_size]
                    docs = [self._row_to_text(new_ids_map[b]) for b in batch_ids]
                    metas = [self._row_to_meta(new_ids_map[b]) for b in batch_ids]
                    self.collection.upsert(documents=docs, metadatas=metas, ids=batch_ids)
            self._write_sync_state(signature, len(new_ids_set))
        except Exception as exc:
            self.vector_ready = False
            self.last_refresh_error = str(exc)
            logger.warning(
                "Knowledge base vector sync is not ready yet. App will continue to serve with degraded semantic retrieval: %s",
                exc,
            )
            return False

        self.vector_ready = True
        self.last_refresh_error = ""
        return True

    def _refresh_worker(self, force: bool) -> None:
        try:
            for attempt in range(1, KB_STARTUP_MAX_RETRIES + 1):
                if not self._load_project_data():
                    success = False
                else:
                    with self._sync_lock:
                        success = self._sync_vector_store(force=force)
                if success:
                    return
                if attempt >= KB_STARTUP_MAX_RETRIES:
                    return
                delay = KB_STARTUP_RETRY_DELAY_SECONDS * attempt
                logger.info(
                    "Knowledge base retry scheduled in %ss (attempt %s/%s).",
                    delay,
                    attempt + 1,
                    KB_STARTUP_MAX_RETRIES,
                )
                time.sleep(delay)
        finally:
            with self._sync_lock:
                self.sync_in_progress = False
                self._sync_thread = None

    def _start_background_refresh(self, force: bool = False) -> bool:
        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                return False
            self.sync_in_progress = True
            self.vector_ready = False
            if not self.last_refresh_error:
                self.last_refresh_error = "Knowledge base semantic index is syncing in the background."
            self._sync_thread = threading.Thread(
                target=self._refresh_worker,
                args=(force,),
                name="knowledge-base-sync",
                daemon=True,
            )
            self._sync_thread.start()
            return True

    def refresh_data(self, force: bool = False, background: bool = False) -> bool:
        if not self._load_project_data():
            logger.warning(
                "Knowledge base startup refresh skipped because dependencies are not ready yet: %s",
                self.last_refresh_error,
            )
            return False

        signature = self._compute_vector_signature()
        if not force and self._vector_store_current(signature):
            self.vector_ready = True
            self.last_refresh_error = ""
            return True

        if background:
            self._start_background_refresh(force=force)
            return True

        with self._sync_lock:
            self.sync_in_progress = True
            try:
                return self._sync_vector_store(force=force)
            finally:
                self.sync_in_progress = False

    def get_exact_context(self, entity: str, topic: str, budget: Optional[str] = None) -> str:
        if self.df is None or self.df.empty or not self._has_required_project_fields(self.df):
            return "No data."
        try:
            match = self.df[(self.df['entity'] == entity) & (self.df['topic'] == topic)]
            if budget and 'budget' in self.df.columns and not match.empty:
                match = match[match['budget'] == budget]
            if not match.empty:
                return "".join([f"- {k.capitalize()}: {v}\n" for k, v in match.iloc[0].to_dict().items()])
            return "No data."
        except Exception:
            return ""

    @staticmethod
    def _rerank_documents(
        query_text: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        distances: Optional[List[float]] = None,
        limit: int = 2,
        enabled: Optional[bool] = None,
    ) -> List[str]:
        if enabled is None:
            enabled = os.getenv("EVIDENCE_QUALITY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return list(documents[:limit])
        try:
            from .evidence_quality import rerank
            candidates = []
            for index, document in enumerate(documents):
                metadata = metadatas[index] if metadatas and index < len(metadatas) else {}
                distance = distances[index] if distances and index < len(distances) else 1.0
                try:
                    semantic_score = 1.0 / (1.0 + max(0.0, float(distance)))
                except (TypeError, ValueError):
                    semantic_score = 0.0
                normalized_meta = dict(metadata or {})
                normalized_meta.setdefault("client", normalized_meta.get("entity", ""))
                normalized_meta.setdefault("project", normalized_meta.get("topic", ""))
                candidates.append({"id": str(index), "text": str(document or ""), "semantic_score": semantic_score, "metadata": normalized_meta})
            retrieval_intent = {
                "goal": "find proposal evidence from internal account, project history, expertise, and framework context",
                "preferred_datasets": [
                    "ReferenceAccount",
                    "ConsultantProjectExpertHistory",
                    "EmployeeExpertise",
                    "ReferenceFramework",
                ],
                "exclude": ["FinanceInvoice", "ProjectStandards"],
                "preferred_terms": [query_text],
            }
            ranked = rerank("proposal", query_text, candidates, limit=limit, retrieval_intent=retrieval_intent)
            ordered = [str(item.get("text") or "") for item in ranked if isinstance(item, dict)]
            return ordered or list(documents[:limit])
        except Exception:
            return list(documents[:limit])

    def query(self, client: str, project: str, context_keywords: str = "") -> str:
        try:
            query_text = f"{project} for {client} {context_keywords}".strip()
            res = self.collection.query(query_texts=[query_text], n_results=8, include=["documents", "metadatas", "distances"])
            documents = (res.get("documents") or [[]])[0]
            if not documents:
                return ""
            metadatas = (res.get("metadatas") or [[]])[0]
            distances = (res.get("distances") or [[]])[0]
            return "\n".join(self._rerank_documents(query_text, documents, metadatas, distances, limit=2))
        except Exception:
            return ""
