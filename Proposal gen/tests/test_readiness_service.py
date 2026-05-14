"""Regression coverage for readiness check composition outside Flask routes."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.df = type(
            "FakeDataFrame",
            (),
            {"empty": False, "columns": ["entity", "topic"]},
        )()
        self.vector_ready = True
        self.last_refresh_error = ""
        self.sync_in_progress = False
        self.vector_store_dir = "/tmp/vector"


class FakeAppState:
    def __init__(self, root: Path) -> None:
        self.db_path = root / "state.db"
        self.generated_dir = root / "generated"
        self.templates_dir = root / "templates"
        self.supporting_docs_dir = root / "docs"
        for path in [self.db_path, self.generated_dir, self.templates_dir, self.supporting_docs_dir]:
            if path.suffix:
                path.write_text("", encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)

    def get_settings(self):
        return {}


class ReadinessServiceTest(unittest.TestCase):
    def test_readiness_builder_keeps_route_free_of_probe_logic(self) -> None:
        from main.readiness import build_readiness_payload

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_db = root / "projects.db"
            project_csv = root / "db.csv"
            project_db.write_text("", encoding="utf-8")
            project_csv.write_text("", encoding="utf-8")

            with patch("main.readiness.requests.get") as fake_get:
                fake_get.return_value.ok = True
                payload, status = build_readiness_payload(
                    project_db_path=project_db,
                    project_csv_path=project_csv,
                    knowledge_base=FakeKnowledgeBase(),
                    app_state_store=FakeAppState(root),
                    ollama_host="http://127.0.0.1:11434",
                )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["knowledge_base"]["ok"])
        self.assertTrue(payload["checks"]["app_state"]["ok"])
        self.assertTrue(payload["checks"]["ollama"]["ok"])


if __name__ == "__main__":
    unittest.main()
