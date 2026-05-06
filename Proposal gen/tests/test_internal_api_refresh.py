"""Regression tests for operator-facing Internal API refresh controls."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class InternalApiRefreshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(cls.tmp.name)
        os.environ.update(
            {
                "APP_STATE_DB_PATH": str(tmp_path / "app_state.db"),
                "APP_ASSET_ROOT": str(tmp_path / "assets"),
                "GENERATED_OUTPUT_DIR": str(tmp_path / "generated"),
                "PROJECT_DB_PATH": str(tmp_path / "projects.db"),
                "PROJECT_CSV_PATH": str(ROOT / "db.csv"),
                "VECTOR_STORE_DIR": str(tmp_path / "chroma"),
                "KB_SYNC_STATE_PATH": str(tmp_path / "kb_state.json"),
                "PROJECT_DATA_SOURCE": "api",
                "APP_PROFILE": "demo",
                "INTERNAL_DATA_SOURCE": "api",
            }
        )
        from main import app as app_module

        cls.app_module = app_module
        cls.app = app_module.app

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_internal_api_refresh_endpoint_forces_background_dataset_reload(self) -> None:
        client = self.app.test_client()
        fake_kb = Mock()
        fake_kb.refresh_data.return_value = True
        fake_kb.project_data_source = "api"
        fake_kb.sync_in_progress = True
        fake_kb.last_refresh_error = ""
        fake_queue = Mock()
        fake_queue.has_live_jobs.return_value = False
        config_path = Path(self.tmp.name) / "internal_api_config.json"
        config_path.write_text("{}", encoding="utf-8")

        with patch.object(self.app_module.auth_flow, "is_authenticated", return_value=True), \
             patch.object(self.app_module.auth_flow, "touch_active_auth_session", return_value=True), \
             patch.object(self.app_module, "knowledge_base", fake_kb), \
             patch.object(self.app_module, "generation_queue", fake_queue), \
             patch.object(self.app_module, "FirmAPIClient") as firm_client, \
             patch.object(self.app_module, "InternalDataClient") as internal_client:
            firm_client.return_value.config_file = str(config_path)
            internal_client.return_value.describe_runtime.return_value = {"provider": "api"}
            response = client.post("/api/internal-api/refresh")

        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["status"], "refreshing")
        self.assertTrue(payload["refresh_started"])
        fake_kb.set_project_data_source.assert_called_once_with("api")
        fake_kb.refresh_data.assert_called_once_with(force=True, background=True)

    def test_settings_modal_shows_refresh_button_and_connected_save_state(self) -> None:
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        soup = BeautifulSoup(template, "html.parser")

        save_button = soup.select_one("#btn-save-internal-api")
        refresh_button = soup.select_one("#btn-refresh-internal-api")
        script = "\n".join(script.string or "" for script in soup.find_all("script"))

        self.assertIsNotNone(save_button)
        self.assertEqual(save_button.text.strip(), "Simpan & Aktifkan Internal API")
        self.assertIsNotNone(refresh_button)
        self.assertIn("Refresh Dataset Sekarang", refresh_button.text)
        self.assertIn("setInternalApiConnectionState", script)
        self.assertIn("api_connection_active", script)
        self.assertIn("/api/internal-api/refresh", script)

    def test_internal_api_setup_treats_environment_resource_config_as_active(self) -> None:
        client = self.app.test_client()
        fake_kb = Mock()
        fake_kb.project_data_source = "api"
        fake_kb.sync_in_progress = False
        fake_kb.last_refresh_error = ""

        with patch.object(self.app_module.auth_flow, "is_authenticated", return_value=True), \
             patch.object(self.app_module.auth_flow, "touch_active_auth_session", return_value=True), \
             patch.object(self.app_module, "knowledge_base", fake_kb), \
             patch.object(self.app_module, "FirmAPIClient") as firm_client:
            firm_client.return_value.config_file = ""
            firm_client.return_value.resource_config = {"project_records": {"request": {}}}
            firm_client.return_value.describe_runtime.return_value = {"provider": "api"}
            firm_client.return_value.validate_config.return_value = {"resources": {}}
            response = client.get("/api/internal-api/setup")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["api_connection_active"])
        self.assertTrue(payload["can_refresh_dataset"])


if __name__ == "__main__":
    unittest.main()
