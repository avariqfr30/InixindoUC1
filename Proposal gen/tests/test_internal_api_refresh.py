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
        fake_generator = Mock()
        config_path = Path(self.tmp.name) / "internal_api_config.json"
        config_path.write_text("{}", encoding="utf-8")
        from main.runtime_services import InternalApiRuntimeService

        service = InternalApiRuntimeService(fake_kb, fake_generator, fake_queue)

        with patch.object(self.app_module.auth_flow, "is_authenticated", return_value=True), \
             patch.object(self.app_module.auth_flow, "touch_active_auth_session", return_value=True), \
             patch.object(self.app_module, "internal_api_service", service), \
             patch("main.runtime_services.FirmAPIClient") as firm_client, \
             patch("main.runtime_services.InternalDataClient") as internal_client:
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
        self.assertEqual(save_button.text.strip(), "Simpan & Aktifkan Sumber Data")
        self.assertIsNotNone(refresh_button)
        self.assertIn("Muat Ulang Data Sekarang", refresh_button.text)
        self.assertIsNone(soup.select_one("#settings-portfolio"))
        self.assertIsNone(soup.select_one("#settings-credentials"))
        self.assertIsNone(soup.select_one("#btn-save-settings"))
        self.assertIsNotNone(soup.select_one("#settings-portfolio-files"))
        self.assertIsNotNone(soup.select_one("#settings-credential-files"))
        self.assertIn("Sumber portofolio tidak diisi manual", soup.get_text(" "))
        self.assertIn("Sumber kapabilitas, sertifikasi, dan tenaga ahli tidak diisi manual", soup.get_text(" "))
        self.assertIn("Pengguna proposal cukup memakai bukti yang sudah diringkas otomatis", soup.get_text(" "))
        self.assertIsNotNone(soup.select_one("#portfolio-internal-evidence"))
        self.assertIsNotNone(soup.select_one("#credential-internal-evidence"))
        self.assertNotIn("btnSaveSettings", script)
        self.assertNotIn("settingsPortfolio:", script)
        self.assertNotIn("settingsCredentials:", script)
        self.assertIn("setInternalApiConnectionState", script)
        self.assertIn("api_connection_active", script)
        self.assertIn("/api/internal-api/refresh", script)
        self.assertNotIn("source-status-grid", template)
        self.assertNotIn("readinessResources", script)
        self.assertIn("readiness: data.readiness", script)
        self.assertIn("renderFrameworkOptions(config)", script)
        self.assertIn("config.framework_options", script)
        self.assertIn("data-framework-versions", script)
        self.assertIn("framework-version-select", script)
        self.assertIn("framework-option-name", script)
        self.assertIsNotNone(soup.select_one("#select-perusahaan[type='text']"))
        self.assertIsNotNone(soup.select_one("#group-framework.framework-options"))
        self.assertIn("applyKakSuggestions(latestKakSuggestions, true)", script)
        self.assertIn("field yang masih kosong", script)
        self.assertNotIn("Sumber data proyek", script)
        self.assertNotIn("data.config_file", script)
        self.assertNotIn("missing_required_mapping", script)
        self.assertNotIn("mapped_fields", script)

    def test_internal_api_setup_treats_environment_resource_config_as_active(self) -> None:
        client = self.app.test_client()
        fake_kb = Mock()
        fake_kb.project_data_source = "api"
        fake_kb.sync_in_progress = False
        fake_kb.last_refresh_error = ""
        from main.runtime_services import InternalApiRuntimeService

        service = InternalApiRuntimeService(fake_kb, Mock(), Mock())

        with patch.object(self.app_module.auth_flow, "is_authenticated", return_value=True), \
             patch.object(self.app_module.auth_flow, "touch_active_auth_session", return_value=True), \
             patch.object(self.app_module, "internal_api_service", service), \
             patch("main.runtime_services.FirmAPIClient") as firm_client:
            firm_client.return_value.config_file = ""
            firm_client.return_value.resource_config = {"project_records": {"request": {}}}
            firm_client.return_value.describe_runtime.return_value = {"provider": "api"}
            firm_client.return_value.validate_config.return_value = {"resources": {}}
            response = client.get("/api/internal-api/setup")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["api_connection_active"])
        self.assertTrue(payload["can_refresh_dataset"])
        self.assertIn("readiness", payload)

    def test_internal_api_activation_payload_shape_is_stable(self) -> None:
        from main.internal_api_runtime import internal_api_activation_payload

        fake_kb = Mock()
        fake_kb.project_data_source = "api"
        fake_kb.sync_in_progress = True
        fake_kb.last_refresh_error = ""

        payload = internal_api_activation_payload(
            Path("/tmp/internal_api_config.json"),
            activated=True,
            refresh_started=True,
            knowledge_base=fake_kb,
            validation={"ok": True},
        )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["config_file"], "/tmp/internal_api_config.json")
        self.assertTrue(payload["api_connection_active"])
        self.assertTrue(payload["can_refresh_dataset"])
        self.assertEqual(payload["connection_label"], "Aktif memakai Internal API/APIDog")
        self.assertIn("readiness", payload)
        self.assertEqual(payload["validation"], {"ok": True})

    def test_internal_api_readiness_hides_mapping_details_behind_labels(self) -> None:
        from main.internal_api_runtime import internal_api_readiness

        payload = internal_api_readiness(
            {
                "resources": {
                    "account_records": {
                        "ok": True,
                        "mapped_fields": ["company_name"],
                        "missing_required_mapping": [],
                    },
                    "project_records": {
                        "ok": False,
                        "mapped_fields": ["project_name"],
                        "missing_required_mapping": ["topic"],
                    },
                }
            }
        )

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["resources"][0]["label"], "Daftar klien")
        self.assertEqual(payload["resources"][0]["status_label"], "Siap otomatis")
        self.assertEqual(payload["resources"][1]["label"], "Riwayat proyek & tenaga ahli")
        self.assertEqual(payload["resources"][1]["missing_required_count"], 1)
        self.assertNotIn("mapped_fields", payload["resources"][0])
        self.assertNotIn("missing_required_mapping", payload["resources"][1])

    def test_internal_api_refresh_payload_shape_is_stable(self) -> None:
        from main.internal_api_runtime import internal_api_refresh_payload

        fake_kb = Mock()
        fake_kb.project_data_source = "api"
        fake_kb.sync_in_progress = False
        fake_kb.last_refresh_error = ""

        payload = internal_api_refresh_payload("/tmp/internal_api_config.json", False, fake_kb)

        self.assertEqual(payload["status"], "current")
        self.assertFalse(payload["refresh_started"])
        self.assertTrue(payload["api_connection_active"])
        self.assertTrue(payload["can_refresh_dataset"])
        self.assertEqual(payload["config_file"], "/tmp/internal_api_config.json")

    def test_firm_api_client_caches_dataset_resource_fetches_per_instance(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.demo_mode = False
        client.resource_config = {
            "project_records": {
                "request": {"body": {"dataset": "ConsultantProjectExpertHistory"}},
                "response_path": "data.dataset_result",
                "field_mapping": {
                    "project_name": "project_name",
                    "product_name": "product_name",
                },
            }
        }
        client.request_defaults = {}
        client._resource_record_cache = {}
        client._request_from_spec = Mock(
            return_value={
                "data": {
                    "dataset_result": [
                        {
                            "project_name": "Penyusunan Arsitektur SPBE",
                            "product_name": "Arsitektur SPBE",
                        }
                    ]
                }
            }
        )

        first = client._get_mapped_resource_records("project_records", apply_filters=False)
        second = client._get_mapped_resource_records("project_records", apply_filters=False)

        self.assertEqual(first, second)
        client._request_from_spec.assert_called_once()


if __name__ == "__main__":
    unittest.main()
