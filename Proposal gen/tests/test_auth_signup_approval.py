"""Regression tests for internal-email signup access behavior."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SignupAccessTest(unittest.TestCase):
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
                "PROJECT_DATA_SOURCE": "local",
                "APP_PROFILE": "demo",
                "INTERNAL_DATA_SOURCE": "demo",
                "AUTH_ALLOW_SIGNUP": "true",
                "AUTH_SIGNUP_EMAIL_DOMAIN": "inixindojogja.co.id",
            }
        )
        os.environ.pop("AUTH_REQUIRE_SIGNUP_APPROVAL", None)
        from main import app as app_module

        app_module.AUTH_ALLOW_SIGNUP = True
        app_module.AUTH_SIGNUP_EMAIL_DOMAIN = "inixindojogja.co.id"
        app_module.AUTH_REQUIRE_SIGNUP_APPROVAL = False

        cls.app = app_module.app
        cls.store = app_module.app_state_store

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_internal_email_signup_can_login_immediately(self) -> None:
        client = self.app.test_client()
        username = f"pending.user.{uuid.uuid4().hex}@inixindojogja.co.id"

        signup_response = client.post(
            "/signup",
            data={
                "username": username,
                "password": "secret123",
                "confirm_password": "secret123",
            },
            follow_redirects=False,
        )

        self.assertEqual(signup_response.status_code, 302)
        signup_query = parse_qs(urlparse(signup_response.headers["Location"]).query)
        self.assertIn("silakan login", signup_query.get("signup_success", [""])[0].lower())
        user = self.store.get_user(username)
        self.assertIsNotNone(user)
        self.assertEqual(user["status"], "approved")

        allowed_login = client.post(
            "/login",
            data={"username": username, "password": "secret123"},
            follow_redirects=False,
        )

        self.assertEqual(allowed_login.status_code, 302)
        self.assertEqual(urlparse(allowed_login.headers["Location"]).path, "/")

    def test_signup_rejects_non_internal_email_domain(self) -> None:
        client = self.app.test_client()

        response = client.post(
            "/signup",
            data={
                "username": "rogue@example.com",
                "password": "secret123",
                "confirm_password": "secret123",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        query = parse_qs(urlparse(response.headers["Location"]).query)
        self.assertIn("@inixindojogja.co.id", query.get("signup_error", [""])[0].lower())
        self.assertIsNone(self.store.get_user("rogue@example.com"))

    def test_signup_page_has_client_side_domain_warning(self) -> None:
        client = self.app.test_client()

        response = client.get("/auth")

        self.assertEqual(response.status_code, 200)
        soup = BeautifulSoup(response.get_data(as_text=True), "html.parser")
        panel = soup.select_one(".auth-panel")
        warning = soup.select_one("#signup-domain-warning")
        script = soup.find_all("script")[-1].string or ""
        self.assertEqual(panel["data-signup-domain"], "inixindojogja.co.id")
        self.assertIsNotNone(warning)
        self.assertEqual(warning["role"], "alert")
        self.assertIn("hidden", warning.attrs)
        self.assertIn("signup-domain-warning", script)
        self.assertIn("preventDefault", script)


if __name__ == "__main__":
    unittest.main()
