"""Regression tests for signup approval behavior."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SignupApprovalTest(unittest.TestCase):
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
                "AUTH_REQUIRE_SIGNUP_APPROVAL": "true",
            }
        )
        from main.app import app, app_state_store

        cls.app = app
        cls.store = app_state_store

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_signup_creates_pending_user_that_cannot_login_until_approved(self) -> None:
        client = self.app.test_client()

        signup_response = client.post(
            "/signup",
            data={
                "username": "pending.user",
                "password": "secret123",
                "confirm_password": "secret123",
            },
            follow_redirects=False,
        )

        self.assertEqual(signup_response.status_code, 302)
        signup_query = parse_qs(urlparse(signup_response.headers["Location"]).query)
        self.assertIn("menunggu konfirmasi admin", signup_query.get("signup_success", [""])[0].lower())
        user = self.store.get_user("pending.user")
        self.assertIsNotNone(user)
        self.assertEqual(user["status"], "pending")

        blocked_login = client.post(
            "/login",
            data={"username": "pending.user", "password": "secret123"},
            follow_redirects=False,
        )

        self.assertEqual(blocked_login.status_code, 302)
        blocked_query = parse_qs(urlparse(blocked_login.headers["Location"]).query)
        self.assertIn("belum dikonfirmasi", blocked_query.get("login_error", [""])[0].lower())

        self.assertTrue(self.store.approve_user("pending.user", approved_by="admin"))

        allowed_login = client.post(
            "/login",
            data={"username": "pending.user", "password": "secret123"},
            follow_redirects=False,
        )

        self.assertEqual(allowed_login.status_code, 302)
        self.assertEqual(urlparse(allowed_login.headers["Location"]).path, "/")
        self.assertEqual(self.store.get_user("pending.user")["status"], "approved")


if __name__ == "__main__":
    unittest.main()
