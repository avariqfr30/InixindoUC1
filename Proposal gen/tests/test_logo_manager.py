"""Focused regression coverage for LogoManager lookup ordering."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.document_rendering import DEFAULT_COLOR, LogoManager


class _JsonResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class LogoManagerTest(unittest.TestCase):
    def test_blocked_hosts_are_excluded_from_official_domain_candidates(self) -> None:
        search_payload = {
            "organic": [
                {
                    "title": "Contoso official profile",
                    "link": "https://wikipedia.org/wiki/Contoso",
                    "snippet": "Official-looking encyclopedia result.",
                },
                {
                    "title": "Contoso Indonesia official",
                    "link": "https://www.facebook.com/contoso",
                    "snippet": "Official social profile.",
                },
                {
                    "title": "Contoso official website",
                    "link": "https://www.contoso.co.id/about",
                    "snippet": "Contoso services and company profile.",
                },
            ]
        }

        with patch("main.document_rendering.Researcher._has_serper_key", return_value=True), patch(
            "main.document_rendering.requests.post",
            return_value=_JsonResponse(search_payload),
        ) as post:
            candidates = LogoManager._official_domain_candidates("PT Contoso Indonesia")

        post.assert_called_once()
        self.assertEqual(candidates, ["contoso.co.id"])
        self.assertNotIn("wikipedia.org", candidates)
        self.assertNotIn("facebook.com", candidates)

    def test_get_logo_prefers_official_domain_fetch_before_serper_image_fallback(self) -> None:
        official_stream = io.BytesIO(b"official-logo")
        call_order = []

        def fetch_from_domain(domain: str, client_name: str):
            call_order.append(("official", domain, client_name))
            return 0.01, official_stream, (12, 34, 56)

        def unexpected_serper_image_call(*args, **kwargs):
            call_order.append(("serper-image", args, kwargs))
            raise AssertionError("Serper image fallback should not run after official logo fetch succeeds")

        with patch("main.document_rendering.Researcher._has_serper_key", return_value=True), patch(
            "main.document_rendering.LogoManager._official_domain_candidates",
            return_value=["contoso.co.id"],
        ), patch(
            "main.document_rendering.LogoManager._fetch_logo_from_domain",
            side_effect=fetch_from_domain,
        ), patch(
            "main.document_rendering.requests.post",
            side_effect=unexpected_serper_image_call,
        ):
            logo_stream, color = LogoManager.get_logo_and_color("PT Contoso Indonesia")

        self.assertIs(logo_stream, official_stream)
        self.assertEqual(color, (12, 34, 56))
        self.assertEqual(call_order, [("official", "contoso.co.id", "PT Contoso Indonesia")])

    def test_get_logo_safely_uses_fallback_when_official_lookup_fails(self) -> None:
        fallback_stream = io.BytesIO(b"fallback-logo")

        with patch("main.document_rendering.Researcher._has_serper_key", return_value=True), patch(
            "main.document_rendering.LogoManager._official_domain_candidates",
            return_value=["contoso.co.id"],
        ), patch(
            "main.document_rendering.LogoManager._fetch_logo_from_domain",
            return_value=None,
        ) as fetch, patch(
            "main.document_rendering.requests.post",
            return_value=_JsonResponse({"images": []}),
        ) as post, patch(
            "main.document_rendering.LogoManager._create_fallback_logo",
            return_value=fallback_stream,
        ) as create_fallback:
            logo_stream, color = LogoManager.get_logo_and_color("PT Contoso Indonesia")

        fetch.assert_called_once_with("contoso.co.id", "PT Contoso Indonesia")
        post.assert_called_once()
        self.assertIn("google.serper.dev/images", post.call_args.args[0])
        create_fallback.assert_called_once_with("PT Contoso Indonesia")
        self.assertIs(logo_stream, fallback_stream)
        self.assertEqual(color, DEFAULT_COLOR)


if __name__ == "__main__":
    unittest.main()
