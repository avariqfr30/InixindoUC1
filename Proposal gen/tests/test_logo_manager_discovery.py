"""Focused coverage for LogoManager discovery and fallback behavior."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LogoManagerDiscoveryTest(unittest.TestCase):
    def test_official_domain_discovery_blocks_non_official_hosts_and_preserves_order(self) -> None:
        from main.document_rendering import LogoManager

        serper_response = Mock()
        serper_response.json.return_value = {
            "organic": [
                {
                    "title": "Example Bank official LinkedIn",
                    "snippet": "Official company page",
                    "link": "https://www.linkedin.com/company/example-bank",
                },
                {
                    "title": "Example Bank official website",
                    "snippet": "Official portal",
                    "link": "https://www.examplebank.co.id/about",
                },
                {
                    "title": "Example Bank official Facebook",
                    "snippet": "Official social channel",
                    "link": "https://facebook.com/examplebank",
                },
                {
                    "title": "Example Bank official investor site",
                    "snippet": "Example Bank investor relations",
                    "link": "https://investor.examplebank.co.id/profile",
                },
                {
                    "title": "Duplicate Example Bank official website",
                    "snippet": "Same host with www prefix",
                    "link": "https://www.examplebank.co.id/contact",
                },
                {
                    "title": "Example Bank official Wikipedia",
                    "snippet": "Reference page",
                    "link": "https://id.wikipedia.org/wiki/Example_Bank",
                },
            ]
        }

        with patch("main.document_rendering.Researcher._has_serper_key", return_value=True), patch(
            "main.document_rendering.requests.post",
            return_value=serper_response,
        ):
            candidates = LogoManager._official_domain_candidates("Example Bank")

        self.assertEqual(candidates, ["examplebank.co.id", "investor.examplebank.co.id"])

    def test_blocked_host_matching_does_not_reject_unrelated_domains_with_same_text(self) -> None:
        from main.document_rendering import LogoManager

        self.assertFalse(LogoManager._host_allowed_for_logo("wikipedia.org"))
        self.assertFalse(LogoManager._host_allowed_for_logo("id.wikipedia.org"))
        self.assertFalse(LogoManager._host_allowed_for_logo("www.linkedin.com"))
        self.assertTrue(LogoManager._host_allowed_for_logo("official-wikipedia.org.examplebank.co.id"))
        self.assertTrue(LogoManager._host_allowed_for_logo("linkedin.examplebank.co.id"))

    def test_get_logo_uses_official_domain_candidates_before_image_search(self) -> None:
        from main.document_rendering import LogoManager

        official_stream = io.BytesIO(b"official-logo")

        with patch(
            "main.document_rendering.LogoManager._official_domain_candidates",
            return_value=["stale.examplebank.co.id", "examplebank.co.id"],
        ), patch(
            "main.document_rendering.LogoManager._fetch_logo_from_domain",
            side_effect=[None, (0.01, official_stream, (10, 20, 30))],
        ) as fetch_logo, patch(
            "main.document_rendering.Researcher._has_serper_key",
            return_value=True,
        ), patch("main.document_rendering.requests.post") as post:
            logo_stream, color = LogoManager.get_logo_and_color("Example Bank")

        self.assertIs(logo_stream, official_stream)
        self.assertEqual(color, (10, 20, 30))
        self.assertEqual(
            [call.args[0] for call in fetch_logo.call_args_list],
            ["stale.examplebank.co.id", "examplebank.co.id"],
        )
        post.assert_not_called()

    def test_get_logo_continues_when_official_domain_fetch_fails(self) -> None:
        from main.document_rendering import DEFAULT_COLOR, LogoManager

        image_response = Mock()
        image_response.json.return_value = {"images": []}

        with patch(
            "main.document_rendering.LogoManager._official_domain_candidates",
            return_value=["broken.examplebank.co.id"],
        ), patch(
            "main.document_rendering.LogoManager._fetch_logo_from_domain",
            side_effect=RuntimeError("official site unavailable"),
        ), patch(
            "main.document_rendering.Researcher._has_serper_key",
            return_value=True,
        ), patch(
            "main.document_rendering.requests.post",
            return_value=image_response,
        ) as post:
            logo_stream, color = LogoManager.get_logo_and_color("Example Bank")

        self.assertIsNotNone(logo_stream)
        self.assertEqual(color, DEFAULT_COLOR)
        self.assertIn("/images", post.call_args.args[0])

    def test_image_search_fallback_skips_blocked_hosts_before_fetching_image(self) -> None:
        from main.document_rendering import DEFAULT_COLOR, LogoManager

        image_response = Mock()
        image_response.json.return_value = {
            "images": [
                {
                    "title": "Example Bank official logo",
                    "source": "facebook.com",
                    "link": "https://facebook.com/examplebank/logo",
                    "imageUrl": "https://facebook.com/examplebank/logo.png",
                }
            ]
        }

        with patch(
            "main.document_rendering.LogoManager._official_domain_candidates",
            return_value=[],
        ), patch(
            "main.document_rendering.Researcher._has_serper_key",
            return_value=True,
        ), patch(
            "main.document_rendering.requests.post",
            return_value=image_response,
        ), patch("main.document_rendering.requests.get") as get:
            logo_stream, color = LogoManager.get_logo_and_color("Example Bank")

        self.assertIsNotNone(logo_stream)
        self.assertEqual(color, DEFAULT_COLOR)
        get.assert_not_called()

    def test_get_logo_falls_back_safely_when_official_domain_discovery_fails(self) -> None:
        from main.document_rendering import DEFAULT_COLOR, LogoManager

        with patch(
            "main.document_rendering.LogoManager._official_domain_candidates",
            side_effect=RuntimeError("serper search unavailable"),
        ), patch(
            "main.document_rendering.Researcher._has_serper_key",
            return_value=False,
        ):
            logo_stream, color = LogoManager.get_logo_and_color("Example Bank")

        self.assertIsNotNone(logo_stream)
        self.assertEqual(color, DEFAULT_COLOR)


if __name__ == "__main__":
    unittest.main()
