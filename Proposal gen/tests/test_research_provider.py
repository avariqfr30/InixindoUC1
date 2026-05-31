"""Focused coverage for configurable research providers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ResearchProviderTest(unittest.TestCase):
    def test_ollama_search_normalizes_results_for_existing_relevance_filters(self) -> None:
        from main.research import Researcher

        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "title": "Ollama result",
                    "url": "https://example.com/source",
                    "content": "Relevant snippet text.",
                }
            ]
        }

        with patch("main.research.SEARCH_PROVIDER", "ollama"), patch(
            "main.research.OLLAMA_API_KEY",
            "ollama-key",
        ), patch("main.research.osint_cache.get", return_value=None), patch(
            "main.research.osint_cache.set",
        ), patch("main.research.requests.post", return_value=response) as post:
            results = Researcher.search("proposal governance", limit=3, recency_bucket="month")

        self.assertEqual(
            results,
            [
                {
                    "title": "Ollama result",
                    "link": "https://example.com/source",
                    "snippet": "Relevant snippet text.",
                    "url": "https://example.com/source",
                    "content": "Relevant snippet text.",
                }
            ],
        )
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "https://ollama.com/api/web_search")
        self.assertEqual(post.call_args.kwargs["json"], {"query": "proposal governance", "max_results": 3})
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer ollama-key")

    def test_ollama_provider_without_key_returns_empty_search_and_fetch_without_requests(self) -> None:
        from main.research import Researcher

        with patch("main.research.SEARCH_PROVIDER", "ollama"), patch(
            "main.research.OLLAMA_API_KEY",
            "",
        ), patch("main.research.osint_cache.get", return_value=None), patch(
            "main.research.osint_cache.set",
        ), patch("main.research.requests.post") as post, patch("main.research.requests.get") as get:
            self.assertEqual(Researcher.search("missing key query"), [])
            self.assertEqual(Researcher.fetch_full_markdown("https://example.com/source"), "")

        post.assert_not_called()
        get.assert_not_called()

    def test_serper_missing_key_falls_back_to_ollama_when_configured(self) -> None:
        from main.research import Researcher

        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "title": "Fallback result",
                    "url": "https://example.com/fallback",
                    "content": "Fallback snippet.",
                }
            ]
        }

        with patch("main.research.SEARCH_PROVIDER", "serper"), patch(
            "main.research.SERPER_API_KEY",
            "SERPER_API",
        ), patch("main.research.OLLAMA_API_KEY", "ollama-key"), patch(
            "main.research.osint_cache.get",
            return_value=None,
        ), patch("main.research.osint_cache.set"), patch(
            "main.research.requests.post",
            return_value=response,
        ) as post:
            results = Researcher.search("backup search", limit=2)

        self.assertEqual(results[0]["title"], "Fallback result")
        self.assertEqual(post.call_args.args[0], "https://ollama.com/api/web_search")

    def test_serper_empty_results_fall_back_to_ollama_when_configured(self) -> None:
        from main.research import Researcher

        serper_response = Mock()
        serper_response.json.return_value = {"organic": []}
        ollama_response = Mock()
        ollama_response.json.return_value = {
            "results": [
                {
                    "title": "Ollama backup",
                    "url": "https://example.com/backup",
                    "content": "Backup snippet.",
                }
            ]
        }

        with patch("main.research.SEARCH_PROVIDER", "serper"), patch(
            "main.research.SERPER_API_KEY",
            "serper-key",
        ), patch("main.research.OLLAMA_API_KEY", "ollama-key"), patch(
            "main.research.osint_cache.get",
            return_value=None,
        ), patch("main.research.osint_cache.set"), patch(
            "main.research.requests.post",
            side_effect=[serper_response, ollama_response],
        ) as post:
            results = Researcher.search("backup search", limit=2)

        self.assertEqual(results[0]["title"], "Ollama backup")
        self.assertEqual(post.call_args_list[0].args[0], "https://google.serper.dev/search")
        self.assertEqual(post.call_args_list[1].args[0], "https://ollama.com/api/web_search")

    def test_ollama_fetch_uses_web_fetch_content(self) -> None:
        from main.research import Researcher

        response = Mock()
        response.json.return_value = {
            "title": "Fetched page",
            "content": "Fetched markdown content.",
            "links": ["https://example.com/next"],
        }

        with patch("main.research.SEARCH_PROVIDER", "ollama"), patch(
            "main.research.OLLAMA_API_KEY",
            "ollama-key",
        ), patch("main.research.osint_cache.get", return_value=None), patch(
            "main.research.osint_cache.set",
        ), patch("main.research.requests.post", return_value=response) as post:
            content = Researcher.fetch_full_markdown("https://example.com/source")

        self.assertEqual(content, "Fetched markdown content.")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "https://ollama.com/api/web_fetch")
        self.assertEqual(post.call_args.kwargs["json"], {"url": "https://example.com/source"})
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer ollama-key")


if __name__ == "__main__":
    unittest.main()
