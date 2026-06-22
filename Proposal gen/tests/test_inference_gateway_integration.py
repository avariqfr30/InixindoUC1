"""Integration coverage for process-wide proposal inference routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RecordingGateway:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class InferenceGatewayIntegrationTests(unittest.TestCase):
    def test_generator_and_research_use_the_same_gateway_without_changing_chat_contract(self):
        from main.proposal_generator import ProposalGenerator
        from main.research import Researcher
        from main.config import LLM_MODEL

        gateway = RecordingGateway(
            {"message": {"content": '{"insight": "Kebutuhan transformasi terverifikasi."}'}}
        )

        with patch("main.proposal_generator.get_inference_gateway", return_value=gateway), patch(
            "main.proposal_generator.InternalDataClient"
        ), patch("main.research.get_inference_gateway", return_value=gateway), patch.object(
            Researcher, "fetch_full_markdown", return_value="Source material"
        ):
            generator = ProposalGenerator(object())
            try:
                insight = Researcher.extract_insight_with_llm.__wrapped__(
                    "https://example.com/source",
                    "the transformation need",
                )
            finally:
                generator.io_pool.shutdown(wait=True)

        self.assertIs(generator.ollama, gateway)
        self.assertEqual(insight, "Kebutuhan transformasi terverifikasi.")
        self.assertEqual(len(gateway.calls), 1)
        args, kwargs = gateway.calls[0]
        self.assertEqual(args, ())
        self.assertEqual(kwargs["model"], LLM_MODEL)
        self.assertEqual(kwargs["options"], {"temperature": 0.0})
        self.assertEqual(kwargs["messages"][0]["role"], "user")

    def test_runtime_profile_extraction_uses_gateway_and_preserves_response_parsing(self):
        from main.runtime_components import FirmAPIClient
        from main.research import Researcher

        gateway = RecordingGateway(
            {
                "message": {
                    "content": (
                        '{"office_address": "Jl. Contoh 1", '
                        '"email": "hello@example.com", "phone": "+62 21 123"}'
                    )
                }
            }
        )
        client = FirmAPIClient.__new__(FirmAPIClient)

        with patch("main.runtime_components.get_inference_gateway", return_value=gateway), patch.object(
            Researcher,
            "search",
            return_value=[{"link": "https://example.com/contact", "title": "Contact"}],
        ), patch.object(
            Researcher,
            "_filter_recent_entity_results",
            side_effect=lambda hits, **_kwargs: hits,
        ), patch.object(
            Researcher, "fetch_full_markdown", return_value="Official contact details"
        ), patch.object(
            FirmAPIClient, "_normalize_firm_profile", side_effect=lambda payload: payload
        ):
            profile = client._build_profile_from_osint()

        self.assertEqual(profile["office_address"], "Jl. Contoh 1")
        self.assertEqual(profile["email"], "hello@example.com")
        self.assertEqual(profile["phone"], "+62 21 123")
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0][1]["options"], {"temperature": 0.0})


if __name__ == "__main__":
    unittest.main()
