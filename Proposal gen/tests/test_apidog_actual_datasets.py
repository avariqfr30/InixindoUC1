"""Regression tests for the actual Inworx/APIDog dataset catalog."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ApidogActualDatasetTest(unittest.TestCase):
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
            }
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_config_defaults_use_real_apidog_dataset_codes(self) -> None:
        from main.internal_api_setup import build_internal_api_config

        config = build_internal_api_config(
            {
                "url": "https://internal-api.example.com/api/Resource/dataset",
                "method": "POST",
                "body_encoding": "form",
                "auth_mode": "basic",
            }
        )

        resources = config["resources"]
        self.assertEqual(
            resources["project_records"]["request"]["body"]["dataset"],
            "ConsultantProjectExpertHistory",
        )
        self.assertEqual(resources["project_records"]["field_mapping"]["entity"], "project_name")
        self.assertEqual(resources["project_records"]["field_mapping"]["topic"], "product_name")
        self.assertEqual(
            resources["client_relationship"]["request"]["body"]["dataset"],
            "ConsultantProjectExpertHistory",
        )
        self.assertEqual(
            resources["client_relationship"]["record_filters"],
            {"project_name__icontains": "{client_name}"},
        )
        self.assertEqual(resources["account_records"]["request"]["body"]["dataset"], "ReferenceAccount")

    def test_resource_filters_support_case_insensitive_contains_for_real_project_names(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.request_defaults = {}
        client.resource_config = {
            "client_relationship": {
                "response_path": "data.dataset_result",
                "record_filters": {"project_name__icontains": "{client_name}"},
                "field_mapping": {"summary": "project_name"},
            }
        }
        client._request_from_spec = lambda spec, **context: {
            "data": {
                "dataset_result": [
                    {
                        "project_name": "PI07 - Penyusunan Roadmap SOC & NOC BPRS Dinar Ashri NTB",
                        "product_name": "Asesmen Jaringan",
                    },
                    {
                        "project_name": "PX01 - Audit Infrastruktur PT Contoh",
                        "product_name": "IT Audit",
                    },
                ]
            }
        }

        payload = client._resolve_resource_payload("client_relationship", client_name="bprs dinar ashri")

        self.assertEqual(payload["summary"], "PI07 - Penyusunan Roadmap SOC & NOC BPRS Dinar Ashri NTB")

    def test_client_relationship_is_derived_from_consultant_project_history(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.demo_mode = False
        client.data_acquisition_mode = "staged"
        client.request_defaults = {}
        client.resource_config = {
            "client_relationship": {
                "response_path": "data.dataset_result",
                "record_filters": {"project_name__icontains": "{client_name}"},
                "field_mapping": {
                    "summary": "project_name",
                    "project_name": "project_name",
                    "product_name": "product_name",
                    "expert_name": "expert_name",
                    "position_name": "position_name",
                },
            },
            "account_records": {
                "response_path": "data.dataset_result",
                "record_filters": {"company_name__icontains": "{client_name}"},
                "field_mapping": {
                    "company_name": "company_name",
                    "company_region_name": "company_region_name",
                    "company_province_name": "company_province_name",
                    "company_segment": "company_segment",
                },
            },
        }

        def request_from_spec(spec, **context):
            body = ((spec or {}).get("request") or {}).get("body") or {}
            dataset = body.get("dataset")
            if dataset == "ReferenceAccount":
                return {
                    "data": {
                        "dataset_result": [
                            {
                                "company_name": "BPRS Dinar Ashri NTB",
                                "company_region_name": "Mataram",
                                "company_province_name": "Nusa Tenggara Barat",
                                "company_segment": "SWASTA",
                            }
                        ]
                    }
                }
            return {
                "data": {
                    "dataset_result": [
                        {
                            "project_name": "PI07 - Penyusunan Roadmap SOC & NOC BPRS Dinar Ashri NTB",
                            "product_name": "Asesmen Jaringan",
                            "expert_name": "Dickyfli Perdana Putra",
                            "position_name": "Project Manager",
                        }
                    ]
                }
            }

        client._request_from_spec = request_from_spec

        relationship = client.get_client_relationship("BPRS Dinar Ashri")

        self.assertEqual(relationship["mode"], "existing")
        self.assertTrue(relationship["verified"])
        self.assertEqual(relationship["source"], "internal_api")
        self.assertIn("Penyusunan Roadmap SOC & NOC", relationship["summary"])
        self.assertIn("Asesmen Jaringan", relationship["summary"])

    def test_project_records_map_consultant_history_into_knowledge_base_rows(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.demo_mode = False
        client.resource_config = {
            "project_records": {
                "request": {"body": {"dataset": "ConsultantProjectExpertHistory"}},
                "response_path": "data.dataset_result",
                "field_mapping": {
                    "entity": "project_name",
                    "topic": "product_name",
                    "project_name": "project_name",
                    "expert_name": "expert_name",
                    "position_name": "position_name",
                },
            }
        }
        client._request_from_spec = lambda spec, **context: {
            "data": {
                "dataset_result": [
                    {
                        "project_name": "PI07 - Penyusunan Roadmap SOC & NOC BPRS Dinar Ashri NTB",
                        "product_name": "Asesmen Jaringan",
                        "expert_name": "Dickyfli Perdana Putra",
                        "position_name": "Project Manager",
                    }
                ]
            }
        }

        records = client.get_project_records()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["entity"], "PI07 - Penyusunan Roadmap SOC & NOC BPRS Dinar Ashri NTB")
        self.assertEqual(records[0]["topic"], "Asesmen Jaringan")
        self.assertEqual(records[0]["expert_name"], "Dickyfli Perdana Putra")


if __name__ == "__main__":
    unittest.main()
