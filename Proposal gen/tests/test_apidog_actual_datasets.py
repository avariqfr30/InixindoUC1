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
        self.assertNotIn("ConsultantProjectExpertHistory", relationship["summary"])
        self.assertNotIn("Data internal", relationship["summary"])

    def test_project_records_map_consultant_history_into_knowledge_base_rows(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.demo_mode = False
        client.resource_config = {
            "project_records": {
                "request": {"body": {"dataset": "ConsultantProjectExpertHistory"}},
                "response_path": "data.dataset_result",
                "record_filters": {"company_name__icontains": "{client_name}"},
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

    def test_reference_account_supplies_clean_client_names_not_project_titles(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.resource_config = {
            "account_records": {
                "request": {"body": {"dataset": "ReferenceAccount"}},
                "response_path": "data.dataset_result",
                "record_filters": {"company_name__icontains": "{client_name}"},
                "field_mapping": {
                    "company_name": "company_name",
                    "company_region_name": "company_region_name",
                    "company_province_name": "company_province_name",
                    "company_segment": "company_segment",
                    "company_sub_segment": "company_sub_segment",
                },
            }
        }
        client._request_from_spec = lambda spec, **context: {
            "data": {
                "dataset_result": [
                    {"company_name": "BPRS Dinar Ashri NTB", "company_region_name": "Mataram"},
                    {"company_name": "PI07 - Penyusunan Roadmap SOC & NOC BPRS Dinar Ashri NTB"},
                    {"company_name": "PT Sumberdaya Andalan Mandiri", "company_region_name": "Jakarta"},
                    {"company_name": "BPRS Dinar Ashri NTB", "company_region_name": "Mataram"},
                ]
            }
        }

        options = client.get_client_options()

        self.assertEqual(options, ["BPRS Dinar Ashri NTB", "PT Sumberdaya Andalan Mandiri"])

    def test_client_context_uses_consultant_history_for_use_cases_and_expert_fit(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client.resource_config = {
            "account_records": {
                "request": {"body": {"dataset": "ReferenceAccount"}},
                "response_path": "data.dataset_result",
                "field_mapping": {
                    "company_name": "company_name",
                    "company_region_name": "company_region_name",
                    "company_province_name": "company_province_name",
                    "company_segment": "company_segment",
                    "company_sub_segment": "company_sub_segment",
                    "company_category_name": "company_category_name",
                },
            },
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
            },
        }

        def request_from_spec(spec, **context):
            dataset = (((spec or {}).get("request") or {}).get("body") or {}).get("dataset")
            if dataset == "ReferenceAccount":
                return {
                    "data": {
                        "dataset_result": [
                            {
                                "company_name": "BPRS Dinar Ashri NTB",
                                "company_region_name": "Mataram",
                                "company_province_name": "Nusa Tenggara Barat",
                                "company_segment": "Banking",
                                "company_sub_segment": "BPRS",
                                "company_category_name": "Financial Services",
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
                        },
                        {
                            "project_name": "PX10 - IT Governance PT Lain",
                            "product_name": "IT Governance",
                            "expert_name": "Nama Lain",
                            "position_name": "Consultant",
                        },
                    ]
                }
            }

        client._request_from_spec = request_from_spec

        context = client.get_client_context("BPRS Dinar Ashri")

        self.assertTrue(context["available"])
        self.assertIn("BPRS Dinar Ashri NTB", context["client_name"])
        self.assertIn("Mataram", context["account_summary"])
        self.assertIn("Konteks akun internal", context["account_summary"])
        self.assertIn("bukan sebagai rumusan tujuan proyek", context["account_summary"])
        self.assertNotIn("ReferenceAccount", context["account_summary"])
        self.assertNotIn("Data internal", context["account_summary"])
        self.assertEqual(len(context["use_cases"]), 1)
        self.assertEqual(context["use_cases"][0]["product_name"], "Asesmen Jaringan")
        self.assertEqual(context["use_cases"][0]["expert_name"], "Dickyfli Perdana Putra")
        self.assertIn("Project Manager", context["expert_guidance"])

    def test_capability_context_uses_broader_consultant_history_without_client_match(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
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
                        "project_name": "PI06 - Kajian Arsitektur SPBE Domain Infrastruktur",
                        "product_name": "Arsitektur SPBE",
                        "expert_name": "Julizar Handi Wijaya",
                        "position_name": "Project Manager",
                    },
                    {
                        "project_name": "PI05 - Pendampingan ISO/IEC 27001:2022 Kota Semarang",
                        "product_name": "Pendampingan ISO 27001",
                        "expert_name": "Citra Arfanudin",
                        "position_name": "Project Manager",
                    },
                    {
                        "project_name": "PX99 - Pelatihan Umum",
                        "product_name": "Office Productivity",
                        "expert_name": "Nama Lain",
                        "position_name": "Trainer",
                    },
                ]
            }
        }

        context = client.get_capability_context(
            project_type="Strategic",
            service_type="Konsultan",
            focus_terms=["Ingin mengadopsi SPBE", "ISO", "Regulasi"],
            limit=3,
        )

        self.assertTrue(context["available"])
        self.assertGreaterEqual(len(context["matches"]), 2)
        self.assertIn("SPBE", context["summary"])
        self.assertIn("ISO 27001", context["summary"])
        self.assertIn("Julizar Handi Wijaya", context["expert_guidance"])
        self.assertIn("Citra Arfanudin", context["expert_guidance"])

    def test_consultant_history_formats_product_expert_position_matrix(self) -> None:
        from main.runtime_components import FirmAPIClient

        formatted = FirmAPIClient._format_project_expert_history(
            [
                {
                    "project_name": "PI06 - Kajian Arsitektur SPBE Domain Infrastruktur",
                    "product_name": "Arsitektur SPBE",
                    "expert_name": "Julizar Handi Wijaya",
                    "position_name": "Project Manager",
                },
                {
                    "project_name": "Penyusunan dokumen arsitektur SPBE Kab Gunung Kidul",
                    "product_name": "Arsitektur SPBE",
                    "expert_name": "Mustofa",
                    "position_name": "Tenaga Ahli",
                },
                {
                    "project_name": "PI05 - Pendampingan ISO/IEC 27001:2022 Kota Semarang",
                    "product_name": "Pendampingan ISO 27001",
                    "expert_name": "Citra Arfanudin",
                    "position_name": "Project Manager",
                },
            ]
        )

        self.assertTrue(formatted["available"])
        self.assertIn("Arsitektur SPBE", formatted["summary"])
        self.assertIn("Pendampingan ISO 27001", formatted["summary"])
        self.assertIn("Arsitektur SPBE: Project Manager - Julizar Handi Wijaya", formatted["expert_guidance"])
        self.assertIn("Tenaga Ahli - Mustofa", formatted["expert_guidance"])
        self.assertEqual(formatted["product_expert_matrix"][0]["product_name"], "Arsitektur SPBE")
        self.assertEqual(formatted["product_expert_matrix"][0]["positions"][0]["position_name"], "Project Manager")
        self.assertIn("Julizar Handi Wijaya", formatted["product_expert_matrix"][0]["positions"][0]["experts"])
        self.assertNotIn("ConsultantProjectExpertHistory", formatted["formatted_summary"])
        self.assertNotIn("PI06 -", formatted["formatted_summary"])

    def test_expert_bench_context_uses_full_consultant_history(self) -> None:
        from main.runtime_components import FirmAPIClient

        client = FirmAPIClient.__new__(FirmAPIClient)
        client._dedupe_records = FirmAPIClient._dedupe_records
        client._format_project_expert_history = FirmAPIClient._format_project_expert_history
        client.get_project_records = lambda: [
            {
                "entity": "Pendampingan ISO Kota",
                "topic": "Pendampingan ISO 27001",
                "expert_name": "Citra Arfanudin",
                "position_name": "Project Manager",
            },
            {
                "entity": "PI06 - Kajian Arsitektur SPBE Domain Infrastruktur",
                "topic": "Arsitektur SPBE",
                "expert_name": "Julizar Handi Wijaya",
                "position_name": "Project Manager",
            },
            {
                "entity": "Peta Rencana SPBE Kabupaten",
                "topic": "Peta Rencana SPBE",
                "expert_name": "Mustofa",
                "position_name": "Tenaga Ahli",
            },
            {
                "entity": "Peta Rencana SPBE Kota",
                "topic": "Peta Rencana SPBE",
                "expert_name": "Umar Affandhi",
                "position_name": "Project Manager",
            },
        ]

        context = client.get_expert_bench_context(limit_products=5)

        self.assertTrue(context["available"])
        self.assertEqual(context["record_count"], 4)
        self.assertEqual(context["product_expert_matrix"][0]["product_name"], "Peta Rencana SPBE")
        self.assertIn("Arsitektur SPBE", context["summary"])
        self.assertIn("Peta Rencana SPBE", context["summary"])
        self.assertIn("Pendampingan ISO 27001", context["summary"])


if __name__ == "__main__":
    unittest.main()
