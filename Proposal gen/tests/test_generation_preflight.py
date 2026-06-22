"""Focused regression coverage for serialized Internal API preflight reads."""

from __future__ import annotations

import concurrent.futures
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.generation_preflight import internal_api_serial_access, load_internal_preflight


class TrackingFirmAPI:
    def __init__(self) -> None:
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def _record(self, name, result):
        with self.lock:
            self.calls.append(name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return result

    def get_expert_bench_context(self, limit_products=8):
        return self._record("expert", {"available": True, "limit": limit_products})

    def get_framework_catalog(self):
        return self._record("framework", [{"name": "COBIT"}])

    def get_delivery_guidance(self, project_type):
        return self._record("delivery", {"project_type": project_type})

    def get_firm_profile(self):
        return self._record("profile", {"name": "Inixindo"})

    def get_client_relationship(self, client_name):
        return self._record("relationship", {"client": client_name, "mode": "new"})


class GenerationPreflightTests(unittest.TestCase):
    def test_internal_reads_are_serial_and_preserve_order_and_shape(self) -> None:
        api = TrackingFirmAPI()

        snapshot = load_internal_preflight(
            api,
            project_type="Advisory",
            base_client="Contoso",
        )

        self.assertEqual(api.calls, ["expert", "framework", "delivery", "profile", "relationship"])
        self.assertEqual(api.max_active, 1)
        self.assertEqual(snapshot.expert_bench_context, {"available": True, "limit": 8})
        self.assertEqual(snapshot.framework_context, [{"name": "COBIT"}])
        self.assertEqual(snapshot.firm_data, {"project_type": "Advisory"})
        self.assertEqual(snapshot.firm_profile, {"name": "Inixindo"})
        self.assertEqual(snapshot.relationship_context, {"client": "Contoso", "mode": "new"})
        with self.assertRaises(FrozenInstanceError):
            snapshot.firm_data = {}

    def test_concurrent_preflights_do_not_overlap_shared_internal_api(self) -> None:
        api = TrackingFirmAPI()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    load_internal_preflight,
                    api,
                    project_type="Advisory",
                    base_client=f"Client {index}",
                    expert_bench_context={"available": False},
                    expert_bench_resolved=True,
                )
                for index in range(2)
            ]
            [future.result(timeout=2) for future in futures]

        self.assertEqual(api.max_active, 1)

    def test_existing_available_context_skips_redundant_reads(self) -> None:
        api = TrackingFirmAPI()
        expert = {"available": True, "source": "request"}
        frameworks = [{"name": "ISO 27001"}]

        snapshot = load_internal_preflight(
            api,
            project_type="Assessment",
            base_client="Fabrikam",
            expert_bench_context=expert,
            framework_context=frameworks,
        )

        self.assertEqual(api.calls, ["delivery", "profile", "relationship"])
        self.assertIs(snapshot.expert_bench_context, expert)
        self.assertIs(snapshot.framework_context, frameworks)

    def test_owner_resolved_unavailable_expert_is_not_fetched_twice(self) -> None:
        api = TrackingFirmAPI()
        expert = {"available": False}

        snapshot = load_internal_preflight(
            api,
            project_type="Assessment",
            base_client="Fabrikam",
            expert_bench_context=expert,
            expert_bench_resolved=True,
        )

        self.assertEqual(api.calls, ["framework", "delivery", "profile", "relationship"])
        self.assertIs(snapshot.expert_bench_context, expert)

    def test_expert_and_framework_failures_retain_non_fatal_fallbacks(self) -> None:
        class FailingOptionalAPI(TrackingFirmAPI):
            def get_expert_bench_context(self, limit_products=8):
                self.calls.append("expert")
                raise RuntimeError("expert unavailable")

            def get_framework_catalog(self):
                self.calls.append("framework")
                raise RuntimeError("framework unavailable")

        api = FailingOptionalAPI()
        with self.assertLogs("main.generation_preflight", level="ERROR"):
            snapshot = load_internal_preflight(api, project_type="Audit", base_client="Northwind")

        self.assertIsNone(snapshot.expert_bench_context)
        self.assertIsNone(snapshot.framework_context)
        self.assertEqual(api.calls, ["expert", "framework", "delivery", "profile", "relationship"])

    def test_required_delivery_failure_still_propagates(self) -> None:
        class FailingDeliveryAPI(TrackingFirmAPI):
            def get_delivery_guidance(self, project_type):
                raise RuntimeError("delivery unavailable")

        with self.assertRaisesRegex(RuntimeError, "delivery unavailable"):
            load_internal_preflight(
                FailingDeliveryAPI(),
                project_type="Audit",
                base_client="Northwind",
            )

    def test_internal_lane_can_overlap_independent_work(self) -> None:
        independent_started = threading.Event()
        internal_started = threading.Event()
        release = threading.Event()

        class BlockingAPI(TrackingFirmAPI):
            def get_framework_catalog(self):
                internal_started.set()
                self.assert_overlap()
                return [{"name": "COBIT"}]

            @staticmethod
            def assert_overlap():
                if not independent_started.wait(timeout=1):
                    raise AssertionError("independent work did not overlap the Internal API lane")
                if not release.wait(timeout=1):
                    raise AssertionError("overlap test was not released")

        api = BlockingAPI()

        def independent_work():
            independent_started.set()
            self.assertTrue(internal_started.wait(timeout=1))
            release.set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            internal_future = pool.submit(
                load_internal_preflight,
                api,
                project_type="Audit",
                base_client="Northwind",
                expert_bench_context={"available": False},
                expert_bench_resolved=True,
            )
            independent_future = pool.submit(independent_work)
            snapshot = internal_future.result(timeout=2)
            independent_future.result(timeout=2)

        self.assertEqual(snapshot.relationship_context["client"], "Northwind")

    def test_engine_runs_research_on_owner_thread_and_merges_after_join(self) -> None:
        source = (ROOT / "main" / "proposal_engine.py").read_text(encoding="utf-8")

        preflight_timer = source.index("preflight_started = time.monotonic()")
        expert_fetch = source.index("expert_bench_context = self.firm_api.get_expert_bench_context")
        ai_context = source.index('ai_context = " ".join')
        logo_submit = source.index("logo_future = self.io_pool.submit")
        internal_submit = source.index("internal_preflight_future = self.io_pool.submit")
        research_call = source.index("research_bundle = self._get_research_bundle")
        internal_join = source.index("internal_preflight = internal_preflight_future.result()")
        technique = source.index("proposal_technique_contract = build_proposal_technique_contract")

        self.assertLess(preflight_timer, expert_fetch)
        self.assertIn("with internal_api_serial_access():", source)
        self.assertLess(expert_fetch, ai_context)
        self.assertLess(ai_context, logo_submit)
        self.assertLess(logo_submit, research_call)
        self.assertLess(internal_submit, research_call)
        self.assertLess(research_call, internal_join)
        self.assertLess(internal_join, technique)
        self.assertNotIn("submit(self._get_research_bundle", source)
        self.assertIn('"performance": performance_metadata', source)


if __name__ == "__main__":
    unittest.main()
