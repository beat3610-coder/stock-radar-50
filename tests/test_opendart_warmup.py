import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from opendart import RunRequestBudget, max_requests_per_run, warmup_workers_from_env
from opendart_warmup import run_warmup
from universe import AUDIT_63_BENCHMARK


class OpenDartWarmupTests(unittest.TestCase):
    def test_workers_use_distinct_clients_and_share_corp_map_and_request_budget(self):
        tickers = list(AUDIT_63_BENCHMARK["KOSPI"] + AUDIT_63_BENCHMARK["KOSDAQ"])

        class FakeClient:
            instances = []
            lock = threading.Lock()
            map_calls = 0

            def __init__(self, cache_root=None, request_budget=None):
                self.cache_root = cache_root or Path("fake-cache")
                self.request_budget = request_budget
                self.key_present = True
                self._requests_n = 0
                with self.lock:
                    self.instances.append(self)

            def set_time_budget(self, _seconds):
                pass

            def set_request_budget(self, budget):
                self.request_budget = budget

            def get_corp_codes(self):
                type(self).map_calls += 1
                self.request_budget.acquire()
                self._requests_n += 1
                return {ticker: f"{int(ticker):08d}" for ticker in tickers}

            def company_financials(self, ticker, **_kwargs):
                self.assert_acquired = self.request_budget.acquire()
                if self.assert_acquired:
                    self._requests_n += 1
                return {"records": [{"rcept_date": "2025-03-11"}], "diagnostics": {
                    "status": "OK", "available": True, "attempted": True,
                    "requests_n": 1 if self.assert_acquired else 0,
                    "first_rcept_date": "2025-03-11", "last_rcept_date": "2025-03-11",
                    "failures": [], "cap_reached": not self.assert_acquired}}

        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            os.chdir(directory)
            try:
                result = run_warmup(time_budget_seconds=60, workers=4,
                    progress=lambda *_args, **_kwargs: None, client_factory=FakeClient)
                saved = json.loads(Path("data/opendart-warmup-latest.json").read_text())
            finally:
                os.chdir(old_cwd)

        self.assertEqual(FakeClient.map_calls, 1)
        self.assertEqual(len(FakeClient.instances), len(tickers) + 1)
        self.assertEqual(len({id(client) for client in FakeClient.instances}), len(tickers) + 1)
        self.assertEqual(result["available_symbols_n"], len(tickers))
        self.assertEqual(result["workers"], 4)
        self.assertEqual(result["requests_n"], len(tickers) + 1)
        self.assertEqual(saved["requests_n"], len(tickers) + 1)
        self.assertFalse(result["cap_reached"])

    def test_request_cap_stops_cleanly_and_records_diagnostics(self):
        class FakeClient:
            def __init__(self, cache_root=None, request_budget=None):
                self.cache_root = cache_root or Path("fake-cache")
                self.request_budget = request_budget
                self.key_present = True
                self._requests_n = 0

            def set_time_budget(self, _seconds):
                pass

            def get_corp_codes(self):
                if self.request_budget.acquire():
                    self._requests_n += 1
                return {}

        with tempfile.TemporaryDirectory() as directory:
            old_cwd = os.getcwd()
            os.chdir(directory)
            try:
                result = run_warmup(time_budget_seconds=60, workers=2,
                    progress=lambda *_args, **_kwargs: None, client_factory=FakeClient,
                    request_budget=RunRequestBudget(limit=0))
            finally:
                os.chdir(old_cwd)

        self.assertEqual(result["requests_n"], 0)
        self.assertTrue(result["cap_reached"])
        self.assertEqual(result["request_limit"], 0)
        self.assertEqual(result["attempted_symbols_n"], 0)
        self.assertEqual(result["available_symbols_n"], 0)
        self.assertEqual(len(result["missing_tickers"]), 60)

    def test_worker_environment_defaults_to_four_and_is_bounded(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(warmup_workers_from_env(), 4)
            self.assertEqual(max_requests_per_run(), 6000)
        with patch.dict(os.environ, {"OPENDART_WARMUP_WORKERS": "99"}):
            self.assertEqual(warmup_workers_from_env(), 8)

    def test_request_budget_is_shared_and_never_exceeds_limit_concurrently(self):
        budget = RunRequestBudget(limit=17)
        threads = [threading.Thread(target=budget.acquire) for _ in range(40)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(budget.requests_n, 17)
        self.assertTrue(budget.cap_reached)


if __name__ == "__main__":
    unittest.main()
