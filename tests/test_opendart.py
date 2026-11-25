from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
import zipfile
from io import BytesIO
from unittest.mock import patch

from opendart import (EARNINGS_FEATURES, REPORT_CODES, REPORT_QUARTERS, OpenDartClient,
                      earnings_features_for_signal, earnings_gate_status, parse_corp_code_zip,
                      parse_financial_response, _quarter_standalones)
from slb import (SLB_FEATURES, SlbClient, isin_for_stock_code,
                 parse_slb_xml, slb_features_for_signal)


SLB_FIXTURE = b"""<?xml version='1.0' encoding='UTF-8'?>
<response><header><resultCode>00</resultCode><resultMsg>NORMAL SERVICE.</resultMsg></header>
<body><items>
<item><stdDt>20261120</stdDt><slbTrRemQty>900</slbTrRemQty><matcQty>90</matcQty><redQty>10</redQty></item>
<item><stdDt>20261121</stdDt><slbTrRemQty>1000</slbTrRemQty><matcQty>100</matcQty><redQty>0</redQty></item>
</items><totalCount>2</totalCount></body></response>"""


def financial_record(year, report, receipt, revenue, operating_income, net_income, fs="CFS"):
    return {"stock_code": "005930", "corp_code": "00126380",
            "bsns_year": year, "reprt_code": report, "fs_div": fs,
            "rcept_no": receipt.replace("-", "") + "000001", "rcept_date": receipt,
            "revenue": revenue, "operating_income": operating_income,
            "net_income": net_income, "currency": "KRW"}


class Response:
    def __init__(self, payload):
        self._payload = payload
        self.content = json.dumps(payload).encode() if isinstance(payload, dict) else payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class OpenDartTests(unittest.TestCase):
    def test_corp_code_zip_parser_maps_six_digit_stock_code(self):
        xml = b"<result><list><corp_code>00126380</corp_code><corp_name>Samsung</corp_name><stock_code>005930</stock_code></list></result>"
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("CORPCODE.xml", xml)
        self.assertEqual(parse_corp_code_zip(stream.getvalue()), {"005930": "00126380"})

    def test_is_rows_and_cis_operating_income_are_used_sce_duplicate_is_ignored(self):
        payload = {"status": "000", "list": [
            {"sj_div": "IS", "account_id": "ifrs-full_ProfitLoss", "account_nm": "당기순이익", "thstrm_amount": " 1,234 ", "rcept_no": "20250311000001", "currency": "KRW"},
            {"sj_div": "SCE", "account_id": "ifrs-full_ProfitLoss", "account_nm": "당기순이익", "thstrm_amount": "99,999", "rcept_no": "20250311000001"},
            {"sj_div": "CIS", "account_id": "other", "account_nm": "영업이익(손실)", "thstrm_amount": "567", "rcept_no": "20250311000001", "currency": "KRW"},
            {"sj_div": "BS", "account_nm": "매출액", "thstrm_amount": "8,000", "rcept_no": "20250311000001"},
        ]}
        status, records, _ = parse_financial_response(payload, stock_code="005930",
            corp_code="00126380", bsns_year=2024, reprt_code="11014", fs_div="CFS")
        self.assertEqual(status, "000")
        self.assertEqual(records[0]["net_income"], 1234.0)
        self.assertEqual(records[0]["operating_income"], 567.0)
        self.assertIsNone(records[0]["revenue"])
        self.assertEqual(records[0]["rcept_date"], "2025-03-11")

    def test_status_013_is_no_data_and_triggers_ofs_fallback(self):
        calls = []
        replies = [Response({"status": "013", "message": "no data"}), Response({
            "status": "000", "list": [
                {"sj_div": "CIS", "account_id": "ifrs-full_Revenue", "account_nm": "매출액",
                 "thstrm_amount": "1,000", "rcept_no": "20250311000001"},
                {"sj_div": "CIS", "account_id": "dart_OperatingIncomeLoss", "account_nm": "영업이익(손실)",
                 "thstrm_amount": "100", "rcept_no": "20250311000001"},
            ]})]

        def fake_get(url, *, params, timeout):
            calls.append(dict(params))
            return replies.pop(0)

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENDART_API_KEY": "test-secret"}):
            client = OpenDartClient(directory, http_get=fake_get, clock=lambda: 1.0, sleeper=lambda _: None)
            result = client.company_financials("005930", corp_code="00126380",
                start_year=2024, end_year=2024, reprt_codes=("11014",))
            self.assertEqual([call["fs_div"] for call in calls], ["CFS", "OFS"])
            self.assertEqual(result["records"][0]["fs_div"], "OFS")
            cache = json.loads((__import__("pathlib").Path(directory) / "financials" / "00126380.json").read_text())
        self.assertEqual(cache["request_status"]["2024:11014:CFS"]["status"], "013")
        self.assertEqual(cache["request_status"]["2024:11014:OFS"]["status"], "000")

    def test_keyless_dart_abstains_without_any_network_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch("opendart.requests.get") as get:
            client = OpenDartClient(directory)
            self.assertEqual(client.get_corp_codes(), {})
            result = client.company_financials("005930", corp_code="00126380",
                start_year=2025, end_year=2025, reprt_codes=("11014",))
            self.assertEqual(client._requests_n, 0)
            get.assert_not_called()
        features = earnings_features_for_signal(result["records"], "2025-03-12",
            corp_code="00126380", abstain=True)
        self.assertEqual(features["earnings_data_available"], 0)
        self.assertTrue(all(math.isnan(features[name]) for name in EARNINGS_FEATURES))

    def test_existing_corp_code_map_is_reused_without_redownload(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch("opendart.requests.get") as get:
            path = __import__("pathlib").Path(directory) / "corp-codes.json"
            path.write_text(json.dumps({"005930": "00126380"}), encoding="utf-8")
            client = OpenDartClient(directory)
            self.assertEqual(client.get_corp_codes(), {"005930": "00126380"})
            get.assert_not_called()

    def test_receipt_date_is_strictly_prior_and_saturday_report_is_available_monday(self):
        report = financial_record(2025, "11013", "2025-03-11", 100, 20, 10)
        same_day = earnings_features_for_signal([report], "2025-03-11", corp_code="00126380")
        next_day = earnings_features_for_signal([report], "2025-03-12", corp_code="00126380")
        self.assertEqual(same_day["earnings_data_available"], 0)
        self.assertTrue(all(math.isnan(same_day[name]) for name in EARNINGS_FEATURES))
        self.assertEqual(next_day["earnings_data_available"], 1)
        self.assertEqual(next_day["earnings_last_rcept_date"], "2025-03-11")
        self.assertEqual(next_day["earnings_days_since_rcept"], 1)
        saturday = financial_record(2025, "11013", "2025-03-15", 110, 25, 12)
        monday = earnings_features_for_signal([saturday], "2025-03-17", corp_code="00126380")
        self.assertEqual(monday["earnings_data_available"], 1)
        self.assertEqual(monday["earnings_last_rcept_date"], "2025-03-15")

    def test_q4_standalone_yoy_qoq_margin_and_momentum_values(self):
        records = [
            financial_record(2023, "11013", "2023-05-15", 80, -10, 2),
            financial_record(2023, "11012", "2023-08-01", 100, 2, 5),
            financial_record(2023, "11014", "2023-11-15", 90, 48, 8),
            financial_record(2023, "11011", "2024-03-01", 395, 100, 40),
            financial_record(2024, "11013", "2024-05-15", 150, -8, 5),
            financial_record(2024, "11012", "2024-08-01", 187.5, -3.2, 10),
            financial_record(2024, "11014", "2024-11-15", 112.5, 67.2, 15),
            financial_record(2024, "11011", "2025-03-10", 600, 140, 80),
        ]
        features = earnings_features_for_signal(records, "2025-03-12", corp_code="00126380")
        self.assertAlmostEqual(features["rev_yoy_pct"], 20.0)
        self.assertAlmostEqual(features["op_yoy_pct"], 40.0)
        self.assertAlmostEqual(features["ni_yoy_pct"], 100.0)
        self.assertAlmostEqual(features["rev_qoq_pct"], 100 / 3)
        self.assertAlmostEqual(features["op_qoq_pct"], 25.0)
        self.assertAlmostEqual(features["opm_pct"], 56.0)
        self.assertAlmostEqual(features["opm_yoy_delta_pp"], 8.0)
        self.assertAlmostEqual(features["earnings_momentum"], 40.0)

    def test_single_quarter_amounts_and_q4_annual_remainder(self):
        records = [
            financial_record(2024, "11013", "2024-05-15", 100, 10, 5),
            financial_record(2024, "11012", "2024-08-14", 110, 20, 10),
            financial_record(2024, "11014", "2024-11-14", 120, 30, 15),
            financial_record(2024, "11011", "2025-03-10", 480, 100, 50),
        ]
        quarterly_revenue = _quarter_standalones(records, "revenue")
        self.assertEqual(quarterly_revenue[(2024, 1)], 100)
        self.assertEqual(quarterly_revenue[(2024, 2)], 110)
        self.assertEqual(quarterly_revenue[(2024, 3)], 120)
        self.assertEqual(quarterly_revenue[(2024, 4)], 150)
        third_quarter = earnings_features_for_signal(records, "2024-11-17", corp_code="00126380")
        self.assertAlmostEqual(third_quarter["rev_qoq_pct"], (120 / 110 - 1) * 100)
        fourth_quarter = earnings_features_for_signal(records, "2025-03-12", corp_code="00126380")
        self.assertAlmostEqual(fourth_quarter["rev_qoq_pct"], (150 / 120 - 1) * 100)

    def test_report_code_mapping_and_november_11014_is_third_quarter(self):
        self.assertEqual(REPORT_QUARTERS, {"11013": 1, "11012": 2, "11014": 3, "11011": 4})
        self.assertEqual(REPORT_CODES, ("11013", "11012", "11014", "11011"))
        records = [
            financial_record(2025, "11013", "2025-05-15", 100, 10, 5),
            financial_record(2025, "11012", "2025-08-14", 220, 30, 15),
            financial_record(2025, "11014", "2025-11-14", 360, 60, 25),
        ]
        features = earnings_features_for_signal(records, "2025-11-17", corp_code="00126380")
        self.assertEqual(features["earnings_last_rcept_date"], "2025-11-14")
        self.assertAlmostEqual(features["rev_qoq_pct"], (360 / 220 - 1) * 100)

    def test_negative_prior_operating_income_turnaround_is_nan(self):
        records = [
            financial_record(2023, "11014", "2023-11-15", 10000, -1800, -100),
            financial_record(2024, "11014", "2024-11-14", 20000, 7000, 200),
        ]
        features = earnings_features_for_signal(records, "2024-11-18", corp_code="00126380")
        self.assertTrue(math.isnan(features["op_yoy_pct"]))
        self.assertTrue(math.isnan(features["ni_yoy_pct"]))

    def test_nonpositive_revenue_and_missing_denominators_are_nan(self):
        records = [financial_record(2024, "11014", "2025-03-10", 0, 20, 10),
                   financial_record(2023, "11014", "2024-03-10", 100, 10, 5)]
        features = earnings_features_for_signal(records, "2025-03-12", corp_code="00126380")
        self.assertTrue(math.isnan(features["rev_yoy_pct"]))
        self.assertTrue(math.isnan(features["opm_pct"]))
        missing_revenue_payload = {"status": "000", "list": [
            {"sj_div": "IS", "account_id": "ifrs-full_ProfitLoss", "account_nm": "당기순이익",
             "thstrm_amount": "1", "rcept_no": "20250311000001"},
        ]}
        _, parsed, _ = parse_financial_response(missing_revenue_payload, stock_code="005930",
            corp_code="00126380", bsns_year=2024, reprt_code="11014", fs_div="CFS")
        self.assertIsNone(parsed[0]["revenue"])

    def test_etf_without_corp_code_is_not_applicable_and_gate_abstains(self):
        features = earnings_features_for_signal([], "2025-03-12", corp_code=None)
        self.assertEqual(features["earnings_status"], "not_applicable")
        self.assertEqual(features["earnings_data_available"], 0)
        self.assertTrue(all(math.isnan(features[name]) for name in EARNINGS_FEATURES))
        self.assertEqual(earnings_gate_status(features, earnings_momentum_min=0), "ABSTAIN")
        self.assertEqual(earnings_gate_status({"earnings_data_available": 1, "earnings_momentum": -1},
            earnings_momentum_min=0), "REJECT")


class SlbTests(unittest.TestCase):
    def test_luhn_isins_match_verified_codes(self):
        self.assertEqual(isin_for_stock_code("005930"), "KR7005930003")
        self.assertEqual(isin_for_stock_code("000660"), "KR7000660001")

    def test_xml_fixture_parses_header_and_fields(self):
        result = parse_slb_xml(SLB_FIXTURE)
        self.assertEqual(result["result_code"], "00")
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["items"][1]["slbTrRemQty"], "1000")

    def test_keyless_slb_abstains_without_network_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch("slb.requests.get") as get:
            client = SlbClient(directory)
            dealing = client.get_dealing_for_ticker("005930")
            rank = client.get_stock_rank("20261121")
            self.assertEqual(client._requests_n, 0)
            get.assert_not_called()
        self.assertEqual(dealing["status"], "ABSTAIN_NO_API_KEY")
        self.assertEqual(rank["status"], "ABSTAIN_NO_API_KEY")

    def test_slb_features_require_exact_signal_date_and_keep_missing_as_nan(self):
        rows = parse_slb_xml(SLB_FIXTURE)["items"]
        outside = slb_features_for_signal("005930", "2026-11-24", dealing_records=rows)
        self.assertEqual(outside["slb_data_available"], 0)
        self.assertTrue(all(math.isnan(outside[name]) for name in SLB_FEATURES))
        current = slb_features_for_signal("005930", "2026-11-21", dealing_records=rows)
        self.assertEqual(current["slb_data_available"], 1)
        self.assertAlmostEqual(current["slb_rem_qty_change_1d_pct"], 100 / 9)
        self.assertAlmostEqual(current["slb_rem_qty_change_5d_pct"], 100 / 9)
        self.assertAlmostEqual(current["slb_matc_to_rem_pct"], 10.0)
        self.assertEqual(current["slb_history_days"], 2)

    def test_slb_one_day_change_uses_row_before_current_date_not_duplicate_current_row(self):
        rows = [
            {"stdDt": "20261120", "slbTrRemQty": "900", "matcQty": "90"},
            {"stdDt": "20261121", "slbTrRemQty": "1000", "matcQty": "100"},
            {"stdDt": "20261121", "slbTrRemQty": "1100", "matcQty": "110"},
        ]
        result = slb_features_for_signal("005930", "2026-11-21", dealing_records=rows)
        self.assertAlmostEqual(result["slb_rem_qty_change_1d_pct"], (1100 / 900 - 1) * 100)


if __name__ == "__main__":
    unittest.main()
