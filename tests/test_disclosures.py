import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit_63_fast import (_legacy_policy_stage,_not_evaluated_fundamental_stage,
                           _disclosure_bad_title_counts)
from disclosures import (collect_company_disclosures,disclosure_features_for_signal,
                         get_company_disclosures,parse_disclosure_html,
                         bad_disclosure_keyword_matches)


FIXTURE="""<table><tr><th>접수일자</th><th>공시대상회사</th><th>보고서명</th><th>제출인</th></tr>
<tr><td>2026.11.16</td><td>삼성전자</td><td><a href='/dsaf001/main.do?rcpNo=20261116000001'>분기보고서 (2026.09)</a></td><td>삼성전자</td></tr>
<tr><td>2026.11.17</td><td>삼성전자</td><td><a href='javascript:openReport(20261117000002)'>유상증자 결정</a></td><td>삼성전자</td></tr></table>"""


class DisclosureTests(unittest.TestCase):
    def test_html_parser_extracts_receipt_title_and_rcp_number(self):
        rows=parse_disclosure_html(FIXTURE)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["receipt_date"],"2026-11-16")
        self.assertEqual(rows[0]["title"],"분기보고서 (2026.09)")
        self.assertEqual(rows[0]["rcpNo"],"20261116000001")

    def test_signal_uses_only_disclosures_strictly_before_signal_day(self):
        idx=pd.bdate_range("2026-01-01",periods=10)
        records=[{"receipt_date":"2026-01-02","title":"분기보고서"},
                 {"receipt_date":"2026-01-02","title":"유상증자 결정"},
                 {"receipt_date":"2026-01-05","title":"전환사채 발행"},
                 {"receipt_date":"2026-01-06","title":"자사주취득 결정"}]
        result=disclosure_features_for_signal(records,"2026-01-05",idx,available=True)
        self.assertEqual(result["disclosure_bad_60d"],1)
        self.assertEqual(result["disclosure_positive_20d"],0)
        self.assertEqual(result["disclosure_periodic_days_since"],1)
        self.assertTrue(np.isnan(disclosure_features_for_signal(records,"2026-01-05",idx,available=False)["disclosure_bad_60d"]))

    def test_bad_titles_use_specific_terms_and_explicit_exclusions(self):
        ordinary_share_change="최대주주등소유주식변동신고서"
        foreign_delisting="주요사항보고서(해외증권시장주권등상장폐지결정)"
        self.assertEqual(bad_disclosure_keyword_matches(ordinary_share_change),())
        self.assertEqual(bad_disclosure_keyword_matches(foreign_delisting),())
        self.assertEqual(bad_disclosure_keyword_matches("임원ㆍ주요주주 소유주식변동"),())
        self.assertEqual(bad_disclosure_keyword_matches("주요주주 주식변동"),())
        self.assertTrue(bad_disclosure_keyword_matches("유상증자결정"))
        self.assertIn("전환사채권발행결정",bad_disclosure_keyword_matches("전환사채권발행결정"))
        self.assertIn("최대주주 변경",bad_disclosure_keyword_matches("최대주주 변경 결정"))

    def test_bad_title_counts_are_reported_by_cohort_input_and_exclude_false_hits(self):
        ticker="005930"
        docs={ticker:{"records":[
            {"title":"최대주주등소유주식변동신고서"},
            {"title":"주요사항보고서(해외증권시장주권등상장폐지결정)"},
            {"title":"유상증자결정"},{"title":"유상증자결정"},
            {"title":"전환사채권발행결정"}]}}
        result=_disclosure_bad_title_counts([(ticker,"삼성전자","KR","IT")],docs)
        self.assertEqual(result,[
            {"title":"유상증자결정","count":2,"matched_keywords":["유상증자"]},
            {"title":"전환사채권발행결정","count":1,"matched_keywords":["전환사채권발행결정"]}])

    def test_ten_year_span_is_requested_as_calendar_years(self):
        class Response:
            status_code=200;text="<table></table>"
            def raise_for_status(self):pass
        class Session:
            def __init__(self):self.headers={};self.params=[]
            def get(self,url,params,timeout):self.params.append(dict(params));return Response()
        session=Session()
        with tempfile.TemporaryDirectory() as d:
            result=get_company_disclosures("005930","삼성전자","2015-12-01","2017-02-01",
                session=session,cache_dir=d,force_refresh=True)
        self.assertEqual([x["startDate"][:4] for x in session.params],["2015","2016","2017"])
        self.assertEqual([x["endDate"][:4] for x in session.params],["2015","2016","2017"])
        self.assertTrue(all(x["maxResults"]==100 and "publicType" not in x for x in session.params))
        self.assertEqual(result["diagnostics"]["requests_n"],3)

    def test_budget_expiry_is_partial_and_missing_features_stay_nan(self):
        result=collect_company_disclosures([("005930","삼성전자"),("247540","에코프로비엠")],
            time_budget_seconds=0,workers=1,fetcher=lambda *a,**k:self.fail("must not fetch"))
        self.assertEqual(result["attempted_symbols"],0)
        self.assertEqual(result["not_attempted"],["005930","247540"])
        features=disclosure_features_for_signal([],"2026-01-05",pd.bdate_range("2025-01-01",periods=300),available=False)
        self.assertTrue(all(np.isnan(v) for v in features.values()))

    def test_risk_off_gate_removes_legacy_entries_and_fundamentals_are_not_evaluated(self):
        rows=[{"ticker":"A","entry_date":"2025-01-02","exit_date":"2025-01-05","market_regime":"RISK_OFF"},
              {"ticker":"B","entry_date":"2025-01-02","exit_date":"2025-01-05","market_regime":"NEUTRAL"}]
        chosen=_legacy_policy_stage(rows,"legacy_risk_off_hard_gate")
        self.assertEqual([x["ticker"] for x in chosen],["B"])
        self.assertEqual(_not_evaluated_fundamental_stage()["status"],"NOT_EVALUATED")
        self.assertEqual(_not_evaluated_fundamental_stage()["periods"],{})


if __name__=="__main__":unittest.main()
