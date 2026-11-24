import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit_63_fast import (_legacy_policy_stage,_not_evaluated_fundamental_stage,
                           _disclosure_bad_title_counts,_disclosure_failure_reasons_top,
                           _risk_adjusted_policy_selection,
                           POLICY_SELECTION_METRIC,_focused_liquidity_ranking,
                           _focused_recommendation,_focused_universe_ablation,
                           _disclosure_gate_input_coverage,_policy_ladder,
                           _policy_stage_trade_records)
from universe import AUDIT_63_BENCHMARK
from disclosures import (collect_company_disclosures,disclosure_features_for_signal,
                         get_cached_company_disclosures,get_company_disclosures,parse_disclosure_html,
                         bad_disclosure_keyword_matches)
from disclosure_warmup import prioritize_benchmark_targets


FIXTURE="""<table><tr><th>접수일자</th><th>공시대상회사</th><th>보고서명</th><th>제출인</th></tr>
<tr><td>2026.11.16</td><td>삼성전자</td><td><a href='/dsaf001/main.do?rcpNo=20261116000001'>분기보고서 (2026.09)</a></td><td>삼성전자</td></tr>
<tr><td>2026.11.17</td><td>삼성전자</td><td><a href='javascript:openReport(20261117000002)'>유상증자 결정</a></td><td>삼성전자</td></tr></table>"""


class DisclosureTests(unittest.TestCase):
    def test_focused_liquidity_ranking_uses_development_turnover_and_quality_only(self):
        dates=pd.bdate_range("2016-01-04","2026-10-30")
        dev_end=pd.Timestamp("2025-11-01")
        dev_mask=dates<dev_end
        turnover_a=np.where(dev_mask,1_000.,1e12)
        turnover_b=np.where(dev_mask,2_000.,1.)
        frame=lambda turnover,index=dates:pd.DataFrame({"Close":100.,"Volume":turnover/100.},index=index)
        delayed=dates[dates>=pd.Timestamp("2016-01-06")]
        missing=dates[np.arange(len(dates))%10!=0]
        items=[("A","A","KR","test"),("B","B","KR","test"),
               ("C","C","KR","test"),("D","D","KR","test")]
        result=_focused_liquidity_ranking("KOSPI",items,{
            "A":frame(turnover_a),"B":frame(turnover_b),"C":frame(np.full(len(delayed),9_000.),delayed),
            "D":frame(np.full(len(missing),8_000.),missing)})
        self.assertEqual(result["eligible_tickers_by_liquidity"][:2],["B","A"])
        by_code={r["ticker"]:r for r in result["symbols"]}
        self.assertIn("FIRST_BAR_NOT_BEFORE_2016-01-05",by_code["C"]["exclusion_reasons"])
        self.assertIn("DEVELOPMENT_BAR_COMPLETENESS_BELOW_95_PERCENT",by_code["D"]["exclusion_reasons"])
        self.assertEqual(by_code["B"]["median_daily_turnover"],2000.0)

    def test_focused_recommendation_uses_development_and_prefers_smallest_qualifying_set(self):
        baseline={"trades":100,"mean_net_pct":1.0,"max_drawdown_sequence_pct":50.0}
        sets={
            "smaller":{"recommendation_candidate":True,"total_symbols_n":13,
                "periods":{"development":{"trades":40,"mean_net_pct":.8,"max_drawdown_sequence_pct":20},
                    "validation":{"trades":100,"mean_net_pct":1000,"max_drawdown_sequence_pct":1},
                    "diagnostic":{"trades":100,"mean_net_pct":1000,"max_drawdown_sequence_pct":1}},
                "selected_tickers":{"KOSPI":["a"],"KOSDAQ":["b"],"TIGER_ETF":[],"SOXL":[]}},
            "larger":{"recommendation_candidate":True,"total_symbols_n":23,
                "periods":{"development":{"trades":60,"mean_net_pct":1.5,"max_drawdown_sequence_pct":20},
                    "validation":{"trades":0,"mean_net_pct":None,"max_drawdown_sequence_pct":None}},
                "selected_tickers":{"KOSPI":["a"],"KOSDAQ":["b"],"TIGER_ETF":[],"SOXL":[]}}}
        result=_focused_recommendation(sets,baseline)
        self.assertEqual(result["selected_set"],"smaller")
        self.assertFalse(result["selection_used_validation_or_diagnostic"])

    def test_focused_universe_keeps_core_tickers_and_does_not_redefine_benchmark(self):
        cohorts={cohort:[(ticker,ticker,"US" if cohort=="SOXL" else "KR","")
                         for ticker in tickers]
                 for cohort,tickers in AUDIT_63_BENCHMARK.items()}
        result=_focused_universe_ablation(cohorts,{cohort:{} for cohort in cohorts},
            {cohort:{period:[] for period in ("development","validation","diagnostic")}
             for cohort in cohorts},{})
        core=result["sets"]["core3_plus_liquidity_fill10"]
        self.assertEqual(core["selected_tickers"]["KOSPI"][:3],
                         ["005930","000660","035420"])
        self.assertEqual(core["core_tickers_included"],["005930","000660","035420"])
        self.assertEqual(result["sets"]["baseline_63"]["selected_tickers"],
                         {cohort:list(tickers) for cohort,tickers in AUDIT_63_BENCHMARK.items()})
        self.assertEqual(result["sample_source"],
                         "already-loaded frozen 63 benchmark OHLCV only; no additional instrument fetches")
        self.assertEqual(result["policy_evaluated"],"selected_policy")

    def test_policy_b_universe_block_reuses_the_same_sets_under_another_policy(self):
        cohorts={cohort:[(ticker,ticker,"US" if cohort=="SOXL" else "KR","")
                         for ticker in tickers]
                 for cohort,tickers in AUDIT_63_BENCHMARK.items()}
        trade=lambda ticker,day:{"ticker":ticker,"signal_date":day,"entry_date":day,
            "exit_date":"2024-01-05","net_pct":1.5,"model_expected_net_pct":0.1,
            "market_regime":"NEUTRAL"}
        trades={cohort:{period:[] for period in ("development","validation","diagnostic")}
                for cohort in cohorts}
        trades["KOSPI"]["development"]=[trade("005930","2024-01-02"),trade("000660","2024-01-02")]
        trades["KOSPI"]["validation"]=[trade("005930","2025-12-02")]
        priced={cohort:{ticker:pd.DataFrame({"Close":100.}
                  ,index=pd.bdate_range("2016-01-04","2026-10-30"))
                  for ticker in tickers} for cohort,tickers in cohorts.items()}
        result=_focused_universe_ablation(cohorts,priced,trades,{},
            policy_label="legacy_risk_off_hard_gate")
        self.assertEqual(result["policy_evaluated"],"legacy_risk_off_hard_gate")
        self.assertIn("policy legacy_risk_off_hard_gate",result["selection_rule"]["policy"])
        self.assertEqual(result["sets"]["baseline_63"]["periods"]["development"]["trades"],2)
        self.assertEqual(result["sets"]["baseline_63"]["periods"]["validation"]["trades"],1)
        smaller=result["sets"]["liquidity_top_KOSPI10_KOSDAQ10"]["periods"]["validation"]
        self.assertEqual(smaller["trades"],0)
        self.assertFalse(result["recommended_universe"]["selection_used_validation_or_diagnostic"])

    def test_policy_stage_trade_records_keep_stage_status_and_signal_time_fields(self):
        rows=[{"ticker":"005930","signal_date":"2024-01-02","entry_date":"2024-01-02",
               "exit_date":"2024-01-05","net_pct":1.5,"market_regime":"RISK_OFF"},
              {"ticker":"000660","signal_date":"2024-01-02","entry_date":"2024-01-02",
               "exit_date":"2024-01-05","net_pct":-0.5,"market_regime":"RISK_ON"}]
        ladder={"legacy_only":{"status":"EVALUATED","params":{}},
                "legacy_risk_off_hard_gate":{"status":"EVALUATED","params":{}},
                "legacy_risk_off_trend_filter":{"status":"INSUFFICIENT_DEVELOPMENT_SAMPLE","params":None}}
        records=_policy_stage_trade_records(ladder,rows)
        development=records["legacy_only"]["periods"]["development"]
        self.assertEqual(sorted(r["ticker"] for r in development),["000660","005930"])
        self.assertEqual({r["signal_date"] for r in development},{"2024-01-02"})
        self.assertEqual({r["market_regime"] for r in development},{"RISK_OFF","RISK_ON"})
        self.assertEqual(records["legacy_risk_off_hard_gate"]["status"],"EVALUATED")
        self.assertEqual(records["legacy_risk_off_trend_filter"]["status"],"INSUFFICIENT_DEVELOPMENT_SAMPLE")
        self.assertEqual(records["legacy_risk_off_trend_filter"]["periods"]["development"],[])

    @staticmethod
    def _selection_stage(trades,mean,sequence_mdd,portfolio_mdd):
        return {"status":"EVALUATED","periods":{"development":{
            "trades":trades,"mean_net_pct":mean,
            "max_drawdown_sequence_pct":sequence_mdd,
            "max_drawdown_portfolio_5_equal_pct":portfolio_mdd}}}

    def test_risk_adjusted_selection_can_prefer_lower_return_with_much_lower_drawdown(self):
        steps={"legacy_only":self._selection_stage(75,.787,28.48,3.2),
               "legacy_risk_off_hard_gate":self._selection_stage(68,.598,8.0,2.1)}
        selected,comparison=_risk_adjusted_policy_selection(steps)
        self.assertEqual(selected,"legacy_risk_off_hard_gate")
        self.assertEqual(comparison["selection_metric"],POLICY_SELECTION_METRIC)
        self.assertGreater(steps["legacy_risk_off_hard_gate"]["development_risk_adjusted_score"],
                           steps["legacy_only"]["development_risk_adjusted_score"])
        self.assertEqual(comparison["stages"]["legacy_risk_off_hard_gate"]["development_5slot_portfolio_mdd_pct"],2.1)

    def test_risk_adjusted_selection_excludes_stages_below_minimum_trades(self):
        steps={"legacy_only":self._selection_stage(20,.5,20,5),
               "legacy_risk_off_hard_gate":self._selection_stage(9,2.0,1,1)}
        selected,comparison=_risk_adjusted_policy_selection(steps)
        self.assertEqual(selected,"legacy_only")
        self.assertFalse(comparison["stages"]["legacy_risk_off_hard_gate"]["candidate"])
        self.assertEqual(comparison["stages"]["legacy_risk_off_hard_gate"]["exclusion_reason"],"BELOW_MIN_POLICY_TRADES")

    def test_risk_adjusted_selection_falls_back_to_legacy_when_no_candidate_qualifies(self):
        steps={"legacy_only":self._selection_stage(8,.5,20,5),
               "legacy_risk_off_hard_gate":self._selection_stage(12,-.1,3,2)}
        selected,comparison=_risk_adjusted_policy_selection(steps)
        self.assertEqual(selected,"legacy_only")
        self.assertEqual(comparison["fallback"],"legacy_only")
        self.assertFalse(comparison["stages"]["legacy_only"]["candidate"])

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

    def test_availability_accepts_eighty_percent_year_coverage_and_marks_partial(self):
        class Response:
            status_code=200
            def __init__(self,text):self.text=text
            def raise_for_status(self):pass
        class Session:
            def __init__(self):self.headers={};self.years=[]
            def get(self,url,params,timeout):
                year=int(params["startDate"][:4]);self.years.append(year)
                if year==2019:return Response("<html></html>")
                rcp=f"{year}0101000001"
                return Response(f"<table><tr><td>{year}.01.01</td><td>회사</td><td><a href='?rcpNo={rcp}'>사업보고서</a></td><td>회사</td></tr></table>")
        session=Session()
        with tempfile.TemporaryDirectory() as d,patch("disclosures.time.sleep"):
            result=get_company_disclosures("005930","회사","2015-12-01","2019-12-31",
                session=session,cache_dir=d)
        diag=result["diagnostics"]
        self.assertEqual(session.years,[2015,2016,2017,2018,2019])
        self.assertEqual(diag["available_years_n"],4)
        self.assertEqual(diag["year_coverage_pct"],80.0)
        self.assertTrue(diag["available"])
        self.assertTrue(diag["partial"])
        self.assertTrue(diag["complete"])

    def test_partial_cache_retries_only_failed_page_and_preserves_completed_years(self):
        class Response:
            status_code=200
            text="<table><tr><td>2016.10.01</td><td>회사</td><td><a href='?rcpNo=20161001000001'>분기보고서</a></td><td>회사</td></tr></table>"
            def raise_for_status(self):pass
        class Session:
            def __init__(self):self.headers={};self.pages=[]
            def get(self,url,params,timeout):self.pages.append((params["startDate"][:4],params["currentPage"]));return Response()
        session=Session()
        cached={"records":[
            {"receipt_date":"2015-12-15","title":"사업보고서","rcpNo":"20151215000001"},
            {"receipt_date":"2016-03-01","title":"분기보고서","rcpNo":"20160301000001"}],
            "diagnostics":{"requested_start":"2015-12-01","requested_end":"2016-12-31",
                "complete":False,"failures":["year=2016 page=2 try=4: ConnectionError"],
                "year_status":{"2015":{"complete":True,"completed_pages":[1],"failed_pages":[],
                    "next_page":None,"window_start":"2015-12-01","window_end":"2015-12-31"},
                    "2016":{"complete":False,"completed_pages":[1],"failed_pages":[2],"next_page":2,
                    "window_start":"2016-01-01","window_end":"2016-12-31"}}}}
        with tempfile.TemporaryDirectory() as d,patch("disclosures.time.sleep"):
            Path(d,"005930.json").write_text(json.dumps(cached),encoding="utf-8")
            result=get_company_disclosures("005930","회사","2015-12-01","2016-12-31",
                session=session,cache_dir=d)
        self.assertEqual(session.pages,[("2016",2)])
        self.assertEqual(result["diagnostics"]["requests_n"],1)
        self.assertEqual(len(result["records"]),3)
        self.assertTrue(result["diagnostics"]["complete"])
        self.assertTrue(result["diagnostics"]["available"])

    def test_four_attempts_use_one_two_four_second_backoff(self):
        class Response:
            status_code=200;text="<html></html>"
            def raise_for_status(self):pass
        class Session:
            def __init__(self):self.headers={};self.calls=0
            def get(self,url,params,timeout):
                self.calls+=1
                if self.calls<4:raise ConnectionError("temporary disconnect")
                return Response()
        session=Session()
        with tempfile.TemporaryDirectory() as d,patch("disclosures.time.sleep") as sleeper:
            result=get_company_disclosures("005930","회사","2016-01-01","2016-12-31",
                session=session,cache_dir=d)
        self.assertEqual(result["diagnostics"]["requests_n"],4)
        backoffs=[call.args[0] for call in sleeper.call_args_list if call.args[0]>=1.0]
        self.assertEqual(backoffs,[1.0,2.0,4.0])

    def test_budget_expiry_is_partial_and_missing_features_stay_nan(self):
        result=collect_company_disclosures([("005930","삼성전자"),("247540","에코프로비엠")],
            time_budget_seconds=0,workers=1,fetcher=lambda *a,**k:self.fail("must not fetch"))
        self.assertEqual(result["attempted_symbols"],0)
        self.assertEqual(result["not_attempted"],["005930","247540"])
        features=disclosure_features_for_signal([],"2026-01-05",pd.bdate_range("2025-01-01",periods=300),available=False)
        self.assertTrue(all(np.isnan(v) for v in features.values()))

    def test_disclosure_collection_clamps_workers_and_counts_unavailable_results(self):
        fetcher=lambda ticker,name,**kwargs:{"records":[],"diagnostics":{"available":False,
            "complete":False,"year_coverage_pct":0.0,"failures":["temporary failure"]}}
        result=collect_company_disclosures([(str(i),str(i)) for i in range(5)],
            time_budget_seconds=10,workers=8,fetcher=fetcher)
        self.assertEqual(result["workers"],3)
        self.assertEqual(result["attempted_symbols"],5)
        self.assertEqual(len(result["failures"]),5)

    def test_d_stage_counts_missing_input_rows_and_marks_low_symbol_coverage_partial(self):
        rows=[{"ticker":"A","signal_date":"2025-01-01","entry_date":"2025-01-02",
            "exit_date":"2025-01-05","market_regime":"NEUTRAL","market_ma200_gap_pct":0,
            "signal_trend_200_pct":1,"relative_strength_60d_pct":1,"signal_volatility_20d_pct":2,
            "disclosure_bad_60d":float("nan"),"disclosure_periodic_days_since":float("nan")}]
        coverage=_disclosure_gate_input_coverage(rows,{
            "market_ma200_gap_min_pct":-5,"volatility_cap_pct":10})
        self.assertEqual(coverage["candidate_rows_before_disclosure_gate_n"],1)
        self.assertEqual(coverage["excluded_missing_disclosure_n"],1)
        ladder=_policy_ladder([],{},30,"KOSPI",{"available_symbols_n":10,
            "requested_symbols_n":30,"coverage_pct":33.33})
        self.assertEqual(ladder["legacy_risk_off_trend_disclosure"]["status"],"INSUFFICIENT_DEVELOPMENT_SAMPLE")
        self.assertEqual(ladder["legacy_risk_off_trend_disclosure"]["disclosure_coverage"]["symbol_coverage_pct"],33.33)
        empty=_policy_ladder([],{},30,"KOSDAQ",{"available_symbols_n":0,
            "requested_symbols_n":30,"coverage_pct":0.0})
        self.assertEqual(empty["legacy_risk_off_trend_disclosure"]["status"],"NOT_EVALUATED")

    def test_audit_cache_reader_never_makes_network_requests_for_missing_cache(self):
        with tempfile.TemporaryDirectory() as d:
            result=get_cached_company_disclosures("005930","삼성전자",cache_dir=d)
        self.assertFalse(result["diagnostics"]["available"])
        self.assertFalse(result["diagnostics"]["attempted"])
        self.assertEqual(result["diagnostics"]["requests_n"],0)
        self.assertIn("manual disclosure-warmup workflow",result["diagnostics"]["reason"])
        self.assertEqual(_disclosure_failure_reasons_top({"005930":result}),
                         {"failure_reasons_top5":[],"cache_missing_symbols_n":1})
        self.assertEqual(_disclosure_failure_reasons_top({}),
                         {"failure_reasons_top5":[],"cache_missing_symbols_n":0})
        failed=dict(result)
        failed["diagnostics"]={**result["diagnostics"],"failures":["RuntimeError: cache read"]}
        self.assertEqual(_disclosure_failure_reasons_top({"005930":failed}),
                         {"failure_reasons_top5":[{"reason":"RuntimeError","count":1}],
                          "cache_missing_symbols_n":1})
        failed["diagnostics"]={**result["diagnostics"],
            "failures":["No warmup cache; use the manual disclosure-warmup workflow.",
                        "HTTPError: HTTP 503 Service Unavailable"]}
        self.assertEqual(_disclosure_failure_reasons_top({"005930":failed}),
                         {"failure_reasons_top5":[{"reason":"HTTP 503","count":1}],
                          "cache_missing_symbols_n":1})

    def test_risk_off_gate_removes_legacy_entries_and_fundamentals_are_not_evaluated(self):
        rows=[{"ticker":"A","entry_date":"2025-01-02","exit_date":"2025-01-05","market_regime":"RISK_OFF"},
              {"ticker":"B","entry_date":"2025-01-02","exit_date":"2025-01-05","market_regime":"NEUTRAL"}]
        chosen=_legacy_policy_stage(rows,"legacy_risk_off_hard_gate")
        self.assertEqual([x["ticker"] for x in chosen],["B"])
        self.assertEqual(_not_evaluated_fundamental_stage()["status"],"NOT_EVALUATED")
        self.assertEqual(_not_evaluated_fundamental_stage()["periods"],{})

    def test_warmup_prioritizes_the_frozen_korean_benchmark_names(self):
        targets=[{"ticker":"999999","name":"other","cohort":"KOSPI"},
                 {"ticker":"005930","name":"삼성전자","cohort":"KOSPI"},
                 {"ticker":"000660","name":"SK하이닉스","cohort":"KOSPI"},
                 {"ticker":"247540","name":"에코프로비엠","cohort":"KOSDAQ"}]
        ordered=prioritize_benchmark_targets(targets,{"KOSPI":["000660","005930"],"KOSDAQ":["247540"]})
        self.assertEqual([t["ticker"] for t in ordered],["000660","005930","247540","999999"])

    def test_disclosure_warmup_request_stays_off_without_a_committed_request(self):
        import audit_63_fast
        missing=Path(tempfile.gettempdir())/"stock-radar-missing-warmup-request.json"
        if missing.exists():missing.unlink()
        with patch.object(audit_63_fast,"DISCLOSURE_WARMUP_REQUEST_PATH",missing):
            self.assertIsNone(audit_63_fast._run_disclosure_warmup_request())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"request.json"
            path.write_text(json.dumps({"enabled":False}),encoding="utf-8")
            with patch.object(audit_63_fast,"DISCLOSURE_WARMUP_REQUEST_PATH",path):
                self.assertEqual(audit_63_fast._run_disclosure_warmup_request()["status"],"DISABLED")



if __name__=="__main__":unittest.main()
