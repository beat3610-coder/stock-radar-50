import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from expanded_direction_audit import (
    FEATURES, FLOW_FEATURES, MIN_DEVELOPMENT_IC, MODEL_KINDS, REFIT_MONTH, _collect_extended_pool_flows,
    _comparison_fit_spec, _edge_label_status, _features, _flow_ablation_metadata,
    _make_rows, _mandatory_market_items, _metrics,
    _trade_perf, direction_label, _freeze_volatility_labels, _fit_temperature,
    _model, _training_rows_for, _selection_rows_for, _refit_anchor, _causal_prior_baseline,
    _probability_skill, BandRegressionModel,
)
from providers import get_historical_investor_flows
from universe import MANDATORY_KR, MANDATORY_US, AUDIT_63_BENCHMARK
from audit_63_fast import _frozen_audit_cohorts
from market_context import _flow_features


class ExpandedDirectionAuditTests(unittest.TestCase):
    def test_mandatory_etfs_normalize_universe_three_field_rows(self):
        self.assertEqual(list(_mandatory_market_items(MANDATORY_US,"US")),[("SOXL","SOXL","US")])
        self.assertEqual(list(_mandatory_market_items(MANDATORY_KR,"KR")),[
            ("472150","TIGER 배당커버드콜액티브","KR"),
            ("0262B0","TIGER 200커버드콜액티브","KR"),
        ])

    def test_fixed_threshold_labels(self):
        self.assertEqual(direction_label(.51, .5), "UP")
        self.assertEqual(direction_label(-.51, .5), "DOWN")
        self.assertEqual(direction_label(.5, .5), "SIDEWAYS")
        with self.assertRaises(ValueError):
            direction_label(float("nan"), .5)

    def test_frozen_benchmark_replaces_unstable_kospi_symbol(self):
        self.assertEqual(len(AUDIT_63_BENCHMARK["KOSPI"]),30)
        self.assertIn("015760",AUDIT_63_BENCHMARK["KOSPI"])
        self.assertNotIn("096770",AUDIT_63_BENCHMARK["KOSPI"])
        self.assertEqual(len(AUDIT_63_BENCHMARK["KOSDAQ"]),30)
        cohorts=_frozen_audit_cohorts({"KR":[("096770","SK이노베이션","KR","화학")],"KQ":[]})
        self.assertEqual([x[0] for x in cohorts["KOSPI"]],list(AUDIT_63_BENCHMARK["KOSPI"]))
        self.assertEqual([x[0] for x in cohorts["TIGER_ETF"]],list(AUDIT_63_BENCHMARK["TIGER_ETF"]))

    def test_volatility_band_k_uses_development_only(self):
        rows=[]
        for i,value in enumerate(range(1,101)):
            rows.append({"cohort":"KOSPI","label_available_date":"2025-10-01" if i<80 else "2025-12-01",
                         "volatility_unit_pct":1.0,"net_pct":float(value)})
        report=_freeze_volatility_labels(rows)["KOSPI"]
        self.assertEqual(report["development_n"],80)
        self.assertAlmostEqual(report["k"],24.7)
        self.assertEqual(sum(report["development_class_counts"].values()),80)

    def test_classifiers_are_not_balanced_and_temperature_is_fitted(self):
        logistic=_model("multinomial_logistic")
        forest=_model("random_forest")
        self.assertIsNone(logistic[-1].class_weight)
        self.assertIsNone(forest[-1].class_weight)
        x=np.asarray([[0.,np.nan],[1.,2.],[-1.,1.],[2.,np.nan],[-2.,0.],[.5,3.]],dtype=float)
        y=np.asarray([0,1,2,1,0,2])
        forest.fit(x,y)
        self.assertEqual(list(forest.classes_),[0,1,2])
        self.assertEqual(forest.predict_proba(x).shape,(6,3))
        rows=[{"label_vol":("UP" if i%2 else "DOWN")} for i in range(120)]
        probs=np.tile([.6,.1,.3],(120,1))
        result=_fit_temperature(rows,probs)
        self.assertEqual(result["fit_n"],120)
        self.assertGreater(result["temperature"],0)

    def test_fixed_schema_feature_matrix(self):
        row={key:0.0 for key in FEATURES}
        x=_features([row])
        self.assertEqual(x.shape,(1,len(FEATURES)))
        self.assertTrue(np.isfinite(x).all())

    def test_fixed_schema_includes_flow_macro_and_relative_features(self):
        self.assertTrue(set(FLOW_FEATURES).issubset(FEATURES))
        row={key:0.0 for key in FEATURES}
        self.assertEqual(_features([row]).shape,(1,len(FEATURES)))

    def test_us_soxl_uses_the_available_ordinary_us_training_pool(self):
        full=[{"ticker":"AAA"},{"ticker":"BBB"}];benchmark=[{"ticker":"SOXL"}]
        self.assertIs(_training_rows_for("US",full,benchmark),full)
        self.assertIs(_training_rows_for("KOSPI",full,benchmark),benchmark)
        self.assertIs(_selection_rows_for("US",full,benchmark),full)
        self.assertIs(_selection_rows_for("KOSPI",full,benchmark),benchmark)

    def test_edge_status_requires_minimum_development_ic_without_changing_selection(self):
        self.assertEqual(MIN_DEVELOPMENT_IC,.01)
        self.assertEqual(_edge_label_status(True,.00054),
                         "DIAGNOSTIC_MODEL_SELECTED_NO_DEVELOPMENT_EDGE")
        self.assertEqual(_edge_label_status(True,.02),"DEVELOPMENT_EDGE_SELECTED")
        self.assertEqual(_edge_label_status(False,None),"NO_MODEL_WITH_NONZERO_DEVELOPMENT_N")

    def test_korean_universe_fits_use_distinct_training_rows_and_frozen_evaluation(self):
        pool=[{"ticker":f"P{i:03d}"} for i in range(200)]
        pool63=pool[:63]
        eval_rows=[{"ticker":f"E{i:03d}"} for i in range(63)]
        for cohort in ("KOSPI","KOSDAQ"):
            audit=_comparison_fit_spec(cohort,"63_audit_symbols",pool,pool63,eval_rows)
            full=_comparison_fit_spec(cohort,"available_full_universe",pool,pool63,eval_rows)
            self.assertIs(audit["training_rows"],pool63)
            self.assertIs(full["training_rows"],pool)
            self.assertIs(audit["evaluation_rows"],eval_rows)
            self.assertIs(full["evaluation_rows"],eval_rows)
            self.assertEqual(audit["training_rows_n"],63)
            self.assertEqual(full["training_rows_n"],200)
            self.assertEqual(audit["evaluation_tickers_n"],63)
            self.assertEqual(full["evaluation_tickers_n"],63)
            audit_ablation=_flow_ablation_metadata("random_forest",10,"spread_0.10",audit)
            full_ablation=_flow_ablation_metadata("random_forest",10,"spread_0.10",full)
            for key in ("same_model","same_horizon","policy_key","policy_threshold_source",
                        "evaluation_rows_n","evaluation_tickers_n"):
                self.assertEqual(audit_ablation[key],full_ablation[key])
            self.assertNotEqual(audit_ablation["training_rows_n"],full_ablation["training_rows_n"])

    def test_us_universe_fits_both_keep_full_us_training_pool(self):
        pool=[{"ticker":f"US{i:03d}"} for i in range(98)]
        pool63=[{"ticker":"SOXL"}]
        eval_rows=[{"ticker":"SOXL"}]
        for fit_name in ("63_audit_symbols","available_full_universe"):
            spec=_comparison_fit_spec("US",fit_name,pool,pool63,eval_rows)
            self.assertIs(spec["training_rows"],pool)
            self.assertIs(spec["evaluation_rows"],eval_rows)
            self.assertEqual(spec["evaluation_tickers_n"],1)

    def test_extended_pool_flow_attempt_flag_comes_from_actual_calls_and_reuses_input(self):
        dates=pd.bdate_range("2026-01-01",periods=3)
        preexisting=pd.DataFrame({"foreign_net_shares":[1,2,3],
                                  "institution_net_shares":[-1,0,1]},index=dates)
        preexisting.attrs["collection_diagnostics"]={"requests_n":4,"first_date":"2026-01-01",
                                                     "last_date":"2026-01-05"}
        fetched=[]
        def fetch(code,**kwargs):
            fetched.append(code)
            frame=preexisting.copy()
            frame.attrs["collection_diagnostics"]={"requests_n":2,"first_date":"2026-01-01",
                "last_date":"2026-01-05","attempted":True,"available":True}
            return frame,"Naver mobile stock trend API"
        universe={"KOSPI":[("005930","삼성전자","KR","전자")],
                  "KOSDAQ":[("247540","에코프로비엠","KQ","화학")],"US":[("AAPL","Apple","US","Tech")]}
        frames={"005930":preexisting}
        result=_collect_extended_pool_flows(universe,frames,budget_seconds=60,workers=1,
                                             flow_fetcher=fetch)
        self.assertTrue(result["attempted_symbols"]>0)
        self.assertEqual(result["attempted_symbols"],1)
        self.assertEqual(result["preexisting_reused_symbols"],1)
        self.assertEqual(fetched,["247540"])
        self.assertIs(frames["005930"],preexisting)
        self.assertNotIn("AAPL",frames)
        self.assertEqual(result["coverage_by_cohort"]["KOSPI"]["available_symbols"],1)

    def test_failed_extended_flow_remains_missing_not_zero(self):
        dates=pd.bdate_range("2026-01-01",periods=45)
        price=pd.DataFrame({"Open":100.,"High":101.,"Low":99.,"Close":100.,"Volume":1000.},index=dates)
        def fail(code,**kwargs):raise RuntimeError("provider unavailable")
        universe={"KOSPI":[],"KOSDAQ":[("247540","에코프로비엠","KQ","화학")],"US":[]}
        frames={}
        result=_collect_extended_pool_flows(universe,frames,budget_seconds=60,workers=1,flow_fetcher=fail)
        self.assertEqual(result["attempted_symbols"],1)
        self.assertTrue(result["extended_296_pool_attempted"])
        features=_flow_features(price,None)
        self.assertTrue(features["foreign_flow_5d_to_turnover"].isna().all())
        self.assertTrue(features["institution_flow_20d_to_turnover"].isna().all())
        self.assertEqual(features["flow_data_available"].iloc[-1],0.0)

    def test_naver_flow_collector_pages_backward_and_records_coverage(self):
        class Response:
            status_code=200
            def __init__(self,rows):self.rows=rows
            def raise_for_status(self):pass
            def json(self):return self.rows
        class Session:
            def __init__(self):self.headers={};self.calls=[]
            def get(self,url,params,timeout):
                self.calls.append(params)
                if len(self.calls)==1:days=pd.bdate_range("2026-11-18",periods=3)
                elif len(self.calls)==2:days=pd.bdate_range("2026-11-04",periods=10)
                else:days=pd.bdate_range("2026-10-28",periods=5)
                rows=[{"bizdate":d.strftime("%Y%m%d"),"foreignerPureBuyQuant":"+1,234",
                       "organPureBuyQuant":"-234"} for d in reversed(days)]
                return Response(rows)
        with tempfile.TemporaryDirectory() as directory:
            from providers import HistoricalFlowError
            with patch("providers.PRICE_CACHE",Path(directory)/"prices"),patch("requests.Session",Session):
                frame,source=get_historical_investor_flows("005930",start_date="2026-11-01",end_date="2026-11-20")
                self.assertIn("Naver mobile",source)
                self.assertEqual(len(frame),15)
                diag=frame.attrs["collection_diagnostics"]
                self.assertEqual(diag["requests_n"],3)
                self.assertEqual(diag["first_date"],"2026-11-02")
                self.assertEqual(diag["last_date"],"2026-11-20")
                self.assertTrue(diag["available"])

    def test_naver_duplicate_page_steps_cursor_back_and_continues(self):
        class Response:
            status_code=200
            def __init__(self,rows):self.rows=rows
            def raise_for_status(self):pass
            def json(self):return self.rows
        class Session:
            def __init__(self):self.headers={};self.calls=[]
            def get(self,url,params,timeout):
                self.calls.append(params.get("bizdate"))
                if len(self.calls)==1:
                    days=pd.bdate_range("2023-06-12",periods=10)
                elif len(self.calls)==2:
                    days=pd.bdate_range("2023-05-29",periods=10)
                else:
                    return Response([])
                rows=[{"bizdate":d.strftime("%Y%m%d"),"foreignerPureBuyQuant":"100",
                       "organPureBuyQuant":"-50"} for d in reversed(days)]
                return Response(rows)
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/"prices"
            flow_dir=cache.parent/"flows";flow_dir.mkdir()
            cached_dates=pd.bdate_range("2023-06-12","2023-06-30")
            cached=pd.DataFrame({"foreign_net_shares":10,"institution_net_shares":-5},index=cached_dates)
            cached.to_csv(flow_dir/"naver_105560.csv",index_label="Date")
            session=Session()
            with patch("providers.PRICE_CACHE",cache),patch("requests.Session",lambda:session):
                frame,_=get_historical_investor_flows("105560",start_date="2023-06-01",end_date="2023-06-30")
            self.assertEqual(session.calls,["20230612","20230611"])
            self.assertEqual(str(frame.index.min().date()),"2023-06-01")
            diag=frame.attrs["collection_diagnostics"]
            self.assertEqual(diag["first_date"],"2023-06-01")
            self.assertEqual(diag["requests_n"],2)
            self.assertEqual(diag["failures"],[])

    def test_refit_anchor_is_calendar_aligned_and_never_reaches_the_block(self):
        self.assertEqual(REFIT_MONTH,24)
        self.assertEqual(_refit_anchor("2024-05-01"),"2024-01-01")
        self.assertEqual(_refit_anchor("2025-11-03"),"2024-01-01")
        self.assertEqual(_refit_anchor("2026-07-15"),"2026-01-01")
        self.assertEqual(_refit_anchor("2023-12-29"),"2022-01-01")
        for day in ("2016-01-04","2021-06-30","2026-10-30"):
            self.assertLess(_refit_anchor(day)[:4]+"-01-01",day)
            self.assertLessEqual(_refit_anchor(day),day)

    def test_band_regression_probabilities_follow_the_band(self):
        rows=[]
        rng=np.random.default_rng(7)
        for i in range(400):
            x=float(rng.normal());net=2.0*x+float(rng.normal())
            rows.append({"x":x,"net_pct":net})
        X=np.asarray([[r["x"]] for r in rows]);y=np.asarray([r["net_pct"] for r in rows])
        model=BandRegressionModel().fit(X,y)
        probs=model.predict_proba(np.asarray([[-5.],[0.],[5.]]),np.asarray([1.,1.,1.]))
        self.assertEqual(probs.shape,(3,3))
        np.testing.assert_allclose(probs.sum(axis=1),1.0)
        self.assertGreater(probs[2,2],probs[0,2])
        self.assertGreater(probs[0,0],probs[2,0])
        # A wider SIDEWAYS band must not increase the probability of an extreme move.
        narrow=model.predict_proba(np.asarray([[3.]]),np.asarray([1.]))[0]
        wide=model.predict_proba(np.asarray([[3.]]),np.asarray([6.]))[0]
        self.assertGreater(narrow[2],wide[2])
        self.assertIsNone(_model("multinomial_logistic")[-1].class_weight)
        self.assertIn("band_regression",MODEL_KINDS)

    def test_causal_prior_baseline_uses_only_labels_before_the_anchor(self):
        train=[]
        for i in range(30):
            train.append({"label_vol":"UP","label_available_date":"2023-03-01"})
        for i in range(30):
            train.append({"label_vol":"DOWN","label_available_date":"2025-03-01"})
        eval_rows=[{"label_vol":"DOWN","signal_date":"2026-04-01"} for _ in range(10)]
        report=_causal_prior_baseline(train,eval_rows)
        self.assertEqual(report["n"],10)
        # 2025-03 labels sit inside the 365-day window ending 2026-01-01 only for
        # the 2025 block; the 2026 anchor window is 2025-01-01..2026-01-01.
        self.assertEqual(report["class_distribution"]["DOWN"],10)
        self.assertIsNotNone(report["accuracy_pct"])

    def test_probability_skill_is_positive_only_when_brier_beats_the_period_prior(self):
        rows=[{"label_vol":"UP"},{"label_vol":"UP"},{"label_vol":"DOWN"},{"label_vol":"UP"}]
        sharp=np.asarray([[.05,.05,.9],[.05,.05,.9],[.8,.1,.1],[.2,.2,.6]])
        flat=np.full((4,3),1/3)
        self.assertGreater(_probability_skill(rows,sharp)["brier_skill"],0)
        self.assertLess(_probability_skill(rows,flat)["brier_skill"],0)

    def test_three_way_report_flags_a_model_worse_than_the_class_prior(self):
        from audit_63_fast import _three_way_report
        samples=([{"label":"DOWN","prediction":"UP","probabilities":{"UP":.9,"DOWN":.05,"SIDEWAYS":.05}}
                  for _ in range(20)]+
                 [{"label":"DOWN","prediction":"DOWN","probabilities":{"UP":.05,"DOWN":.9,"SIDEWAYS":.05}}
                  for _ in range(20)])
        report=_three_way_report(samples)
        self.assertLess(report["brier_skill_vs_period_prior"],0)
        self.assertEqual(report["probability_edge_status"],"NO_DEMONSTRATED_EDGE_WORSE_THAN_CLASS_PRIOR")
        self.assertGreater(report["accuracy_standard_error_pp"],0)

    def test_three_class_metrics_and_confusion_labels(self):
        rows=[{"label_0.5":"DOWN","net_pct":-1,"signal_date":"2025-01-01"},
              {"label_0.5":"SIDEWAYS","net_pct":0,"signal_date":"2025-01-01"},
              {"label_0.5":"UP","net_pct":1,"signal_date":"2025-01-01"}]
        result=_metrics(rows,np.asarray([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]))
        self.assertEqual(result["accuracy_pct"],100)
        self.assertEqual(result["confusion_matrix"]["counts"],[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(set(result["class_metrics"]),{"DOWN","SIDEWAYS","UP"})

    def test_selected_trades_include_drawdowns_and_avoidance(self):
        rows=[{"ticker":"A","entry_date":"2025-01-01","label_available_date":"2025-01-05","net_pct":-2,"label_0.5":"DOWN"},
              {"ticker":"B","entry_date":"2025-01-02","label_available_date":"2025-01-06","net_pct":1,"label_0.5":"UP"},
              {"ticker":"C","entry_date":"2025-01-03","label_available_date":"2025-01-07","net_pct":.5,"label_0.5":"UP"}]
        result=_trade_perf(rows[1:],rows,np.asarray([[.8,.1,.1],[.1,.1,.8],[.1,.1,.8]]))
        self.assertEqual(result["n"],2)
        self.assertEqual(result["portfolio_5slot_trade_n"],2)
        self.assertEqual(result["rejected_down_n"],1)
        self.assertEqual(result["rejected_down_loss_avoided_pct"],2)
        self.assertIsNotNone(result["max_drawdown_5slot_pct"])

    def test_labels_enter_next_open_and_mature_at_horizon_close(self):
        dates=pd.bdate_range("2016-01-01",periods=400)
        px=np.linspace(100,120,len(dates))
        frame=pd.DataFrame({"Open":px*.999,"High":px*1.01,"Low":px*.99,
                            "Close":px,"Volume":np.full(len(px),10000.)},index=dates)
        rows=_make_rows(frame,"TEST","KR","KOSPI")
        first=rows[0]
        self.assertEqual(first["entry_date"],str(dates[201].date()))
        self.assertEqual(first["label_available_date"],str(dates[205].date()))
        self.assertEqual(first["exit_date"],first["label_available_date"])
        self.assertEqual({r["horizon"] for r in rows},{5,10,20})


if __name__ == "__main__":
    unittest.main()
