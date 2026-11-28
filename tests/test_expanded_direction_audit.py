import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
from sklearn.metrics import log_loss
import pandas as pd

from expanded_direction_audit import (
    DECISION_TAU_GRID, FEATURES, FLOW_FEATURES, MAX_REFERENCE_PRIOR_MIX, MIN_DEVELOPMENT_IC, MODEL_KINDS,
    PRIOR_HALF_LIFE_GRID, PRIOR_SHRINKAGE_WINDOW_DAYS, REFIT_MONTH, TEMPERATURE_LOG_BOUNDS,
    REGIME_GATE_DRAWDOWN_GRID, REGIME_GATE_FEATURES, REGIME_GATE_INDEX_RET20_GRID,
    REGIME_GATE_MA200_GAP_GRID, REGIME_GATE_MAX_MACRO_F1_DROP_PP, REGIME_GATE_MIN_ACCURACY_GAIN_PP,
    REGIME_GATE_MIN_DEV_OVERRIDES, REGIME_GATE_STATES, REGIME_GATE_TARGETS,
    _apply_prior_mix, _apply_regime_gate, _apply_temperature, _beats_causal_prior_on_development, _causal_class_priors, _class_prior,
    _collect_extended_pool_flows, _comparison_fit_spec, _decision_diversity, _edge_label_status, _features,
    _fit_calibration, _flow_ablation_metadata, _make_rows, _mandatory_market_items, _metrics,
    _prior_corrected_decisions, _prior_shift_decisions, _select_prior_half_life, _stance_dispersion,
    _trade_perf, direction_label, _freeze_volatility_labels,
    _fit_temperature, _band_z, _band_pct, _model, _training_rows_for, _selection_rows_for,
    _refit_anchor, _causal_prior_baseline, _probability_skill, BandRegressionModel,
    _fit_regime_gate, _regime_off,
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

    def test_causal_priors_only_use_labels_observable_before_each_signal(self):
        train=[{"label_vol":"UP","label_available_date":"2025-12-01"} for _ in range(40)]
        train+=[{"label_vol":"DOWN","label_available_date":"2026-05-10"} for _ in range(40)]
        early=[{"signal_date":"2026-04-01"}]
        late=[{"signal_date":"2026-06-01"}]
        early_prior=_causal_class_priors(train,early)[0]
        late_prior=_causal_class_priors(train,late)[0]
        self.assertAlmostEqual(early_prior[2],1.0)
        # A label maturing after the signal date (or on the same day) must not leak in.
        self.assertAlmostEqual(late_prior[0],40/80,places=6)
        same_day=[{"signal_date":"2026-05-10"}]
        self.assertAlmostEqual(_causal_class_priors(train,same_day)[0][2],1.0)
        self.assertEqual(PRIOR_SHRINKAGE_WINDOW_DAYS,250)

    def test_prior_mix_and_class_balanced_decision_are_bounded(self):
        probs=np.asarray([[.6,.3,.1],[.2,.5,.3],[.1,.2,.7]])
        priors=np.asarray([[.3,.4,.3]]*3)
        self.assertTrue(np.allclose(_apply_prior_mix(probs,priors,0.),probs))
        mixed=_apply_prior_mix(probs,priors,1.)
        self.assertTrue(np.allclose(mixed,priors))
        np.testing.assert_allclose(mixed.sum(axis=1),1.0)
        self.assertTrue(np.array_equal(_prior_corrected_decisions(probs,priors,0.),probs.argmax(axis=1)))
        # A rare class the prior does not expect can win once tau > 0.
        skewed=np.asarray([[.36,.34,.30]])
        rare=np.asarray([[.40,.40,.20]])
        self.assertEqual(int(_prior_corrected_decisions(skewed,rare,0.)[0]),0)
        # Dividing by the causal prior lifts the class the prior does not expect.
        self.assertEqual(int(_prior_corrected_decisions(skewed,rare,1.)[0]),2)

    def test_calibration_is_fitted_on_development_and_reports_its_scope(self):
        rng=np.random.default_rng(11); rows=[];probs=[]
        names=("DOWN","SIDEWAYS","UP")
        for i in range(600):
            truth=names[i%3];rows.append({"label_vol":truth})
            sharp=np.zeros(3);sharp[i%3]=.8;sharp[(i+1)%3]=.15;sharp[(i+2)%3]=.05
            probs.append(sharp)
        priors=np.tile([1/3,1/3,1/3],(600,1))
        report=_fit_calibration(rows,np.asarray(probs),priors)
        self.assertEqual(report["fit_n"],600)
        # The prior mixture no longer feeds the probability; only the reference is reported.
        self.assertEqual(report["prior_mix"],0.)
        self.assertGreaterEqual(report["reference_prior_mix"],0.)
        self.assertLessEqual(report["reference_prior_mix"],0.5)
        self.assertIn(report["decision_tau"],DECISION_TAU_GRID)
        self.assertEqual(report["prior_window_days"],PRIOR_SHRINKAGE_WINDOW_DAYS)
        self.assertIn("development_brier_reference_prior_mix",report)
        self.assertIn("development_brier_temperature_only",report)
        self.assertEqual(report["development_decision_diversity"]["collapse_status"],"DISTRIBUTED")
        self.assertIn("temperature_log_bounds",report)

    def test_winsor_and_normalised_band_models_use_the_frozen_band(self):
        rng=np.random.default_rng(3);rows=[]
        for i in range(500):
            x=float(rng.normal());z=1.2*x+float(rng.normal())*.8
            rows.append({"x":x,"z":z})
        X=np.asarray([[r["x"]] for r in rows]);z=np.asarray([r["z"] for r in rows])
        for kind in ("band_winsor","band_z"):
            model=_model(kind).fit(X,z)
            probs=model.predict_proba(np.asarray([[-2.],[0.],[2.]]),np.asarray([1.6,1.6,1.6]))
            self.assertEqual(probs.shape,(3,3))
            np.testing.assert_allclose(probs.sum(axis=1),1.0)
            self.assertGreater(probs[2,2],probs[0,2])
        self.assertIn("band_winsor",MODEL_KINDS);self.assertIn("band_z",MODEL_KINDS)
        self.assertAlmostEqual(_band_z({"label_vol_band_pct":3.0,"volatility_unit_pct":1.5}),2.0)
        self.assertEqual(_band_pct({"label_vol_band_pct":-3.0}),3.0)

    def test_development_selection_requires_beating_the_causal_prior(self):
        def result(accuracy,ic=.02,shares=(30.,30.,40.)):
            return {"metrics":{"datewise_spearman_mean":ic,
                "decision":{"accuracy_pct":accuracy,
                    "predicted_share_pct":{"DOWN":shares[0],"SIDEWAYS":shares[1],"UP":shares[2]}},
                "probability_skill":{"causal_prior_accuracy_pct":36.0}}}
        self.assertTrue(_beats_causal_prior_on_development(result(37.0)))
        self.assertFalse(_beats_causal_prior_on_development(result(35.9)))
        self.assertFalse(_beats_causal_prior_on_development(result(40.0,ic=-.01)))
        # A one-class development decision is not a classifier, whatever its accuracy.
        self.assertFalse(_beats_causal_prior_on_development(result(40.0,shares=(100.,0.,0.))))
        self.assertFalse(_beats_causal_prior_on_development(result(40.0,shares=(0.5,0.,99.5))))

    def test_three_way_report_exposes_causal_prior_and_plain_argmax(self):
        from audit_63_fast import _three_way_report
        samples=[]
        for i in range(30):
            label="DOWN" if i<15 else "UP"
            samples.append({"label":label,"prediction":"UP" if i<10 else ("DOWN" if i<15 else "UP"),
                "argmax_prediction":"UP",
                "probabilities":{"UP":.5,"DOWN":.3,"SIDEWAYS":.2},
                "causal_prior":{"UP":.3,"DOWN":.4,"SIDEWAYS":.3}})
        report=_three_way_report(samples)
        self.assertIn("causal_prior_brier",report)
        self.assertIn("brier_skill_vs_causal_prior",report)
        self.assertEqual(report["causal_prior_accuracy_pct"],50.0)
        self.assertEqual(report["plain_argmax_decision"]["predicted_counts"]["UP"],30)
        self.assertIn(report["causal_probability_edge_status"],
                      ("PROBABILITY_SKILL_PRESENT_VS_CAUSAL_PRIOR","NO_DEMONSTRATED_EDGE_WORSE_THAN_CAUSAL_PRIOR"))


class CalibrationDecisionFixTests(unittest.TestCase):
    """The decision stage must express the model's ranking instead of a class pin."""

    def test_out_of_fold_sigma_is_a_forecast_error_not_an_in_sample_fit(self):
        rng=np.random.default_rng(5);rows=[]
        for _ in range(1200):
            x=float(rng.normal());rows.append((x,2.0*x+float(rng.normal())*2.0))
        X=np.asarray([[r[0]] for r in rows]);y=np.asarray([r[1] for r in rows])
        model=BandRegressionModel().fit(X,y)
        # In-sample residuals of a 2-parameter fit understate a 2.0 noise scale.
        in_sample=float(np.std(y-model.routine.predict(X)))
        self.assertGreater(model.sigma,in_sample)
        self.assertTrue(model.sigma_source.startswith("out_of_fold"))
        self.assertLess(abs(model.sigma-2.0),0.35)
        probs=model.predict_proba(np.asarray([[0.],[0.],[0.]]),np.asarray([2.,2.,2.]))
        self.assertTrue(np.all(np.abs(probs-.5)<.45))
        self.assertGreater(probs[0,1],0.1)

    def test_temperature_reaches_a_softer_optimum_than_the_old_4_482_bound(self):
        rng=np.random.default_rng(3);rows=[];probs=[]
        names=("DOWN","SIDEWAYS","UP")
        # Labels drawn from probabilities the model overstates by a factor of 10;
        # the development optimum therefore sits well above the old bound e^1.5.
        z=rng.normal(0,1.6,(1500,3))
        p=np.exp(z)/np.exp(z).sum(1,keepdims=True)
        p_true=np.exp(z/10.)/np.exp(z/10.).sum(1,keepdims=True)
        for i in range(len(z)):
            rows.append({"label_vol":names[int(rng.choice(3,p=p_true[i]))]})
            probs.append(p[i])
        probs=np.asarray(probs)
        report=_fit_temperature(rows,probs)
        self.assertGreater(report["temperature"],4.482)
        self.assertLess(report["temperature"],54.)
        self.assertFalse(report["temperature_at_bound"])
        self.assertEqual(list(report["temperature_log_bounds"]),list(TEMPERATURE_LOG_BOUNDS))
        y=np.asarray([names.index(r["label_vol"]) for r in rows])
        self.assertLess(report["development_oof_logloss_after"],
                        float(log_loss(y,_apply_temperature(probs,4.482),labels=[0,1,2])))

    def test_label_shift_rule_moves_with_the_class_mix_and_keeps_row_differences(self):
        probs=np.asarray([[.40,.30,.30],[.45,.25,.30],[.30,.30,.40],[.50,.20,.30]])
        train=np.asarray([.20,.30,.50])
        # The market's class mix turned DOWN while the model still leans UP.
        now=np.tile([.55,.30,.15],(4,1))
        plain=_prior_shift_decisions(probs,now,train,0.)
        shifted=_prior_shift_decisions(probs,now,train,1.)
        self.assertTrue(np.array_equal(plain,probs.argmax(axis=1)))
        self.assertTrue(np.all(shifted==0))
        # A row the model is genuinely bullish on can still be chosen.
        strong=np.asarray([[.02,.02,.96]]);now_one=np.asarray([[.55,.30,.15]])
        self.assertEqual(int(_prior_shift_decisions(strong,now_one,train,1.)[0]),2)
        # Dividing by the canonical prior alone (PR #18 rule) is a different rule.
        self.assertFalse(np.array_equal(shifted,_prior_corrected_decisions(probs,now,1.)))

    def test_decision_diversity_labels_pins_dominance_and_spread(self):
        pinned=_decision_diversity(np.zeros(100,dtype=int))
        self.assertEqual(pinned["collapse_status"],"ONE_CLASS_PINNED")
        self.assertEqual(pinned["distinct_classes"],1)
        dominant=_decision_diversity(np.asarray([0]*95+[1]*5))
        self.assertEqual(dominant["collapse_status"],"ONE_CLASS_DOMINANT")
        spread=_decision_diversity(np.asarray([0]*40+[1]*30+[2]*30))
        self.assertEqual(spread["collapse_status"],"DISTRIBUTED")
        self.assertEqual(spread["distinct_classes"],3)
        self.assertGreater(spread["prediction_entropy_nats"],0)

    def test_half_life_prior_follows_a_regime_turn_without_using_unmatured_labels(self):
        train=[]
        for i in range(120):
            train.append({"label_vol":"UP","label_available_date":"2025-12-01"})
        for i in range(40):
            train.append({"label_vol":"DOWN","label_available_date":"2026-03-20"})
        row=[{"signal_date":"2026-04-01"}]
        flat=_causal_class_priors(train,row)[0]
        short=_causal_class_priors(train,row,half_life_days=40.)[0]
        self.assertGreater(flat[2],.6)
        self.assertGreater(short[0],flat[0])
        self.assertGreater(short[0],.25)
        # A label maturing the same day as the signal must never be counted.
        leaky=train+[{"label_vol":"UP","label_available_date":"2026-04-01"}]*200
        self.assertAlmostEqual(float(_causal_class_priors(leaky,row,half_life_days=40.)[0][0]),
                               float(short[0]),places=9)

    def test_half_life_is_chosen_on_development_prior_log_loss_only(self):
        train=[]
        for i in range(200):
            train.append({"label_vol":"UP","label_available_date":"2025-01-01"})
        for i in range(200):
            train.append({"label_vol":"DOWN","label_available_date":"2026-03-20"})
        dev=[{"signal_date":"2026-04-01","label_vol":"DOWN"} for _ in range(50)]
        dev+=[{"signal_date":"2026-04-01","label_vol":"UP"} for _ in range(10)]
        half_life,report=_select_prior_half_life(train,dev)
        self.assertIn(half_life,tuple(PRIOR_HALF_LIFE_GRID)+(None,))
        table=report["candidates"]
        self.assertEqual(set(table),{"flat_250d"}|{f"half_life_{int(h)}d" for h in PRIOR_HALF_LIFE_GRID})
        best=min(table.values(),key=lambda v:v["prior_log_loss_nats"])
        self.assertAlmostEqual(best["prior_log_loss_nats"],
            table["flat_250d" if half_life is None else f"half_life_{int(half_life)}d"]["prior_log_loss_nats"],places=12)
        self.assertLess(half_life or 1e9,250.)

    def test_metrics_score_the_supplied_decision_not_the_probability_argmax(self):
        rows=[{"label_vol":label,"net_pct":net,"signal_date":"2025-01-01"}
              for label,net in (("DOWN",-1.),("DOWN",-2.),("UP",1.))]
        probs=np.asarray([[.1,.1,.8]]*3)
        report=_metrics(rows,probs,label_key="label_vol",predictions=np.asarray([0,0,2]))
        self.assertEqual(report["accuracy_pct"],100.)
        self.assertEqual(report["confusion_matrix"]["counts"],[[2,0,0],[0,0,0],[0,0,1]])
        argmax_only=_metrics(rows,probs,label_key="label_vol")
        self.assertAlmostEqual(argmax_only["accuracy_pct"],100/3)

    def test_development_calibration_is_shared_and_future_rows_never_leak(self):
        import warnings
        from expanded_direction_audit import _backtest_model
        warnings.simplefilter("ignore")
        days=pd.bdate_range("2020-01-02",periods=1250)
        def build(cohort,future=False):
            rng=np.random.default_rng(21);rows=[]
            span=list(days)+([pd.Timestamp("2026-02-02")+pd.Timedelta(days=3*i) for i in range(60)] if future else [])
            for i,day in enumerate(span):
                for ticker in ("A","B"):
                    entry=pd.Timestamp(day)+pd.Timedelta(days=1)
                    exit_=pd.Timestamp(day)+pd.Timedelta(days=30)
                    ret=float(rng.normal(0,4))+(2.0 if ticker=="A" else -2.0)*float(np.sin(i/9))
                    label="UP" if ret>.5 else "DOWN" if ret< -.5 else "SIDEWAYS"
                    rows.append({"cohort":cohort,"ticker":ticker,"market":"KR",
                        "signal_date":str(day.date()),"entry_date":str(entry.date()),
                        "exit_date":str(exit_.date()),"label_available_date":str(exit_.date()),
                        "horizon":20,"net_pct":ret,"label_vol":label,"label_0.3":label,"label_0.5":label,"label_1.0":label,
                        "label_vol_band_pct":1.5,"volatility_unit_pct":1.0,
                        "target_touch_theta_0.5":"TARGET_FIRST" if ret>0 else "STOP_FIRST",
                        "ret20":float(np.sin(i/7))*3,"rsi":50+float(np.cos(i/5))*10})
            if future:
                for row in rows:
                    if row["signal_date"]>="2026-01-02":row["label_vol"]="UP"
            return rows
        out={}
        for cohort in ("KOSPI","KOSDAQ"):
            rows=build(cohort)
            dev=_backtest_model(rows,rows,"multinomial_logistic",20,"development")
            # Adding labels that only mature after the holdout must not move any
            # development-fitted parameter or the causal priors used there.
            augmented=build(cohort,future=True)
            other=_backtest_model(augmented,augmented,"multinomial_logistic",20,"development")
            # Identical up to floating-point noise in the decay-weight scaling.
            self.assertAlmostEqual(dev["temperature"]["temperature"],other["temperature"]["temperature"],places=6)
            self.assertAlmostEqual(dev["metrics"]["decision"]["accuracy_pct"],
                                   other["metrics"]["decision"]["accuracy_pct"],places=6)
            metrics=dev["metrics"]
            for key in ("decision","raw_argmax_decision","plain_argmax_decision","prior_divide_decision",
                        "probability_dispersion","probability_skill"):
                self.assertIn(key,metrics,cohort)
            self.assertIn(metrics["decision"]["collapse_status"],
                          ("DISTRIBUTED","ONE_CLASS_DOMINANT","ONE_CLASS_PINNED"))
            for part in ("raw","calibrated"):
                self.assertIn(part,metrics["probability_dispersion"])
            holdout=_backtest_model(rows,rows,"multinomial_logistic",20,
                "validation",temperature_info=dev["temperature"],payoffs=dev["class_payoffs_from_development"])
            self.assertEqual(holdout["temperature"],dev["temperature"])
            out[cohort]=dev["n"]
        self.assertTrue(all(n>0 for n in out.values()))


class RegimeTrendGateTests(unittest.TestCase):
    """The trend gate must be causal, development-fitted, and decision-only."""

    @staticmethod
    def _row(label="UP",gap=-8.,ret20=-5.,drawdown=None,**extra):
        row={"label_vol":label,"signal_date":"2026-05-04",
             "market_index_ma200_gap_pct":gap,"market_index_return_20d_pct":ret20}
        if drawdown is not None:row["market_drawdown_120_pct"]=drawdown
        row.update(extra)
        return row

    @staticmethod
    def _params(target="DOWN"):
        return {"mode":"trend_override","ma200_gap_max_pct":-3.,
                "drawdown_120_min_pct":-8.,"index_return_20d_max_pct":-2.,
                "target":target}

    def test_gate_moves_only_up_decisions_and_only_inside_a_risk_off_trend(self):
        params=self._params()
        rows=[self._row("DOWN",-8.,-5.),          # below trend: UP -> DOWN
              self._row("UP",5.,3.),              # above trend: kept
              self._row("DOWN",-8.,None),        # no 20-day value: no state
              self._row("DOWN",None,-5.),        # no 200-day value, no drawdown: no state
              self._row("SIDEWAYS",-9.,-6.),      # not an UP decision: kept
              self._row("SIDEWAYS",-1.,-1.)]      # above the fitted threshold: kept
        probs=np.asarray([[.1,.2,.7]]*len(rows))
        pred=np.asarray([2,2,2,2,1,2])
        before=pred.copy()
        gated,active,override=_apply_regime_gate(rows,probs,pred,params)
        self.assertEqual(gated.tolist(),[0,2,2,2,1,2])
        self.assertEqual(active.tolist(),[True,False,False,False,True,False])
        self.assertEqual(override.tolist(),[True,False,False,False,False,False])
        self.assertEqual(pred.tolist(),before.tolist(),"input decisions must not be mutated")
        self.assertEqual(probs.tolist(),[[.1,.2,.7]]*6,"probabilities must not be touched")

    def test_drawdown_clause_fires_when_the_index_is_above_its_200_day_line(self):
        # Index still above its own 200-day mean but 12% off its 120-day high and
        # falling over 20 days: that is exactly the failing breadth case.
        params=self._params()
        row=self._row("DOWN",3.,-3.,drawdown=-12.)
        self.assertTrue(_regime_off(row,params))
        self.assertFalse(_regime_off({**row,"market_drawdown_120_pct":-2.,
                                     "market_index_ma200_gap_pct":3.},params))
        # A weak 20-day return alone is not enough: the longer trend must break.
        self.assertFalse(_regime_off({"market_index_return_20d_pct":-6.,
            "market_index_ma200_gap_pct":8.,"market_drawdown_120_pct":-1.},params))

    def test_gate_target_can_be_sideways_or_the_second_choice(self):
        rows=[self._row("DOWN",-8.,-5.,drawdown=-10.)]
        probs=np.asarray([[.2,.6,.2]])
        pred=np.asarray([2])
        for target,expected in (("DOWN",0),("SIDEWAYS",1),("SECOND_CHOICE",1)):
            gated,_,_=_apply_regime_gate(rows,probs,pred,self._params(target))
            self.assertEqual(int(gated[0]),expected,target)
        # Down-leaning second choice picks DOWN.
        gated,_,_=_apply_regime_gate(rows,np.asarray([[.6,.2,.2]]),pred,
            self._params("SECOND_CHOICE"))
        self.assertEqual(int(gated[0]),0)

    def test_instrument_state_uses_the_instruments_own_trend(self):
        params={**self._params(),"state":"instrument"}
        weak={"label_vol":"UP","ma200_gap":-5.,"ret20":-4.}
        self.assertTrue(_regime_off(weak,params))
        self.assertFalse(_regime_off({**weak,"ma200_gap":5.},params))
        self.assertFalse(_regime_off({**weak,"ret20":3.},params))
        # A weak index alone is not an instrument-level trend break.
        self.assertFalse(_regime_off({"market_index_return_20d_pct":-6.,
            "market_index_ma200_gap_pct":-9.,"market_drawdown_120_pct":-12.},params))

    def test_index_and_instrument_state_also_needs_the_instrument_below_its_200_day_mean(self):
        params={**self._params(),"state":"index_and_instrument"}
        base={"market_index_return_20d_pct":-4.,"market_index_ma200_gap_pct":-6.}
        self.assertTrue(_regime_off({**base,"ma200_gap":-2.},params))
        self.assertFalse(_regime_off({**base,"ma200_gap":2.},params))
        self.assertFalse(_regime_off({**base},params))

    def test_gate_off_leaves_every_decision_unchanged(self):
        rows=[self._row("DOWN",-9.,-9.),self._row("UP",-9.,-9.)]
        probs=np.asarray([[.1,.1,.8],[.8,.1,.1]])
        pred=np.asarray([2,0])
        gated,active,override=_apply_regime_gate(rows,probs,pred,{"mode":"none"})
        self.assertEqual(gated.tolist(),[2,0])
        self.assertFalse(active.any())
        self.assertFalse(override.any())

    def test_gate_state_never_reads_the_label_probabilities_or_other_rows(self):
        params=self._params()
        base=self._row("UP",-8.,-5.,drawdown=-4.)
        for label in ("UP","DOWN","SIDEWAYS"):
            self.assertTrue(_regime_off({**base,"label_vol":label},params),label)
        # Unrelated future-looking or payoff fields cannot change the state.
        self.assertEqual(_regime_off(base,params),
            _regime_off({**base,"net_pct":99.,"label_available_date":"2027-01-01",
                         "forward_return_pct":12.},params))
        self.assertFalse(_regime_off(self._row("UP",float("nan"),-5.),params))
        self.assertFalse(_regime_off({},params))

    def test_gate_thresholds_are_fitted_on_development_rows_only(self):
        rng=np.random.default_rng(11);rows=[];probs=[];pred=[]
        for _ in range(2000):
            gap=float(rng.uniform(-12.,8.));ret=float(rng.uniform(-8.,8.))
            risk=gap<-2. and ret<-2.
            rows.append({"label_vol":"DOWN" if risk else "UP","signal_date":"2024-03-04",
                         "market_index_ma200_gap_pct":gap,"market_index_return_20d_pct":ret})
            probs.append([.05,.05,.9]);pred.append(2)
        fit=_fit_regime_gate(rows,np.asarray(probs),np.asarray(pred))
        chosen=fit["chosen"]
        self.assertEqual(chosen["mode"],"trend_override")
        self.assertEqual(chosen["target"],"DOWN")
        self.assertAlmostEqual(chosen["ma200_gap_max_pct"],-2.)
        self.assertAlmostEqual(chosen["index_return_20d_max_pct"],-2.)
        # Without a drawdown column only the 200-day clause can fire.
        self.assertAlmostEqual(chosen["drawdown_120_min_pct"],REGIME_GATE_DRAWDOWN_GRID[0])
        self.assertGreaterEqual(chosen["development_accuracy_gain_pp"],REGIME_GATE_MIN_ACCURACY_GAIN_PP)
        self.assertGreaterEqual(chosen["development_overrides_n"],REGIME_GATE_MIN_DEV_OVERRIDES)
        self.assertEqual(fit["selection"]["grid"]["state"],list(REGIME_GATE_STATES))
        self.assertEqual(fit["selection"]["grid"]["ma200_gap_max_pct"],list(REGIME_GATE_MA200_GAP_GRID))
        self.assertEqual(fit["selection"]["grid"]["index_return_20d_max_pct"],list(REGIME_GATE_INDEX_RET20_GRID))
        self.assertEqual(fit["selection"]["grid"]["drawdown_120_min_pct"],list(REGIME_GATE_DRAWDOWN_GRID))
        self.assertEqual(fit["selection"]["gate_features"],list(REGIME_GATE_FEATURES))
        # Every reported candidate is a development-only evaluation.
        self.assertTrue(fit["selection"]["candidates"])
        self.assertTrue(all(c["overrides_n"]>0 for c in fit["selection"]["candidates"]))

    def test_no_development_benefit_keeps_the_gate_off(self):
        # A perfect development decision cannot be improved by overriding it, so
        # no candidate can pass the gain guard and the gate must stay off.
        rows=[{"label_vol":"UP","signal_date":"2024-03-04",
               "market_index_ma200_gap_pct":-9.,"market_index_return_20d_pct":-6.}]*600
        rows+=[{"label_vol":"DOWN","signal_date":"2024-03-04",
                "market_index_ma200_gap_pct":6.,"market_index_return_20d_pct":4.}]*600
        probs=np.asarray([[.05,.05,.9]]*600+[[.9,.05,.05]]*600)
        pred=np.asarray([2]*600+[0]*600)
        fit=_fit_regime_gate(rows,probs,pred)
        self.assertEqual(fit["chosen"]["mode"],"none")
        self.assertEqual(fit["selection"]["status"],"NO_DEVELOPMENT_GAIN")
        gated,_,override=_apply_regime_gate(rows,probs,pred,fit["chosen"])
        self.assertTrue(np.array_equal(gated,pred))
        self.assertEqual(int(override.sum()),0)

    def test_gate_pins_are_rejected_by_the_macro_f1_guard(self):
        # Labels are UP 55% of the time and DOWN 45%; always-downgrade inside a
        # risk-off day block would raise raw accuracy only by pinning everything.
        rows=[];probs=[];pred=[]
        for i in range(1000):
            risk=i%10<5
            label="UP" if i%20<11 else "DOWN"
            rows.append({"label_vol":label,"signal_date":"2024-03-04",
                         "market_index_ma200_gap_pct":(-9. if risk else 6.),
                         "market_index_return_20d_pct":(-6. if risk else 4.)})
            probs.append([.3,.2,.5]);pred.append(2)
        fit=_fit_regime_gate(rows,np.asarray(probs),np.asarray(pred))
        for candidate in fit["selection"]["candidates"]:
            if candidate["distinct_classes"]<2:
                self.assertNotEqual(fit["chosen"].get("target"),candidate["target"])
        self.assertLessEqual(REGIME_GATE_MAX_MACRO_F1_DROP_PP,.5)

    def test_development_gate_is_reused_unchanged_for_validation(self):
        import warnings
        from expanded_direction_audit import _backtest_model
        warnings.simplefilter("ignore")
        days=pd.bdate_range("2020-01-02",periods=1250)
        def build(future=False):
            rng=np.random.default_rng(31);rows=[]
            span=list(days)+list(pd.bdate_range("2025-11-10","2026-03-05",freq="5D"))+\
                ([pd.Timestamp("2026-02-02")+pd.Timedelta(days=3*i) for i in range(60)] if future else [])
            for i,day in enumerate(span):
                gap=float(9*np.sin(i/45.));ret20=float(5*np.cos(i/31.))
                risk=gap<-2. and ret20<-1.
                for ticker in ("A","B"):
                    entry=pd.Timestamp(day)+pd.Timedelta(days=1)
                    exit_=pd.Timestamp(day)+pd.Timedelta(days=30)
                    ret=float(rng.normal(0,2.0))+(-3.2 if risk else .7)
                    label="UP" if ret>.8 else "DOWN" if ret< -.8 else "SIDEWAYS"
                    rows.append({"cohort":"KOSPI","ticker":ticker,"market":"KR",
                        "signal_date":str(day.date()),"entry_date":str(entry.date()),
                        "exit_date":str(exit_.date()),"label_available_date":str(exit_.date()),
                        "horizon":20,"net_pct":ret,"label_vol":label,"label_0.3":label,
                        "label_0.5":label,"label_1.0":label,"label_vol_band_pct":1.5,
                        "volatility_unit_pct":1.0,
                        "target_touch_theta_0.5":"TARGET_FIRST" if ret>0 else "STOP_FIRST",
                        "ret20":float(np.sin(i/7))*3,"rsi":50+float(np.cos(i/5))*10,
                        "market_index_ma200_gap_pct":gap,"market_index_return_20d_pct":ret20})
            if future:
                for row in rows:
                    if row["signal_date"]>="2026-01-02":row["label_vol"]="UP"
            return rows
        rows=build()
        dev=_backtest_model(rows,rows,"multinomial_logistic",20,"development",include_samples=True)
        augmented=build(future=True)
        other=_backtest_model(augmented,augmented,"multinomial_logistic",20,"development")
        self.assertEqual(dev["temperature"]["regime_gate"],other["temperature"]["regime_gate"])
        self.assertEqual(dev["temperature"]["regime_gate_selection"]["scope"],
            "development OOF decisions only; validation and diagnostic never enter this fit")
        gate=dev["temperature"]["regime_gate"]
        self.assertIn(gate["mode"],("trend_override","none"))
        for key in ("regime_gate","regime_gate_decision"):
            self.assertIn(key,dev["metrics"])
        self.assertEqual(dev["metrics"]["regime_gate"]["params"],gate)
        self.assertEqual(dev["metrics"]["regime_gate_decision"]["n"],dev["n"])
        summary=dev["gated_decision_summary"]
        self.assertEqual(sum(summary["predicted_counts"].values()),dev["n"])
        sample=dev["evaluation_samples"][0]
        for key in ("gated_prediction","regime_gate_active","regime_gate_override",
                    "market_index_ma200_gap_pct","market_index_return_20d_pct"):
            self.assertIn(key,sample)
        holdout=_backtest_model(rows,rows,"multinomial_logistic",20,"validation",
            temperature_info=dev["temperature"],payoffs=dev["class_payoffs_from_development"],
            include_samples=True)
        self.assertEqual(holdout["temperature"]["regime_gate"],gate)
        self.assertEqual(holdout["metrics"]["regime_gate"]["params"],gate)
        self.assertEqual(holdout["metrics"]["regime_gate"]["scope"],
            "fitted parameters reused unchanged from development")


if __name__ == "__main__":
    unittest.main()
