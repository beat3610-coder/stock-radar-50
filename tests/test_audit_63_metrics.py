import unittest
import numpy as np
import pandas as pd

from audit_63_fast import (_calibration,_equal_weight_replay,_overlap_report,
                           _auc,_development_legacy_coefficient,_policy_results,_policy_thresholds,
                           _select_config,_select_policy,_select_r_policy,_rank_metrics,_decorate_features,
                           _target_variant_choices,_legacy_policy_rows,_three_way_report)

def trade(ticker,entry,exit_date,net,prob=.4,ev=-1.0):
    return {"ticker":ticker,"signal_date":entry,"entry_date":entry,"exit_date":exit_date,
            "exit_price":100.0,"entry":100.0,"stop":95.0,"take":108.0,
            "net_pct":net,"model_probability":prob,"model_expected_net_pct":ev,
            "model_predicted_win_pct":100*prob,
            "score":50.0,"strategy":"x","candidate_family":"broad"}

class Audit63MetricsTests(unittest.TestCase):
    def test_target_variant_selection_handles_empty_cohort(self):
        self.assertEqual(_target_variant_choices({"pooled":{"selected_dev_metrics":None}}),[])
        choices=_target_variant_choices({"pooled":{"selected_dev_metrics":{"datewise_spearman_mean":.1}},
                                         "date_cs":{"selected_dev_metrics":None}})
        self.assertEqual(choices,[(.1,"pooled")])

    def test_legacy_model_and_regime_gates_are_both_applied(self):
        base="2025-01-02"
        legacy=[trade("A",base,"2025-01-10",1),trade("B",base,"2025-01-10",2)]
        legacy[0]["market_regime"]="RISK_ON";legacy[1]["market_regime"]="RISK_OFF"
        scores=[{**trade(t["ticker"],base,"2025-01-10",1),"model_predicted_r":.8}
                for t in legacy]
        scores.append({**trade("C",base,"2025-01-10",1),"model_predicted_r":.2})
        chosen=_legacy_policy_rows(legacy,scores,"legacy_model_filter_and_regime_gate",.5)
        self.assertEqual([r["ticker"] for r in chosen],["A"])

    def test_model_payoff_features_do_not_use_next_open(self):
        dates=pd.bdate_range("2025-01-01",periods=230)
        frame=pd.DataFrame({"Close":np.full(230,100.),"Volume":np.full(230,1000.)},index=dates)
        computed=pd.DataFrame({"MA200":np.full(230,95.)},index=dates)
        base={"ticker":"T","signal_date":str(dates[-1].date()),"stop":95.,"take":108.,
              "entry_net_rr":0.2,"entry_net_upside_pct":1.,"entry_net_downside_pct":8.}
        alt={**base,"entry":115.,"entry_net_rr":8.,"entry_net_upside_pct":40.,"entry_net_downside_pct":1.}
        first=_decorate_features(base.copy(),frame,computed,{},{});second=_decorate_features(alt.copy(),frame,computed,{},{})
        for key in ("entry_net_rr","entry_net_upside_pct","entry_net_downside_pct"):
            self.assertEqual(first[key],second[key])

    def test_reliability_and_ece_are_reported(self):
        rows=[{"model_probability":.2,"net_pct":-1},{"model_probability":.8,"net_pct":1}]
        result=_calibration(rows)
        self.assertEqual(result["n"],2)
        self.assertEqual(result["ece"],0.2)
        self.assertEqual(len(result["reliability_bins"]),10)

    def test_no_development_edge_fails_closed(self):
        rows=[trade(f"T{i}",f"2025-01-{i+1:02d}",f"2025-01-{i+2:02d}",-1) for i in range(12)]
        thresholds=_policy_thresholds(rows)
        policy,_,status=_select_policy(rows,thresholds,rows)
        self.assertIsNone(policy)
        self.assertEqual(status,"NO_DEVELOPMENT_EDGE")

    def test_configuration_selection_uses_development_auroc_only(self):
        good=[];bad=[]
        for i in range(12):
            win=i%2==0
            base={"signal_date":"2025-01-02","entry_date":"2025-01-02","exit_date":"2025-01-03",
                  "net_pct":1 if win else -1}
            good.append({**base,"model_probability":0.9 if win else 0.1})
            bad.append({**base,"model_probability":0.1 if win else 0.9})
        # Validation is deliberately reversed and must not alter config choice.
        good += [{"signal_date":"2025-12-02","entry_date":"2025-12-02","exit_date":"2025-12-03",
                  "model_probability":0.1,"net_pct":1}]
        bad += [{"signal_date":"2025-12-02","entry_date":"2025-12-02","exit_date":"2025-12-03",
                 "model_probability":0.9,"net_pct":1}]
        chosen,_,_=_select_config({"good":good,"bad":bad})
        self.assertEqual(chosen,"good")

    def test_auroc_and_ev_quantile_gates_have_selection(self):
        self.assertEqual(_auc([{"model_probability":.9,"net_pct":1},
                               {"model_probability":.1,"net_pct":-1}])["auroc"],1.0)
        rows=[trade(f"T{i}",f"2025-01-{i+1:02d}",f"2025-01-{i+2:02d}",1,ev=float(i)) for i in range(20)]
        thresholds=_policy_thresholds(rows)
        self.assertEqual(set(thresholds),{"ev_p70","ev_p80","ev_p90"})
        diag=[{**r,"signal_date":"2026-05-01","entry_date":"2026-05-01","exit_date":"2026-05-02"} for r in rows]
        result=_policy_results(diag,thresholds,"diagnostic")
        self.assertEqual(result["ev_p70"]["by_family"]["broad"]["gate_passed_candidates"],
                         result["ev_p70"]["gate_passed_candidates"])
        rates=[result[k]["acceptance_rate_pct"] for k in ("ev_p70","ev_p80","ev_p90")]
        self.assertGreater(rates[0],rates[1]);self.assertGreater(rates[1],rates[2])
        with_oos_extreme=rows+[{**diag[0],"model_expected_net_pct":1000}]
        self.assertEqual(_policy_thresholds(with_oos_extreme),thresholds)

    def test_legacy_coefficient_uses_chronological_rolling_window(self):
        def candidate(i,entry,family,win):
            return {"ticker":f"T{i}","signal_date":entry,"entry_date":entry,
                    "exit_date":str(pd.Timestamp(entry)+pd.Timedelta(days=1))[:10],
                    "net_pct":1.0 if win else -1.0,"candidate_legacy":float(family),
                    "signal_rsi":50,"signal_adx":22,"signal_atr_pct":2,
                    "signal_ma20_ma60_pct":1,"signal_volume_ratio":1,
                    "signal_5d_return_pct":0,"entry_gap_pct":0,
                    "entry_net_rr":1.5,"entry_net_upside_pct":4,
                    "entry_net_downside_pct":4,"signal_stop_atr":1.8,
                    "signal_trend_200_pct":1,"signal_macdh_pct":0,
                    "signal_drawdown_20_pct":-2,"signal_rsi_5d_change":0,
                    "signal_vol20_60_ratio":1}
        broad_dates=pd.date_range("2024-08-01",periods=240,freq="D")
        legacy_dates=pd.date_range("2025-05-01",periods=60,freq="D")
        broad=[candidate(i,str(d.date()),0,i%2==0) for i,d in enumerate(broad_dates)]
        legacy=[candidate(1000+i,str(d.date()),1,True) for i,d in enumerate(legacy_dates)]
        report=_development_legacy_coefficient(legacy+broad,"rolling240_full_shrink50")
        self.assertEqual(report["training_candidates"],240)
        self.assertGreater(report["standardized_log_odds_coefficient"],0)

    def test_paired_overlap_decomposes_signals(self):
        legacy=[trade("A","2025-01-02","2025-01-04",-1),trade("B","2025-01-03","2025-01-05",1)]
        model=[trade("A","2025-01-02","2025-01-05",1),trade("C","2025-01-04","2025-01-06",2)]
        result=_overlap_report(legacy,model)
        self.assertEqual(result["legacy_only"]["n"],1)
        self.assertEqual(result["model_only"]["n"],1)
        self.assertEqual(result["both"]["n"],1)
        self.assertEqual(result["both"]["paired_mean_net_delta_pp"],2.0)

    def test_r_policy_rank_gate_blocks_nonpositive_development_ic_but_keeps_table(self):
        rows=[]
        for i in range(20):
            day=pd.Timestamp("2025-01-01")+pd.Timedelta(days=i*2)
            rows.append({**trade(f"T{i}",str(day.date()),str((day+pd.Timedelta(days=1)).date()),
                                  1.0 if i%2 else -0.2,ev=float(i)),
                         "signal_date":str(day.date()),"entry_date":str(day.date()),
                         "exit_date":str((day+pd.Timedelta(days=1)).date()),
                         "model_predicted_r":i/20,"realized_r":(1.0 if i%2 else -0.2)/2,
                         "entry_net_downside_pct":2.0,"strategy":"broad_market_ml"})
        cuts={"r_p70":{"global":0.05,"by_strategy":{}},
              "r_p80":{"global":0.10,"by_strategy":{}},
              "r_p90":{"global":0.15,"by_strategy":{}}}
        policy,table,status=_select_r_policy(rows,cuts,development_r_ic=-0.0459)
        self.assertIsNone(policy)
        self.assertEqual(status,"NO_DEVELOPMENT_EDGE")
        self.assertEqual(set(table),{"r_p70","r_p80","r_p90"})
        self.assertTrue(all("acceptance_rate_pct" in table[k] and "selected" in table[k]
                            for k in table))

    def test_rank_metrics_report_iid_standard_error_and_interval(self):
        rows=[{"realized_r":float(i),"model_predicted_r":float(i),
               "net_pct":float(i-5)} for i in range(10)]
        m=_rank_metrics(rows)
        self.assertEqual(m["n"],10)
        self.assertEqual(m["r_ic_standard_error_approx"],round(1/(9**0.5),6))
        self.assertEqual(len(m["r_ic_95_ci_approx"]),2)
        self.assertIn("dependence ignored",m["r_ic_uncertainty_method"])

    def test_portfolio_caps_at_five_concurrent_equal_slots(self):
        dates=pd.bdate_range("2026-04-01",periods=5)
        frames={f"T{i}":pd.DataFrame({"Close":[100,101,102,103,104]},index=dates) for i in range(6)}
        rows=[{**trade(f"T{i}",str(dates[1].date()),str(dates[3].date()),1,ev=1),
               "exit_price":103.0} for i in range(6)]
        result=_equal_weight_replay(rows,frames,"2026-04-01","2026-04-07",cap=5)
        self.assertEqual(result["trades_executed"],5)
        self.assertEqual(result["max_concurrent_positions"],5)
        self.assertIn("equal initial",result["method"])

    def test_three_way_report_shows_the_regime_gate_beside_the_ungated_decision(self):
        def sample(label,pred,gated,override=False):
            return {"label":label,"prediction":pred,"gated_prediction":gated,
                    "regime_gate_active":override,"regime_gate_override":override,
                    "probabilities":{"UP":.4,"DOWN":.3,"SIDEWAYS":.3},
                    "raw_probabilities":{"UP":.4,"DOWN":.3,"SIDEWAYS":.3},
                    "argmax_prediction":"UP","causal_prior":{"UP":.3,"DOWN":.4,"SIDEWAYS":.3}}
        samples=[sample("DOWN","UP","DOWN",True) for _ in range(6)]+\
                [sample("UP","UP","UP") for _ in range(4)]
        report=_three_way_report(samples)
        # Headline is the gated decision; the ungated one stays beside it.
        self.assertEqual(report["accuracy_pct"],100.)
        self.assertEqual(report["regime_gate_decision"]["predicted_counts"]["DOWN"],6)
        self.assertAlmostEqual(report["ungated_decision"]["accuracy_pct"],40.)
        self.assertAlmostEqual(report["accuracy_gate_minus_ungated_pp"],60.)
        self.assertEqual(report["regime_gate_overrides_n"],6)
        self.assertEqual(report["decision_collapse_status"],"DISTRIBUTED")
        # Samples without a gated column keep the previous report untouched.
        legacy=_three_way_report([{k:v for k,v in row.items()
            if k not in ("gated_prediction","regime_gate_active","regime_gate_override")}
            for row in samples])
        self.assertNotIn("regime_gate_decision",legacy)
        self.assertAlmostEqual(legacy["accuracy_pct"],40.)

if __name__=="__main__":unittest.main()
