import unittest
import numpy as np
import pandas as pd

from audit_63_fast import (_calibration,_equal_weight_replay,_overlap_report,
                           _auc,_development_legacy_coefficient,_policy_results,_policy_thresholds,
                           _select_config,_select_policy,_decorate_features)

def trade(ticker,entry,exit_date,net,prob=.4,ev=-1.0):
    return {"ticker":ticker,"signal_date":entry,"entry_date":entry,"exit_date":exit_date,
            "exit_price":100.0,"entry":100.0,"stop":95.0,"take":108.0,
            "net_pct":net,"model_probability":prob,"model_expected_net_pct":ev,
            "model_predicted_win_pct":100*prob,
            "score":50.0,"strategy":"x","candidate_family":"broad"}

class Audit63MetricsTests(unittest.TestCase):
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

    def test_portfolio_caps_at_five_concurrent_equal_slots(self):
        dates=pd.bdate_range("2026-04-01",periods=5)
        frames={f"T{i}":pd.DataFrame({"Close":[100,101,102,103,104]},index=dates) for i in range(6)}
        rows=[{**trade(f"T{i}",str(dates[1].date()),str(dates[3].date()),1,ev=1),
               "exit_price":103.0} for i in range(6)]
        result=_equal_weight_replay(rows,frames,"2026-04-01","2026-04-07",cap=5)
        self.assertEqual(result["trades_executed"],5)
        self.assertEqual(result["max_concurrent_positions"],5)
        self.assertIn("equal initial",result["method"])

if __name__=="__main__":unittest.main()
