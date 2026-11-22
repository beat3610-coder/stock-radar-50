import unittest
import pandas as pd

from audit_63_fast import (_calibration,_equal_weight_replay,_overlap_report,
                           _select_config,_select_policy)

def trade(ticker,entry,exit_date,net,prob=.4,ev=-1.0):
    return {"ticker":ticker,"signal_date":entry,"entry_date":entry,"exit_date":exit_date,
            "exit_price":100.0,"entry":100.0,"stop":95.0,"take":108.0,
            "net_pct":net,"model_probability":prob,"model_expected_net_pct":ev,
            "model_predicted_win_pct":100*prob,
            "score":50.0,"strategy":"x"}

class Audit63MetricsTests(unittest.TestCase):
    def test_reliability_and_ece_are_reported(self):
        rows=[{"model_probability":.2,"net_pct":-1},{"model_probability":.8,"net_pct":1}]
        result=_calibration(rows)
        self.assertEqual(result["n"],2)
        self.assertEqual(result["ece"],0.2)
        self.assertEqual(len(result["reliability_bins"]),10)

    def test_no_development_edge_fails_closed(self):
        rows=[trade(f"T{i}",f"2025-01-{i+1:02d}",f"2025-01-{i+2:02d}",-1) for i in range(12)]
        policy,_,status=_select_policy(rows)
        self.assertIsNone(policy)
        self.assertEqual(status,"NO_DEVELOPMENT_EDGE")

    def test_configuration_selection_uses_only_development_brier(self):
        good=[{"signal_date":"2025-01-02","entry_date":"2025-01-02","exit_date":"2025-01-03",
               "model_probability":0.9,"net_pct":1} for _ in range(12)]
        bad=[{"signal_date":"2025-01-02","entry_date":"2025-01-02","exit_date":"2025-01-03",
              "model_probability":0.1,"net_pct":1} for _ in range(12)]
        # Validation is deliberately reversed and must not alter config choice.
        good += [{"signal_date":"2025-11-02","entry_date":"2025-11-02","exit_date":"2025-11-03",
                  "model_probability":0.1,"net_pct":1}]
        bad += [{"signal_date":"2025-11-02","entry_date":"2025-11-02","exit_date":"2025-11-03",
                 "model_probability":0.9,"net_pct":1}]
        chosen,_,_=_select_config({"good":good,"bad":bad})
        self.assertEqual(chosen,"good")

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
