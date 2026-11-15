"""Prediction math, reproducibility and future-leak tests."""
import unittest
from predictive_model import vector, fit_probabilities,walkforward_model_filter,walkforward_model_policies,POLICIES,MIN_COMPLETED

def record(i,win,signal="2025-02-01",exit_date="2025-02-03",hot=False):
    return {"ticker":f"S{i}","entry_date":signal,"signal_date":signal,
            "exit_date":exit_date,"net_pct":3.0 if win else -2.5,
            "signal_rsi":63.0 if hot else 48.0,
            "signal_adx":35.0 if hot else 19.0,
            "signal_atr_pct":1.7 if hot else 4.5,
            "signal_ma20_ma60_pct":4.0 if hot else 0.2,
            "signal_volume_ratio":1.6 if hot else 0.8,
            "signal_5d_return_pct":3.5 if hot else -2.0,
            "entry_gap_pct":0.2 if hot else 1.0}

class ModelTest(unittest.TestCase):
    def history(self):
        return [record(i,i%3!=0,"2025-01-01","2025-01-04",hot=i%3!=0) for i in range(60)]
    def test_missing_inputs_rejected(self):
        self.assertIsNone(vector({"signal_rsi":60}))
        self.assertIsNone(fit_probabilities([],record(1,True)))
    def test_deterministic_and_bounded(self):
        history=self.history()
        p=fit_probabilities(history,record(999,False,hot=True))
        self.assertEqual(p,fit_probabilities(history,record(999,False,hot=True)))
        self.assertGreaterEqual(p["predicted_win_pct"],0)
        self.assertLessEqual(p["predicted_win_pct"],100)
        self.assertEqual(p["training_completed"],60)
    def test_signal_before_exit_never_learns_future(self):
        history=self.history()
        test=record(999,False,"2025-02-01","2025-02-15",hot=True)
        later=record(1000,False,"2025-02-01","2025-02-12",hot=True)
        a,_=walkforward_model_filter(history+[test])
        b,_=walkforward_model_filter(history+[test,later])
        self.assertEqual([x for x in a if x["ticker"]=="S999"],
                         [x for x in b if x["ticker"]=="S999"])
    def test_single_snapshot_fit_for_many_same_day_signals(self):
        history=self.history()
        same_day=[record(2000+i,bool(i%2),"2025-02-01","2025-02-14",hot=bool(i%2))
                  for i in range(120)]
        out,stats=walkforward_model_policies(history+same_day)
        self.assertEqual(stats["total"],180)
        self.assertEqual(stats["models_fitted"],2)
        self.assertEqual(stats["scored"],120)
        self.assertGreater(stats["total"],stats["models_fitted"])
    def test_predeclared_policy_thresholds_nested(self):
        self.assertLess(POLICIES["coverage"]["win_pct"],POLICIES["balanced"]["win_pct"])
        self.assertLess(POLICIES["balanced"]["win_pct"],POLICIES["strict"]["win_pct"])
        history=self.history()
        candidates=[record(999,False,"2025-02-01","2025-02-15",hot=True)]
        result,diagnostics=walkforward_model_policies(history+candidates)
        self.assertEqual(diagnostics["total"],61)
        self.assertGreaterEqual(len(result["coverage"]),len(result["balanced"]))
        self.assertGreaterEqual(len(result["balanced"]),len(result["strict"]))
    def test_policy_scores_exclude_later_same_day_results(self):
        hist=self.history()
        item=record(999,False,"2025-02-01","2025-02-10",hot=True)
        future=record(1000,True,"2025-02-01","2025-02-03",hot=True)
        first,_=walkforward_model_policies(hist+[item])
        second,_=walkforward_model_policies(hist+[item,future])
        for policy in POLICIES:
            self.assertEqual([t for t in first[policy] if t["ticker"]=="S999"],
                             [t for t in second[policy] if t["ticker"]=="S999"])

    def test_no_future_cannot_generate_signal(self):
        all_future=[record(i,True,"2026-01-01","2026-01-15",hot=True) for i in range(60)]
        selected,counts=walkforward_model_filter(all_future)
        self.assertFalse(selected)
        self.assertEqual(counts["insufficient"],60)
    def test_directional_prediction_from_features(self):
        h=self.history()
        strong=fit_probabilities(h,record(1001,False,hot=True))
        weak=fit_probabilities(h,record(1002,True,hot=False))
        self.assertGreater(strong["predicted_win_pct"],weak["predicted_win_pct"])

if __name__=="__main__":
    unittest.main()
