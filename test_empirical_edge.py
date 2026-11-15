"""Probability model anti-leak and shrinking regression tests."""
import unittest
from empirical_edge import posterior_from_completed, walkforward_filter, walkforward_regime_filter, _wilson_lower

def trade(i,profit,signal,exit_date):
    return {"ticker":f"S{i}","signal_date":signal,
            "entry_date":signal,"exit_date":exit_date,
            "net_pct":profit,"exit_reason":"take" if profit>0 else "stop"}

class CalibratedEdgeTest(unittest.TestCase):
    def test_no_optimism_without_samples(self):
        p=posterior_from_completed([])
        self.assertFalse(p["eligible"])
        self.assertEqual(p["sample_count"],0)
    def test_positive_probability_is_smoothed(self):
        p=posterior_from_completed([trade(i,4,"2025-01-01","2025-01-02") for i in range(30)])
        self.assertTrue(p["eligible"])
        self.assertLess(p["win_probability_pct"],100.0)
        self.assertGreater(p["expected_net_pct"],0)
    def test_recent_regime_deterioration_blocks_stale_long_mean(self):
        old=[trade(i,5,"2025-01-01","2025-01-02") for i in range(70)]
        recent=[trade(200+i,-3,"2025-01-01","2025-01-02") for i in range(30)]
        p=posterior_from_completed(old+recent)
        self.assertGreater(p["expected_net_pct"],0)
        self.assertLess(p["recent_expected_net_pct"],0)
        self.assertFalse(p["eligible"])

    def test_wilson_uncertainty_never_calls_ten_of_ten_certain(self):
        self.assertLess(_wilson_lower(10,10),1)
        self.assertEqual(_wilson_lower(0,0),0)
        self.assertGreater(_wilson_lower(30,30),0.55)

    def test_regime_gate_uses_only_matching_past_exits(self):
        training=[]
        for i in range(30):
            t=trade(i,4,"2025-01-01","2025-01-02")
            t.update(signal_atr_pct=2,signal_ma20_ma60_pct=3)
            training.append(t)
        for i in range(30):
            t=trade(i+100,-8,"2025-01-01","2025-01-02")
            t.update(signal_atr_pct=5,signal_ma20_ma60_pct=0)
            training.append(t)
        good=trade(999,-5,"2025-02-01","2025-02-12")
        good.update(signal_atr_pct=2,signal_ma20_ma60_pct=3)
        bad=trade(998,5,"2025-02-01","2025-02-12")
        bad.update(signal_atr_pct=5,signal_ma20_ma60_pct=0)
        chosen,status=walkforward_regime_filter(training+[good,bad])
        self.assertIn("S999",[x["ticker"] for x in chosen])
        self.assertNotIn("S998",[x["ticker"] for x in chosen])
        self.assertGreater(status["poor_expectancy"],0)

    def test_regime_gate_no_future_results(self):
        training=[]
        for i in range(30):
            t=trade(i,4,"2025-01-01","2025-01-02")
            t.update(signal_atr_pct=2,signal_ma20_ma60_pct=3)
            training.append(t)
        candidate=trade(999,-8,"2025-02-01","2025-02-12")
        candidate.update(signal_atr_pct=2,signal_ma20_ma60_pct=3)
        future=trade(1000,-99,"2025-02-01","2025-02-10")
        future.update(signal_atr_pct=2,signal_ma20_ma60_pct=3)
        first,_=walkforward_regime_filter(training+[candidate])
        second,_=walkforward_regime_filter(training+[candidate,future])
        self.assertEqual([x for x in first if x["ticker"]=="S999"],
                         [x for x in second if x["ticker"]=="S999"])

    def test_negative_history_blocked(self):
        p=posterior_from_completed([trade(i,-2,"2025-01-01","2025-01-02") for i in range(30)])
        self.assertFalse(p["eligible"])
    def test_lookahead_completely_excluded(self):
        hist=[trade(i,4,"2025-01-01","2025-01-02") for i in range(30)]
        candidate=trade(999,-5,"2025-02-01","2025-02-12")
        future=[trade(300+i,-90,"2025-02-01","2025-02-10") for i in range(50)]
        a,_=walkforward_filter(hist+[candidate])
        b,_=walkforward_filter(hist+[candidate]+future)
        aa=[x for x in a if x["ticker"]=="S999"]
        bb=[x for x in b if x["ticker"]=="S999"]
        self.assertEqual(len(aa),1)
        self.assertEqual(aa,bb)
    def test_overlapping_trades_not_seen_until_exit(self):
        incomplete=[trade(i,5,"2025-01-01","2025-03-01") for i in range(30)]
        candidate=trade(999,2,"2025-02-01","2025-02-20")
        a,status=walkforward_filter(incomplete+[candidate])
        self.assertFalse(any(x["ticker"]=="S999" for x in a))
        self.assertGreater(status["insufficient"],0)

if __name__=="__main__":
    unittest.main()
