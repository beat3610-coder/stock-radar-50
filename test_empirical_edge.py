"""Probability model anti-leak and shrinking regression tests."""
import unittest
from empirical_edge import posterior_from_completed, walkforward_filter

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
