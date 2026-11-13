"""Deterministic execution tests; synthetic bars are NOT evidence of returns."""
import unittest
import pandas as pd
from audit_300 import simulate_exit, metrics

def bars(records):
    return pd.DataFrame(records,columns=["Open","High","Low","Close"],
                        index=pd.date_range("2026-01-01",periods=len(records),freq="B"))

class ExecutionTests(unittest.TestCase):
    def test_both_levels_hit_stop_first(self):
        d=bars([(100,101,99,100),(100,111,89,105)])
        r=simulate_exit(d,0,100,95,110,max_days=1)
        self.assertEqual(r["exit_reason"],"stop")
        self.assertAlmostEqual(r["net_pct"],-5.35)
    def test_gap_below_stop_is_filled_at_open(self):
        d=bars([(100,101,99,100),(90,94,89,92)])
        r=simulate_exit(d,0,90,95,110,max_days=1)
        self.assertIsNone(r) # entry below original stop invalid; skipped
    def test_gap_after_valid_entry(self):
        d=bars([(100,101,99,100),(100,101,98,99),(90,92,88,90)])
        r=simulate_exit(d,0,100,95,110,max_days=2)
        self.assertEqual(r["exit_reason"],"stop_gap")
        self.assertAlmostEqual(r["net_pct"],-10.35)
    def test_take_gap_at_open(self):
        d=bars([(100,101,99,100),(112,113,110,112)])
        r=simulate_exit(d,0,112,95,110,max_days=1)
        self.assertIsNone(r) # entry above original target invalid
    def test_incomplete_horizon_not_tested(self):
        d=bars([(100,101,99,100),(100,105,96,103)])
        self.assertIsNone(simulate_exit(d,0,100,95,110,max_days=10))
    def test_negative_strategy_does_not_pass(self):
        m=metrics([{"net_pct":-5},{"net_pct":2},{"net_pct":-3}])
        self.assertLess(m["mean_net_pct"],0)
        self.assertLess(m["profit_factor"],1)

if __name__=="__main__":
    unittest.main()
