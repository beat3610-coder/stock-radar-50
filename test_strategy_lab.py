"""Mechanical and anti-leak tests; no performance or trading claims."""
import unittest
import pandas as pd
from strategy_lab import (train_choice,_interval,qualifying,TRAIN_END,VALID_END,PARAMS)

class StrategyLabTests(unittest.TestCase):
    def test_train_never_looks_at_test(self):
        earlier=[{"entry_date":"2025-08-11","exit_date":"2025-08-20","net_pct":1.2,
                 "ticker":"A"} for _ in range(30)]
        test=[{"entry_date":"2026-07-11","exit_date":"2026-07-20","net_pct":-20.,
               "ticker":"B"} for _ in range(100)]
        leader1,score1=train_choice({"trend_breakout":earlier,
                                    "trend_pullback":[]})
        leader2,score2=train_choice({"trend_breakout":earlier+test,
                                    "trend_pullback":[]})
        self.assertEqual(leader1,leader2)
        self.assertEqual(score1,score2)
    def test_fixed_periods_disjoint(self):
        r=[{"entry_date":"2025-10-29","exit_date":"2025-10-30"},
           {"entry_date":TRAIN_END,"exit_date":"2025-11-10"},
           {"entry_date":"2026-03-20","exit_date":"2026-03-25"},
           {"entry_date":VALID_END,"exit_date":"2026-04-15"}]
        parts=[_interval(r,z) for z in ("train","validate","test")]
        self.assertEqual([len(x) for x in parts],[1,2,1])
    def test_unsupported_variant_cannot_signal(self):
        r=pd.Series({"Close":100,"MA20":99,"MA60":95,"MA200":90,"RSI":55,
             "MACDH":1,"ADX":25,"PDI":25,"MDI":10,"ATR":2,
             "Volume":200,"VOL20":100})
        self.assertFalse(qualifying(r,r,"KR","made_up",101))
    def test_params_finite(self):
        self.assertTrue(all(v["rr"]>1 and v["stop_atr"]>0 for v in PARAMS.values()))

if __name__=="__main__":
    unittest.main()
