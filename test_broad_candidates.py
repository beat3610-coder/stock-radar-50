"""Regression tests for broad-population candidate generation."""
import unittest
import numpy as np
import pandas as pd
from broad_candidates import broad_candidates_from_computed

class BroadPoolTest(unittest.TestCase):
    def frame(self):
        idx=pd.bdate_range("2025-01-01",periods=300)
        n=len(idx)
        # Weak trend is intentionally represented in the broad population.
        return pd.DataFrame({
            "Open":np.full(n,100.),"High":np.full(n,101.),
            "Low":np.full(n,99.),"Close":np.full(n,100.),
            "Volume":np.full(n,100000.),"VOL20":np.full(n,80000.),
            "MA20":np.full(n,95.),"MA60":np.full(n,99.),
            "MA200":np.full(n,120.),"RSI":np.full(n,35.),
            "ADX":np.full(n,14.),"ATR":np.full(n,2.)
        },index=idx)
    def test_negative_trend_is_not_excluded_from_training(self):
        x=self.frame()
        out=broad_candidates_from_computed(x,"TEST","KR",end_date="2027-01-01")
        self.assertGreater(len(out),0)
        self.assertTrue(all(t["signal_ma20_ma60_pct"]<0 for t in out))
        self.assertTrue(all(t["net_pct"]<0 for t in out))
    def test_invalid_prices_never_produce_candidates(self):
        x=self.frame()
        x.loc[:,"Open"]=0
        self.assertFalse(broad_candidates_from_computed(x,"TEST","KR",end_date="2027-01-01"))
    def test_no_future_signal_features(self):
        x=self.frame()
        first=broad_candidates_from_computed(x,"TEST","KR",end_date="2025-12-31")[0]
        y=x.copy()
        y.loc[y.index[255]:,"RSI"]=90.
        other=broad_candidates_from_computed(y,"TEST","KR",end_date="2025-12-31")[0]
        self.assertEqual(first["signal_date"],other["signal_date"])
        self.assertEqual(first["signal_rsi"],other["signal_rsi"])

if __name__=="__main__":
    unittest.main()
