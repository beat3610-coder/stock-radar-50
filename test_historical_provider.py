"""Avoid silently shrinking explicitly requested historical coverage to two years."""
import unittest
import types
import sys
from unittest.mock import patch
import pandas as pd
import numpy as np
from providers import get_prices,get_historical_prices

class HistoricalProviderTests(unittest.TestCase):
    def price_frame(self):
        index=pd.date_range("2012-01-01",periods=10,freq="B")
        return pd.DataFrame({"Open":np.full(10,100.),"High":np.full(10,110.),
                             "Low":np.full(10,90.),"Close":np.full(10,101.),
                             "Volume":np.full(10,10000.)},index=index)
    def test_explicit_historical_start_flows_to_provider(self):
        called=[]
        fdr=types.SimpleNamespace(DataReader=lambda ticker,start:
           (called.append((ticker,start)) or self.price_frame()))
        with patch.dict(sys.modules,{"FinanceDataReader":fdr}):
            prices,source=get_historical_prices("005930","KR",start_date="2011-01-01")
        self.assertEqual(called[0][1],"2011-01-01")
        self.assertEqual(str(prices.index[0].date()),"2012-01-02")
        self.assertEqual(len(prices),10)
    def test_live_default_stays_recent(self):
        called=[]
        fdr=types.SimpleNamespace(DataReader=lambda ticker,start:
           (called.append(start) or self.price_frame()))
        with patch.dict(sys.modules,{"FinanceDataReader":fdr}):
            get_prices("005930","KR")
        self.assertNotEqual(called[0],"2011-01-01")

if __name__=="__main__":
    unittest.main()
