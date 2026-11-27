"""Avoid silently shrinking explicitly requested historical coverage to two years."""
import os
import tempfile
import unittest
import types
import sys
from unittest.mock import patch
import pandas as pd
import numpy as np

os.environ.setdefault("STOCK_RADAR_PRICE_CACHE", tempfile.mkdtemp(prefix="stock-radar-prices-"))

from providers import get_prices,get_historical_prices

def price_frame(start,periods):
    index=pd.date_range(start,periods=periods,freq="B")
    return pd.DataFrame({"Open":np.full(periods,100.),"High":np.full(periods,110.),
                         "Low":np.full(periods,90.),"Close":np.full(periods,101.),
                         "Volume":np.full(periods,10000.)},index=index)

class HistoricalProviderTests(unittest.TestCase):
    def test_explicit_historical_start_flows_to_provider(self):
        called=[]
        frame=price_frame("2011-01-03",300)
        fdr=types.SimpleNamespace(DataReader=lambda ticker,start:
           (called.append((ticker,start)) or frame))
        with patch.dict(sys.modules,{"FinanceDataReader":fdr,"yfinance":None}):
            prices,source=get_historical_prices("005930","KR",start_date="2011-01-01")
        self.assertEqual(called[0][1],"2011-01-01")
        self.assertEqual(str(prices.index[0].date()),"2011-01-03")
        self.assertEqual(len(prices),300)
    def test_short_history_is_rejected_not_silently_shortened(self):
        short=price_frame("2012-01-02",10)
        fdr=types.SimpleNamespace(DataReader=lambda ticker,start:short)
        with patch.dict(sys.modules,{"FinanceDataReader":fdr,"yfinance":None}):
            with self.assertRaises(RuntimeError):
                get_historical_prices("000660","KR",start_date="2011-01-01")
    def test_live_default_stays_recent(self):
        called=[]
        fdr=types.SimpleNamespace(DataReader=lambda ticker,start:
           (called.append(start) or price_frame("2024-01-02",60)))
        with patch.dict(sys.modules,{"FinanceDataReader":fdr,"yfinance":None}):
            get_prices("005930","KR")
        self.assertNotEqual(called[0],"2011-01-01")
        self.assertGreater(pd.Timestamp(called[0]).year,2020)

if __name__=="__main__":
    unittest.main()
