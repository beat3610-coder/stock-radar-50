import unittest

import numpy as np
import pandas as pd

from market_context import _macro_feature_row, build_market_context


class MarketContextTests(unittest.TestCase):
    def frame(self, dates, offset=0):
        x=np.arange(len(dates),dtype=float)+100+offset
        return pd.DataFrame({"Open":x,"High":x+1,"Low":x-1,"Close":x,
                            "Volume":np.arange(len(dates),dtype=float)+1000},index=dates)

    def test_korean_macro_alignment_excludes_same_date_us_close(self):
        dates=pd.date_range("2025-01-01",periods=35,freq="D")
        macro={"KOSPI":self.frame(dates),"SP500":self.frame(dates,1000),
               "SOX":self.frame(dates,2000),"VIX":self.frame(dates,3000),
               "USDKRW":self.frame(dates,4000)}
        kr=_macro_feature_row(dates[30],"KOSPI",macro)
        us=_macro_feature_row(dates[30],"SOXL",macro)
        # Values available only on the signal date do not affect the Korean as-of row.
        macro2={**macro,"SP500":macro["SP500"].copy()}
        macro2["SP500"].loc[dates[30],"Close"]=macro2["SP500"].loc[dates[30],"Close"]*2
        changed=_macro_feature_row(dates[30],"KOSPI",macro2)
        self.assertAlmostEqual(kr["sp500_return_20d_pct"],changed["sp500_return_20d_pct"])

    def test_cross_section_ranks_exclude_unlisted_rows_and_report_counts(self):
        dates=pd.bdate_range("2024-01-01",periods=80)
        frames={"A":self.frame(dates),"B":self.frame(dates[20:],offset=10)}
        context=build_market_context(frames,{},"KOSPI")
        day=str(dates[18].date());later=str(dates[40].date())
        self.assertNotIn(day,context["features"]["B"])
        self.assertEqual(context["features"]["A"][day]["xs_turnover20_rank_sample_n"],1)
        self.assertEqual(context["features"]["A"][later]["xs_return20_rank_sample_n"],2)
        self.assertEqual(context["features"]["B"][later]["xs_return20_rank_sample_n"],2)

    def test_model_feature_names_for_high52_correlation_and_gap_are_populated(self):
        dates=pd.bdate_range("2023-01-01",periods=100)
        frame=self.frame(dates)
        context=build_market_context({"A":frame},{},"KOSPI")
        row=context["features"]["A"][str(dates[-1].date())]
        for key in ("high52_distance_pct","price_volume_corr20","signal_gap_pct"):
            self.assertIn(key,row)
            self.assertTrue(np.isfinite(row[key]))

    def test_fixed_risk_rule_distinguishes_off_and_unknown(self):
        dates=pd.bdate_range("2020-01-01",periods=270)
        down=np.linspace(200,100,len(dates))
        index=pd.DataFrame({"Close":down},index=dates)
        vix=pd.DataFrame({"Close":np.r_[np.full(269,15.),50.]},index=dates)
        fx=pd.DataFrame({"Close":np.r_[np.full(len(dates)-21,1300.),np.linspace(1300.,1360.,21)]},index=dates)
        row=_macro_feature_row(dates[-1],"KOSPI",{"KOSPI":index,"VIX":vix,"USDKRW":fx})
        self.assertEqual(row["market_regime"],"RISK_OFF")
        unknown=_macro_feature_row(dates[0],"KOSPI",{})
        self.assertEqual(unknown["market_regime"],"UNKNOWN")


if __name__=="__main__":unittest.main()
