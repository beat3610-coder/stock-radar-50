import unittest

import numpy as np
import pandas as pd

from market_context import _flow_features, _macro_feature_row, build_market_context


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

    def test_us_signal_uses_strict_prior_sox_close(self):
        dates=pd.bdate_range("2025-01-01",periods=35)
        sox=self.frame(dates,2000)
        before=_macro_feature_row(dates[-1],"US",{"SOX":sox})
        changed=sox.copy();changed.loc[dates[-1],"Close"]*=3
        after=_macro_feature_row(dates[-1],"US",{"SOX":changed})
        self.assertAlmostEqual(before["market_index_return_20d_pct"],after["market_index_return_20d_pct"])

    def test_missing_flows_stay_nan_and_are_not_fabricated_as_zero(self):
        dates=pd.bdate_range("2025-01-01",periods=45);prices=self.frame(dates)
        flows=pd.DataFrame({"foreign_net_shares":np.r_[np.nan*np.ones(15),np.arange(30)-15],
                            "institution_net_shares":np.r_[np.nan*np.ones(15),np.arange(30)[::-1]-15]},index=dates)
        values=_flow_features(prices,flows)
        self.assertTrue(values["foreign_flow_5d_to_turnover"].iloc[:15].isna().all())
        self.assertTrue(values["institution_flow_20d_to_turnover"].iloc[:34].isna().any())
        self.assertEqual(values["flow_data_available"].iloc[0],0.0)
        self.assertGreater(values["flow_data_available"].nunique(),2)
        agreement=values["flow_direction_agreement"].dropna()
        self.assertTrue(set(agreement.unique()).issubset({-1.0,0.0,1.0}))
        self.assertFalse((values["foreign_buy_streak"].dropna()==0).all())
        changed=flows.copy();changed.iloc[-1,:]=1e12
        unchanged_before=_flow_features(prices,changed)
        pd.testing.assert_series_equal(values["foreign_flow_5d_to_turnover"].iloc[:-1],
            unchanged_before["foreign_flow_5d_to_turnover"].iloc[:-1])

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
        self.assertTrue(np.isnan(unknown["market_risk_off"]))
        self.assertTrue(np.isnan(unknown["market_risk_on"]))


if __name__=="__main__":unittest.main()
