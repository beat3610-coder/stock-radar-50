import unittest

import numpy as np

from r_rank_model import (MODEL_CONFIGS, STRATEGIES, accepts, design_matrix,
                          FEATURE_GROUPS, FEATURE_LIMIT, NUMERIC_FEATURES, _fit_snapshot,
                          make_variant_thresholds, realized_r, walk_forward_scores)


class RRankModelTests(unittest.TestCase):
    def row(self, i, pred=0.0):
        return {"ticker":f"T{i%12}","strategy":STRATEGIES[i%len(STRATEGIES)],
            "signal_date":"2025-06-01","entry_date":"2025-06-02","exit_date":"2025-06-10",
            "net_pct":1.0 if i%2 else -1.0,"entry_net_downside_pct":2.0,
            "entry_net_rr":1.5,"model_predicted_r":pred}

    def test_target_is_realized_r_and_features_have_fixed_width(self):
        r=self.row(0);self.assertEqual(realized_r(r),-.5)
        x=design_matrix([r]);self.assertEqual(x.shape,(1,len(NUMERIC_FEATURES)+12));self.assertTrue(np.isfinite(x).all())
        self.assertLessEqual(x.shape[1],FEATURE_LIMIT)
        dropped=design_matrix([r],excluded_features=("signal_rsi","strategy_flag:trend_breakout"))
        self.assertEqual(dropped.shape,(1,len(NUMERIC_FEATURES)+10))

    def test_predeclared_ablation_feature_group_widths(self):
        row=self.row(0)
        widths={name:design_matrix([row],name).shape[1] for name in FEATURE_GROUPS}
        self.assertEqual(widths["base30"],30)
        self.assertEqual(widths["plus_regime"],49)
        self.assertEqual(widths["plus_relative"],61)
        self.assertEqual(widths["plus_flow"],71)

    def test_only_fixed_predeclared_model_configs(self):
        self.assertEqual({v["kind"] for v in MODEL_CONFIGS.values()},
                         {"ridge","rf","gbr","hgb"})

    def test_date_cross_section_target_centers_only_closed_snapshot_rows(self):
        rows=[]
        for day,values in (("2025-01-01",(1.0,3.0)),("2025-01-02",(2.0,6.0))):
            for i,value in enumerate(values):
                rows.append({**self.row(i),"signal_date":day,"net_pct":value,
                    "entry_net_downside_pct":1.0,"exit_date":"2025-01-05"})
        snapshot=_fit_snapshot(rows,np.zeros((4,1),dtype=float),"ridge_a1",min_train_samples=4,
                               target_variant="date_cs")
        self.assertIsNotNone(snapshot)
        np.testing.assert_allclose(snapshot["train_r"],[-1.0,1.0,-2.0,2.0])

    def test_variant_thresholds_and_acceptance(self):
        rows=[{**self.row(i,pred=i/20),"market_regime":"RISK_OFF" if i%2 else "RISK_ON"} for i in range(60)]
        thresholds=make_variant_thresholds(rows)
        self.assertEqual(set(thresholds),{"r_p70","r_p80","r_p90"})
        self.assertIn("RISK_OFF",thresholds["r_p70"]["by_regime"])
        self.assertIn("RISK_ON",thresholds["r_p70"]["by_strategy_regime"][STRATEGIES[0]])
        self.assertTrue(accepts(rows[-1],"r_p70",thresholds))
        self.assertFalse(accepts(rows[0],"r_p90",thresholds))

    def test_cohort_can_declare_a_lower_minimum_for_single_instrument_history(self):
        from datetime import date, timedelta
        rows=[]
        start=date(2023,1,1)
        for i in range(100):
            day=start+timedelta(days=i)
            rows.append({"ticker":"SOXL","strategy":"broad_market_ml",
                "signal_date":str(day),"entry_date":str(day),
                "exit_date":str(day+timedelta(days=1)),"net_pct":1.0 if i%2 else -1.0,
                "entry_net_downside_pct":2.0,"entry_net_rr":1.5})
        rows.append({"ticker":"SOXL","strategy":"broad_market_ml",
            "signal_date":"2024-01-03","entry_date":"2024-01-04","exit_date":"2024-01-10",
            "net_pct":1.0,"entry_net_downside_pct":2.0,"entry_net_rr":1.5})
        scored,status=walk_forward_scores(rows,"ridge_a1",min_train_samples=100)
        self.assertEqual(status["min_training_samples"],100)
        self.assertEqual(len(scored),1)


if __name__=="__main__":unittest.main()
