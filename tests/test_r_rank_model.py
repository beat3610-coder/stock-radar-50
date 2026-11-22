import unittest

import numpy as np

from r_rank_model import (MODEL_CONFIGS, STRATEGIES, accepts, design_matrix,
                          make_variant_thresholds, realized_r)


class RRankModelTests(unittest.TestCase):
    def row(self, i, pred=0.0):
        return {"ticker":f"T{i%12}","strategy":STRATEGIES[i%len(STRATEGIES)],
            "signal_date":"2025-06-01","entry_date":"2025-06-02","exit_date":"2025-06-10",
            "net_pct":1.0 if i%2 else -1.0,"entry_net_downside_pct":2.0,
            "entry_net_rr":1.5,"model_predicted_r":pred}

    def test_target_is_realized_r_and_features_have_fixed_width(self):
        r=self.row(0);self.assertEqual(realized_r(r),-.5)
        x=design_matrix([r]);self.assertEqual(x.shape,(1,30));self.assertTrue(np.isfinite(x).all())

    def test_only_fixed_predeclared_model_configs(self):
        self.assertEqual({v["kind"] for v in MODEL_CONFIGS.values()},
                         {"ridge","rf","gbr","hgb"})

    def test_variant_thresholds_and_acceptance(self):
        rows=[self.row(i,pred=i/20) for i in range(60)]
        thresholds=make_variant_thresholds(rows)
        self.assertEqual(set(thresholds),{"r_p70","r_p80","r_p90"})
        self.assertTrue(accepts(rows[-1],"r_p70",thresholds))
        self.assertFalse(accepts(rows[0],"r_p90",thresholds))


if __name__=="__main__":unittest.main()
