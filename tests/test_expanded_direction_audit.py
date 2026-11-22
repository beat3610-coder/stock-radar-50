import unittest

import numpy as np
import pandas as pd

from expanded_direction_audit import (
    FEATURES, _features, _make_rows, _metrics, _trade_perf, direction_label,
)


class ExpandedDirectionAuditTests(unittest.TestCase):
    def test_fixed_threshold_labels(self):
        self.assertEqual(direction_label(.51, .5), "UP")
        self.assertEqual(direction_label(-.51, .5), "DOWN")
        self.assertEqual(direction_label(.5, .5), "SIDEWAYS")
        with self.assertRaises(ValueError):
            direction_label(float("nan"), .5)

    def test_fixed_schema_feature_matrix(self):
        row={key:0.0 for key in FEATURES}
        x=_features([row])
        self.assertEqual(x.shape,(1,len(FEATURES)))
        self.assertTrue(np.isfinite(x).all())

    def test_three_class_metrics_and_confusion_labels(self):
        rows=[{"label_0.5":"DOWN","net_pct":-1,"signal_date":"2025-01-01"},
              {"label_0.5":"SIDEWAYS","net_pct":0,"signal_date":"2025-01-01"},
              {"label_0.5":"UP","net_pct":1,"signal_date":"2025-01-01"}]
        result=_metrics(rows,np.asarray([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]))
        self.assertEqual(result["accuracy_pct"],100)
        self.assertEqual(result["confusion_matrix"]["counts"],[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(set(result["class_metrics"]),{"DOWN","SIDEWAYS","UP"})

    def test_selected_trades_include_drawdowns_and_avoidance(self):
        rows=[{"ticker":"A","entry_date":"2025-01-01","label_available_date":"2025-01-05","net_pct":-2,"label_0.5":"DOWN"},
              {"ticker":"B","entry_date":"2025-01-02","label_available_date":"2025-01-06","net_pct":1,"label_0.5":"UP"}]
        result=_trade_perf(rows[1:],rows,np.asarray([[.8,.1,.1],[.1,.1,.8]]))
        self.assertEqual(result["n"],1)
        self.assertEqual(result["rejected_down_n"],1)
        self.assertEqual(result["rejected_down_loss_avoided_pct"],2)
        self.assertIsNotNone(result["max_drawdown_5slot_pct"])

    def test_labels_enter_next_open_and_mature_at_horizon_close(self):
        dates=pd.bdate_range("2016-01-01",periods=400)
        px=np.linspace(100,120,len(dates))
        frame=pd.DataFrame({"Open":px*.999,"High":px*1.01,"Low":px*.99,
                            "Close":px,"Volume":np.full(len(px),10000.)},index=dates)
        rows=_make_rows(frame,"TEST","KR","KOSPI")
        first=rows[0]
        self.assertEqual(first["entry_date"],str(dates[201].date()))
        self.assertEqual(first["label_available_date"],str(dates[205].date()))
        self.assertEqual(first["exit_date"],first["label_available_date"])
        self.assertEqual({r["horizon"] for r in rows},{5,10,20})


if __name__ == "__main__":
    unittest.main()
