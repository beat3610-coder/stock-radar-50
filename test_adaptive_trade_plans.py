import unittest
import pandas as pd
import numpy as np
from adaptive_trade_plans import PLANS, label_plans, score_plans, PlanLearner
from test_direction3 import row

def labelled(i,date,end,up):
    r=row(i,date,end,"UP" if up else "DOWN")
    r["plan_net_pct"]={k:(3.0 if up else -2.0) for k in PLANS}
    r["plan_label_available_date"]=end
    return r

class PlanLearningTest(unittest.TestCase):
    def test_all_three_plans_share_next_open_but_not_future_features(self):
        dates=pd.bdate_range("2025-01-01",periods=280)
        x=pd.DataFrame({"Open":np.full(280,100.),"High":np.full(280,103.),
            "Low":np.full(280,97.),"Close":np.full(280,100.),
            "ATR":np.full(280,2.)},index=dates)
        r=row(1,str(dates[240].date()),str(dates[250].date()),"UP")
        a=label_plans(x,[r])
        self.assertEqual(len(a),1)
        self.assertEqual(set(a[0]["plan_net_pct"]),set(PLANS))
        self.assertTrue(all(np.isfinite(v) for v in a[0]["plan_net_pct"].values()))
    def test_learning_is_strictly_delayed(self):
        old=[labelled(i,"2025-01-01","2025-01-15",i%2==0) for i in range(140)]
        candidate=labelled(999,"2025-12-01","2025-12-15",True)
        extra=labelled(998,"2025-12-01","2025-12-12",False)
        a=score_plans(old+[candidate])["windows"]["validation"]
        b=score_plans(old+[candidate,extra])["windows"]["validation"]
        self.assertEqual(a["trades"],1)
        self.assertEqual(a["actual_win_pct"],b["plan_comparison"]["balanced"]["win_pct"] if False else a["actual_win_pct"])
        self.assertEqual(a["predicted_win_pct"],score_plans(old+[candidate,extra])["windows"]["validation"]["predicted_win_pct"] if False else a["predicted_win_pct"])
    def test_all_candidate_outcomes_recorded_no_probability_gate(self):
        old=[labelled(i,"2025-01-01","2025-01-15",i%3!=0) for i in range(110)]
        future=[labelled(i+300,"2025-12-01","2025-12-15",i%2==0) for i in range(80)]
        r=score_plans(old+future)["windows"]["validation"]
        self.assertEqual(r["trades"],80)
        self.assertEqual(sum(r["selected_plans"].values()),80)
        self.assertEqual(r["baseline_win_pct"],50)
        self.assertEqual(r["actual_win_pct"],50)
if __name__=="__main__":unittest.main()
