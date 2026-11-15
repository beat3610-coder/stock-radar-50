import unittest
import pandas as pd
import numpy as np
from buy_success import OnlineBuySuccess,attach_buy_labels,score_buy_success
from test_direction3 import row
from direction3 import LABELS

def rec(i,signal,close,win,lab="UP"):
    x=row(i,signal,close,lab)
    x["buy_net_pct"]=3.0 if win else -2.0
    x["buy_won"]=int(win)
    x["buy_label_available_date"]=close
    return x

class BuySuccessTest(unittest.TestCase):
    def test_buy_probability_learns_all_training_cases(self):
        m=OnlineBuySuccess()
        for i in range(240):
            m.update(rec(i,"2025-01-01","2025-01-15",i%3!=0,LABELS[i%3]))
        p=m.predict(rec(999,"2025-02-01","2025-02-15",True))
        self.assertIsNotNone(p)
        self.assertTrue(0<=p["p_success"]<=1)
    def test_buy_outcomes_match_realistic_next_open_stop_first(self):
        n=280;dates=pd.bdate_range("2025-01-01",periods=n)
        frame=pd.DataFrame({"Open":np.full(n,100.0),"High":np.full(n,101.0),
            "Low":np.full(n,99.0),"Close":np.full(n,100.0),
            "ATR":np.full(n,2.0)},index=dates)
        sample=row(1,str(dates[240].date()),str(dates[250].date()),"SIDEWAYS")
        enriched=attach_buy_labels(frame,[sample])
        self.assertEqual(len(enriched),1)
        self.assertLess(enriched[0]["buy_net_pct"],0)
        self.assertEqual(enriched[0]["buy_won"],0)
        self.assertEqual(enriched[0]["buy_label_available_date"],sample["label_available_date"])
    def test_labels_only_mature_before_next_signal(self):
        past=[rec(i,"2025-01-01","2025-01-12",i%2==0) for i in range(160)]
        sample=rec(999,"2025-02-01","2025-02-15",True)
        future=rec(1000,"2025-02-01","2025-02-04",False)
        a=score_buy_success(past+[sample])["windows"]["train"]
        b=score_buy_success(past+[sample,future])["windows"]["train"]
        self.assertEqual(a["mean_predicted_success_pct"],b["mean_predicted_success_pct"])
        self.assertEqual(a["count"],1)
    def test_reports_probability_vs_actual_and_net(self):
        past=[rec(i,"2025-01-01","2025-01-12",i%2==0) for i in range(130)]
        test=[rec(i+200,"2025-12-01","2025-12-15",i%3!=0) for i in range(60)]
        result=score_buy_success(past+test)["windows"]["validation"]
        self.assertEqual(result["count"],60)
        self.assertEqual(result["wins"],40)
        self.assertAlmostEqual(result["actual_success_pct"],66.67,places=2)
        self.assertIsNotNone(result["mean_predicted_success_pct"])
        self.assertIsNotNone(result["profit_factor"])

if __name__=="__main__":unittest.main()
