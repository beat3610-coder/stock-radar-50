import unittest
import numpy as np
import pandas as pd
from probability_walkforward_audit import audit, transform

class ProbabilityAuditTests(unittest.TestCase):
    def make_data(self, dates=145):
        rng=np.random.default_rng(42)
        dates_list=pd.bdate_range("2025-01-01",periods=dates)
        rows=[]
        for i,date in enumerate(dates_list):
            actual=("up","down","flat")[i%3]
            # Deliberately overconfident, occasionally wrong historical forecasts.
            probs=[0.02,0.02,0.02]
            probs[(i+int(i%5==0))%3]=0.96
            rows.append(dict(date=date,market="KR",actual=actual,
                             p_up=probs[0],p_down=probs[1],p_flat=probs[2],
                             symbol="000001"))
        return pd.DataFrame(rows)

    def test_probabilities_sum_one(self):
        p=np.array([[0.99,0.005,0.005],[0.1,0.2,0.7]])
        q=transform(p,np.array([0.3,0.4,0.3]),1.5,0.25)
        np.testing.assert_allclose(q.sum(axis=1),np.ones(2))

    def test_chronological_audit_runs(self):
        result=audit(self.make_data(),min_train_dates=100,min_test_dates=30)
        self.assertEqual(result["KR"]["status"],"RESEARCH_ONLY_NOT_LIVE_APPROVED")
        self.assertEqual(result["KR"]["baseline"]["n"],45)
        self.assertEqual(result["KR"]["calibrated"]["n"],45)

    def test_missing_data_rejected(self):
        bad=self.make_data()
        bad.loc[0,"p_up"]=2.0
        with self.assertRaises(ValueError): audit(bad)

    def test_insufficient_history_not_approved(self):
        result=audit(self.make_data(90))
        self.assertEqual(result["KR"]["status"],"INSUFFICIENT_DATA")

    def test_duplicate_rows_rejected(self):
        data=self.make_data()
        data=pd.concat([data,data.iloc[[0]]],ignore_index=True)
        with self.assertRaises(ValueError): audit(data)

if __name__ == "__main__":
    unittest.main()
