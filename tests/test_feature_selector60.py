import unittest
from datetime import date,timedelta
from feature_selector60 import audit, _clean
def data(n=600,flip_after=None):
    out=[]
    for i in range(n):
        d=date(2024,1,1)+timedelta(days=2*i)
        good=(i%4)!=0
        if flip_after is not None and i>=flip_after:good=not good
        out.append({"market":"KR","ticker":f"T{i%20}","signal_date":d.isoformat(),
          "exit_date":(d+timedelta(days=1)).isoformat(),
          "net_pct":2 if good else -1,
          "signal_rsi":56 if good else 35,"signal_adx":30 if good else 15,
          "signal_atr_pct":2 if good else 5,"signal_ma20_ma60_pct":2 if good else -2,
          "signal_volume_ratio":1.4 if good else 0.8,
          "signal_5d_return_pct":1 if good else -3,
          "entry_gap_pct":0,"entry_net_rr":1.8,"entry_net_upside_pct":5})
    return out
class Tests(unittest.TestCase):
 def test_no_live_certification(self):
    r=audit(data())
    self.assertFalse(r["eligible_for_live"])
    self.assertIn(r["status"],("VALIDATION_PASS_RESEARCH_ONLY","VALIDATION_REJECTED","INSUFFICIENT_DATA","REJECTED_NO_POSITIVE_DEVELOPMENT_POLICY"))
 def test_future_diagnostic_outcomes_do_not_influence_train_selection(self):
    a=audit(data())
    b=audit(data(flip_after=410))
    self.assertEqual(a["selected"]["model"],b["selected"]["model"])
    self.assertEqual(a["selected"]["threshold"],b["selected"]["threshold"])
 def test_missing_features_excluded(self):
    x=data(5);x[0].pop("signal_rsi")
    self.assertEqual(len(_clean(x,"KR")),4)
 def test_insufficient_rows_fail_closed(self):
    self.assertEqual(audit(data(20))["status"],"INSUFFICIENT_DATA")
if __name__=="__main__":unittest.main()
