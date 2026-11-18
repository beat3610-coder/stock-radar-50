import unittest
from datetime import date,timedelta
from profit_selector51 import discover, obey

def example(n=560, future_flip=False):
 out=[]
 for i in range(n):
  day=date(2024,1,1)+timedelta(days=i)
  good=i%4!=0
  if future_flip and i>400:good=not good
  out.append({"market":"KR","signal_date":day.isoformat(),
   "exit_date":(day+timedelta(days=1)).isoformat(),
   "ticker":"A","net_pct":2.0 if good else -1.0,
   "signal_rsi":56 if good else 30,"signal_adx":25,"signal_atr_pct":2,
   "signal_ma20_ma60_pct":2,"signal_volume_ratio":1,
   "signal_5d_return_pct":2,"entry_gap_pct":0,
   "entry_net_rr":1.8,"entry_net_upside_pct":6})
 return out
class TestSelector51(unittest.TestCase):
 def test_not_promoted(self):
  r=discover(example(),market="KR")
  self.assertFalse(r["eligible_for_live"])
  self.assertIn(r["status"],("VALIDATION_PASS_RESEARCH_ONLY","VALIDATION_REJECTED","NO_TRAINING_CANDIDATE"))
 def test_selection_does_not_use_future_outcomes(self):
  a=discover(example())
  b=discover(example(future_flip=True))
  self.assertEqual(a["selected"]["conditions"],b["selected"]["conditions"])
 def test_refuse_tiny_samples(self):
  self.assertEqual(discover(example(30))["status"],"NO_TRAINING_CANDIDATE")
 def test_missing_features_do_not_pass(self):
  self.assertFalse(obey({},(("signal_rsi",">=",40),)))
if __name__=="__main__":unittest.main()
