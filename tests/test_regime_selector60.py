import unittest
from datetime import date,timedelta
from regime_selector60 import score_all,report

def sample(n=360,change_future=False):
 rows=[]
 for i in range(n):
  d=date(2024,1,1)+timedelta(days=i*3)
  win=i%5!=0
  if change_future and i>=320:win=not win
  rows.append({"market":"KR","signal_date":d.isoformat(),
    "exit_date":(d+timedelta(days=2)).isoformat(),
    "ticker":f"A{i%30}","net_pct":2 if win else -1,
    "signal_ma20_ma60_pct":1,"signal_atr_pct":2,
    "signal_rsi":55,"signal_adx":30})
 return rows

class Regime60Tests(unittest.TestCase):
 def test_valid_probability_and_no_live_claim(self):
  r=report(sample(),market="KR")
  self.assertEqual(r["target_win_pct"],60)
  self.assertFalse(r["live_approved"])
  self.assertTrue(r["scored"]>100)
  for row in score_all(sample()):
   self.assertTrue(0<=row["probability"]<=1)
 def test_future_outcome_does_not_change_earlier_predictions(self):
  a=score_all(sample())
  b=score_all(sample(change_future=True))
  cutoff=sample()[320]["signal_date"]
  self.assertEqual([(r["signal_date"],r["probability"]) for r in a if r["signal_date"]<cutoff],
                   [(r["signal_date"],r["probability"]) for r in b if r["signal_date"]<cutoff])
 def test_zero_sample_fails_closed(self):
  r=report(sample(20))
  self.assertIsNone(r["selected_policy"])
  self.assertEqual(r["status"],"REJECTED")
 def test_market_filter(self):
  self.assertEqual(report(sample(),market="US")["scored"],0)

if __name__=="__main__":unittest.main()
