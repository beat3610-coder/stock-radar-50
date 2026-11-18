import unittest
from datetime import date,timedelta
from trend_guard_model import audit,GUARDS,guarded
def examples(n=650,flip=999999):
    rows=[]
    for i in range(n):
        dt=date(2024,1,1)+timedelta(days=2*i)
        good=(i%5)!=0
        if i>=flip:good=not good
        rows.append({"market":"KR","ticker":f"A{i%17}","signal_date":dt.isoformat(),
          "exit_date":(dt+timedelta(days=1)).isoformat(),
          "net_pct":2 if good else -2,"exit_reason":"take" if good else "stop",
          "signal_rsi":57 if good else 33,"signal_adx":28 if good else 12,
          "signal_atr_pct":2 if good else 5,
          "signal_ma20_ma60_pct":2 if good else -2,
          "signal_volume_ratio":1.2 if good else 0.5,
          "signal_5d_return_pct":2 if good else -5,
          "entry_gap_pct":0,"entry_net_rr":1.8,"entry_net_upside_pct":5})
    return rows
class GuardTests(unittest.TestCase):
    def test_signal_only_guards(self):
        x=examples(5)
        self.assertTrue(GUARDS["trend_quality"](x[1]))
        self.assertFalse(GUARDS["trend_quality"](x[0]))
    def test_fixed_development_choice(self):
        a=audit(examples())
        b=audit(examples(flip=475))
        self.assertEqual(a["selected"] and [a["selected"][k] for k in ("model","base_policy","guard")],
                         b["selected"] and [b["selected"][k] for k in ("model","base_policy","guard")])
        self.assertFalse(a["live_approved"])
    def test_small_data_fails_closed(self):
        self.assertEqual(audit(examples(20))["status"],"INSUFFICIENT_TRAINING")
if __name__=="__main__":unittest.main()
