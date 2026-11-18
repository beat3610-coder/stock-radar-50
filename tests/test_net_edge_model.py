import unittest
from datetime import date,timedelta
from net_edge_model import audit,_losses
def trades(n=650,flip_at=None):
    result=[]
    for i in range(n):
        d=date(2024,1,1)+timedelta(days=i*2)
        good=i%4!=0
        if flip_at is not None and i>=flip_at:good=not good
        result.append({"market":"KR","ticker":f"T{i%23}",
           "signal_date":d.isoformat(),"exit_date":(d+timedelta(days=1)).isoformat(),
           "entry_net_rr":1.8,"net_pct":2 if good else -2,
           "exit_reason":"target" if good else "stop",
           "signal_rsi":55 if good else 34,"signal_adx":30 if good else 15,
           "signal_atr_pct":2 if good else 5,
           "signal_ma20_ma60_pct":2 if good else -2,
           "signal_volume_ratio":1.5 if good else 0.7,
           "signal_5d_return_pct":2 if good else -3,
           "entry_gap_pct":0,"entry_net_upside_pct":5})
    return result
class NetEdgeTest(unittest.TestCase):
    def test_profit_attribution(self):
        self.assertEqual(_losses([{"net_pct":-2,"exit_reason":"stop"}])["loss_exit_reasons"]["stop"],1)
    def test_not_approved(self):
        r=audit(trades())
        self.assertFalse(r["live_approved"])
        self.assertEqual(r["target_win_pct"],60)
        self.assertIn(r["status"],("VALIDATION_PASS_RESEARCH_ONLY","VALIDATION_REJECTED","INSUFFICIENT_TIME_FOLDS","NO_POSITIVE_DEVELOPMENT_POLICY"))
    def test_diagnostic_not_used_to_choose(self):
        a=audit(trades())
        b=audit(trades(flip_at=450))
        self.assertEqual(a["selected_policy"],b["selected_policy"])
        self.assertEqual(a.get("selected_model"),b.get("selected_model"))
    def test_empty_fails_closed(self):
        self.assertEqual(audit([])["status"],"INSUFFICIENT_DEVELOPMENT")
if __name__=="__main__": unittest.main()
