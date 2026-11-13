"""Synthetic mechanics tests; never evidence of investing profitability."""
import unittest
import pandas as pd
from portfolio_audit import replay
from certification import assess,wilson_lower

def history():
    ix=pd.bdate_range("2026-04-01",periods=30)
    return pd.DataFrame({"Open":[100.]*30,"High":[105.]*30,
        "Low":[95.]*30,"Close":[100.]*30},index=ix)

class PortfolioAuditTests(unittest.TestCase):
    def test_cash_limited_and_daily_equity(self):
        frame=history()
        t={"ticker":"X","entry_date":str(frame.index[0].date()),
            "exit_date":str(frame.index[2].date()),
            "entry":100.,"exit_price":105.,"stop":95.,"take":110.,
            "score":80,"net_pct":4.65}
        r=replay([t],{"X":frame})
        self.assertTrue(r["valid"])
        self.assertEqual(r["trades_executed"],1)
        self.assertGreater(r["end_equity"],r["initial_equity"])
        self.assertLessEqual(r["executions"][0]["shares"]*100,20_000)
    def test_loss_and_gap_marked_in_money(self):
        f=history()
        t={"ticker":"X","entry_date":str(f.index[0].date()),
           "exit_date":str(f.index[1].date()),"entry":100.,
           "exit_price":85.,"stop":95.,"take":110.,"score":85}
        r=replay([t],{"X":f})
        self.assertLess(r["total_return_pct"],0)
        self.assertGreater(r["max_drawdown_pct"],0)
    def test_same_day_exit(self):
        f=history();d=str(f.index[0].date())
        t={"ticker":"X","entry_date":d,"exit_date":d,"entry":100.,
           "exit_price":96.,"stop":96.,"take":110.,"score":80}
        r=replay([t],{"X":f})
        self.assertEqual(r["trades_executed"],1)
        self.assertLess(r["total_return_pct"],0)
    def test_intraday_sale_proceeds_cannot_fund_same_morning_open(self):
        f=history()
        d0=str(f.index[0].date()); d1=str(f.index[1].date()); d2=str(f.index[2].date())
        prev={"ticker":"X","entry_date":d0,"exit_date":d1,
              "entry":100.,"exit_price":105.,"stop":95.,"take":110.,"score":80}
        next_trade={"ticker":"Y","entry_date":d1,"exit_date":d2,
                    "entry":100.,"exit_price":105.,"stop":95.,"take":110.,"score":80}
        r=replay([prev,next_trade],{"X":f,"Y":f},max_positions=1)
        self.assertEqual(r["trades_executed"],1)
        self.assertEqual(r["executions"][0]["ticker"],"X")
    def test_no_data_fails(self):
        self.assertFalse(replay([],{})["valid"])
    def test_certification_cannot_approve_unobserved_paper(self):
        sample={"markets":{m:{"requested":100,"loaded":100,
            "holdout":{"trades":50,"win_rate_pct":70,
                       "profit_factor":2.1,"mean_net_pct":2}}
            for m in ("KR","KQ","US")},
            "survivorship_bias":False,
            "historical_data_previously_inspected":False}
        r=assess(sample,portfolio={"valid":True,"max_drawdown_pct":5,
                "benchmark_verified":True})
        self.assertEqual(r["status"],"NOT_APPROVED")
        self.assertTrue(any("paper" in reason for reason in r["reasons"]))
    def test_wilson_bound_is_conservative(self):
        self.assertLess(wilson_lower(6,10),0.6)
        self.assertIsNone(wilson_lower(0,0))

if __name__=="__main__":
    unittest.main()
