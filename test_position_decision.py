"""Position math/decision tests (not predictive accuracy tests)."""
import unittest
from position_decision import position_review

class PositionReviewTests(unittest.TestCase):
    def test_cost_and_cash_pnl(self):
        x=position_review(entry=100,shares=10,current=110,stop=95,target=120,cost_pct=.35)
        self.assertTrue(x["valid"])
        self.assertAlmostEqual(x["liquidation_net"],96.5)
        self.assertAlmostEqual(x["stop_net"],-53.5)
        self.assertAlmostEqual(x["target_net"],196.5)
    def test_stop_priority(self):
        x=position_review(entry=100,shares=10,current=89,stop=90,target=120)
        self.assertIn("손절선",x["decision"])
    def test_target_priority(self):
        x=position_review(entry=100,shares=10,current=125,stop=90,target=120)
        self.assertIn("익절",x["decision"])
    def test_trend_downgrade(self):
        x=position_review(entry=100,shares=10,current=98,stop=90,target=120,
                          ma20=100,macdh=-.5)
        self.assertIn("비중 축소",x["decision"])
    def test_invalid_fails(self):
        self.assertFalse(position_review(entry=100,shares=0,current=98,stop=90,target=120)["valid"])

if __name__=="__main__":
    unittest.main()
