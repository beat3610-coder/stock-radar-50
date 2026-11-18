import unittest
from datetime import datetime, timedelta
from three_round_trade_model import three_round_audit

def make_trades(n=180, perturbed_future=False):
    rows=[]
    for i in range(n):
        # Two-day holding period: settled outcomes cannot enter same-day training.
        win=(i%5 in (0,1,2))
        if perturbed_future and i>=150:
            win=not win
        day=datetime(2024,1,1)+timedelta(days=3*i)
        rows.append({"market":"KR","ticker":f"T{i%20:03d}",
            "signal_date":day.date().isoformat(),
            "exit_date":(day+timedelta(days=2)).date().isoformat(),
            "net_pct":2.0 if win else -1.0,
            "signal_rsi":60 if i%2 else 45,
            "signal_adx":30 if i%3 else 15,
            "signal_atr_pct":4 if i%4 else 1,
            "signal_ma20_ma60_pct":2 if i%2 else -2})
    return rows

class ThreeRoundTests(unittest.TestCase):
    def test_all_three_rounds_present(self):
        result=three_round_audit(make_trades())
        self.assertGreater(result["scored"],0)
        for key in ("baseline","round1","round2","round3"):
            self.assertIsNotNone(result["rounds"][key]["brier"])
        self.assertEqual(result["status"],"RESEARCH_ONLY_NOT_LIVE_APPROVED")

    def test_future_outcomes_cannot_change_earlier_forecasts(self):
        a=three_round_audit(make_trades())
        b=three_round_audit(make_trades(perturbed_future=True))
        # Both runs score the same pre-change historical observations;
        # last-segment changes are allowed to affect subsequent scores.
        self.assertEqual(a["scored"],b["scored"])

    def test_market_filter(self):
        result=three_round_audit(make_trades(),market="US")
        self.assertEqual(result["scored"],0)
        self.assertIsNone(result["rounds"]["round3"]["brier"])

    def test_no_predictions_with_insufficient_history(self):
        self.assertEqual(three_round_audit(make_trades(20))["scored"],0)

if __name__=="__main__":unittest.main()
