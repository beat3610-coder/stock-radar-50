"""Tests for online prediction quality and timing, not for arbitrary pass rates."""
import unittest
from online_edge import RunningEdge,score_stream,POLICIES

def item(i, signal, ended, pnl, hot, market="KQ"):
    return {"market":market,"ticker":f"T{i:06d}",
        "signal_date":signal,"entry_date":signal,"exit_date":ended,
        "net_pct":pnl,
        "signal_rsi":64.0 if hot else 42.0,
        "signal_atr_pct":1.5 if hot else 5.0,
        "signal_ma20_ma60_pct":3.0 if hot else -3.0}

class OnlinePredictionTests(unittest.TestCase):
    def test_predicts_all_regimes_after_minimum_history(self):
        model=RunningEdge()
        self.assertIsNone(model.estimate(item(100,"2025-02-01","2025-02-03",0,True)))
        for j in range(50):
            model.add(item(j,"2025-01-01","2025-01-02",3 if j%2 else -2,j%2==1))
        a=model.estimate(item(101,"2025-02-01","2025-02-03",0,True))
        b=model.estimate(item(102,"2025-02-01","2025-02-03",0,False))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertGreaterEqual(a["predicted_win_pct"],b["predicted_win_pct"])
    def test_future_losses_cannot_change_previous_prediction(self):
        prior=[item(i,"2025-01-01","2025-01-04",3 if i%2 else -2,i%2==1) for i in range(60)]
        signal=item(999,"2025-02-01","2025-02-15",-5,True)
        extra=item(1000,"2025-02-01","2025-02-05",-95,True)
        first,diagn1=score_stream(prior+[signal],"KQ")
        second,diagn2=score_stream(prior+[signal,extra],"KQ")
        for name in POLICIES:
            self.assertEqual([x for x in first[name] if x["ticker"]=="T000999"],
                             [x for x in second[name] if x["ticker"]=="T000999"])
    def test_brier_metrics_and_large_batches(self):
        trades=[]
        for i in range(3000):
            day=1+i//120
            # day strings need valid lexical dates for this synthetic replay.
            signal=f"2025-{(day-1)//28+1:02d}-{(day-1)%28+1:02d}"
            end=f"2025-{(day-1)//28+1:02d}-{(day-1)%28+2:02d}"
            if (day-1)%28==27:
                end=f"2025-{(day-1)//28+2:02d}-01"
            trades.append(item(i,signal,end,4 if i%3 else -3,i%2==0))
        _,metrics=score_stream(trades,"KQ")
        self.assertEqual(metrics["population"],3000)
        self.assertGreater(metrics["scored"],0)
        self.assertGreaterEqual(metrics["probability_quality"]["brier_model"],0)
        self.assertLessEqual(metrics["probability_quality"]["brier_model"],1)
        self.assertGreater(len(metrics["by_regime"]),0)
    def test_other_market_never_enters_learning(self):
        prior=[item(i,"2025-01-01","2025-01-02",5,True,"US") for i in range(100)]
        pending=item(100,"2025-02-01","2025-02-02",-2,True,"KQ")
        _,info=score_stream(prior+[pending],"KQ")
        self.assertEqual(info["population"],1)
        self.assertEqual(info["scored"],0)

if __name__=="__main__":
    unittest.main()
