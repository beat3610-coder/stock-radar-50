import unittest
from adaptive_online import OnlineLogistic,signal_features
from test_online_edge import item
from online_edge import score_stream

def enriched(i,signal,ended,pnl,hot,market="KQ"):
    t=item(i,signal,ended,pnl,hot,market)
    t.update(signal_adx=34 if hot else 18,
        signal_volume_ratio=1.8 if hot else .7,
        signal_5d_return_pct=3 if hot else -2,entry_gap_pct=.2)
    return t

class AdaptiveLearningTests(unittest.TestCase):
    def test_weights_really_update_and_separate_features(self):
        model=OnlineLogistic()
        before=model.w[:]
        for i in range(100):
            model.update(enriched(i,"2025-01-01","2025-01-02",
                         4 if i%2 else -3,i%2==1))
        self.assertNotEqual(before,model.w)
        self.assertGreater(model.predict(enriched(201,"2025-02-01","2025-02-02",0,True)),
                           model.predict(enriched(202,"2025-02-01","2025-02-02",0,False)))
    def test_future_same_date_does_not_leak(self):
        history=[enriched(i,"2025-01-01","2025-01-03",3 if i%2 else -2,i%2==1) for i in range(60)]
        sample=enriched(999,"2025-02-01","2025-02-10",-5,True)
        extra=enriched(1000,"2025-02-01","2025-02-03",-100,True)
        _,a=score_stream(history+[sample],"KQ")
        _,b=score_stream(history+[sample,extra],"KQ")
        self.assertEqual(a["by_window"]["train"]["brier_adaptive"],
                         b["by_window"]["train"]["brier_adaptive"])
    def test_large_stream_has_adaptive_and_baseline_metrics(self):
        seq=[]
        for d in range(2,30):
            sig=f"2025-01-{d:02d}"
            end=f"2025-01-{d+1:02d}" if d<29 else "2025-02-01"
            for i in range(65):
                seq.append(enriched(d*100+i,sig,end,3 if i%2 else -2,i%2==1))
        _,d=score_stream(seq,"KQ")
        self.assertGreater(d["adaptive_scored"],100)
        self.assertIsNotNone(d["by_window"]["train"]["brier_adaptive"])
    def test_missing_features_safe(self):
        self.assertIsNone(signal_features({"signal_rsi":30}))
        self.assertFalse(OnlineLogistic().update({"net_pct":3}))

if __name__=="__main__":unittest.main()
