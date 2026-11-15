import unittest
from online_mixture import OnlineExpertMixture
from test_adaptive_online import enriched
from online_edge import score_stream

class ExpertMixtureTests(unittest.TestCase):
    def test_learning_updates_only_when_label_arrives(self):
        model=OnlineExpertMixture()
        first=model.weights()
        self.assertAlmostEqual(model.predict((0.9,0.2,0.3)),sum((0.9,0.2,0.3))/3)
        self.assertEqual(first,model.weights())
        for i in range(60):
            self.assertTrue(model.update((0.9,0.2,0.3),3.0))
        self.assertGreater(model.weights()[0],model.weights()[1])
        self.assertGreater(model.weights()[0],model.weights()[2])
    def test_no_updating_from_same_day_or_future_outcomes(self):
        history=[enriched(i,"2025-01-01","2025-01-03",3 if i%2 else -2,i%2==1) for i in range(60)]
        candidate=enriched(999,"2025-02-01","2025-02-15",-5,True)
        future=enriched(1000,"2025-02-01","2025-02-05",-99,True)
        _,a=score_stream(history+[candidate],"KQ")
        _,b=score_stream(history+[candidate,future],"KQ")
        self.assertEqual(a["by_window"]["train"]["brier_mixture"],b["by_window"]["train"]["brier_mixture"])
        self.assertEqual(a["by_window"]["train"]["average_mixture_predicted_win_pct"],b["by_window"]["train"]["average_mixture_predicted_win_pct"])
    def test_every_candidate_gets_score_after_warmup(self):
        trades=[]
        for day in range(2,29):
            for i in range(30):
                trades.append(enriched(day*100+i,f"2025-01-{day:02d}",f"2025-01-{day+1:02d}",3 if i%3 else -4,i%2==0))
        _,r=score_stream(trades,"KQ")
        self.assertGreater(r["probability_quality"]["mixture_scored"],300)
        self.assertTrue(0<=r["probability_quality"]["brier_mixture"]<=1)
        self.assertEqual(r["probability_quality"]["mixture_scored"],r["probability_quality"]["adaptive_scored"])
    def test_invalid_inputs_do_not_change_weights(self):
        m=OnlineExpertMixture()
        before=m.weights()
        self.assertFalse(m.update((0.2,None,0.3),1.0))
        self.assertFalse(m.update((0.2,0.5,0.3),float("nan")))
        self.assertEqual(before,m.weights())

if __name__=="__main__":unittest.main()
