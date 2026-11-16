import unittest
from direction3 import _empty_direction_stat, _record_direction, _finalize_direction_stat, _weakest_f1, _candidate_rank, score_directions, LABELS, FEATURE_KEYS

class ThreeWayRankingTests(unittest.TestCase):
    def test_sideways_failure_is_not_hidden_by_average(self):
        s=_empty_direction_stat()
        for _ in range(30):
            _record_direction(s,0,0,[0.8,0.1,0.1],[1/3]*3)
        for _ in range(30):
            _record_direction(s,1,1,[0.1,0.8,0.1],[1/3]*3)
        for _ in range(30):
            _record_direction(s,2,0,[0.8,0.1,0.1],[1/3]*3)
        _finalize_direction_stat(s)
        self.assertEqual(s["class_metrics"]["SIDEWAYS"]["recall_pct"],0)
        self.assertEqual(_weakest_f1(s),0)
    def test_worse_brier_is_penalized(self):
        bad={"weakest_class_f1_pct":35,"whole_history_macro_f1_pct":50,
             "worst_period_macro_f1_pct":30,"whole_history_brier":0.8,
             "whole_history_prior_brier":0.65}
        good={"weakest_class_f1_pct":28,"whole_history_macro_f1_pct":43,
              "worst_period_macro_f1_pct":28,"whole_history_brier":0.62,
              "whole_history_prior_brier":0.65}
        self.assertGreater(_candidate_rank(good),_candidate_rank(bad))
    def test_weakest_class_matters(self):
        a={"weakest_class_f1_pct":15,"whole_history_macro_f1_pct":55,
           "worst_period_macro_f1_pct":35,"whole_history_brier":0.6,
           "whole_history_prior_brier":0.7}
        b={**a,"weakest_class_f1_pct":30,"whole_history_macro_f1_pct":53}
        self.assertGreater(_candidate_rank(b),_candidate_rank(a))
# A replay must always report apples-to-apples incumbent comparisons.
class ChampionSelectionTests(unittest.TestCase):
    def test_previous_champion_is_compared_on_identical_observations(self):
        from test_direction3 import row
        training=[row(i,"2025-01-01","2025-01-15",LABELS[i%3]) for i in range(150)]
        evaluation=[row(i+200,"2025-12-01","2025-12-15",LABELS[i%3]) for i in range(60)]
        champion={"learning_rate":0.13,"l2":0.06,"prior_mix":0.15,
                  "sideways_weight":1.0}
        result=score_directions(training+evaluation,previous_best=champion)
        tune=result["tuning"]
        self.assertIsNotNone(tune["previous_champion_same_sample_comparison"])
        self.assertTrue(tune["previous_champion_same_sample_comparison"]["same_observations"])
        self.assertIn("previous_champion_retained_by_guard",tune)

if __name__=="__main__":unittest.main()
