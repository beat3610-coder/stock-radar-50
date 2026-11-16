import unittest
from direction3 import _empty_direction_stat, _record_direction, _finalize_direction_stat, _weakest_f1, _candidate_rank

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
if __name__=="__main__":unittest.main()
