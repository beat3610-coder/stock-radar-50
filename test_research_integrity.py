import unittest
from research_integrity import market_health, certify_market, certify_all

GOOD={"requested":100,"loaded":100,"errors":[],
      "adaptive_plans":{"selective_challenger":{
        "validation_eligible":True,
        "windows":{"diagnostic":{"actual_win_pct":65,"mean_net_pct":1.2,
                 "profit_factor":1.4,"trades":350,"coverage_pct":25}}}}}

class TestResearchIntegrity(unittest.TestCase):
    def test_good_is_only_research(self):
        report=certify_market(GOOD)
        self.assertTrue(report["reached"])
        self.assertFalse(report["live_approved"])
        self.assertFalse(report["independent_holdout"])
    def test_stale_previous_not_approved(self):
        old={**GOOD,"learning_attempt_status":"RETAINED_PREVIOUS_DATA_FAILURE",
             "latest_attempt_loaded":0}
        self.assertFalse(certify_market(old)["reached"])
        self.assertTrue(market_health(old)["retained_previous_model"])
    def test_partial_coverage_not_approved(self):
        partial={**GOOD,"loaded":99}
        self.assertFalse(certify_market(partial)["reached"])
    def test_invalid_metrics_not_approved(self):
        bad={**GOOD,"adaptive_plans":{"selective_challenger":{
             "validation_eligible":True,
             "windows":{"diagnostic":{"actual_win_pct":float("nan"),
                 "mean_net_pct":1,"profit_factor":1.5,"trades":350,"coverage_pct":30}}}}}
        self.assertFalse(certify_market(bad)["reached"])
    def test_missing_market_not_all(self):
        states,all_reached=certify_all({"markets":{"KR":GOOD,"KQ":GOOD}})
        self.assertFalse(all_reached)
        self.assertFalse(states["US"]["fresh_training"])
if __name__=="__main__":unittest.main()
