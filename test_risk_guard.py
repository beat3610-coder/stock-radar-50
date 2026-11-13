"""Risk-screen regression tests. Synthetic fixtures are never performance proof."""
import unittest
from risk_guard import screen_candidate, RESEARCH, RESEARCH_CAUTION, HOLD

BASE=dict(signal="매수 검토",market="KR",price=100.,score=85.,
          stop=95.,target=112.,plan={"history_samples":90,
          "historical_net_expectancy_pct":2.0},
          rsi=58.,atr=2.,trend_ok=True,ma20=99.,ma60=96.,ma200=92.,
          previous_ma200=90.,macdh=1.,adx=28.,pdi=30.,mdi=11.)

def screen(**kwargs):
    args={**BASE,**kwargs}
    return screen_candidate(**args)

class RiskGateTests(unittest.TestCase):
    def test_valid_candidate_is_research_only(self):
        self.assertEqual(screen()[0],RESEARCH)
    def test_holds_never_upgraded_to_buy(self):
        self.assertEqual(screen(signal="축소·매도 검토")[0],"축소·매도 검토")
    def test_downtrend_blocks_even_very_high_score(self):
        self.assertEqual(screen(score=100, ma200=105)[0],HOLD)
    def test_missing_values_fail_closed(self):
        self.assertEqual(screen(atr=None)[0],HOLD)
    def test_market_specific_risk(self):
        self.assertEqual(screen(market="US",score=100,stop=86)[0],HOLD)
    def test_cost_adjusted_negative_history_blocks(self):
        self.assertEqual(screen(plan={"history_samples":90,
                                     "historical_net_expectancy_pct":-1})[0],HOLD)
    def test_insufficient_observations_block(self):
        self.assertEqual(screen(plan={"history_samples":10,
                                     "historical_net_expectancy_pct":3})[0],HOLD)
    def test_narrow_profit_to_risk_blocks(self):
        self.assertEqual(screen(target=105)[0],RESEARCH_CAUTION)
    def test_moderate_risk_warns_without_fabricating_buy(self):
        self.assertEqual(screen(market="US",score=65,stop=92)[0],RESEARCH_CAUTION)
    def test_unsupported_market_blocks(self):
        self.assertEqual(screen(market="unknown")[0],HOLD)

if __name__=="__main__":
    unittest.main()
