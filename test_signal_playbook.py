"""Test UI playbook math without claims about investment accuracy."""
import unittest
import pandas as pd
from signal_playbook import playbook

class ResearchPlaybookTest(unittest.TestCase):
    def test_safe_no_signal_for_missing_history(self):
        self.assertEqual(playbook(pd.DataFrame(), "KR")["setups"],[])
    def test_stop_target_around_signal_close_and_cap(self):
        dates=pd.bdate_range("2024-01-01",periods=221)
        frame=pd.DataFrame({
            "Close":[100.]*220+[105.],"MA20":[100.]*221,
            "MA60":[98.]*221,"MA200":[90.]*221,
            "RSI":[55.]*221,"MACDH":[-0.1]*220+[0.4],
            "ADX":[25.]*221,"PDI":[25.]*221,"MDI":[12.]*221,
            "ATR":[2.]*221,"VOL20":[100.]*221,"Volume":[170.]*221,
            "High":[102.]*220+[106.]
        },index=dates)
        r=playbook(frame,"KR",capital=10_000_000)
        self.assertTrue(r["setups"])
        for idea in r["setups"]:
            self.assertLess(idea["stop"],idea["signal_price"])
            self.assertGreater(idea["target"],idea["signal_price"])
            self.assertLessEqual(idea["capital_used"],2_000_000)
            self.assertLessEqual(idea["planned_loss"],100_000)
            self.assertFalse(r["live_approved"])
    def test_no_live_claim(self):
        dates=pd.bdate_range("2024-01-01",periods=221)
        frame=pd.DataFrame({"Close":[100.]*221,"ATR":[2.]*221,"High":[101.]*221},index=dates)
        # Qualifying requires indicator columns; no claim of a possible buy on invalid data.
        with self.assertRaises((AttributeError,KeyError)):
            playbook(frame,"KR")

if __name__=="__main__":
    unittest.main()
