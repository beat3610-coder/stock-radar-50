"""Deterministic regression tests for the causal regime challenger."""
import unittest
from datetime import date,timedelta
from direction3 import FEATURE_KEYS, LABELS
from regime_direction import classify_regime, score_regime_direction

def event(i, *, regime="RANGE", label="SIDEWAYS", maturity=11):
    day=date(2024,1,1)+timedelta(days=i)
    row={
        "ticker":str(i%10).zfill(6),
        "signal_date":day.isoformat(),
        "label_available_date":(day+timedelta(days=maturity)).isoformat(),
        "direction_label":label,
    }
    row.update({
        "signal_rsi":55, "signal_adx":12 if regime=="RANGE" else 30,
        "signal_atr_pct":2.5,
        "signal_ma20_ma60_pct":0 if regime=="RANGE" else
            (3 if regime=="UPTREND" else -3),
        "signal_volume_ratio":1.3, "signal_5d_return_pct":2,
        "signal_macdh_atr":0, "signal_price_ma200_pct":0 if regime=="RANGE" else
            (3 if regime=="UPTREND" else -3),
        "signal_bb_position":0, "signal_rsi_wilder":55,
        "signal_adx_wilder":25, "signal_cmf20":0,
        "signal_donchian20_pos":0.5,"signal_roc20":0,
        "signal_bb_width_pct":7,
    })
    assert all(k in row for k in FEATURE_KEYS)
    return row

class RegimeDirectionTest(unittest.TestCase):
    def test_classifier(self):
        for regime in ("RANGE","UPTREND","DOWNTREND"):
            self.assertEqual(classify_regime(event(0,regime=regime)),regime)
        item=event(0)
        item["signal_adx"]=float("nan")
        self.assertIsNone(classify_regime(item))

    def test_same_sample_paired_and_probabilities(self):
        regimes=("RANGE","UPTREND","DOWNTREND")
        labels=("SIDEWAYS","UP","DOWN")
        rows=[event(i,regime=regimes[i%3],label=labels[i%3])
              for i in range(550)]
        out=score_regime_direction(rows,train_end="2024-06-01",
            validation_end="2025-01-01",eval_end="2026-01-01")
        self.assertTrue(out["research_only"])
        self.assertFalse(out["live_approved"])
        self.assertFalse(out["independent_holdout"])
        for window in ("train","validation","diagnostic"):
            comp=out["comparison"][window]
            self.assertEqual(comp["paired_samples"],
                out["summary"]["regime"][window]["total"])
            self.assertEqual(comp["paired_samples"],
                out["summary"]["pooled"][window]["total"])
        self.assertGreater(sum(x["paired_samples"] for x in out["comparison"].values()),0)
        self.assertGreater(out["regime_expert_predictions"],0)

    def test_calibration_choice_ignores_later_diagnostic_labels(self):
        regimes=("RANGE","UPTREND","DOWNTREND")
        labels=("SIDEWAYS","UP","DOWN")
        rows=[event(i,regime=regimes[i%3],label=labels[i%3])
              for i in range(550)]
        params={"train_end":"2024-06-01",
                "validation_end":"2025-01-01","eval_end":"2026-01-01"}
        a=score_regime_direction(rows,**params)
        altered=[{**r,"direction_label":"DOWN" if r["direction_label"]!="DOWN" else "UP"}
                 if r["signal_date"]>="2025-01-01" else r for r in rows]
        b=score_regime_direction(altered,**params)
        self.assertEqual(a["calibrated_selected_on_validation"],
                         b["calibrated_selected_on_validation"])
        self.assertIn("calibrated_diagnostic",a)
        self.assertEqual(a["calibrated_diagnostic"]["samples"],
                         a["summary"]["pooled"]["diagnostic"]["total"])
    def test_no_unmatured_future_training(self):
        # No labels mature before next signal; neither learner can predict.
        rows=[event(i, maturity=1000) for i in range(150)]
        result=score_regime_direction(rows,eval_end="2030-01-01")
        self.assertEqual(sum(x["paired_samples"] for x in
            result["comparison"].values()),0)
        self.assertEqual(result["skipped"]["warmup"],len(rows))
        self.assertEqual(result["regime_expert_predictions"],0)

    def test_invalid_regime_is_skipped(self):
        rows=[event(i) for i in range(150)]
        rows[100]["signal_bb_width_pct"]=float("nan")
        out=score_regime_direction(rows,eval_end="2026-01-01")
        self.assertEqual(out["skipped"]["missing_features"],1)
if __name__=="__main__":
    unittest.main()
