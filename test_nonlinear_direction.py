"""Synthetic tests: feature causality, nonlinear fit and no diagnostic-label selection."""
import unittest
from datetime import date,timedelta
from test_regime_direction import event
from nonlinear_direction import market_features,vectors,run_nonlinear

class NonlinearTests(unittest.TestCase):
    def samples(self):
        start=date(2024,1,1)
        rows=[]
        regimes=("UPTREND","DOWNTREND","RANGE")
        labels=("UP","DOWN","SIDEWAYS")
        for day in range(800):
            if day%7 in (5,6):
                continue
            for ticker in range(5):
                ix=(day+ticker)%3
                r=event(day,regime=regimes[ix],label=labels[ix],maturity=11)
                dt=start+timedelta(days=day)
                r["ticker"]=f"S{ticker:05d}"
                r["signal_date"]=dt.isoformat()
                r["label_available_date"]=(dt+timedelta(days=11)).isoformat()
                rows.append(r)
        return rows
    def test_same_day_context(self):
        rows=self.samples()[:100]
        context=market_features(rows)
        self.assertTrue(context)
        data=vectors(rows,context)
        self.assertEqual(len(data),len(rows))
        self.assertEqual(len(data[0][1]),22)
    def test_holdout_relabel_does_not_change_selected_parameters(self):
        rows=self.samples()
        args=dict(train_end="2024-09-01",validation_end="2025-03-01",eval_end="2026-01-01")
        a=run_nonlinear(rows,**args)
        self.assertEqual(a["status"],"RESEARCH_ONLY")
        changed=[{**r,"direction_label":"DOWN"} if r["signal_date"]>="2025-03-01" else r for r in rows]
        b=run_nonlinear(changed,**args)
        self.assertEqual(a["chosen_on_validation"],b["chosen_on_validation"])
        self.assertGreater(a["diagnostic"]["samples"],100)
    def test_three_class_probabilities_have_finite_metrics(self):
        rows=self.samples()
        result=run_nonlinear(rows,train_end="2024-09-01",
                             validation_end="2025-03-01",eval_end="2026-01-01")
        self.assertLessEqual(result["diagnostic"]["accuracy_pct"],100)
        self.assertGreaterEqual(result["diagnostic"]["brier"],0)
        self.assertEqual(sum(result["diagnostic"]["predicted"].values()),
                         result["diagnostic"]["samples"])

if __name__=="__main__":
    unittest.main()
