"""Synthetic tests: feature causality, nonlinear fit and no diagnostic-label selection."""
import unittest
from datetime import date,timedelta
from test_regime_direction import event
from nonlinear_direction import market_features,vectors,run_nonlinear,_causal_recalibration
from rolling_refit_direction import run_refit

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
    def test_online_prior_excludes_current_and_future_labels(self):
        import numpy as np
        rows=self.samples()[:100]
        ctx=market_features(rows)
        data=vectors(rows,ctx)
        # Same signal timestamp: changing outcome must not affect prediction.
        ref=np.array([1/3,1/3,1/3])
        p=np.tile(ref,(len(data),1))
        a=_causal_recalibration(data[:30],data[30:],p[:30],ref,alpha=1.0)
        changed=[(r,x,0) for r,x,y in data[:30]]
        b=_causal_recalibration(changed,data[30:],p[:30],ref,alpha=1.0)
        self.assertTrue(np.allclose(a[0],b[0]))
    def test_base_rate_blend_is_normalized_and_causal(self):
        import numpy as np
        rows=self.samples()[:100]
        data=vectors(rows,market_features(rows))
        ref=np.array([0.2,0.5,0.3])
        probs=np.tile(np.array([0.8,0.1,0.1]),(30,1))
        a=_causal_recalibration(data[:30],data[30:],probs,ref,
                                alpha=0.5,prior_blend=0.5)
        self.assertTrue(np.allclose(a.sum(axis=1),1.0))
        self.assertTrue(np.all(a>=0))
        changed=[(r,v,0) for r,v,y in data[:30]]
        b=_causal_recalibration(changed,data[30:],probs,ref,
                                alpha=0.5,prior_blend=0.5)
        self.assertTrue(np.allclose(a[0],b[0]))
    def test_rolling_refit_uses_only_matured_labels(self):
        rows=self.samples()
        params={"model":"extratrees","recent_mature_window":800,
                "prior_adaptation_strength":0.5,"temperature":1.0,"prior_blend":0.25}
        a=run_refit(rows,params,eval_end="2026-11-01")
        self.assertEqual(a["status"],"RESEARCH_ONLY")
        changed=[{**r,"direction_label":"DOWN"} if r["signal_date"]>="2026-09-01"
                 else r for r in rows]
        b=run_refit(changed,params,eval_end="2026-11-01")
        # Earlier periods cannot change when late outcomes are modified.
        self.assertEqual(a["periods"][0]["diagnostic"],
                         b["periods"][0]["diagnostic"])
        self.assertGreater(a["samples"],0)
    def test_three_class_probabilities_have_finite_metrics(self):
        rows=self.samples()
        result=run_nonlinear(rows,train_end="2024-09-01",
                             validation_end="2025-03-01",eval_end="2026-01-01")
        self.assertLessEqual(result["diagnostic"]["accuracy_pct"],100)
        self.assertGreaterEqual(result["diagnostic"]["brier"],0)
        self.assertEqual(sum(result["diagnostic"]["predicted"].values()),
                         result["diagnostic"]["samples"])
        self.assertIn(result["chosen_on_validation"]["model"],
                      ("histgb","extratrees","ensemble50"))
        self.assertIn("selective_signal",result["diagnostic"])
        chosen_signal=result["diagnostic"]["selective_signal"]
        self.assertGreaterEqual(chosen_signal["coverage_pct"],0)
        self.assertLessEqual(chosen_signal["coverage_pct"],100)
        if not chosen_signal["validation_target_met"]:
            self.assertEqual(chosen_signal["signals"],0)

if __name__=="__main__":
    unittest.main()
