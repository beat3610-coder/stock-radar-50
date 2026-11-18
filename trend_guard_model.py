"""Trend-first entry gating: compare guarded and unguarded entries on same
candidate population, with development-only model/policy selection.

Conservative no-trade is a valid outcome and is always reported with coverage.
Historical diagnostics have been inspected previously and are NOT pristine.
"""
import numpy as np
from collections import Counter
from net_edge_model import _fit, _score, _window, _losses, POLICIES, MIN_FIT
from feature_selector60 import _clean
from profit_selector51 import TRAIN, VALID, END, stats

GUARDS={
 "no_extra_guard": lambda r: True,
 "trend_relaxed": lambda r: (r["signal_ma20_ma60_pct"]>=-0.5 and
    40<=r["signal_rsi"]<=75 and r["signal_adx"]>=14 and
    r["signal_5d_return_pct"]>=-5),
 "trend_basic": lambda r: (r["signal_ma20_ma60_pct"]>0 and
    43<=r["signal_rsi"]<=72 and r["signal_adx"]>=18 and
    r["signal_5d_return_pct"]>-3),
 "trend_quality": lambda r: (r["signal_ma20_ma60_pct"]>=1 and
    48<=r["signal_rsi"]<=68 and r["signal_adx"]>=22 and
    r["signal_atr_pct"]<=4.0 and
    0.85<=r["signal_volume_ratio"]<=2.5 and
    -2<=r["signal_5d_return_pct"]<=6),
 "trend_volatility": lambda r: (r["signal_atr_pct"]<=3.0 and
    r["signal_5d_return_pct"]>=0 and r["signal_adx"]>=20),
 "trend_momentum": lambda r: (r["signal_ma20_ma60_pct"]>=0 and
    0<=r["signal_5d_return_pct"]<=8 and
    45<=r["signal_rsi"]<=70),
 "trend_lowrisk": lambda r: (r["signal_ma20_ma60_pct"]>0 and
    48<=r["signal_rsi"]<=68 and r["signal_adx"]>=20 and
    r["signal_atr_pct"]<=3.0 and
    abs(r["entry_gap_pct"])<=1.0 and
    r["entry_net_rr"]>=1.5),
}
# All gating constraints use signal-time observations, except next-open entry
# gap/RR which are known at execution, never future prices/exit labels.
def guarded(rows,guard):
    fn=GUARDS[guard]
    return [z for z in rows if fn(z[0])]

def audit(trades,market=None):
    items=_clean(trades,market)
    dev=[z for z in items if z[0]["signal_date"]<TRAIN and z[0]["exit_date"]<TRAIN]
    val=_window(items,TRAIN,VALID)
    diag=_window(items,VALID,END)
    baseline={k:stats([z[0] for z in v]) for k,v in
              (("train",dev),("validation",val),("diagnostic",diag))}
    report={"market":market,"target_win_pct":60,"status":"INSUFFICIENT_TRAINING",
       "live_approved":False,"selected":None,"baseline":baseline,
       "guard_diagnostic_comparison":{name:{"summary":stats([z[0] for z in guarded(diag,name)]),
          "coverage_pct":round(100*len(guarded(diag,name))/len(diag),2) if diag else None}
          for name in GUARDS},
       "note":"Diagnostic comparisons are reporting only, never used for selection."}
    dates=sorted(set(z[0]["signal_date"] for z in dev))
    if len(dev)<90 or len(dates)<12:return report
    cuts=[dates[int(len(dates)*q)] for q in (.45,.70)]
    folds=[]
    for i,cut in enumerate(cuts):
        end=cuts[i+1] if i+1<len(cuts) else TRAIN
        fit=[z for z in dev if z[0]["exit_date"]<cut]
        check=_window(dev,cut,end)
        if len(fit)<MIN_FIT or len(check)<15:
            report["status"]="INSUFFICIENT_TIME_FOLDS"
            return report
        folds.append((fit,check))
    comparisons=[]
    rejection_counts=Counter()
    candidate_diagnostics=[]
    for kind in ("ridge","forest"):
        trained=[(_fit(fit,kind),check) for fit,check in folds]
        if any(m is None for m,_ in trained):continue
        for policy in POLICIES:
            plain_folds=[_score(m,check,policy) for m,check in trained]
            plain=[r for fold in plain_folds for r in fold]
            plain_s=stats(plain)
            for guard in GUARDS:
                selected=[r for r in plain if GUARDS[guard](r)]
                scores=stats(selected)
                fold_scores=[stats([r for r in fold if GUARDS[guard](r)])
                             for fold in plain_folds]
                # Reject candidates that depend on a single lucky development fold.
                unstable=(any(f["n"]<8 for f in fold_scores) or
                          any(f["mean_net_pct"] is None or f["mean_net_pct"]<=0
                              for f in fold_scores))
                reason = ("unstable_folds" if scores["n"]>=25 and unstable else ("too_few_trades" if scores["n"]<25 else
                          "nonpositive_net" if scores["mean_net_pct"]<=0 else
                          "no_improvement" if guard!="no_extra_guard" and plain_s["n"]>0 and
                          (scores["win_pct"]<=plain_s["win_pct"] and
                           scores["mean_net_pct"]<=plain_s["mean_net_pct"]) else "eligible"))
                rejection_counts[reason]+=1
                candidate_diagnostics.append({"model":kind,"policy":policy,
                    "guard":guard,"reason":reason,"development":scores,
                    "development_folds":fold_scores,
                    "unguarded":plain_s})
                if reason!="eligible":continue
                # Prefer net expectancy, penalize small samples and weak win rates.
                # Rank only pre-validation development results.
                # Lower-tail penalty prevents large winners masking frequent stops.
                downside=np.mean([min(0,r["net_pct"])**2 for r in selected])**0.5
                reward=(scores["mean_net_pct"]-1.5/np.sqrt(scores["n"])
                        -0.02*max(0,60-scores["win_pct"])-0.15*downside)
                comparisons.append((reward,scores["n"],kind,policy,guard,scores,plain_s))
    report["development_rejection_counts"]=dict(rejection_counts)
    report["development_candidates"]=candidate_diagnostics
    if not comparisons:
        report["status"]="NO_TRAINING_EDGE"
        # Explain rejection without presenting a rejected candidate as a buy signal.
        # This is a development-only diagnosis; never relax deployment standards.
        eligible_counts=sorted(
            (c for c in candidate_diagnostics if c["development"]["n"]>=15),
            key=lambda c:(-c["development"]["n"],c["model"],c["policy"],c["guard"]))
        report["largest_rejected_candidate"]=eligible_counts[0] if eligible_counts else None
        report["rejected_candidate_count"]=len(candidate_diagnostics)
        return report
    comparisons.sort(key=lambda z:(-z[0],-z[1],z[2],z[3],z[4]))
    _,_,kind,policy,guard,internal,unfiltered_internal=comparisons[0]
    trained=_fit(dev,kind)
    if trained is None:return report
    def metrics_at(rows):
        raw=_score(trained,rows,policy)
        selected=[z for z in raw if GUARDS[guard](z)]
        return {"unguarded":stats(raw),"selected":stats(selected),
            "population":len(rows),
            "coverage_pct":round(100*len(selected)/len(rows),2) if rows else None,
            "selected_loss_attribution":_losses(selected)}
    v=metrics_at(val)
    d=metrics_at(diag)
    score=v["selected"]
    passed=(score["n"]>=30 and score["win_pct"]>=60 and
      score["mean_net_pct"]>0 and
      (score["profit_factor"] is None and score["win_pct"]==100 or
       score["profit_factor"] is not None and score["profit_factor"]>1))
    report.update({"status":"VALIDATION_PASS_RESEARCH_ONLY" if passed else "VALIDATION_REJECTED",
       "selected":{"model":kind,"base_policy":policy,"guard":guard,
         "development_guarded":internal,"development_unguarded":unfiltered_internal,
         "validation":v,"diagnostic":d,"validation_pass":passed}})
    return report
