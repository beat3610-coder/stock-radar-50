"""Market-level two-year reproducibility check for direction forecasts.

Evaluate fixed and quarterly-refitted models on identical *past* signals.
The model configuration is frozen in advance for this protocol: it is not
selected from either evaluation window. All training labels mature before
their fit cutoff. This remains retrospective research with survivor bias.
"""
from nonlinear_direction import market_features,vectors,_fit,_predict,_stats,_causal_recalibration
import numpy as np
from price_volume_features import interact_rows

# This is a fixed protocol, not a winner tuned on 2025/2026 outcomes.
FROZEN_CONFIG={"model":"extratrees","recent_mature_window":800,
               "prior_adaptation_strength":0.5,"temperature":1.0,
               "prior_blend":0.25}
WINDOWS={
 "2025":(("2025-01-01","2025-04-01","2025-07-01","2025-10-01"),
         ("2025-04-01","2025-07-01","2025-10-01","2026-01-01")),
 "2026":(("2026-01-01","2026-04-01","2026-07-01","2026-09-01"),
         ("2026-04-01","2026-07-01","2026-09-01","2026-11-01"))
}
def _one_fit(train,test,interaction=False):
    if interaction:
        train=interact_rows(train)
        test=interact_rows(test)
    model=_fit(train,FROZEN_CONFIG["model"])
    prior=np.bincount([r[2] for r in train],minlength=3).astype(float)+1
    prior/=prior.sum()
    return _causal_recalibration(test,train,_predict(model,test),prior,
        FROZEN_CONFIG["recent_mature_window"],
        FROZEN_CONFIG["prior_adaptation_strength"],
        FROZEN_CONFIG["temperature"],
        prior_blend=FROZEN_CONFIG["prior_blend"])

def _diagnostic(y,p):
    s=_stats(np.asarray(y,dtype=int),np.asarray(p))
    counts=s["predicted"]
    s["three_class_coverage"]=all(v>=10 for v in counts.values())
    return s

def run_two_year_robustness(rows):
    data=vectors(rows,market_features(rows))
    reports={}
    for year,(starts,ends) in WINDOWS.items():
        base=[r for r in data if r[0]["label_available_date"]<starts[0]]
        # Exclude outcomes not known by end of year for a fair 2025 audit.
        end_of_window=ends[-1]
        segments=[]
        for start,end in zip(starts,ends):
            period=[r for r in data if start<=r[0]["signal_date"]<end and
                    r[0]["label_available_date"]<end_of_window]
            segments.append((start,end,period))
        if len(base)<240 or len({r[2] for r in base})<3 or sum(len(x[2]) for x in segments)<100:
            reports[year]={"status":"INSUFFICIENT","base_mature":len(base)}
            continue
        all_test=[item for _,_,period in segments for item in period]
        fixed_probs=_one_fit(base,all_test)
        y=[r[2] for r in all_test]
        if not len(y):
            reports[year]={"status":"INSUFFICIENT"}
            continue
        rolling_probs=[]
        interaction_probs=[]
        periods=[]
        for start,end,period in segments:
            if not period:
                periods.append({"start":start,"status":"NO_DATA"})
                continue
            matured=[r for r in data if r[0]["label_available_date"]<start]
            p=_one_fit(matured,period)
            p_interaction=_one_fit(matured,period,interaction=True)
            rolling_probs.extend(list(p))
            interaction_probs.extend(list(p_interaction))
            periods.append({"start":start,"end":end,"train_mature":len(matured),
                "samples":len(period),"accuracy_pct":_stats(
                    np.asarray([r[2] for r in period],dtype=int),p)["accuracy_pct"]})
        fixed=_diagnostic(y,fixed_probs)
        rolling=_diagnostic(y,rolling_probs)
        extended=_diagnostic(y,interaction_probs)
        reports[year]={
            "status":"RESEARCH_ONLY",
            "paired_samples":len(y),
            "frozen_config":FROZEN_CONFIG,
            "fixed":fixed,"rolling":rolling,"price_volume":extended,"periods":periods,
            "price_volume_accuracy_change_pp":round(extended["accuracy_pct"]-rolling["accuracy_pct"],2),
            "price_volume_macro_f1_change_pp":round(extended["macro_f1_pct"]-rolling["macro_f1_pct"],2),
            "price_volume_brier_change":round(extended["brier"]-rolling["brier"],6),
            "direction_f1_delta_pp":{label:round(extended["direction_f1_pct"][label]-rolling["direction_f1_pct"][label],2) for label in extended["direction_f1_pct"]},
            "accuracy_change_pp":round(rolling["accuracy_pct"]-fixed["accuracy_pct"],2),
            "macro_f1_change_pp":round(rolling["macro_f1_pct"]-fixed["macro_f1_pct"],2),
            "brier_change":round(rolling["brier"]-fixed["brier"],6),
            "rolling_better_brier":rolling["brier"]<fixed["brier"]}
    eligible=[v for v in reports.values() if v.get("status")=="RESEARCH_ONLY"]
    price_volume_gate=(len(eligible)==2 and all(r["price_volume_brier_change"]<0 and r["price_volume_accuracy_change_pp"]>=0 and r["price_volume_macro_f1_change_pp"]>=-0.5 for r in eligible))
    return {"status":"RESEARCH_ONLY" if len(eligible)==2 else "INCOMPLETE",
            "independent_forward_test":False,
            "price_volume_research_gate_passed":price_volume_gate,
            "live_approved":False,
            "frozen_configuration":FROZEN_CONFIG,
            "calendar_windows":reports,
            "two_years_probability_improved":len(eligible)==2 and all(
                v["rolling_better_brier"] for v in eligible),
            "limitations":["Earlier periods were already inspected in ongoing research",
                           "Present-day constituents introduce survivorship bias",
                           "10-session overlapping outcomes not independent",
                           "No trading transaction costs measured"]}
