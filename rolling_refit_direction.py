"""Quarterly walk-forward refits of an already validation-selected model.

Never choose model parameters using diagnostic labels. Each period only trains
on labels that matured before its beginning, and updates calibration only from
strictly matured observations. Research only, not a future-performance claim.
"""
from nonlinear_direction import market_features,vectors,_fit,_predict,_stats,_causal_recalibration
import numpy as np

REFIT_STARTS=("2026-04-01","2026-07-01","2026-09-01")
REFIT_ENDS=("2026-07-01","2026-09-01","2026-11-01")

def run_refit(rows, chosen, eval_end="2026-11-01",\n              starts=REFIT_STARTS, ends=REFIT_ENDS):
    data=vectors(rows,market_features(rows))
    combined=[]
    periods=[]
    for start,end in zip(starts,ends):
        if start>=eval_end:break
        end=min(end,eval_end)
        train=[r for r in data if r[0]["label_available_date"]<start]
        test=[r for r in data if start<=r[0]["signal_date"]<end
              and r[0]["label_available_date"]<eval_end]
        if len(train)<240 or not test or len({r[2] for r in train})<3:
            periods.append({"start":start,"status":"INSUFFICIENT",
                            "train":len(train),"test":len(test)})
            continue
        kind=chosen["model"]
        if kind=="ensemble50":
            p=(_predict(_fit(train,"histgb"),test)+
               _predict(_fit(train,"extratrees"),test))/2
        else:
            p=_predict(_fit(train,kind),test)
        prior=np.bincount([r[2] for r in train],minlength=3).astype(float)+1
        prior/=prior.sum()
        calibrated=_causal_recalibration(
            test,train,p,prior,chosen["recent_mature_window"],
            chosen["prior_adaptation_strength"],chosen["temperature"],
            prior_blend=chosen["prior_blend"])
        combined.extend((r[0]["ticker"],r[0]["signal_date"],r[2],prob)
                        for r,prob in zip(test,calibrated))
        periods.append({"start":start,"end":end,"train_mature":len(train),
                        "test":len(test),"status":"RESEARCH_ONLY",
                        "diagnostic":_stats(np.asarray([r[2] for r in test]),calibrated)})
    if not combined:
        return {"status":"INSUFFICIENT","periods":periods}
    y=np.array([r[2] for r in combined],dtype=int)
    p=np.array([r[3] for r in combined])
    return {"status":"RESEARCH_ONLY","refit_strategy":"quarterly fixed starts",
            "model_selected_before_diagnostic":chosen,
            "periods":periods,"diagnostic":_stats(y,p),
            "samples":len(combined),
            "per_ticker":{ticker:_stats(
                 np.array([r[2] for r in combined if r[0]==ticker],dtype=int),
                 np.array([r[3] for r in combined if r[0]==ticker]))
                 for ticker in sorted({r[0] for r in combined})}}
