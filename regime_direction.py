"""Causal three-way regime challenger for STOCK RADAR 50 research.

The pooled incumbent and regime ensemble predict the exact same samples.
Outcomes update all models ONLY once their label-availability date is earlier
than the next signal. Historical reports are diagnostic, not live approval.
"""
from direction3 import (
    LABELS, OnlineDirectionModel, _empty_direction_stat,
    _record_direction, _finalize_direction_stat, _macro_f1
)

REGIMES=("UPTREND","DOWNTREND","RANGE")

def classify_regime(row):
    """Features observed at signal close, never forward returns."""
    try:
        spread=float(row["signal_ma20_ma60_pct"])
        above=float(row["signal_price_ma200_pct"])
        adx=float(row["signal_adx"])
        width=float(row["signal_bb_width_pct"])
    except (KeyError,ValueError,TypeError):
        return None
    import math
    if not all(math.isfinite(x) for x in (spread,above,adx,width)):
        return None
    # Rules fixed before observing any outcomes, with distinct range behavior.
    if adx<20 or (abs(spread)<0.8 and width<8):
        return "RANGE"
    if spread>0 and above>0:
        return "UPTREND"
    if spread<0 and above<0:
        return "DOWNTREND"
    return "RANGE"

def _bucket(signal,train_end,validation_end):
    return ("train" if signal<train_end else
            "validation" if signal<validation_end else "diagnostic")

def _summary(stats):
    for s in stats.values():
        _finalize_direction_stat(s)
        s["macro_f1_pct"]=_macro_f1(s)
    return stats

def score_regime_direction(rows, *, train_end="2025-11-01",
                           validation_end="2026-04-01",
                           eval_end="2026-11-01", config=None,
                           regime_weight=0.55):
    """Same-observation pooled vs mixture benchmark (no tuning on holdout).

    The initial pooled model is unchanged by the existence of regime experts.
    Expert predictions are ignored until their own causal minimum sample count.
    Both predictions are scored for the same labeled signal.
    """
    config=config or {}
    params={
        "learning_rate":float(config.get("learning_rate",0.13)),
        "l2":float(config.get("l2",0.06)),
        "prior_mix":float(config.get("prior_mix",0.15)),
        "sideways_weight":float(config.get("sideways_weight",1.0)),
    }
    weight=max(0.0,min(1.0,float(regime_weight)))
    pooled=OnlineDirectionModel(**params)
    experts={k:OnlineDirectionModel(**params) for k in REGIMES}
    ordered=sorted(rows,key=lambda t:(t["signal_date"],t["ticker"]))
    finished=iter(sorted(rows,key=lambda t:(t["label_available_date"],t["ticker"])))
    mature=next(finished,None)
    summary={model:{window:_empty_direction_stat() for window in
                    ("train","validation","diagnostic")}
             for model in ("pooled","regime")}
    per_regime={r:{model:_empty_direction_stat() for model in ("pooled","regime")}
                for r in REGIMES}
    skipped={"missing_features":0,"warmup":0,"late_labels":0}
    expert_coverage=0
    for event in ordered:
        signal=event["signal_date"]
        while mature is not None and mature["label_available_date"]<signal:
            pooled.update(mature)
            regime=classify_regime(mature)
            if regime is not None:
                experts[regime].update(mature)
            mature=next(finished,None)
        if event["label_available_date"]>=eval_end:
            skipped["late_labels"]+=1
            continue
        regime=classify_regime(event)
        if regime is None:
            skipped["missing_features"]+=1
            continue
        base=pooled.predict(event)
        if base is None:
            skipped["warmup"]+=1
            continue
        local=experts[regime].predict(event)
        if local is not None:
            expert_coverage+=1
        mixed=([(1-weight)*p+weight*q for p,q in zip(base,local)]
               if local is not None else base)
        label=event.get("direction_label")
        if label not in LABELS:
            skipped["missing_features"]+=1
            continue
        actual=LABELS.index(label)
        window=_bucket(signal,train_end,validation_end)
        prior=pooled.prior()
        for model,probs in (("pooled",base),("regime",mixed)):
            predicted=max(range(3),key=lambda j:probs[j])
            _record_direction(summary[model][window],actual,predicted,probs,prior)
            if window=="diagnostic":
                _record_direction(per_regime[regime][model],
                                  actual,predicted,probs,prior)
    summary={key:_summary(stat) for key,stat in summary.items()}
    per_regime={reg:_summary(stat) for reg,stat in per_regime.items()}
    comparison={}
    for window in ("train","validation","diagnostic"):
        old=summary["pooled"][window]
        new=summary["regime"][window]
        comparison[window]={
            "paired_samples":old["total"],
            "pooled_macro_f1_pct":old["macro_f1_pct"],
            "regime_macro_f1_pct":new["macro_f1_pct"],
            "pooled_brier":old["multiclass_brier"],
            "regime_brier":new["multiclass_brier"],
            "pooled_accuracy_pct":old["accuracy_pct"],
            "regime_accuracy_pct":new["accuracy_pct"],
            "brier_improved":(
                old["multiclass_brier"] is not None
                and new["multiclass_brier"] is not None
                and new["multiclass_brier"]<old["multiclass_brier"]),
            "macro_f1_improved":(
                old["macro_f1_pct"] is not None
                and new["macro_f1_pct"] is not None
                and new["macro_f1_pct"]>old["macro_f1_pct"]),
        }
    return {
        "research_only":True,"live_approved":False,
        "independent_holdout":False,
        "model":"causal_pooled_plus_regime_mixture",
        "regime_weight":weight,
        "config_source":"historical champion / whole-history diagnostic",
        "incumbent_config":params,
        "summary":summary,
        "regimes":per_regime,
        "comparison":comparison,
        "skipped":skipped,
        "regime_expert_predictions":expert_coverage,
        "warning":("기존 우승 수식과 동일 과거 데이터를 재사용한 연구 비교. "
                   "검증/진단 결과로 자동 실전 배포하거나 우승식을 교체하지 않음."),
    }
