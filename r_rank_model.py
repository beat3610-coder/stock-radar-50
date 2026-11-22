"""Leakage-controlled R-multiple regression/ranking models for audit-63.

Features use signal-close values with fixed domain scaling. Quarterly snapshots
are fitted only from trades whose exits precede the quarter's first calendar
day. The classifier is a train-only logistic calibration of each regressor's
R predictions and is an auxiliary diagnostic, not the ranking target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge

STRATEGIES=("trend_breakout","trend_pullback","momentum_reentry",
            "trend_pullback_balanced","momentum_reentry_balanced",
            "broad_market_ml")
NUMERIC_FEATURES=("signal_rsi","signal_adx","signal_atr_pct",
    "signal_ma20_ma60_pct","signal_volume_ratio","signal_5d_return_pct",
    "signal_trend_200_pct","signal_drawdown_20_pct","entry_net_rr",
    "entry_net_upside_pct","entry_net_downside_pct","signal_stop_atr",
    "market_ma200_gap_pct","market_breadth_pct","signal_return_20d_pct",
    "signal_return_60d_pct","signal_volatility_20d_pct","signal_turnover_log10")
CENTERS={"signal_rsi":50,"signal_adx":25,"signal_atr_pct":3,
    "signal_ma20_ma60_pct":0,"signal_volume_ratio":1.2,"signal_5d_return_pct":0,
    "signal_trend_200_pct":0,"signal_drawdown_20_pct":-3,"entry_net_rr":1.5,
    "entry_net_upside_pct":4,"entry_net_downside_pct":4,"signal_stop_atr":1.8,
    "market_ma200_gap_pct":0,"market_breadth_pct":50,"signal_return_20d_pct":0,
    "signal_return_60d_pct":0,"signal_volatility_20d_pct":3,"signal_turnover_log10":8}
SCALES={"signal_rsi":25,"signal_adx":20,"signal_atr_pct":3,
    "signal_ma20_ma60_pct":8,"signal_volume_ratio":1.5,"signal_5d_return_pct":10,
    "signal_trend_200_pct":30,"signal_drawdown_20_pct":12,"entry_net_rr":2,
    "entry_net_upside_pct":6,"entry_net_downside_pct":6,"signal_stop_atr":2,
    "market_ma200_gap_pct":25,"market_breadth_pct":25,"signal_return_20d_pct":20,
    "signal_return_60d_pct":35,"signal_volatility_20d_pct":5,"signal_turnover_log10":2.5}
INTERACTION_FEATURES=("entry_net_rr",)
FEATURE_COUNT=len(NUMERIC_FEATURES)+len(STRATEGIES)*(1+len(INTERACTION_FEATURES))
FEATURE_LIMIT=30
MIN_TRAIN_SAMPLES=300
MIN_TRAIN_SAMPLES_BY_COHORT={"KOSPI":300,"KOSDAQ":300,"TIGER_ETF":100,"SOXL":100}
MAX_TRAIN_SAMPLES=4000
MIN_VARIANT_SAMPLES=300
VARIANT_SHRINKAGE_PRIOR=100.0
RANK_TARGET="net_pct / entry_net_downside_pct"

# Fixed before evaluating the current validation/diagnostic periods.
MODEL_CONFIGS={
    "ridge_a1":{"kind":"ridge","alpha":1.0},
    "ridge_a10":{"kind":"ridge","alpha":10.0},
    "rf_d4_leaf25":{"kind":"rf","n_estimators":48,"max_depth":4,"min_samples_leaf":25},
    "rf_d6_leaf40":{"kind":"rf","n_estimators":48,"max_depth":6,"min_samples_leaf":40},
    "gbr_d1_lr03":{"kind":"gbr","n_estimators":50,"max_depth":1,"learning_rate":0.03,"min_samples_leaf":25},
    "gbr_d2_lr03":{"kind":"gbr","n_estimators":50,"max_depth":2,"learning_rate":0.03,"min_samples_leaf":25},
    "hgb_leaf7":{"kind":"hgb","max_iter":60,"max_leaf_nodes":7,"min_samples_leaf":30,"l2_regularization":5.0},
    "hgb_leaf15":{"kind":"hgb","max_iter":60,"max_leaf_nodes":15,"min_samples_leaf":40,"l2_regularization":10.0},
}
RANDOM_SEED=3610

if FEATURE_COUNT>FEATURE_LIMIT:
    raise RuntimeError(f"fixed feature count {FEATURE_COUNT} exceeds limit {FEATURE_LIMIT}")

def realized_r(trade):
    try:
        net=float(trade["net_pct"]);risk=float(trade["entry_net_downside_pct"])
        if np.isfinite(net) and np.isfinite(risk) and risk>0:return net/risk
    except (KeyError,TypeError,ValueError,ZeroDivisionError):pass
    return None

def design_matrix(trades):
    """Fixed-scale features, 6 variant one-hots, and variant×net-RR terms."""
    x=np.zeros((len(trades),FEATURE_COUNT),dtype=np.float32)
    nnum=len(NUMERIC_FEATURES)
    for i,t in enumerate(trades):
        for j,key in enumerate(NUMERIC_FEATURES):
            try:value=float(t[key])
            except (KeyError,TypeError,ValueError):value=CENTERS[key]
            if not np.isfinite(value):value=CENTERS[key]
            x[i,j]=np.clip((value-CENTERS[key])/SCALES[key],-4,4)
        strategy=t.get("strategy") or t.get("candidate_variant")
        if strategy not in STRATEGIES and t.get("candidate_family")=="broad":strategy="broad_market_ml"
        if strategy not in STRATEGIES:strategy="broad_market_ml"
        k=STRATEGIES.index(strategy)
        x[i,nnum+k]=1.0
        rr=x[i,NUMERIC_FEATURES.index("entry_net_rr")]
        x[i,nnum+len(STRATEGIES)+k]=rr
    return x

def _estimator(config):
    cfg=MODEL_CONFIGS[config] if isinstance(config,str) else config
    kind=cfg["kind"]
    if kind=="ridge":return Ridge(alpha=cfg["alpha"])
    if kind=="rf":return RandomForestRegressor(n_jobs=1,random_state=RANDOM_SEED,**{k:v for k,v in cfg.items() if k!="kind"})
    if kind=="gbr":return GradientBoostingRegressor(random_state=RANDOM_SEED,loss="huber",**{k:v for k,v in cfg.items() if k!="kind"})
    if kind=="hgb":return HistGradientBoostingRegressor(random_state=RANDOM_SEED,early_stopping=False,
        **{k:v for k,v in cfg.items() if k!="kind"})
    raise ValueError(f"unknown model kind: {kind}")

def _fit_snapshot(train_rows,train_x,config,min_train_samples=MIN_TRAIN_SAMPLES):
    y=np.asarray([realized_r(t) for t in train_rows],dtype=float)
    valid=np.isfinite(y)
    if valid.sum()<min_train_samples:return None
    # Keep a deterministic chronological training window to bound CPU/memory.
    positions=np.flatnonzero(valid)[-MAX_TRAIN_SAMPLES:]
    y=y[positions];x=train_x[positions]
    if np.std(y)<1e-8:return None
    model=_estimator(config);model.fit(x,y)
    # Calibrate win probability from this estimator's own R predictions using
    # a chronological tail that the calibrating regressor did not train on.
    # This keeps AUROC model-specific and avoids in-sample fitted-score leakage.
    labels=(y>0).astype(int)
    calibrator=None
    split=int(len(y)*0.8)
    if len(y)-split>=30 and split>=MIN_TRAIN_SAMPLES//2 and np.unique(labels[split:]).size==2:
        calibration_model=_estimator(config);calibration_model.fit(x[:split],y[:split])
        calibration_r=np.asarray(calibration_model.predict(x[split:]),dtype=float)
        calibrator=LogisticRegression(C=0.2,solver="lbfgs",max_iter=250,random_state=RANDOM_SEED)
        calibrator.fit(calibration_r.reshape(-1,1),labels[split:])
    return {"model":model,"calibrator":calibrator,"train_n":len(y),"train_r":y}

def walk_forward_scores(candidates,config,min_train_samples=MIN_TRAIN_SAMPLES):
    """Quarterly prequential snapshots; all labels have exit_date < anchor."""
    if not candidates:return [],{"candidate_n":0,"scored_n":0,"insufficient_history_n":0,"snapshot_n":0,"max_train_n":0}
    ordered=sorted(candidates,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"],t.get("strategy","")))
    exits=sorted(range(len(ordered)),key=lambda i:(ordered[i]["exit_date"],ordered[i]["ticker"],ordered[i]["entry_date"]))
    xall=design_matrix(ordered);ys=np.asarray([realized_r(t) if realized_r(t) is not None else np.nan for t in ordered])
    scored=[];cursor=0;past_idx=[];last_quarter=None;snapshot=None;insufficient=0;max_train=0;snapshots=0
    for idx,row in enumerate(ordered):
        day=pd.Timestamp(row["signal_date"])
        quarter=day.to_period("Q")
        if quarter!=last_quarter:
            anchor=quarter.start_time.strftime("%Y-%m-%d")
            while cursor<len(exits) and ordered[exits[cursor]]["exit_date"]<anchor:
                past_idx.append(exits[cursor]);cursor+=1
            snapshot=_fit_snapshot([ordered[j] for j in past_idx],xall[past_idx],config,min_train_samples) if len(past_idx)>=min_train_samples else None
            last_quarter=quarter;snapshots+=1
            if snapshot:max_train=max(max_train,snapshot["train_n"])
        if snapshot is None:
            insufficient+=1;continue
        row_x=xall[idx:idx+1]
        pred_r=float(snapshot["model"].predict(row_x)[0])
        cal=snapshot["calibrator"]
        prob=float(cal.predict_proba(np.asarray([[pred_r]]))[0,1]) if cal is not None else float(np.mean(snapshot["train_r"]>0))
        scored.append({**row,"realized_r":float(ys[idx]),"model_predicted_r":pred_r,
            "model_probability":prob,"model_predicted_win_pct":100*prob,
            "model_expected_net_pct":pred_r*float(row["entry_net_downside_pct"]),
            "model_training_n":snapshot["train_n"],"model_config":config})
    return scored,{"candidate_n":len(ordered),"scored_n":len(scored),
        "insufficient_history_n":insufficient,"snapshot_n":snapshots,"max_train_n":max_train,
        "min_training_samples":min_train_samples,"max_training_samples":MAX_TRAIN_SAMPLES,
        "snapshot_cadence":"quarterly; exits strictly before quarter start"}

def make_variant_thresholds(dev_scored,quantiles=(.70,.80,.90)):
    """Variant-specific predicted-R quantiles shrunk toward cohort quantiles."""
    values=np.asarray([float(t["model_predicted_r"]) for t in dev_scored if np.isfinite(float(t.get("model_predicted_r",np.nan)))])
    if not len(values):return {}
    out={f"r_p{int(q*100)}":{"global":float(np.quantile(values,q)),"by_strategy":{}}
         for q in quantiles}
    for strategy in STRATEGIES:
        group=np.asarray([float(t["model_predicted_r"]) for t in dev_scored
                          if t.get("strategy")==strategy and np.isfinite(float(t.get("model_predicted_r",np.nan)))])
        if not len(group):continue
        weight=len(group)/(len(group)+VARIANT_SHRINKAGE_PRIOR)
        for q in quantiles:
            key=f"r_p{int(q*100)}";local=float(np.quantile(group,q));global_cut=out[key]["global"]
            out[key]["by_strategy"][strategy]={"candidate_n":len(group),"shrinkage_weight":round(weight,4),
                "raw_quantile":round(local,6),"threshold":round(weight*local+(1-weight)*global_cut,6)}
    return out

def accepts(trade,policy,thresholds):
    row=thresholds.get(policy)
    if not row:return False
    strategy=trade.get("strategy")
    spec=row.get("by_strategy",{}).get(strategy)
    cut=spec["threshold"] if spec else row["global"]
    return float(trade["model_predicted_r"])>=float(cut)

