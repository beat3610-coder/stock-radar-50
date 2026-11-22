"""Past-only probability and candidate-payoff model (research only).

Features are read at the signal close. Labels enter training only after their
exit date is strictly earlier than the candidate signal date. All transforms
use fixed, documented domain scales; no evaluation-window fit is performed.
"""
import math
import numpy as np

MIN_COMPLETED=35
MAX_TRAIN=240
MIN_POLICY_TRADES=10
MIN_DEVELOPMENT_EDGE=0.0

FEATURES=("signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct",
          "signal_volume_ratio","signal_5d_return_pct","entry_gap_pct",
          "signal_trend_200_pct","signal_macdh_pct","signal_drawdown_20_pct",
          "signal_rsi_5d_change","signal_vol20_60_ratio",
          "entry_net_rr","entry_net_upside_pct","entry_net_downside_pct",
          "signal_stop_atr","candidate_legacy")
CORE_FEATURES=("signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct",
               "signal_volume_ratio","signal_5d_return_pct","entry_gap_pct",
               "entry_net_rr","entry_net_upside_pct","entry_net_downside_pct",
               "signal_stop_atr","candidate_legacy")
# Fixed scale centers and ranges; never estimated from validation/diagnostic.
CENTER={"signal_rsi":55.0,"signal_adx":25.0,"signal_atr_pct":2.5,
        "signal_ma20_ma60_pct":2.0,"signal_volume_ratio":1.3,
        "signal_5d_return_pct":2.0,"entry_gap_pct":0.0,
        "signal_trend_200_pct":2.0,"signal_macdh_pct":0.0,
        "signal_drawdown_20_pct":-3.0,"signal_rsi_5d_change":0.0,
        "signal_vol20_60_ratio":1.0,"entry_net_rr":1.5,
        "entry_net_upside_pct":4.0,"entry_net_downside_pct":4.0,
        "signal_stop_atr":1.8,"candidate_legacy":0.0}
SCALE={"signal_rsi":20.0,"signal_adx":20.0,"signal_atr_pct":3.0,
       "signal_ma20_ma60_pct":4.0,"signal_volume_ratio":1.5,
       "signal_5d_return_pct":8.0,"entry_gap_pct":2.5,
       "signal_trend_200_pct":20.0,"signal_macdh_pct":1.5,
       "signal_drawdown_20_pct":8.0,"signal_rsi_5d_change":20.0,
       "signal_vol20_60_ratio":0.75,"entry_net_rr":1.0,
       "entry_net_upside_pct":5.0,"entry_net_downside_pct":5.0,
       "signal_stop_atr":1.5,"candidate_legacy":1.0}

# Each candidate explores a predeclared training/shrinkage choice. The audit
# chooses among them using development-only prequential ranking quality.
CONFIGS={
    "rolling240_full_shrink50":{"window":"rolling","max_train":240,"half_life":None,"shrink":0.50,"features":FEATURES},
    "expanding_full_shrink35":{"window":"expanding","max_train":None,"half_life":None,"shrink":0.35,"features":FEATURES},
    "ewma120_full_shrink50":{"window":"ewma","max_train":None,"half_life":120,"shrink":0.50,"features":FEATURES},
    "rolling240_core_shrink65":{"window":"rolling","max_train":240,"half_life":None,"shrink":0.65,"features":CORE_FEATURES},
}
DEFAULT_CONFIG=next(iter(CONFIGS))
REQUIRED_FEATURES=("signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct",
                  "signal_volume_ratio","signal_5d_return_pct","entry_gap_pct",
                  "entry_net_rr","entry_net_upside_pct","entry_net_downside_pct","signal_stop_atr")

# Thresholds are fixed before evaluation. Candidate-specific EV is computed
# from predicted probability and the candidate's own net target/stop amounts.
POLICIES={
    "coverage":{"win_pct":40.0,"min_ev_pct":0.0},
    "balanced":{"win_pct":45.0,"min_ev_pct":0.15},
    "strict":{"win_pct":50.0,"min_ev_pct":0.35},
}

def vector(trade,features=FEATURES):
    try:
        if any(key not in trade for key in REQUIRED_FEATURES):return None
        values=np.asarray([float(trade.get(key,CENTER[key])) for key in features],dtype=float)
        if not np.isfinite(values).all():return None
        scaled=np.asarray([(value-CENTER[key])/SCALE[key]
                           for value,key in zip(values,features)],dtype=float)
        return np.clip(scaled,-3,3)
    except (KeyError,TypeError,ValueError,OverflowError):
        return None

def _sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-35,35)))

def _config(config):
    if config is None:return CONFIGS[DEFAULT_CONFIG]
    if isinstance(config,str):return CONFIGS[config]
    return config

def prepare_snapshot(history,config=None):
    """Fit one model from already closed candidates before a signal date."""
    cfg=_config(config);usable=[]
    for trade in history:
        v=vector(trade,cfg["features"])
        try:pnl=float(trade["net_pct"])
        except (ValueError,TypeError,KeyError):continue
        if v is not None and np.isfinite(pnl):usable.append((v,pnl,trade.get("exit_date","")))
    if cfg["window"]=="rolling":usable=usable[-int(cfg["max_train"]):]
    if len(usable)<MIN_COMPLETED:return None
    x=np.asarray([r[0] for r in usable]);pnl=np.asarray([r[1] for r in usable])
    y=(pnl>0).astype(float);n=len(y)
    if np.unique(y).size<2:return None
    sample_weight=np.ones(n,dtype=float)
    if cfg["window"]=="ewma":
        # Weight by order of fully closed trades, not by evaluation outcomes.
        age=np.arange(n-1,-1,-1,dtype=float)
        sample_weight=np.exp(-math.log(2)*age/float(cfg["half_life"]))
        sample_weight*=n/sample_weight.sum()
    X=np.column_stack([np.ones(n),x]);beta=np.zeros(X.shape[1],dtype=float)
    penalty=np.diag([0.0]+[8.0]*x.shape[1])
    for _ in range(16):
        p=_sigmoid(X@beta);weights=np.maximum(p*(1-p),0.02)*sample_weight
        grad=X.T@(sample_weight*(y-p))-penalty@beta
        hessian=X.T@(weights[:,None]*X)+penalty+np.eye(X.shape[1])*1e-7
        delta=np.linalg.solve(hessian,grad);beta+=np.clip(delta,-1.5,1.5)
        if np.max(np.abs(delta))<1e-5:break
    return {"n":n,"beta":beta,"base":float((y.sum()+2)/(n+4)),
            "features":cfg["features"],"shrink":float(cfg["shrink"]),
            "window":cfg["window"]}

def predict_from_snapshot(model,candidate):
    """Return calibrated probability and EV using this candidate's own payoff."""
    if model is None:return None
    query=vector(candidate,model["features"])
    if query is None:return None
    try:
        upside=float(candidate["entry_net_upside_pct"])
        downside=float(candidate["entry_net_downside_pct"])
    except (KeyError,TypeError,ValueError):return None
    if not(np.isfinite(upside) and np.isfinite(downside) and upside>0 and downside>0):return None
    raw=float(_sigmoid(np.r_[1.0,query]@model["beta"]))
    p=float(model["shrink"]*raw+(1-model["shrink"])*model["base"])
    ev=p*upside-(1-p)*downside
    return {"training_completed":model["n"],"predicted_win_pct":round(100*p,4),
            "baseline_win_pct":round(100*model["base"],4),
            "predicted_uplift_pct":round(100*(p-model["base"]),4),
            "expected_net_pct":round(ev,5),"candidate_net_upside_pct":round(upside,5),
            "candidate_net_downside_pct":round(downside,5)}

def fit_probabilities(history,candidate,config=None):
    return predict_from_snapshot(prepare_snapshot(history,config),candidate)

def walkforward_model_filter(trades,config=None):
    ordered=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"]))
    closes=sorted(trades,key=lambda t:(t["exit_date"],t["ticker"],t["entry_date"]))
    accepted=[];info={"total":len(trades),"missing_features":0,"insufficient":0,"weak_prediction":0,"accepted":0}
    past=[];j=0;active=None;model=None
    for item in ordered:
        if item["signal_date"]!=active:
            while j<len(closes) and closes[j]["exit_date"]<item["signal_date"]:
                past.append(closes[j]);j+=1
            model=prepare_snapshot(past,config);active=item["signal_date"]
        if vector(item,_config(config)["features"]) is None:info["missing_features"]+=1;continue
        score=predict_from_snapshot(model,item)
        if score is None:info["insufficient"]+=1;continue
        if score["predicted_win_pct"]<POLICIES["strict"]["win_pct"] or score["expected_net_pct"]<POLICIES["strict"]["min_ev_pct"]:
            info["weak_prediction"]+=1;continue
        accepted.append({**item,**{"model_"+k:v for k,v in score.items()}});info["accepted"]+=1
    return accepted,info

def walkforward_model_policies(trades,config=None):
    """Score candidates prequentially, retaining fixed diagnostic policies."""
    ordered=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"],t["entry_date"]))
    closes=sorted(trades,key=lambda t:(t["exit_date"],t["ticker"],t["entry_date"]))
    selected={key:[] for key in POLICIES}
    status={"total":len(trades),"missing_features":0,"insufficient":0,"scored":0,"models_fitted":0,
            "policy_accepted":{key:0 for key in POLICIES}}
    j=0;history=[];active=None;model=None;cfg=_config(config)
    for item in ordered:
        signal=item["signal_date"]
        if signal!=active:
            while j<len(closes) and closes[j]["exit_date"]<signal:
                history.append(closes[j]);j+=1
            model=prepare_snapshot(history,cfg);active=signal;status["models_fitted"]+=1
        if vector(item,cfg["features"]) is None:status["missing_features"]+=1;continue
        score=predict_from_snapshot(model,item)
        if score is None:status["insufficient"]+=1;continue
        status["scored"]+=1
        for name,policy in POLICIES.items():
            if score["predicted_win_pct"]>=policy["win_pct"] and score["expected_net_pct"]>=policy["min_ev_pct"]:
                selected[name].append({**item,**{"model_"+k:v for k,v in score.items()}})
                status["policy_accepted"][name]+=1
    return selected,status
