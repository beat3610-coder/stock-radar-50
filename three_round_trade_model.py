"""Three chronological research challengers for candidate-trade win probabilities.
Round 1: market-wide recent-win shrinkage.
Round 2: regime-conditioned shrinkage.
Round 3: online Brier-weighted ensemble (weights from *settled prior forecasts*).
Every signal is scored before its outcome is available. Research only.
"""
from collections import defaultdict
import math

MIN_PAST=35
REGIME_PRIOR=24.0
DECAY=0.985
TRADE_THRESHOLD=0.55

def _date(value):
    return str(value)[:10]

def _regime(row):
    try:
        vals=tuple(float(row[k]) for k in
                   ("signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct"))
        if not all(math.isfinite(x) for x in vals):return None
        rsi,adx,atr,slope=vals
        return (int(rsi>=55),int(adx>=25),int(atr>=3),int(slope>=0))
    except (KeyError,ValueError,TypeError):
        return None

def _p(state, prior, strength):
    n,w=state
    return (w+strength*prior)/(n+strength)

def _brier(rows,key):
    if not rows:return None
    return round(sum((r[key]-r["observed"])**2 for r in rows)/len(rows),6)

def _metrics(rows):
    if not rows:return {"n":0,"brier":None,"win_pct":None,"mean_net_pct":None}
    wins=sum(r["observed"] for r in rows)
    return {"n":len(rows),"brier":None,
            "win_pct":round(100*wins/len(rows),3),
            "mean_net_pct":round(sum(r["net_pct"] for r in rows)/len(rows),4)}

def three_round_audit(trades, market=None, min_past=MIN_PAST):
    """Outcomes are used for training only once exit_date < signal_date.
    Never pick an iterative winner based on diagnostic test-period scores.
    """
    valid=[]
    for row in trades:
        if market is not None and row.get("market")!=market:continue
        sig=_date(row.get("signal_date","")); end=_date(row.get("exit_date",""))
        if len(sig)!=10 or len(end)!=10 or end<sig:continue
        try: net=float(row["net_pct"])
        except (KeyError,TypeError,ValueError):continue
        if not math.isfinite(net):continue
        valid.append({**row,"signal_date":sig,"exit_date":end,"net_pct":net})
    signals=sorted(enumerate(valid),key=lambda pair:(pair[1]["signal_date"],pair[0]))
    exits=sorted(enumerate(valid),key=lambda pair:(pair[1]["exit_date"],pair[0]))
    pending={}
    history=[0.,0.]
    recent=[0.,0.]
    regime_history=defaultdict(lambda:[0.,0.])
    regime_recent=defaultdict(lambda:[0.,0.])
    expert_losses=[0.0,0.0,0.0]
    expert_count=0
    j=0
    audit=[]
    for row_id,row in signals:
        sig=row["signal_date"]
        while j<len(exits) and exits[j][1]["exit_date"]<sig:
            finished_id,finished=exits[j]
            j+=1
            y=float(finished["net_pct"]>0)
            key=_regime(finished)
            history[0]+=1;history[1]+=y
            recent[0]=DECAY*recent[0]+1
            recent[1]=DECAY*recent[1]+y
            if key is not None:
                h=regime_history[key];h[0]+=1;h[1]+=y
                z=regime_recent[key]
                z[0]=DECAY*z[0]+1;z[1]=DECAY*z[1]+y
            forecast=pending.pop(finished_id,None)
            if forecast is not None:
                for i,p in enumerate(forecast):
                    expert_losses[i]+=(p-y)**2
                expert_count+=1
        if history[0]<min_past:continue
        # Round 0: smoothed market-wide historical base rate.
        baseline=_p(history,0.5,4.0)
        # Round 1: recency shift, shrunk against expanding base.
        recent_prob=_p(recent,baseline,16)
        one=0.55*baseline+0.45*recent_prob
        # Round 2: condition on 4 contemporaneous market features.
        key=_regime(row)
        local=_p(regime_history[key],one,REGIME_PRIOR) if key is not None else one
        local_recent=_p(regime_recent[key],one,REGIME_PRIOR) if key is not None else one
        two=0.55*local+0.45*local_recent
        # Round 3: weights estimated from historical *closed* forecasts only.
        if expert_count<25:
            three=(one+two)/2
        else:
            # Cumulative loss differences are normalized by sample count.
            averages=[s/expert_count for s in expert_losses]
            weights=[math.exp(-12*loss) for loss in averages]
            z=sum(weights)
            three=(weights[0]*baseline+weights[1]*one+weights[2]*two)/z
        p=(baseline,one,two,three)
        pending[row_id]=p[:3]
        audit.append({"signal_date":sig,"exit_date":row["exit_date"],
            "observed":float(row["net_pct"]>0),"net_pct":row["net_pct"],
            "baseline":baseline,"round1":one,"round2":two,"round3":three})
    result={"market":market,"population":len(valid),"scored":len(audit),
            "status":"RESEARCH_ONLY_NOT_LIVE_APPROVED","rounds":{}}
    for name in ("baseline","round1","round2","round3"):
        result["rounds"][name]={
            "all_trades":len(audit),
            "brier":_brier(audit,name),
            "fixed_55pct_trades":_metrics([r for r in audit if r[name]>=TRADE_THRESHOLD])}
    baseline_brier=result["rounds"]["baseline"]["brier"]
    for name in ("round1","round2","round3"):
        b=result["rounds"][name]["brier"]
        result["rounds"][name]["brier_improvement_vs_base"] = (
            round(baseline_brier-b,6) if b is not None and baseline_brier is not None else None)
    result["limitations"]=[
        "Historical forecasts are computed on previously inspected periods; not untouched holdout",
        "3 rounds are candidate formulas, not three proven accuracy gains",
        "Win-rate improvement on selected trades is not direction classification accuracy",
        "No model promotion without new prospective test and positive net expectancy"]
    return result
