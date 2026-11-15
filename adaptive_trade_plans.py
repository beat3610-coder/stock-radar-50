"""Research-only walk-forward learning of a risk plan, not just a buy score.

Three PREDECLARED next-open-executable risk/target plans are compared using
outcomes of past completed trades. A plan is chosen at signal close based on
strictly prior matured labels, without consulting candidate's own future.
No 80% win guarantee. Result metrics are research diagnostics only.
"""
import math
import pandas as pd
from buy_success import OnlineBuySuccess, COST_PCT, HOLD
from audit_300 import simulate_exit

# Risk is ATR-relative; reward is a ratio of risk. Fixed before diagnostics.
PLANS={
    "compact":(1.25,1.1),
    "balanced":(1.8,1.8),
    "patient":(2.4,2.2),
}
MIN_HISTORY=90

def label_plans(frame,features_rows):
    index={str(pd.Timestamp(d).date()):i for i,d in enumerate(frame.index)}
    out=[]
    for base in features_rows:
        i=index.get(base["signal_date"])
        if i is None or i+HOLD>=len(frame):continue
        try:
            entry=float(frame.Open.iloc[i+1])
            close=float(frame.Close.iloc[i])
            atr=float(frame.ATR.iloc[i])
            if not all(map(math.isfinite,(entry,close,atr))) or min(entry,close,atr)<=0:
                continue
            # No overnight-limit filter is applied, but never permit invalid
            # or unprofitable risk plans based on stale gaps.
            plans={}
            for name,(risk_atr,reward_rr) in PLANS.items():
                risk=max(risk_atr*atr,entry*.02)
                if risk>=entry:break
                pnl=simulate_exit(frame,i,entry,entry-risk,entry+reward_rr*risk,
                                  max_days=HOLD,cost_pct=COST_PCT)
                if pnl is None:break
                plans[name]=float(pnl["net_pct"])
            if len(plans)!=len(PLANS):continue
            out.append({**base,"plan_net_pct":plans,
              "plan_label_available_date":base["label_available_date"],
              "entry_gap_pct":round(100*(entry/close-1),3)})
        except (ValueError,KeyError,TypeError,OverflowError,ZeroDivisionError):
            continue
    return out

class PlanLearner:
    def __init__(self):
        self.learners={name:OnlineBuySuccess() for name in PLANS}
    def update(self,row):
        for name,model in self.learners.items():
            item={**row,"buy_net_pct":row["plan_net_pct"][name]}
            model.update(item)
    def choose(self,row):
        predictions={name:model.predict(row)
                     for name,model in self.learners.items()}
        if any(p is None for p in predictions.values()):return None
        # Predeclared, strict, earliest-matured-only utility: 60% expected
        # after-cost return and 40% probability calibrated expected return.
        # Deterministic tie-break keeps reproducibility.
        pick=max(PLANS,key=lambda n:(
            predictions[n]["expected_net_pct"],-list(PLANS).index(n)))
        return pick,predictions[pick],predictions

def score_plans(rows,*,train_end="2025-11-01",
                validation_end="2026-04-01",eval_end="2026-11-01"):
    ordered=sorted(rows,key=lambda r:(r["signal_date"],r["ticker"]))
    matured=iter(sorted(rows,key=lambda r:(r["plan_label_available_date"],r["ticker"])))
    nxt=next(matured,None)
    model=PlanLearner()
    def base():
        return {"trades":0,"wins":0,"p_sum":0.0,"net_sum":0.0,
                "positive":0.0,"negative":0.0,"predicted_net_sum":0.0,
                "baseline_wins":0,"baseline_net_sum":0.0,"baseline_positive":0.0,
                "baseline_negative":0.0,"selected_plans":{k:0 for k in PLANS},
                "plan_comparison":{k:{"wins":0,"net_sum":0.0} for k in PLANS}}
    stats={part:base() for part in ("train","validation","diagnostic")}
    for row in ordered:
        while nxt is not None and nxt["plan_label_available_date"]<row["signal_date"]:
            model.update(nxt)
            nxt=next(matured,None)
        if row["plan_label_available_date"]>=eval_end:continue
        picked=model.choose(row)
        if picked is None:continue
        name,prediction,_=picked
        part=("train" if row["signal_date"]<train_end else
              "validation" if row["signal_date"]<validation_end else "diagnostic")
        s=stats[part]
        net=row["plan_net_pct"][name]
        baseline=row["plan_net_pct"]["balanced"]
        s["trades"]+=1
        s["wins"]+=int(net>0)
        s["p_sum"]+=prediction["p_success"]
        s["net_sum"]+=net
        s["predicted_net_sum"]+=prediction["expected_net_pct"]
        s["positive"]+=max(net,0)
        s["negative"]+=max(-net,0)
        s["baseline_wins"]+=int(baseline>0)
        s["baseline_net_sum"]+=baseline
        s["baseline_positive"]+=max(baseline,0)
        s["baseline_negative"]+=max(-baseline,0)
        s["selected_plans"][name]+=1
        for k,pnl in row["plan_net_pct"].items():
            s["plan_comparison"][k]["wins"]+=int(pnl>0)
            s["plan_comparison"][k]["net_sum"]+=pnl
    for s in stats.values():
        n=s["trades"]
        s["actual_win_pct"]=round(100*s["wins"]/n,2) if n else None
        s["baseline_win_pct"]=round(100*s["baseline_wins"]/n,2) if n else None
        s["predicted_win_pct"]=round(100*s.pop("p_sum")/n,2) if n else None
        s["mean_net_pct"]=round(s["net_sum"]/n,3) if n else None
        s["baseline_mean_net_pct"]=round(s["baseline_net_sum"]/n,3) if n else None
        s["predicted_mean_net_pct"]=round(s.pop("predicted_net_sum")/n,3) if n else None
        s["profit_factor"]=round(s.pop("positive")/s.pop("negative"),3) if s["negative"] else None
        s["baseline_profit_factor"]=round(s.pop("baseline_positive")/s.pop("baseline_negative"),3) if s["baseline_negative"] else None
        for p in s["plan_comparison"].values():
            p["win_pct"]=round(p["wins"]*100/n,2) if n else None
            p["mean_net_pct"]=round(p.pop("net_sum")/n,3) if n else None
        s.pop("net_sum");s.pop("baseline_net_sum")
    return {"windows":stats,"strategy":"delay-trained plan selection",
      "plans":PLANS,"cost_pct":COST_PCT,"independent_holdout":False,
      "live_approved":False}
