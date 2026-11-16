"""Research-only walk-forward evolution of executable stop/take/hold plans.

Every prediction and plan choice uses only labels that were already mature at
the signal date. Completed historical half-years are deliberately reused for
formula search, so results are research fit rather than independent evidence.
"""
import math
import random
import numpy as np
import pandas as pd
from buy_success import OnlineBuySuccess, COST_PCT, HOLD
from audit_300 import simulate_exit

PLANS={
    "compact":(1.25,1.1),
    "balanced":(1.8,1.8),
    "patient":(2.4,2.2),
}
MIN_HISTORY=90
TARGET_WIN_PCT=60.0
SELECTOR_QUANTILES=(0.50,0.60,0.70,0.80,0.90)
SELECTOR_MIN_HISTORY=250
SELECTOR_MIN_BUCKET=60
SELECTOR_MIN_COVERAGE_PCT=10.0
SELECTOR_MIN_DIAGNOSTIC_TRADES=300


def _normalise_plans(plans=None):
    source=plans or PLANS
    result={}
    for name,value in source.items():
        if isinstance(value,dict):
            risk_atr=float(value.get("risk_atr",1.8))
            reward_rr=float(value.get("reward_rr",1.8))
            hold_sessions=int(value.get("hold_sessions",HOLD))
        else:
            risk_atr=float(value[0])
            reward_rr=float(value[1])
            hold_sessions=int(value[2]) if len(value)>2 else HOLD
        result[name]={"risk_atr":round(max(.5,min(5.0,risk_atr)),4),
                      "reward_rr":round(max(.25,min(4.0,reward_rr)),4),
                      "hold_sessions":max(2,min(HOLD,hold_sessions))}
    return result


def make_plan_candidates(search_generation=1,previous_best=None):
    """Keep anchor plans and add deterministic mutations around the winner."""
    plans={
        "compact":(1.25,1.1,7),
        "balanced":(1.8,1.8,10),
        "patient":(2.4,2.2,10),
        "quick_target":(1.7,.65,5),
        "wide_quick":(2.5,.75,7),
        "short_hold":(1.8,1.0,3),
    }
    center=previous_best or {"risk_atr":1.8,"reward_rr":1.0,
                             "hold_sessions":7}
    seed=(15485863*int(search_generation)
          +int(float(center.get("risk_atr",1.8))*10000)
          +int(float(center.get("reward_rr",1.0))*10000)
          +int(center.get("hold_sessions",7))*100)
    rng=random.Random(seed)
    for i in range(18):
        risk=float(center.get("risk_atr",1.8))*math.exp(rng.uniform(-.65,.65))
        reward=float(center.get("reward_rr",1.0))*math.exp(rng.uniform(-.75,.75))
        hold=int(round(float(center.get("hold_sessions",7))+rng.uniform(-3,3)))
        plans[f"pg{search_generation}_mutation{i+1}"]=(risk,reward,hold)
    return _normalise_plans(plans)


def label_plans(frame,features_rows,plans=None):
    """Label every candidate plan; short holds still mature conservatively at +10."""
    specs=_normalise_plans(plans)
    index={str(pd.Timestamp(d).date()):i for i,d in enumerate(frame.index)}
    out=[]
    for base in features_rows:
        i=index.get(base["signal_date"])
        if i is None or i+HOLD>=len(frame):
            continue
        try:
            entry=float(frame.Open.iloc[i+1])
            close=float(frame.Close.iloc[i])
            atr=float(frame.ATR.iloc[i])
            if not all(map(math.isfinite,(entry,close,atr))) or min(entry,close,atr)<=0:
                continue
            outcomes={}
            for name,spec in specs.items():
                risk=max(spec["risk_atr"]*atr,entry*.02)
                if risk>=entry:
                    break
                pnl=simulate_exit(
                    frame,i,entry,entry-risk,entry+spec["reward_rr"]*risk,
                    max_days=spec["hold_sessions"],cost_pct=COST_PCT)
                if pnl is None:
                    break
                outcomes[name]=float(pnl["net_pct"])
            if len(outcomes)!=len(specs):
                continue
            out.append({**base,"plan_net_pct":outcomes,
                "plan_label_available_date":base["label_available_date"],
                "entry_gap_pct":round(100*(entry/close-1),3)})
        except (ValueError,KeyError,TypeError,OverflowError,ZeroDivisionError):
            continue
    return out


class PlanLearner:
    def __init__(self,plans=None):
        self.plans=_normalise_plans(plans)
        self.learners={name:OnlineBuySuccess() for name in self.plans}
        self.history={name:[] for name in self.plans}
        self.profile_cache={}
        self.profile_dirty=0

    def update(self,row):
        snapshot=row.pop("_plan_prediction_snapshot",None)
        if snapshot:
            for name,(score,probability) in snapshot.items():
                net=float(row["plan_net_pct"][name])
                history=self.history[name]
                history.append((float(score),float(probability),net))
                if len(history)>2000:
                    del history[:len(history)-2000]
            self.profile_dirty+=1
        for name,model in self.learners.items():
            item={**row,"buy_net_pct":row["plan_net_pct"][name]}
            model.update(item)

    def choose(self,row):
        predictions={name:model.predict(row)
                     for name,model in self.learners.items()}
        if any(p is None for p in predictions.values()):
            return None
        row["_plan_prediction_snapshot"]={
            name:(prediction["expected_net_pct"],prediction["p_success"])
            for name,prediction in predictions.items()}
        viable=[name for name,p in predictions.items()
                if p["p_success"]>=TARGET_WIN_PCT/100
                and p["expected_net_pct"]>=0]
        pool=viable or list(self.plans)
        pick=max(pool,key=lambda name:(
            predictions[name]["expected_net_pct"],
            predictions[name]["p_success"],
            -list(self.plans).index(name)))
        return pick,predictions[pick],predictions

    def _refresh_profiles(self):
        if self.profile_cache and self.profile_dirty<64:
            return
        cache={q:{} for q in SELECTOR_QUANTILES}
        for name,history in self.history.items():
            if len(history)<SELECTOR_MIN_HISTORY:
                continue
            values=np.asarray(history,dtype=float)
            scores=values[:,0]
            nets=values[:,2]
            for quantile in SELECTOR_QUANTILES:
                threshold=float(np.quantile(scores,quantile))
                selected=nets[scores>=threshold]
                if len(selected)<SELECTOR_MIN_BUCKET:
                    continue
                gains=float(selected[selected>0].sum())
                losses=float(-selected[selected<0].sum())
                cache[quantile][name]={
                    "threshold":threshold,
                    "samples":int(len(selected)),
                    "win_pct":100*float((selected>0).mean()),
                    "mean_net_pct":float(selected.mean()),
                    "profit_factor":gains/losses if losses>0 else None,
                }
        self.profile_cache=cache
        self.profile_dirty=0

    def selective_choose(self,predictions,quantile):
        """Choose only from causally profitable historical score buckets."""
        self._refresh_profiles()
        viable=[]
        for name,prediction in predictions.items():
            profile=(self.profile_cache.get(quantile) or {}).get(name)
            if not profile:
                continue
            pf=profile.get("profit_factor") or 0
            if (prediction["expected_net_pct"]<profile["threshold"]
                    or prediction["p_success"]<0.50
                    or profile["win_pct"]<50.0
                    or profile["mean_net_pct"]<=0
                    or pf<1.05):
                continue
            viable.append((name,profile))
        if not viable:
            return None
        name,profile=max(viable,key=lambda item:(
            item[1]["mean_net_pct"],item[1].get("profit_factor") or 0,
            item[1]["win_pct"],predictions[item[0]]["expected_net_pct"]))
        return name,predictions[name],profile


def _blank_stats(plan_names):
    return {"trades":0,"wins":0,"p_sum":0.0,"net_sum":0.0,
            "positive":0.0,"negative":0.0,"predicted_net_sum":0.0,
            "baseline_wins":0,"baseline_net_sum":0.0,
            "baseline_positive":0.0,"baseline_negative":0.0,
            "selected_plans":{k:0 for k in plan_names},
            "plan_comparison":{k:{"wins":0,"net_sum":0.0,
                                  "positive":0.0,"negative":0.0}
                               for k in plan_names}}


def _blank_selective_stats(plan_names):
    result=_blank_stats(plan_names)
    result["signals_seen"]=0
    return result


def _record_stats(stat,row,name,prediction,baseline_name):
    net=row["plan_net_pct"][name]
    baseline=row["plan_net_pct"][baseline_name]
    stat["trades"]+=1
    stat["wins"]+=int(net>0)
    stat["p_sum"]+=prediction["p_success"]
    stat["net_sum"]+=net
    stat["predicted_net_sum"]+=prediction["expected_net_pct"]
    stat["positive"]+=max(net,0)
    stat["negative"]+=max(-net,0)
    stat["baseline_wins"]+=int(baseline>0)
    stat["baseline_net_sum"]+=baseline
    stat["baseline_positive"]+=max(baseline,0)
    stat["baseline_negative"]+=max(-baseline,0)
    stat["selected_plans"][name]+=1
    for plan,pnl in row["plan_net_pct"].items():
        p=stat["plan_comparison"][plan]
        p["wins"]+=int(pnl>0)
        p["net_sum"]+=pnl
        p["positive"]+=max(pnl,0)
        p["negative"]+=max(-pnl,0)


def _finalize_stats(stat):
    n=stat["trades"]
    stat["actual_win_pct"]=round(100*stat["wins"]/n,2) if n else None
    stat["baseline_win_pct"]=round(100*stat["baseline_wins"]/n,2) if n else None
    stat["predicted_win_pct"]=round(100*stat.pop("p_sum")/n,2) if n else None
    stat["mean_net_pct"]=round(stat["net_sum"]/n,3) if n else None
    stat["baseline_mean_net_pct"]=round(stat["baseline_net_sum"]/n,3) if n else None
    stat["predicted_mean_net_pct"]=round(
        stat.pop("predicted_net_sum")/n,3) if n else None
    negative=stat.pop("negative")
    baseline_negative=stat.pop("baseline_negative")
    stat["profit_factor"]=round(stat.pop("positive")/negative,3) if negative else None
    stat["baseline_profit_factor"]=round(
        stat.pop("baseline_positive")/baseline_negative,3) if baseline_negative else None
    for p in stat["plan_comparison"].values():
        p["win_pct"]=round(p["wins"]*100/n,2) if n else None
        p["mean_net_pct"]=round(p.pop("net_sum")/n,3) if n else None
        loss=p.pop("negative")
        p["profit_factor"]=round(p.pop("positive")/loss,3) if loss else None
    stat.pop("net_sum")
    stat.pop("baseline_net_sum")


def _finalize_selective_stats(stat):
    signals=stat.get("signals_seen",0)
    _finalize_stats(stat)
    trades=stat.get("trades",0)
    stat["skipped_signals"]=max(0,signals-trades)
    stat["coverage_pct"]=round(100*trades/signals,2) if signals else None


def _period_key(signal_date):
    return signal_date[:4]+("-H1" if signal_date[5:7]<="06" else "-H2")


def score_plans(rows,*,train_end="2025-11-01",
                validation_end="2026-04-01",eval_end="2026-11-01",
                plans=None,search_generation=1,previous_best=None,
                target_success_pct=TARGET_WIN_PCT):
    specs=_normalise_plans(plans)
    names=list(specs)
    baseline_name="balanced" if "balanced" in specs else names[0]
    ordered=sorted(rows,key=lambda r:(r["signal_date"],r["ticker"]))
    matured=iter(sorted(rows,key=lambda r:(
        r["plan_label_available_date"],r["ticker"])))
    nxt=next(matured,None)
    model=PlanLearner(specs)
    stats={part:_blank_stats(names)
           for part in ("train","validation","diagnostic")}
    selective={q:{part:_blank_selective_stats(names)
                  for part in ("train","validation","diagnostic")}
               for q in SELECTOR_QUANTILES}
    periods={}
    for row in ordered:
        while nxt is not None and nxt["plan_label_available_date"]<row["signal_date"]:
            model.update(nxt)
            nxt=next(matured,None)
        if row["plan_label_available_date"]>=eval_end:
            continue
        picked=model.choose(row)
        if picked is None:
            continue
        name,prediction,predictions=picked
        part=("train" if row["signal_date"]<train_end else
              "validation" if row["signal_date"]<validation_end else "diagnostic")
        _record_stats(stats[part],row,name,prediction,baseline_name)
        for quantile in SELECTOR_QUANTILES:
            selective_stat=selective[quantile][part]
            selective_stat["signals_seen"]+=1
            selected=model.selective_choose(predictions,quantile)
            if selected is None:
                continue
            selected_name,selected_prediction,_=selected
            _record_stats(selective_stat,row,selected_name,
                          selected_prediction,baseline_name)
        period=_period_key(row["signal_date"])
        if period not in periods:
            periods[period]=_blank_stats(names)
        _record_stats(periods[period],row,name,prediction,baseline_name)
    for stat in stats.values():
        _finalize_stats(stat)
    for stat in periods.values():
        _finalize_stats(stat)
    for policy in selective.values():
        for stat in policy.values():
            _finalize_selective_stats(stat)

    policy_reports=[]
    for quantile,windows in selective.items():
        validation=windows["validation"]
        policy_reports.append({
            "quantile":quantile,"windows":windows,
            "validation_eligible":(
                validation.get("trades",0)>=200
                and (validation.get("coverage_pct") or 0)>=SELECTOR_MIN_COVERAGE_PCT
                and (validation.get("mean_net_pct") or 0)>0
                and (validation.get("profit_factor") or 0)>=1.05),
        })
    eligible_policies=[p for p in policy_reports if p["validation_eligible"]]
    pool=eligible_policies or [p for p in policy_reports
                               if p["windows"]["validation"].get("trades",0)>=100]
    if not pool:
        pool=policy_reports
    chosen_policy=max(pool,key=lambda p:(
        (p["windows"]["validation"].get("actual_win_pct") or 0)
            >=target_success_pct,
        p["windows"]["validation"].get("mean_net_pct") or -999,
        p["windows"]["validation"].get("profit_factor") or 0,
        p["windows"]["validation"].get("actual_win_pct") or 0,
        p["windows"]["validation"].get("coverage_pct") or 0))

    candidates=[]
    for name,spec in specs.items():
        slices=[]
        for period,stat in sorted(periods.items()):
            if stat.get("trades",0)<100:
                continue
            value=stat["plan_comparison"][name]
            slices.append({"period":period,"samples":stat["trades"],
                "win_pct":value.get("win_pct"),
                "mean_net_pct":value.get("mean_net_pct"),
                "profit_factor":value.get("profit_factor")})
        if slices:
            mean_win=round(sum(x["win_pct"] for x in slices)/len(slices),2)
            mean_net=round(sum(x["mean_net_pct"] for x in slices)/len(slices),3)
            worst_net=min(x["mean_net_pct"] for x in slices)
        else:
            diag=stats["diagnostic"]["plan_comparison"][name]
            mean_win=diag.get("win_pct") or 0
            mean_net=diag.get("mean_net_pct") or -999
            worst_net=mean_net
        candidates.append({"name":name,**spec,"period_count":len(slices),
            "whole_history_win_pct":mean_win,
            "whole_history_mean_net_pct":mean_net,
            "worst_period_mean_net_pct":worst_net,
            "period_scores":slices})

    profitable=[c for c in candidates if c["whole_history_mean_net_pct"]>0]
    reached=[c for c in profitable
             if c["whole_history_win_pct"]>=target_success_pct]
    if reached:
        chosen=max(reached,key=lambda c:(
            c["whole_history_mean_net_pct"],
            c["worst_period_mean_net_pct"],
            c["whole_history_win_pct"]))
    elif profitable:
        chosen=max(profitable,key=lambda c:(
            c["whole_history_win_pct"],
            c["whole_history_mean_net_pct"],
            c["worst_period_mean_net_pct"]))
    else:
        chosen=max(candidates,key=lambda c:(
            c["whole_history_mean_net_pct"],
            c["whole_history_win_pct"],
            c["worst_period_mean_net_pct"]))

    diagnostic=chosen_policy["windows"]["diagnostic"]
    goal_reached=(
        chosen_policy["validation_eligible"]
        and diagnostic.get("actual_win_pct") is not None
        and diagnostic["actual_win_pct"]>=target_success_pct
        and (diagnostic.get("mean_net_pct") or 0)>0
        and (diagnostic.get("profit_factor") or 0)>=1.1
        and diagnostic.get("trades",0)>=SELECTOR_MIN_DIAGNOSTIC_TRADES
        and (diagnostic.get("coverage_pct") or 0)>=SELECTOR_MIN_COVERAGE_PCT)
    return {"windows":stats,"periods":periods,
        "strategy":"generation-evolved delayed-label plan selection",
        "plans":specs,"cost_pct":COST_PCT,
        "target_success_pct":target_success_pct,
        "target_quality_requirements":{
            "actual_win_pct_min":target_success_pct,
            "mean_net_pct_min_exclusive":0,
            "profit_factor_min":1.1,
            "diagnostic_trades_min":SELECTOR_MIN_DIAGNOSTIC_TRADES,
            "coverage_pct_min":SELECTOR_MIN_COVERAGE_PCT},
        "target_reached":goal_reached,
        "selective_challenger":{
            "selection_window":"validation only; diagnostic is not used to choose the policy",
            "chosen_quantile":chosen_policy["quantile"],
            "validation_eligible":chosen_policy["validation_eligible"],
            "windows":chosen_policy["windows"],
            "policies":[{
                "quantile":p["quantile"],
                "validation_eligible":p["validation_eligible"],
                "validation":p["windows"]["validation"],
                "diagnostic":p["windows"]["diagnostic"],
            } for p in policy_reports],
            "rules":{
                "probability_min":50.0,
                "matured_bucket_win_pct_min":50.0,
                "matured_bucket_mean_net_pct_min_exclusive":0,
                "matured_bucket_profit_factor_min":1.05,
                "minimum_history":SELECTOR_MIN_HISTORY,
                "minimum_bucket_samples":SELECTOR_MIN_BUCKET},
            "warning":"거래를 선별한 연구 결과이며 전체 신호 성과와 함께 coverage를 반드시 확인해야 합니다."},
        "independent_holdout":False,"live_approved":False,
        "tuning":{"generation":int(search_generation),
            "selection_window":"완료된 전체 과거기간의 반기별 순차평가",
            "selection_metric":"검증구간에서 비용 후 순손익·손익비·표본·coverage를 통과한 선택 정책만 진단구간에 적용",
            "previous_best":previous_best,
            "chosen_plan":{k:v for k,v in chosen.items()
                           if k!="period_scores"},
            "chosen_period_scores":chosen["period_scores"],
            "candidate_count":len(candidates),
            "candidates":[{k:v for k,v in c.items()
                           if k!="period_scores"} for c in candidates],
            "whole_history_used_for_selection":True,
            "independent_holdout":False,
            "warning":"전체 과거에서 반복 선택한 연구 적합도이며 미래 50% 성공을 보장하지 않습니다."}}
