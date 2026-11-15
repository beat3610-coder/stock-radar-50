"""Online buy-success forecasting paired with the three-way direction model.

Research only: signal-at-close; hypothetical entry at next OPEN. Exit uses
conservative stop-first 10-session OHLC, fixed 1.8 ATR minimum 2.5% stop and
1.8R take. Estimated round-trip cost: 0.35%. NO buy-signal/score filters.
Labels mature at +10 sessions even when the stop or target hit earlier;
predictions always precede training on labels.
"""
import math
import random
import pandas as pd
from direction3 import features,FEATURE_COUNT
from audit_300 import simulate_exit

HOLD=10
COST_PCT=0.35
TRAIN_MIN=90

class OnlineBuySuccess:
    def __init__(self, lr=0.11, l2=0.065, prior_mix=0.15):
        self.weights=[0.0]*FEATURE_COUNT
        self.adagrad=[4.0]*FEATURE_COUNT
        self.lr=lr
        self.l2=l2
        self.prior_mix=max(0.,min(1.,float(prior_mix)))
        self.count=0
        self.wins=0
        self.positive_sum=0.0
        self.negative_sum=0.0
    def _p(self,v):
        z=sum(a*b for a,b in zip(self.weights,v))
        return 1.0/(1.0+math.exp(-max(-30,min(30,z))))
    def prior(self):
        return (self.wins+2)/(self.count+4)
    def predict(self,row):
        v=features(row)
        if v is None or self.count<TRAIN_MIN:return None
        p=(1-self.prior_mix)*self._p(v)+self.prior_mix*self.prior()
        avg_up=self.positive_sum/max(1,self.wins)
        avg_down=self.negative_sum/max(1,self.count-self.wins)
        return {"p_success":p,
                "expected_net_pct":p*avg_up+(1-p)*avg_down,
                "baseline_p":self.prior()}
    def update(self,row):
        v=features(row)
        try:
            net=float(row["buy_net_pct"])
            if v is None or not math.isfinite(net):return False
        except (KeyError,TypeError,ValueError):return False
        y=float(net>0)
        p=self._p(v)
        for j,val in enumerate(v):
            grad=(p-y)*val+(self.l2*self.weights[j] if j else 0)
            self.adagrad[j]+=grad*grad
            self.weights[j]-=self.lr*grad/math.sqrt(self.adagrad[j])
        self.count+=1
        self.wins+=int(y)
        if y:self.positive_sum+=net
        else:self.negative_sum+=net
        return True

def attach_buy_labels(frame,rows):
    """Score every market-direction sample with identical buy execution rules.

    Price at t+1 OPEN is an outcome/execution input only, not a predictor.
    """
    lookup={str(pd.Timestamp(d).date()):i for i,d in enumerate(frame.index)}
    output=[]
    for base in rows:
        i=lookup.get(base["signal_date"])
        if i is None or i+HOLD>=len(frame):continue
        try:
            entry=float(frame.Open.iloc[i+1])
            atr=float(frame.ATR.iloc[i])
            if entry<=0 or atr<=0 or not all(math.isfinite(x) for x in (entry,atr)):
                continue
            risk=max(1.8*atr,0.025*entry)
            stop=entry-risk
            take=entry+1.8*risk
            if stop<=0:continue
            completed=simulate_exit(frame,i,entry,stop,take,
                                    max_days=HOLD,cost_pct=COST_PCT)
            if completed is None:continue
            net=float(completed["net_pct"])
            if not math.isfinite(net):continue
            output.append({**base,"buy_net_pct":net,
                           "buy_won":int(net>0),
                           "buy_exit_reason":completed["exit_reason"],
                           "buy_label_available_date":base["label_available_date"],
                           "buy_stop_pct":100*risk/entry})
        except (KeyError,ValueError,TypeError,OverflowError,ZeroDivisionError):
            continue
    return output

def _empty_buy_stat():
    return {"count":0,"wins":0,"pred_sum":0.0,"brier":0.0,
            "base_brier":0.0,"net_sum":0.0,"expected_net_sum":0.0,
            "actual_profit_sum":0.0,"actual_loss_sum":0.0}


def _record_buy_stat(stat,y,net,predicted):
    p=predicted["p_success"]
    stat["count"]+=1
    stat["wins"]+=int(y)
    stat["pred_sum"]+=p
    stat["brier"]+=(p-y)**2
    stat["base_brier"]+=(predicted["baseline_p"]-y)**2
    stat["net_sum"]+=net
    stat["expected_net_sum"]+=predicted["expected_net_pct"]
    if net>0:
        stat["actual_profit_sum"]+=net
    else:
        stat["actual_loss_sum"]+=-net


def _finalize_buy_stat(stat):
    n=stat["count"]
    p=stat.pop("pred_sum")
    actual_win=stat["wins"]
    stat["mean_predicted_success_pct"]=round(100*p/n,2) if n else None
    stat["actual_success_pct"]=round(100*actual_win/n,2) if n else None
    stat["calibration_gap_pp"]=round(100*(p-actual_win)/n,2) if n else None
    stat["brier_model"]=round(stat.pop("brier")/n,6) if n else None
    stat["brier_baseline"]=round(stat.pop("base_brier")/n,6) if n else None
    stat["mean_net_pct"]=round(stat.pop("net_sum")/n,3) if n else None
    stat["mean_predicted_net_pct"]=round(
        stat.pop("expected_net_sum")/n,3) if n else None
    gain=stat.pop("actual_profit_sum")
    loss=stat.pop("actual_loss_sum")
    stat["profit_factor"]=round(gain/loss,3) if loss else None


def _period_key(signal_date):
    return signal_date[:4]+("-H1" if signal_date[5:7]<="06" else "-H2")


def _score_buy_success_candidate(trades,*,train_end="2025-11-01",
                      validation_end="2026-04-01",eval_end="2026-11-01",
                      lr=0.11,l2=0.065,prior_mix=0.15):
    """Causal probability scoring plus whole-history half-year slices."""
    stream=sorted(trades,key=lambda r:(r["signal_date"],r["ticker"]))
    done=iter(sorted(trades,key=lambda r:(r["buy_label_available_date"],r["ticker"])))
    next_done=next(done,None)
    model=OnlineBuySuccess(lr=lr,l2=l2,prior_mix=prior_mix)
    stats={k:_empty_buy_stat()
           for k in ("train","validation","diagnostic")}
    periods={}
    for row in stream:
        signal=row["signal_date"]
        while next_done is not None and next_done["buy_label_available_date"]<signal:
            model.update(next_done)
            next_done=next(done,None)
        if row["buy_label_available_date"]>=eval_end:
            continue
        predicted=model.predict(row)
        if predicted is None:
            continue
        bucket=("train" if signal<train_end else
                "validation" if signal<validation_end else "diagnostic")
        y=float(row["buy_won"])
        net=float(row["buy_net_pct"])
        _record_buy_stat(stats[bucket],y,net,predicted)
        period=_period_key(signal)
        if period not in periods:
            periods[period]=_empty_buy_stat()
        _record_buy_stat(periods[period],y,net,predicted)
    for stat in stats.values():
        _finalize_buy_stat(stat)
    for stat in periods.values():
        _finalize_buy_stat(stat)
    return {"windows":stats,"periods":periods,
            "training_updates":model.count,
            "trade_assumptions":{"entry":"next_open","max_hold_sessions":HOLD,
               "stop":"max(1.8*signal_ATR,2.5% of entry)",
               "take":"1.8R from next open","roundtrip_cost_pct":COST_PCT,
               "same_candle_stop_take":"stop_first"},
            "target_success_pct":80,"live_approved":False}


def _buy_candidates(search_generation,previous_best):
    candidates=[]
    seen=set()
    def add(name,lr,l2,prior_mix):
        lr=round(max(0.005,min(0.5,float(lr))),5)
        l2=round(max(0.001,min(0.5,float(l2))),5)
        prior_mix=round(max(0.0,min(1.0,float(prior_mix))),5)
        key=(lr,l2,prior_mix)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"name":name,"learning_rate":lr,
                           "l2":l2,"prior_mix":prior_mix})
    for lr in (0.03,0.06,0.10,0.11,0.20):
        for l2 in (0.02,0.065,0.12):
            for prior_mix in (0.0,0.15,0.30,0.50,1.0):
                add(f"blr{lr:g}_l2{l2:g}_prior{prior_mix:g}",
                    lr,l2,prior_mix)
    center=previous_best or {"learning_rate":0.10,"l2":0.065,
                             "prior_mix":0.15}
    seed=(130363*int(search_generation)
          +int(float(center.get("learning_rate",0.10))*100000)
          +int(float(center.get("l2",0.065))*100000)
          +int(float(center.get("prior_mix",0.15))*10000))
    rng=random.Random(seed)
    for i in range(35):
        lr=float(center.get("learning_rate",0.10))*math.exp(rng.uniform(-0.8,0.8))
        reg=float(center.get("l2",0.065))*math.exp(rng.uniform(-0.9,0.9))
        prior=float(center.get("prior_mix",0.15))+rng.uniform(-0.25,0.25)
        add(f"bg{search_generation}_mutation{i+1}",lr,reg,prior)
    return candidates


def score_buy_success(trades,*,train_end="2025-11-01",
                      validation_end="2026-04-01",eval_end="2026-11-01",
                      search_generation=1,previous_best=None):
    """Evolve probability formulas over all completed historical half-years."""
    candidates=_buy_candidates(search_generation,previous_best)
    scored=[]
    for candidate in candidates:
        report=_score_buy_success_candidate(
            trades,train_end=train_end,validation_end=validation_end,
            eval_end=eval_end,lr=candidate["learning_rate"],
            l2=candidate["l2"],prior_mix=candidate["prior_mix"])
        validation=report["windows"]["validation"]
        period_scores=[]
        for period,stat in sorted(report["periods"].items()):
            if stat.get("count",0)<100 or stat.get("brier_model") is None:
                continue
            period_scores.append({
                "period":period,"samples":stat["count"],
                "brier":stat["brier_model"],
                "baseline_brier":stat.get("brier_baseline"),
                "calibration_gap_pp":stat.get("calibration_gap_pp"),
                "actual_success_pct":stat.get("actual_success_pct"),
                "predicted_success_pct":stat.get("mean_predicted_success_pct")})
        if period_scores:
            mean_brier=round(sum(x["brier"] for x in period_scores)
                             /len(period_scores),6)
            worst_brier=max(x["brier"] for x in period_scores)
            mean_gap=round(sum(abs(x["calibration_gap_pp"] or 0)
                               for x in period_scores)
                           /len(period_scores),3)
        else:
            mean_brier=validation.get("brier_model")
            worst_brier=mean_brier
            mean_gap=abs(validation.get("calibration_gap_pp") or 0)
        scored.append({"candidate":candidate,"report":report,
            "period_scores":period_scores,
            "whole_history_brier":mean_brier,
            "worst_period_brier":worst_brier,
            "whole_history_calibration_gap_pp":mean_gap,
            "period_count":len(period_scores),
            "validation_brier":validation.get("brier_model"),
            "validation_baseline_brier":validation.get("brier_baseline"),
            "validation_samples":validation.get("count",0),
            "validation_calibration_gap_pp":validation.get("calibration_gap_pp")})
    eligible=[r for r in scored if r["period_count"]>=2
              and r["whole_history_brier"] is not None]
    if not eligible:
        eligible=[r for r in scored if r["validation_samples"]>=30
                  and r["validation_brier"] is not None]
    chosen=min(eligible,key=lambda r:(
        r["whole_history_brier"],
        r["worst_period_brier"],
        r["whole_history_calibration_gap_pp"]))
    result=chosen["report"]
    result["tuning"]={
        "generation":int(search_generation),
        "selection_window":"완료된 전체 과거기간의 반기별 순차평가",
        "selection_metric":"반기별 Brier 평균 최소화, 최악 반기와 확률 보정오차 순 보조",
        "chosen":chosen["candidate"],
        "previous_best":previous_best,
        "chosen_period_scores":chosen["period_scores"],
        "candidates":[{k:v for k,v in item.items()
                       if k not in ("report","period_scores")}
                      for item in scored],
        "candidate_count":len(candidates),
        "period_count":chosen["period_count"],
        "whole_history_used_for_selection":True,
        "independent_holdout":False,
        "warning":"확률식은 예측 오차를 줄이지만 모든 거래의 실제 승률 자체를 임의로 바꾸지 않습니다."
    }
    return result
