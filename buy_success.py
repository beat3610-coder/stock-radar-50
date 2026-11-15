"""Online buy-success forecasting paired with the three-way direction model.

Research only: signal-at-close; hypothetical entry at next OPEN. Exit uses
conservative stop-first 10-session OHLC, fixed 1.8 ATR minimum 2.5% stop and
1.8R take. Estimated round-trip cost: 0.35%. NO buy-signal/score filters.
Labels mature at +10 sessions even when the stop or target hit earlier;
predictions always precede training on labels.
"""
import math
import pandas as pd
from direction3 import features
from audit_300 import simulate_exit

HOLD=10
COST_PCT=0.35
TRAIN_MIN=90

class OnlineBuySuccess:
    def __init__(self, lr=0.11, l2=0.065, prior_mix=0.15):
        self.weights=[0.0]*10
        self.adagrad=[4.0]*10
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

def _score_buy_success_candidate(trades,*,train_end="2025-11-01",
                      validation_end="2026-04-01",eval_end="2026-11-01",
                      lr=0.11,l2=0.065,prior_mix=0.15):
    """All-sample observed win and forecast success percentage by time window."""
    stream=sorted(trades,key=lambda r:(r["signal_date"],r["ticker"]))
    done=iter(sorted(trades,key=lambda r:(r["buy_label_available_date"],r["ticker"])))
    next_done=next(done,None)
    model=OnlineBuySuccess(lr=lr,l2=l2,prior_mix=prior_mix)
    def state():
        return {"count":0,"wins":0,"pred_sum":0.0,"brier":0.0,
                "base_brier":0.0,"net_sum":0.0,"expected_net_sum":0.0,
                "actual_profit_sum":0.0,"actual_loss_sum":0.0}
    stats={k:state() for k in ("train","validation","diagnostic")}
    for row in stream:
        signal=row["signal_date"]
        while next_done is not None and next_done["buy_label_available_date"]<signal:
            model.update(next_done)
            next_done=next(done,None)
        if row["buy_label_available_date"]>=eval_end:continue
        predicted=model.predict(row)
        if predicted is None:continue
        bucket=("train" if signal<train_end else
                "validation" if signal<validation_end else "diagnostic")
        s=stats[bucket]
        y=float(row["buy_won"]);net=float(row["buy_net_pct"])
        p=predicted["p_success"]
        s["count"]+=1;s["wins"]+=int(y)
        s["pred_sum"]+=p
        s["brier"]+=(p-y)**2
        s["base_brier"]+=(predicted["baseline_p"]-y)**2
        s["net_sum"]+=net
        s["expected_net_sum"]+=predicted["expected_net_pct"]
        if net>0:s["actual_profit_sum"]+=net
        else:s["actual_loss_sum"]+=-net
    for s in stats.values():
        n=s["count"]
        p=s.pop("pred_sum")
        actual_win=s["wins"]
        s["mean_predicted_success_pct"]=round(100*p/n,2) if n else None
        s["actual_success_pct"]=round(100*actual_win/n,2) if n else None
        s["calibration_gap_pp"]=round(100*(p-actual_win)/n,2) if n else None
        s["brier_model"]=round(s.pop("brier")/n,6) if n else None
        s["brier_baseline"]=round(s.pop("base_brier")/n,6) if n else None
        s["mean_net_pct"]=round(s.pop("net_sum")/n,3) if n else None
        s["mean_predicted_net_pct"]=round(s.pop("expected_net_sum")/n,3) if n else None
        gain=s.pop("actual_profit_sum")
        loss=s.pop("actual_loss_sum")
        s["profit_factor"]=round(gain/loss,3) if loss else None
    return {"windows":stats,"training_updates":model.count,
            "trade_assumptions":{"entry":"next_open","max_hold_sessions":HOLD,
               "stop":"max(1.8*signal_ATR,2.5% of entry)",
               "take":"1.8R from next open","roundtrip_cost_pct":COST_PCT,
               "same_candle_stop_take":"stop_first"},
            "target_success_pct":80,"live_approved":False}


def score_buy_success(trades,*,train_end="2025-11-01",
                      validation_end="2026-04-01",eval_end="2026-11-01"):
    """Tune probability parameters on validation Brier; report all outcomes unfiltered."""
    candidates=[]
    for lr in (0.03,0.06,0.10,0.11,0.20):
        for l2 in (0.02,0.065,0.12):
            for prior_mix in (0.0,0.15,0.30,0.50,1.0):
                candidates.append({"name":f"blr{lr:g}_l2{l2:g}_prior{prior_mix:g}",
                                   "learning_rate":lr,"l2":l2,"prior_mix":prior_mix})
    scored=[]
    for candidate in candidates:
        report=_score_buy_success_candidate(
            trades,train_end=train_end,validation_end=validation_end,eval_end=eval_end,
            lr=candidate["learning_rate"],l2=candidate["l2"],
            prior_mix=candidate["prior_mix"])
        validation=report["windows"]["validation"]
        scored.append({"candidate":candidate,"report":report,
            "validation_brier":validation.get("brier_model"),
            "validation_baseline_brier":validation.get("brier_baseline"),
            "validation_samples":validation.get("count",0),
            "validation_calibration_gap_pp":validation.get("calibration_gap_pp")})
    eligible=[r for r in scored if r["validation_samples"]>=30 and
              r["validation_brier"] is not None]
    if not eligible:
        chosen=next(r for r in scored if r["candidate"]["name"]=="blr0.11_l20.065_prior0.15")
    else:
        chosen=min(eligible,key=lambda r:(
            r["validation_brier"],abs(r["validation_calibration_gap_pp"] or 0)))
    result=chosen["report"]
    result["tuning"]={
        "selection_window":f"signal date < {validation_end}",
        "selection_metric":"검증 Brier 최소화; 동률이면 예측확률-실제승률 차이 최소화",
        "chosen":chosen["candidate"],
        "candidates":[{k:v for k,v in item.items() if k!="report"} for item in scored],
        "candidate_count":len(candidates),
        "diagnostic_used_for_selection":False,
        "warning":"전체 표본 승률은 모든 종목 결과의 실제 빈도이며, 확률식 조정으로 바뀌지 않습니다."
    }
    return result
