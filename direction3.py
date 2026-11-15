"""Three-way UP / DOWN / SIDEWAYS forecasting with strictly causal online updates.

Research output, NOT guaranteed 80% accuracy. Forecast at the signal CLOSE,
label using the CLOSE exactly HORIZON subsequent sessions later, and only train
on labels whose observation_date is strictly BEFORE the next signal date.

Definitions (10 sessions, fixed as a hypothesis before evaluation):
    UP       forward close-to-close return > +3%
    DOWN     forward close-to-close return < -3%
    SIDEWAYS all other returns, including +/-3% boundaries.
Each signal produces three probabilities summing to one.
"""
import math

LABELS=("UP","DOWN","SIDEWAYS")
HORIZON=10
BAND_PCT=3.0
WARMUP=240
SAMPLE_STEP=2
TRAIN_MIN=90
CENTERS=(55.,25.,2.5,2.,1.3,2.,0.,0.,0.)
SCALES=(20.,20.,3.,4.,1.5,8.,3.,1.,1.)

def direction_label(forward_return_pct,band_pct=BAND_PCT):
    r=float(forward_return_pct)
    if not math.isfinite(r):raise ValueError("nonfinite return")
    if r>band_pct:return "UP"
    if r< -band_pct:return "DOWN"
    return "SIDEWAYS"

def features(row):
    try:
        keys=("signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct",
              "signal_volume_ratio","signal_5d_return_pct",
              "signal_macdh_atr","signal_price_ma200_pct","signal_bb_position")
        x=[float(row[k]) for k in keys]
        if not all(math.isfinite(v) for v in x):return None
        return [1.]+[max(-3.,min(3.,(a-c)/s)) for a,c,s in zip(x,CENTERS,SCALES)]
    except (KeyError,TypeError,ValueError,OverflowError):
        return None

def stable_softmax(values):
    top=max(values)
    unnorm=[math.exp(v-top) for v in values]
    total=sum(unnorm)
    return [v/total for v in unnorm]

class OnlineDirectionModel:
    """9 observables + intercept, 3-class softmax with AdaGrad SGD.

    Learns signal-to-future-label relationship, never changing the input
    features based on outcome. Independent learner per market.
    """
    def __init__(self,learning_rate=0.13,l2=0.06,prior_mix=0.15,sideways_weight=1.0):
        self.weights=[[0.]*10 for _ in LABELS]
        self.grad_sq=[[4.]*10 for _ in LABELS]
        self.lr=learning_rate
        self.l2=l2
        self.prior_mix=max(0.,min(1.,float(prior_mix)))
        self.sideways_weight=max(1.,float(sideways_weight))
        self.class_counts=[0]*3
        self.samples=0
    def prior(self):
        n=self.samples+3
        return [(count+1)/n for count in self.class_counts]
    def predict(self,row):
        vec=features(row)
        if vec is None or self.samples<TRAIN_MIN:return None
        probs=stable_softmax([sum(w*x for w,x in zip(ws,vec)) for ws in self.weights])
        # Shrink early, noisy 3-way predictions toward online class frequencies.
        prior=self.prior()
        return [(1-self.prior_mix)*p+self.prior_mix*q for p,q in zip(probs,prior)]
    def update(self,row):
        vec=features(row)
        label=row.get("direction_label")
        if vec is None or label not in LABELS:return False
        j=LABELS.index(label)
        old=stable_softmax([sum(w*x for w,x in zip(ws,vec)) for ws in self.weights])
        for k in range(3):
            for m,x in enumerate(vec):
                grad=self.sideways_weight*(old[k]-(1. if k==j else 0.))*x if j==2 else (old[k]-(1. if k==j else 0.))*x
                if m:grad+=self.l2*self.weights[k][m]
                self.grad_sq[k][m]+=grad*grad
                self.weights[k][m]-=self.lr*grad/math.sqrt(self.grad_sq[k][m])
        self.samples+=1
        self.class_counts[j]+=1
        return True

def make_direction_rows(frame,ticker,market,*,end_date="2026-11-01",
                        horizon=HORIZON,step=SAMPLE_STEP,band_pct=BAND_PCT):
    """Emit independent fixed-horizon outcomes, without BUY gating.

    Signal features are computed at session i close only. Label is revealed
    only at session i+horizon close.
    """
    import pandas as pd
    import math
    result=[]
    for i in range(WARMUP,len(frame)-horizon,step):
        signal_date=str(pd.Timestamp(frame.index[i]).date())
        if signal_date>=end_date:break
        r=frame.iloc[i]
        try:
            close=float(r.Close)
            due=float(frame.Close.iloc[i+horizon])
            p5=float(frame.Close.iloc[i-5])
            p200=float(r.MA200)
            ma60=float(r.MA60)
            atr=float(r.ATR)
            v20=float(r.VOL20)
            sd=float(r.BBUP-r.BBLOW)
            if min(close,due,p5,p200,ma60,atr,v20,sd)<=0:continue
            values={
                "signal_rsi":float(r.RSI),
                "signal_adx":float(r.ADX),
                "signal_atr_pct":100*atr/close,
                "signal_ma20_ma60_pct":100*(float(r.MA20)/ma60-1),
                "signal_volume_ratio":float(r.Volume)/v20,
                "signal_5d_return_pct":100*(close/p5-1),
                "signal_macdh_atr":float(r.MACDH)/atr,
                "signal_price_ma200_pct":100*(close/p200-1),
                "signal_bb_position":2*(close-float(r.BBLOW))/sd-1,
            }
            if features(values) is None:continue
            fw=100*(due/close-1)
            result.append({"market":market,"ticker":ticker,
                "signal_date":signal_date,
                "label_available_date":str(pd.Timestamp(frame.index[i+horizon]).date()),
                "horizon_sessions":horizon,"band_pct":band_pct,
                "forward_return_pct":round(fw,6),
                "direction_label":direction_label(fw,band_pct),**values})
        except (TypeError,ValueError,ZeroDivisionError,OverflowError,KeyError):
            continue
    return result

def _score_direction_candidate(trades,*,train_end="2025-11-01",
                     validation_end="2026-04-01",eval_end="2026-11-01",
                     learning_rate=0.13,l2=0.06,prior_mix=0.15,sideways_weight=1.0):
    """Prequential 3-class accuracy, precision/recall, Brier, baseline.

    For any prediction, only outcomes whose label_available_date < signal
    date have been observed and used for training. Predictions are recorded
    for EVERY eligible observation, including low-confidence examples.
    """
    ordered=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"]))
    completions=iter(sorted(trades,key=lambda t:(t["label_available_date"],t["ticker"])))
    next_done=next(completions,None)
    model=OnlineDirectionModel(learning_rate=learning_rate,l2=l2,prior_mix=prior_mix,sideways_weight=sideways_weight)
    stats={k:{"total":0,"correct":0,"brier_sum":0.0,
              "baseline_brier_sum":0.0,
              "confusion":{t:{p:0 for p in LABELS} for t in LABELS},
              "actual":{a:0 for a in LABELS},
              "predicted":{a:0 for a in LABELS},
              "sum_probability":{a:0.0 for a in LABELS}}
           for k in ("train","validation","diagnostic")}
    skipped=0
    for t in ordered:
        signal=t["signal_date"]
        while next_done is not None and next_done["label_available_date"]<signal:
            model.update(next_done)
            next_done=next(completions,None)
        if t["label_available_date"]>=eval_end:continue
        probs=model.predict(t)
        if probs is None:
            skipped+=1;continue
        k="train" if signal<train_end else "validation" if signal<validation_end else "diagnostic"
        a=LABELS.index(t["direction_label"])
        pred=max(range(3),key=lambda j:probs[j])
        base=model.prior()
        st=stats[k]
        st["total"]+=1
        st["correct"]+=int(a==pred)
        st["brier_sum"]+=sum((p-(1 if j==a else 0))**2 for j,p in enumerate(probs))
        st["baseline_brier_sum"]+=sum((p-(1 if j==a else 0))**2 for j,p in enumerate(base))
        st["confusion"][LABELS[a]][LABELS[pred]]+=1
        st["actual"][LABELS[a]]+=1
        st["predicted"][LABELS[pred]]+=1
        for j,label in enumerate(LABELS):
            st["sum_probability"][label]+=probs[j]
    for st in stats.values():
        n=st["total"]
        st["accuracy_pct"]=round(100*st["correct"]/n,2) if n else None
        st["multiclass_brier"]=round(st.pop("brier_sum")/n,6) if n else None
        st["prior_brier"]=round(st.pop("baseline_brier_sum")/n,6) if n else None
        st["mean_probabilities_pct"]={k:round(100*v/n,2) if n else None
                                     for k,v in st.pop("sum_probability").items()}
        st["class_metrics"]={}
        for label in LABELS:
            tp=st["confusion"][label][label]
            predicted=st["predicted"][label]
            actual=st["actual"][label]
            st["class_metrics"][label]={
                "precision_pct":round(100*tp/predicted,2) if predicted else None,
                "recall_pct":round(100*tp/actual,2) if actual else None,
                "correct":tp,"predicted":predicted,"actual":actual}
    return {"windows":stats,"skipped_warmup":skipped,"trained_outcomes":model.samples,
            "horizon_sessions":HORIZON,"band_pct":BAND_PCT,
            "target_precision_pct_each_direction":80,
            "note":"80% is a research target, NOT a verified result."}


def score_directions(trades,*,train_end="2025-11-01",
                     validation_end="2026-04-01",eval_end="2026-11-01"):
    """Tune a small frozen hyperparameter menu on validation, then report all samples.

    Selection uses validation macro F1 across UP, DOWN, and SIDEWAYS, then accuracy, then Brier score.
    The already reviewed diagnostic window is never used to select a candidate.
    """
    candidates=[{"name":"baseline_lr0.13_l20.06_prior0.15_side1",
                  "learning_rate":0.13,"l2":0.06,"prior_mix":0.15,"sideways_weight":1.0}]
    for learning_rate in (0.03,0.10,0.20):
        for l2 in (0.02,0.12):
            for prior_mix in (0.0,0.30,0.60):
                for sideways_weight in (1.0,2.0,4.0):
                    candidates.append({
                        "name":f"lr{learning_rate:g}_l2{l2:g}_prior{prior_mix:g}_side{sideways_weight:g}",
                        "learning_rate":learning_rate,"l2":l2,
                        "prior_mix":prior_mix,"sideways_weight":sideways_weight})
    scored=[]
    for candidate in candidates:
        report=_score_direction_candidate(
            trades,train_end=train_end,validation_end=validation_end,
            eval_end=eval_end,learning_rate=candidate["learning_rate"],
            l2=candidate["l2"],prior_mix=candidate["prior_mix"],
            sideways_weight=candidate["sideways_weight"])
        validation=report["windows"]["validation"]
        class_metrics=validation.get("class_metrics") or {}
        f1_scores=[]
        for label in LABELS:
            item=class_metrics.get(label) or {}
            precision=(item.get("precision_pct") or 0)/100
            recall=(item.get("recall_pct") or 0)/100
            f1_scores.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
        macro_f1_pct=round(100*sum(f1_scores)/len(LABELS),2)
        scored.append({"candidate":candidate,"report":report,
            "validation_macro_f1_pct":macro_f1_pct,
            "validation_accuracy_pct":validation.get("accuracy_pct"),
            "validation_brier":validation.get("multiclass_brier"),
            "validation_samples":validation.get("total",0)})
    eligible=[r for r in scored if r["validation_samples"]>=30 and
              r["validation_macro_f1_pct"] is not None]
    if not eligible:
        chosen=next(r for r in scored if r["candidate"]["name"]=="baseline_lr0.13_l20.06_prior0.15_side1")
    else:
        chosen=max(eligible,key=lambda r:(
            r["validation_macro_f1_pct"],r["validation_accuracy_pct"],
            -(r["validation_brier"] if r["validation_brier"] is not None else float("inf"))))
    result=chosen["report"]
    result["tuning"]={
        "selection_window":f"signal date < {validation_end}",
        "selection_metric":"UP·DOWN·횡보 3개 클래스 Macro F1 최대화, 동률이면 전체정확도 후 Brier",
        "chosen":chosen["candidate"],
        "candidates":[{k:v for k,v in item.items() if k!="report"} for item in scored],
        "candidate_count":len(candidates),
        "diagnostic_used_for_selection":False,
        "warning":"후보 반복 비교로 검증구간도 선택에 사용되었습니다. 이 구간은 이후 독립 검증이 아닙니다."
    }
    result["note"]="80%는 목표이며 미달 결과를 그대로 기록합니다. 진단구간은 후보 선택에 쓰지 않았습니다."
    return result
