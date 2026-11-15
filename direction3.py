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
    def __init__(self,learning_rate=0.13,l2=0.06):
        self.weights=[[0.]*10 for _ in LABELS]
        self.grad_sq=[[4.]*10 for _ in LABELS]
        self.lr=learning_rate
        self.l2=l2
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
        return [0.85*p+0.15*q for p,q in zip(probs,prior)]
    def update(self,row):
        vec=features(row)
        label=row.get("direction_label")
        if vec is None or label not in LABELS:return False
        j=LABELS.index(label)
        old=stable_softmax([sum(w*x for w,x in zip(ws,vec)) for ws in self.weights])
        for k in range(3):
            for m,x in enumerate(vec):
                grad=(old[k]-(1. if k==j else 0.))*x
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

def score_directions(trades,*,train_end="2025-11-01",
                     validation_end="2026-04-01",eval_end="2026-11-01"):
    """Prequential 3-class accuracy, precision/recall, Brier, baseline.

    For any prediction, only outcomes whose label_available_date < signal
    date have been observed and used for training. Predictions are recorded
    for EVERY eligible observation, including low-confidence examples.
    """
    ordered=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"]))
    completions=iter(sorted(trades,key=lambda t:(t["label_available_date"],t["ticker"])))
    next_done=next(completions,None)
    model=OnlineDirectionModel()
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
