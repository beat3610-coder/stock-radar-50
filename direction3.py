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
import random

LABELS=("UP","DOWN","SIDEWAYS")
HORIZON=10
BAND_PCT=3.0
WARMUP=240
SAMPLE_STEP=2
TRAIN_MIN=90
FEATURE_KEYS=(
    "signal_rsi","signal_adx","signal_atr_pct","signal_ma20_ma60_pct",
    "signal_volume_ratio","signal_5d_return_pct","signal_macdh_atr",
    "signal_price_ma200_pct","signal_bb_position","signal_rsi_wilder",
    "signal_adx_wilder","signal_cmf20","signal_donchian20_pos",
    "signal_roc20","signal_bb_width_pct",
)
CENTERS=(55.,25.,2.5,2.,1.3,2.,0.,0.,0.,55.,25.,0.,.5,0.,8.)
SCALES=(20.,20.,3.,4.,1.5,8.,3.,1.,1.,20.,20.,.25,.35,10.,8.)
FEATURE_COUNT=len(FEATURE_KEYS)+1

def direction_label(forward_return_pct,band_pct=BAND_PCT):
    r=float(forward_return_pct)
    if not math.isfinite(r):raise ValueError("nonfinite return")
    if r>band_pct:return "UP"
    if r< -band_pct:return "DOWN"
    return "SIDEWAYS"

def features(row):
    try:
        x=[float(row[k]) for k in FEATURE_KEYS]
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
        self.weights=[[0.]*FEATURE_COUNT for _ in LABELS]
        self.grad_sq=[[4.]*FEATURE_COUNT for _ in LABELS]
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
                "signal_rsi_wilder":float(r.RSI_WILDER),
                "signal_adx_wilder":float(r.ADX_WILDER),
                "signal_cmf20":float(r.CMF20),
                "signal_donchian20_pos":float(r.DONCHIAN20_POS),
                "signal_roc20":float(r.ROC20),
                "signal_bb_width_pct":float(r.BBWIDTH20_PCT),
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

def _empty_direction_stat():
    return {"total":0,"correct":0,"brier_sum":0.0,
            "baseline_brier_sum":0.0,
            "confusion":{t:{p:0 for p in LABELS} for t in LABELS},
            "actual":{a:0 for a in LABELS},
            "predicted":{a:0 for a in LABELS},
            "sum_probability":{a:0.0 for a in LABELS}}


def _record_direction(st,actual,predicted,probs,baseline):
    st["total"]+=1
    st["correct"]+=int(actual==predicted)
    st["brier_sum"]+=sum((p-(1 if j==actual else 0))**2
                         for j,p in enumerate(probs))
    st["baseline_brier_sum"]+=sum((p-(1 if j==actual else 0))**2
                                  for j,p in enumerate(baseline))
    st["confusion"][LABELS[actual]][LABELS[predicted]]+=1
    st["actual"][LABELS[actual]]+=1
    st["predicted"][LABELS[predicted]]+=1
    for j,label in enumerate(LABELS):
        st["sum_probability"][label]+=probs[j]


def _finalize_direction_stat(st):
    n=st["total"]
    st["accuracy_pct"]=round(100*st["correct"]/n,2) if n else None
    st["multiclass_brier"]=round(st.pop("brier_sum")/n,6) if n else None
    st["prior_brier"]=round(st.pop("baseline_brier_sum")/n,6) if n else None
    st["mean_probabilities_pct"]={
        k:round(100*v/n,2) if n else None
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


def _macro_f1(stat):
    scores=[]
    for label in LABELS:
        item=(stat.get("class_metrics") or {}).get(label) or {}
        precision=(item.get("precision_pct") or 0)/100
        recall=(item.get("recall_pct") or 0)/100
        scores.append(2*precision*recall/(precision+recall)
                      if precision+recall else 0.0)
    return round(100*sum(scores)/len(LABELS),2)


def _period_key(signal_date):
    return signal_date[:4]+("-H1" if signal_date[5:7]<="06" else "-H2")


def _score_direction_candidate(trades,*,train_end="2025-11-01",
                     validation_end="2026-04-01",eval_end="2026-11-01",
                     learning_rate=0.13,l2=0.06,prior_mix=0.15,sideways_weight=1.0):
    """Causal prequential scoring with additional whole-history half-year slices."""
    ordered=sorted(trades,key=lambda t:(t["signal_date"],t["ticker"]))
    completions=iter(sorted(trades,key=lambda t:(t["label_available_date"],t["ticker"])))
    next_done=next(completions,None)
    model=OnlineDirectionModel(learning_rate=learning_rate,l2=l2,
        prior_mix=prior_mix,sideways_weight=sideways_weight)
    stats={k:_empty_direction_stat()
           for k in ("train","validation","diagnostic")}
    periods={}
    skipped=0
    for t in ordered:
        signal=t["signal_date"]
        while next_done is not None and next_done["label_available_date"]<signal:
            model.update(next_done)
            next_done=next(completions,None)
        if t["label_available_date"]>=eval_end:
            continue
        probs=model.predict(t)
        if probs is None:
            skipped+=1
            continue
        bucket=("train" if signal<train_end else
                "validation" if signal<validation_end else "diagnostic")
        actual=LABELS.index(t["direction_label"])
        predicted=max(range(3),key=lambda j:probs[j])
        baseline=model.prior()
        _record_direction(stats[bucket],actual,predicted,probs,baseline)
        period=_period_key(signal)
        if period not in periods:
            periods[period]=_empty_direction_stat()
        _record_direction(periods[period],actual,predicted,probs,baseline)
    for st in stats.values():
        _finalize_direction_stat(st)
    for st in periods.values():
        _finalize_direction_stat(st)
    return {"windows":stats,"periods":periods,
            "skipped_warmup":skipped,"trained_outcomes":model.samples,
            "horizon_sessions":HORIZON,"band_pct":BAND_PCT,
            "target_precision_pct_each_direction":80,
            "note":"80% is a research target, NOT a verified result."}


def _direction_candidates(search_generation,previous_best):
    candidates=[]
    seen=set()
    def add(name,learning_rate,l2,prior_mix,sideways_weight):
        learning_rate=round(max(0.005,min(0.5,float(learning_rate))),5)
        l2=round(max(0.001,min(0.5,float(l2))),5)
        prior_mix=round(max(0.0,min(0.95,float(prior_mix))),5)
        sideways_weight=round(max(1.0,min(6.0,float(sideways_weight))),5)
        key=(learning_rate,l2,prior_mix,sideways_weight)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"name":name,"learning_rate":learning_rate,
            "l2":l2,"prior_mix":prior_mix,
            "sideways_weight":sideways_weight})
    add("baseline_lr0.13_l20.06_prior0.15_side1",0.13,0.06,0.15,1.0)
    for learning_rate in (0.03,0.10,0.20):
        for l2 in (0.02,0.12):
            for prior_mix in (0.0,0.30,0.60):
                for sideways_weight in (1.0,2.0,4.0):
                    add(f"lr{learning_rate:g}_l2{l2:g}_prior{prior_mix:g}_side{sideways_weight:g}",
                        learning_rate,l2,prior_mix,sideways_weight)
    center=previous_best or {"learning_rate":0.10,"l2":0.02,
                             "prior_mix":0.0,"sideways_weight":1.0}
    if previous_best:
        add("previous_champion",center.get("learning_rate",0.10),
            center.get("l2",0.02),center.get("prior_mix",0.0),
            center.get("sideways_weight",1.0))
    seed=(104729*int(search_generation)
          +int(float(center.get("learning_rate",0.10))*100000)
          +int(float(center.get("l2",0.02))*100000)
          +int(float(center.get("prior_mix",0.0))*10000)
          +int(float(center.get("sideways_weight",1.0))*1000))
    rng=random.Random(seed)
    for i in range(40):
        lr=float(center.get("learning_rate",0.10))*math.exp(rng.uniform(-0.8,0.8))
        reg=float(center.get("l2",0.02))*math.exp(rng.uniform(-0.9,0.9))
        prior=float(center.get("prior_mix",0.0))+rng.uniform(-0.24,0.24)
        side=float(center.get("sideways_weight",1.0))*math.exp(rng.uniform(-0.45,0.55))
        add(f"g{search_generation}_mutation{i+1}",lr,reg,prior,side)
    return candidates


def score_directions(trades,*,train_end="2025-11-01",
                     validation_end="2026-04-01",eval_end="2026-11-01",
                     search_generation=1,previous_best=None):
    """Evolve formulas and rank them over every eligible historical half-year.

    Each prediction remains causal: a label is learned only after its availability
    date. Formula selection deliberately uses all completed historical periods,
    so the resulting score is a research fit and not an independent holdout.
    """
    candidates=_direction_candidates(search_generation,previous_best)
    scored=[]
    for candidate in candidates:
        report=_score_direction_candidate(
            trades,train_end=train_end,validation_end=validation_end,
            eval_end=eval_end,learning_rate=candidate["learning_rate"],
            l2=candidate["l2"],prior_mix=candidate["prior_mix"],
            sideways_weight=candidate["sideways_weight"])
        validation=report["windows"]["validation"]
        period_scores=[]
        for period,stat in sorted(report["periods"].items()):
            if stat.get("total",0)<100:
                continue
            period_scores.append({
                "period":period,"samples":stat["total"],
                "macro_f1_pct":_macro_f1(stat),
                "accuracy_pct":stat.get("accuracy_pct"),
                "brier":stat.get("multiclass_brier")})
        if period_scores:
            mean_f1=round(sum(x["macro_f1_pct"] for x in period_scores)
                          /len(period_scores),2)
            worst_f1=round(min(x["macro_f1_pct"] for x in period_scores),2)
            mean_accuracy=round(sum(x["accuracy_pct"] for x in period_scores
                                    if x["accuracy_pct"] is not None)
                                /len(period_scores),2)
            mean_brier=round(sum(x["brier"] for x in period_scores
                                 if x["brier"] is not None)
                             /len(period_scores),6)
        else:
            mean_f1=_macro_f1(validation)
            worst_f1=mean_f1
            mean_accuracy=validation.get("accuracy_pct")
            mean_brier=validation.get("multiclass_brier")
        scored.append({"candidate":candidate,"report":report,
            "period_scores":period_scores,
            "whole_history_macro_f1_pct":mean_f1,
            "worst_period_macro_f1_pct":worst_f1,
            "whole_history_accuracy_pct":mean_accuracy,
            "whole_history_brier":mean_brier,
            "period_count":len(period_scores),
            "validation_macro_f1_pct":_macro_f1(validation),
            "validation_accuracy_pct":validation.get("accuracy_pct"),
            "validation_brier":validation.get("multiclass_brier"),
            "validation_samples":validation.get("total",0)})
    eligible=[r for r in scored if r["period_count"]>=2]
    if not eligible:
        eligible=[r for r in scored if r["validation_samples"]>=30]
    if not eligible:
        eligible=scored
    chosen=max(eligible,key=lambda r:(
        r["whole_history_macro_f1_pct"] or 0,
        r["worst_period_macro_f1_pct"] or 0,
        r["whole_history_accuracy_pct"] or 0,
        -(r["whole_history_brier"] if r["whole_history_brier"] is not None
          else float("inf"))))
    result=chosen["report"]
    result["tuning"]={
        "generation":int(search_generation),
        "selection_window":"완료된 전체 과거기간의 반기별 순차평가",
        "selection_metric":"반기별 Macro F1 평균 최대화, 최악 반기·정확도·Brier 순 보조",
        "chosen":chosen["candidate"],
        "previous_best":previous_best,
        "chosen_period_scores":chosen["period_scores"],
        "candidates":[{k:v for k,v in item.items()
                       if k not in ("report","period_scores")}
                      for item in scored],
        "candidate_count":len(candidates),
        "period_count":chosen["period_count"],
        "whole_history_used_for_selection":True,
        "diagnostic_used_for_selection":True,
        "independent_holdout":False,
        "warning":"전체 과거 구간을 수식 선택에 사용한 연구 적합도입니다. 미래 성공률 보장이 아닙니다."
    }
    result["note"]="전 기간을 반기별로 순환 학습·비교하며 미달 결과도 그대로 기록합니다."
    return result
