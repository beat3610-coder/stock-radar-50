"""Causal market-regime conditional probability shrinkage (research candidate).

A regime is computed from today's observed market-wide breadth and MA spread.
Only events with labels available strictly BEFORE the prediction date enter the
conditional prior. Never uses forecasted returns or future market conditions.
"""
from collections import deque
import numpy as np

def regime_of(row, context):
    ctx=context.get(row["signal_date"])
    if ctx is None:
        return 1
    breadth=float(ctx[2])
    spread=float(ctx[3])
    if breadth>=0.58 and spread>=0.5:
        return 0
    if breadth<=0.42 and spread<=-0.5:
        return 2
    return 1

def causal_regime_shrink(events, history, base_probs, context, *, blend=0.25,
                         recent_window=1500, min_regime_samples=80):
    if not 0<=blend<=1:
        raise ValueError("blend must be a probability")
    if len(events)!=len(base_probs):
        raise ValueError("events and probabilities are not paired")
    base=np.asarray(base_probs,dtype=float)
    output=np.array(base,copy=True)
    if not len(events):
        return output
    # Eligible labels enter only after their maturity date, never at the
    # signal date. Observed regime belongs to the historical signal date.
    observations=sorted(
        [(r[0]["label_available_date"],regime_of(r[0],context),int(r[2]))
         for r in list(history)+list(events)],
        key=lambda x:x[0])
    order=sorted(range(len(events)),key=lambda i:events[i][0]["signal_date"])
    queue=deque()
    counts=np.zeros((3,3),dtype=float)
    j=0
    for index in order:
        day=events[index][0]["signal_date"]
        while j<len(observations) and observations[j][0]<day:
            maturity,regime,label=observations[j]
            queue.append((regime,label))
            counts[regime,label]+=1
            if len(queue)>recent_window:
                old_regime,old_label=queue.popleft()
                counts[old_regime,old_label]-=1
            j+=1
        regime=regime_of(events[index][0],context)
        n=counts[regime].sum()
        if n<min_regime_samples:
            continue
        overall=counts.sum(axis=0)
        global_prior=(overall+1)/(overall.sum()+3)
        # Strong shrinkage prevents regime groups with small counts
        # from producing artificially high predictions.
        conditional=(counts[regime]+100*global_prior)/(n+100)
        output[index]=(1-blend)*base[index]+blend*conditional
    return output
