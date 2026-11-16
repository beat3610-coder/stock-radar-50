"""Causal stock-specific probability calibration on top of the market model.

Not a separate per-stock classifier: small ticker samples are strongly shrunk
towards pooled mature class counts. Labels are added only after maturity.
The candidate parameters are fixed BEFORE inspecting 2025/2026 scores.
"""
from collections import defaultdict,deque
import numpy as np

def causal_ticker_shrink(events, history, base_probs, *, blend=0.3,
                         prior_strength=80.0, min_ticker_count=25,
                         global_window=2500, ticker_window=180):
    if not 0<=blend<=1 or prior_strength<=0:
        raise ValueError("Invalid probability shrinkage parameters")
    base=np.asarray(base_probs,dtype=float)
    if base.shape!=(len(events),3):
        raise ValueError("Expected exactly three class probabilities per event")
    output=base.copy()
    observations=sorted([
        (r["label_available_date"],r["ticker"],int(y))
        for r,_,y in list(history)+list(events)
    ])
    order=sorted(range(len(events)),key=lambda i:events[i][0]["signal_date"])
    global_counts=np.zeros(3,dtype=float)
    global_fifo=deque()
    ticker_counts=defaultdict(lambda:np.zeros(3,dtype=float))
    ticker_fifos=defaultdict(deque)
    j=0
    for i in order:
        current=events[i][0]["signal_date"]
        while j<len(observations) and observations[j][0]<current:
            _,ticker,label=observations[j]
            global_counts[label]+=1
            global_fifo.append(label)
            if len(global_fifo)>global_window:
                global_counts[global_fifo.popleft()]-=1
            ticker_counts[ticker][label]+=1
            ticker_fifos[ticker].append(label)
            if len(ticker_fifos[ticker])>ticker_window:
                ticker_counts[ticker][ticker_fifos[ticker].popleft()]-=1
            j+=1
        ticker=events[i][0]["ticker"]
        n=ticker_counts[ticker].sum()
        if n<min_ticker_count:
            continue
        pooled=(global_counts+1)/(global_counts.sum()+3)
        posterior=(ticker_counts[ticker]+prior_strength*pooled)/(n+prior_strength)
        output[i]=(1-blend)*base[i]+blend*posterior
    return output
