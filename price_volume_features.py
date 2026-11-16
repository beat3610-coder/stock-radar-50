"""Deterministic, observation-date-safe price/volume interaction features.

The candidate adds cross terms using indicators already present at the
signal close. It changes neither labels nor future access; it is compared
against the exact same stock/date cohort and fixed ExtraTrees setup.
"""
import numpy as np

def interact_rows(rows):
    out=[]
    for row,vec,label in rows:
        x=np.asarray(vec,dtype=float)
        # The existing model's raw features occupy positions 0..14.
        # All terms use only contemporaneous signal-close observations.
        rsi=x[0]
        atr=x[2]
        spread=x[3]
        volume=x[4]
        ret5=x[5]
        macdh=x[6]
        price_ma200=x[7]
        cmf=x[11]
        bb_width=x[14]
        # Clipping caps unusual volume spikes and keeps units comparable.
        vol=np.clip(volume,0,5)
        trend=np.clip(spread,-20,20)
        ret=np.clip(ret5,-30,30)
        atr_safe=max(abs(atr),0.5)
        new=[
            np.clip(ret/atr_safe,-12,12),
            np.clip(trend/atr_safe,-12,12),
            np.clip(ret*vol/5,-20,20),
            np.clip(macdh*vol/5,-20,20),
            np.clip(cmf*vol,-5,5),
            np.clip((rsi-50)*trend/100,-12,12),
            np.clip(price_ma200*trend/100,-20,20),
            np.clip(bb_width/atr_safe,-5,25),
        ]
        out.append((row,list(vec)+[float(v) for v in new],label))
    return out
