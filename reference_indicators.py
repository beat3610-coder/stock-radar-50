"""Permissive-reference technical indicators, independently implemented.

No third-party source code is copied here.  The public mathematical definitions
are reimplemented so STOCK RADAR can compare its legacy features with common
industry definitions.  Reference projects and licences are documented in
OPEN_SOURCE_REFERENCES.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_average(values: pd.Series, period: int = 14) -> pd.Series:
    """SMA-seeded Wilder recursive average (RMA), preserving the index."""
    data=pd.to_numeric(values,errors="coerce").to_numpy(dtype=float)
    out=np.full(len(data),np.nan,dtype=float)
    if period<1 or len(data)<period:
        return pd.Series(out,index=values.index,dtype=float)
    for end in range(period-1,len(data)):
        window=data[end-period+1:end+1]
        if np.isfinite(window).all():
            out[end]=float(window.mean())
            for i in range(end+1,len(data)):
                if not np.isfinite(data[i]):
                    break
                out[i]=(out[i-1]*(period-1)+data[i])/period
            break
    return pd.Series(out,index=values.index,dtype=float)


def add_reference_indicators(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add non-lookahead Wilder, flow, breakout and volatility features."""
    x=frame.copy()
    high=x.High.astype(float)
    low=x.Low.astype(float)
    close=x.Close.astype(float)
    volume=x.Volume.astype(float)

    delta=close.diff()
    avg_gain=wilder_average(delta.clip(lower=0),period)
    avg_loss=wilder_average((-delta.clip(upper=0)),period)
    total=avg_gain+avg_loss
    x["RSI_WILDER"]=100*avg_gain/total.replace(0,np.nan)
    x.loc[(avg_loss==0)&(avg_gain>0),"RSI_WILDER"]=100.0
    x.loc[(avg_gain==0)&(avg_loss>0),"RSI_WILDER"]=0.0
    x.loc[total==0,"RSI_WILDER"]=50.0

    previous_close=close.shift(1)
    true_range=pd.concat([
        high-low,(high-previous_close).abs(),(low-previous_close).abs()
    ],axis=1).max(axis=1)
    true_range.iloc[0]=np.nan
    x["ATR_WILDER"]=wilder_average(true_range,period)

    up=high.diff()
    down=-low.diff()
    plus_dm=up.where((up>down)&(up>0),0.0)
    minus_dm=down.where((down>up)&(down>0),0.0)
    plus_dm.iloc[0]=np.nan
    minus_dm.iloc[0]=np.nan
    plus_avg=wilder_average(plus_dm,period)
    minus_avg=wilder_average(minus_dm,period)
    atr=x["ATR_WILDER"].replace(0,np.nan)
    x["PDI_WILDER"]=100*plus_avg/atr
    x["MDI_WILDER"]=100*minus_avg/atr
    denom=(x.PDI_WILDER+x.MDI_WILDER).replace(0,np.nan)
    dx=100*(x.PDI_WILDER-x.MDI_WILDER).abs()/denom
    x["ADX_WILDER"]=wilder_average(dx,period)

    spread=(high-low).replace(0,np.nan)
    money_multiplier=((close-low)-(high-close))/spread
    money_flow=(money_multiplier.fillna(0)*volume)
    volume_sum=volume.rolling(20,min_periods=20).sum().replace(0,np.nan)
    x["CMF20"]=money_flow.rolling(20,min_periods=20).sum()/volume_sum

    # Shift the channel one session: today's breakout never changes its own
    # historical threshold.
    prior_high=high.shift(1).rolling(20,min_periods=20).max()
    prior_low=low.shift(1).rolling(20,min_periods=20).min()
    channel=(prior_high-prior_low).replace(0,np.nan)
    x["DONCHIAN20_POS"]=(close-prior_low)/channel
    x["ROC20"]=100*(close/close.shift(20)-1)

    mid=close.rolling(20,min_periods=20).mean()
    width=4*close.rolling(20,min_periods=20).std(ddof=0)
    x["BBWIDTH20_PCT"]=100*width/mid.replace(0,np.nan)
    return x
