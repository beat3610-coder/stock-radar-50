"""Fixed, point-in-time macro and cross-sectional feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

INDEX_FOR_COHORT={"KOSPI":"KOSPI","KOSDAQ":"KOSDAQ","TIGER_ETF":"KOSPI","SOXL":"SOX","US":"SOX"}
US_COHORTS={"SOXL","US"}
REGIME_RULE={
    "risk_off":"(cohort index below MA200 and realized volatility ratio > 1.2) OR VIX 250-observation expanding percentile > 0.80 OR USD/KRW 20-day change > +3%.",
    "risk_on":"cohort index above MA200 and volatility ratio <= 1.2 and VIX percentile <= 0.80 and USD/KRW 20-day change <= +3%; otherwise NEUTRAL when inputs exist.",
    "risk_unknown":"NEUTRAL/RISK_ON cannot be asserted when required market inputs are unavailable; label UNKNOWN and log source failures.",
}

def _clean_series(frame):
    if frame is None or frame.empty or "Close" not in frame:return pd.Series(dtype=float)
    x=pd.to_numeric(frame.Close,errors="coerce").dropna().copy()
    idx=pd.DatetimeIndex(pd.to_datetime(x.index))
    if idx.tz is not None:idx=idx.tz_localize(None)
    x.index=idx.normalize()
    return x[~x.index.duplicated(keep="last")].sort_index()

def _indicators(frame):
    close=_clean_series(frame)
    if close.empty:return {}
    ret=close.pct_change()
    vol20=ret.rolling(20,min_periods=20).std(ddof=0)*np.sqrt(252)*100
    vol60=ret.rolling(60,min_periods=60).std(ddof=0)*np.sqrt(252)*100
    ind={
        "ma20":100*(close/close.rolling(20,min_periods=20).mean()-1),
        "ma60":100*(close/close.rolling(60,min_periods=60).mean()-1),
        "ma200":100*(close/close.rolling(200,min_periods=200).mean()-1),
        "ret5":100*close.pct_change(5),
        "ret20":100*close.pct_change(20),
        "ret60":100*close.pct_change(60),
        "ret120":100*close.pct_change(120),
        "vol20":vol20,
        "vol_ratio":vol20/vol60.replace(0,np.nan),
        "dd120":100*(close/close.rolling(120,min_periods=120).max()-1),
    }
    return ind

def _macro_indicators(macro_frames):
    out={key:_indicators(frame) for key,frame in macro_frames.items()}
    vix=_clean_series(macro_frames.get("VIX"))
    out["VIX"]={"level":vix,
        "change5":100*vix.pct_change(5) if len(vix) else pd.Series(dtype=float),
        "expanding_pct":vix.rolling(250,min_periods=250).rank(pct=True) if len(vix) else pd.Series(dtype=float)}
    fx=_clean_series(macro_frames.get("USDKRW"))
    out["USDKRW"]={"change5":100*fx.pct_change(5) if len(fx) else pd.Series(dtype=float),
        "change20":100*fx.pct_change(20) if len(fx) else pd.Series(dtype=float),
        "z60":(fx-fx.rolling(60,min_periods=60).mean())/fx.rolling(60,min_periods=60).std(ddof=0).replace(0,np.nan) if len(fx) else pd.Series(dtype=float)}
    return out

def _at(series,day,strict_prior):
    if series is None or len(series)==0:return np.nan
    ix=pd.DatetimeIndex(series.index)
    stamp=pd.Timestamp(day).normalize()
    pos=int(ix.searchsorted(stamp,side="left" if strict_prior else "right")-1)
    if pos<0:return np.nan
    try:return float(series.iloc[pos])
    except (TypeError,ValueError):return np.nan

def _macro_feature_row(day,cohort,macro_frames,indicators=None,strict_prior_override=None):
    index_key=INDEX_FOR_COHORT[cohort]
    # Korean signals are generated at the Korean close: same-day US closes are
    # later information and therefore excluded. Direction3 additionally uses
    # strict prior-session macro alignment for all cohorts, including US/SOXL.
    # Macro feeds and US indices are published on separate market clocks. For
    # every signal use only the last completed US session; Korean local index
    # features may use the signal-day Korean close. US cohort SOXL/SOX trend is
    # likewise shifted to the last completed session by strict_prior_index.
    strict_prior=True if strict_prior_override is None else bool(strict_prior_override)
    strict_prior_index=cohort in US_COHORTS
    indicators=indicators or _macro_indicators(macro_frames)
    index_ind=indicators.get(index_key,{})
    vals={}
    for key,label in (("ma20","market_index_ma20_gap_pct"),("ma60","market_index_ma60_gap_pct"),
                      ("ma200","market_index_ma200_gap_pct"),("ret5","market_index_return_5d_pct"),
                      ("ret20","market_index_return_20d_pct"),("ret60","market_index_return_60d_pct"),
                      ("vol20","market_index_volatility_20d_pct"),("vol_ratio","market_volatility_ratio"),
                      ("dd120","market_drawdown_120_pct")):
        vals[label]=_at(index_ind.get(key),day,strict_prior_index)
    vix=indicators.get("VIX",{})
    fx=indicators.get("USDKRW",{})
    vals.update({
        "vix_level":_at(vix.get("level"),day,strict_prior),
        "vix_change_5d_pct":_at(vix.get("change5"),day,strict_prior),
        "vix_expanding_percentile":_at(vix.get("expanding_pct"),day,strict_prior),
        "usdkrw_change_5d_pct":_at(fx.get("change5"),day,False),
        "usdkrw_change_20d_pct":_at(fx.get("change20"),day,False),
        "usdkrw_z60":_at(fx.get("z60"),day,False),
    })
    for key,macro_key in (("sp500_return_20d_pct","SP500"),("sox_return_20d_pct","SOX")):
        vals[key]=_at(indicators.get(macro_key,{}).get("ret20"),day,strict_prior)
    vals["_benchmark_return_120d_pct"]=_at(indicators.get(index_key,{}).get("ret120"),day,strict_prior_index)
    idx_ma=vals.get("market_index_ma200_gap_pct")
    vol_ratio=vals.get("market_volatility_ratio")
    vix_pct=vals.get("vix_expanding_percentile")
    fx20=vals.get("usdkrw_change_20d_pct")
    clauses=[]
    if np.isfinite(idx_ma) and np.isfinite(vol_ratio):
        clauses.append(idx_ma<0 and vol_ratio>1.2)
    if np.isfinite(vix_pct):clauses.append(vix_pct>0.80)
    if np.isfinite(fx20):clauses.append(fx20>3.0)
    enough=bool(clauses)
    if not enough:regime="UNKNOWN"
    elif any(clauses):regime="RISK_OFF"
    elif all(np.isfinite(v) for v in (idx_ma,vol_ratio,vix_pct,fx20)) and idx_ma>0 and vol_ratio<=1.2 and vix_pct<=0.80 and fx20<=3.0:
        regime="RISK_ON"
    else:regime="NEUTRAL"
    vals["market_risk_on"]=(1.0 if regime=="RISK_ON" else 0.0) if regime!="UNKNOWN" else np.nan
    vals["market_risk_off"]=(1.0 if regime=="RISK_OFF" else 0.0) if regime!="UNKNOWN" else np.nan
    vals["market_regime"]=regime
    return vals

def _price_features(frame):
    close=pd.to_numeric(frame.Close,errors="coerce")
    vol=pd.to_numeric(frame.get("Volume",pd.Series(0,index=frame.index)),errors="coerce").fillna(0)
    ret=close.pct_change()
    dollar=(close*vol).replace([np.inf,-np.inf],np.nan)
    high=pd.to_numeric(frame.get("High",close),errors="coerce")
    low=pd.to_numeric(frame.get("Low",close),errors="coerce")
    open_=pd.to_numeric(frame.get("Open",close),errors="coerce")
    price_volume=ret.rolling(20,min_periods=15).corr(vol.pct_change())
    intraday=((close-low)/(high-low).replace(0,np.nan)).rolling(20,min_periods=15).mean()
    return pd.DataFrame({
        "ret20":100*close.pct_change(20),"ret60":100*close.pct_change(60),"ret120":100*close.pct_change(120),
        "vol20":ret.rolling(20,min_periods=15).std(ddof=0)*np.sqrt(252)*100,
        "turnover20":dollar.rolling(20,min_periods=15).mean(),
        "high52_dist":100*(close/close.rolling(252,min_periods=60).max()-1),
        "pv_corr20":price_volume,"intraday_strength20":intraday,
        "gap_pct":100*(open_/close.shift(1)-1),
        "turnover_ratio20":dollar/dollar.rolling(20,min_periods=15).mean().replace(0,np.nan),
        "signed_volume_flow5":(np.sign(ret)*dollar).rolling(5,min_periods=4).sum()/dollar.rolling(20,min_periods=15).mean().replace(0,np.nan)/5,
    },index=frame.index)

def _flow_features(price_frame,flow_frame):
    if flow_frame is None or flow_frame.empty:
        out=pd.DataFrame(index=price_frame.index)
        for label in ("foreign","institution"):
            out[f"{label}_flow_5d_to_turnover"]=np.nan
            out[f"{label}_flow_20d_to_turnover"]=np.nan
            out[f"{label}_buy_streak"]=np.nan
        out["flow_direction_agreement"]=np.nan
        out["flow_data_available"]=0.0
        return out
    flow=flow_frame.copy()
    flow_index=pd.DatetimeIndex(pd.to_datetime(flow.index))
    if flow_index.tz is not None:flow_index=flow_index.tz_localize(None)
    flow.index=flow_index.normalize()
    flow=flow[~flow.index.duplicated(keep="last")].sort_index()
    for key in ("foreign_net_shares","institution_net_shares"):
        if key not in flow:flow[key]=np.nan
    close=pd.to_numeric(price_frame.Close,errors="coerce")
    turnover=(close*pd.to_numeric(price_frame.get("Volume",0),errors="coerce").fillna(0))
    avg_turnover=turnover.rolling(20,min_periods=15).mean().replace(0,np.nan)
    # Exact-date join only: signal-day close flows may feed the next-open
    # decision, but there is no forward/backward fill from a later session.
    data=flow.reindex(price_frame.index)
    observed=data[["foreign_net_shares","institution_net_shares"]].notna().any(axis=1)
    out={}
    for label,key in (("foreign","foreign_net_shares"),("institution","institution_net_shares")):
        shares=pd.to_numeric(data.get(key,pd.Series(np.nan,index=price_frame.index)),errors="coerce")
        value=shares*close
        out[f"{label}_flow_5d_to_turnover"]=value.rolling(5,min_periods=5).sum()/avg_turnover
        out[f"{label}_flow_20d_to_turnover"]=value.rolling(20,min_periods=20).sum()/avg_turnover
        streak=[];n=0
        for share in shares.to_numpy():
            if not np.isfinite(share):
                n=0;streak.append(np.nan)
            else:
                n=n+1 if share>0 else 0
                streak.append(float(n))
        out[f"{label}_buy_streak"]=pd.Series(streak,index=price_frame.index,dtype=float)
    inst20=out["institution_flow_20d_to_turnover"]
    foreign20=out["foreign_flow_20d_to_turnover"]
    both=inst20.notna()&foreign20.notna()
    agreement=np.sign(inst20)*np.sign(foreign20)
    out["flow_direction_agreement"]=agreement.where(both,np.nan)
    out["flow_data_available"]=observed.rolling(20,min_periods=1).mean().fillna(0.0)
    return pd.DataFrame(out,index=price_frame.index)

def build_market_context(frames,macro_frames,cohort,flow_frames=None):
    """Return per-ticker/date features, regime dates, and rank sample diagnostics."""
    flow_frames=flow_frames or {}
    price_data={ticker:_price_features(frame) for ticker,frame in frames.items()}
    flow_data={ticker:_flow_features(frame,flow_frames.get(ticker)) for ticker,frame in frames.items()}
    macro_indicators=_macro_indicators(macro_frames)
    for ticker,frame in frames.items():
        flows=flow_data[ticker]
        if not flows.index.equals(frame.index):
            flows=flows.reindex(frame.index)
        if "flow_data_available" not in flows:
            flows["flow_data_available"]=0.0
        flow_data[ticker]=flows
    union=sorted(set(pd.Timestamp(d).normalize() for frame in frames.values() for d in frame.index))
    market_rows={str(day.date()):_macro_feature_row(day,cohort,macro_frames,macro_indicators) for day in union}
    # Same-date cross-sectional ranks use only constituents with an observed value
    # on that date; later listings and missing fields are not backfilled.
    rank_fields={"xs_return20_rank":"ret20","xs_return60_rank":"ret60","xs_return120_rank":"ret120",
                 "xs_volatility20_rank":"vol20","xs_turnover20_rank":"turnover20"}
    counts={k:[] for k in rank_fields}
    features={}
    for ticker,frame in frames.items():
        features[ticker]={}
    for day in union:
        stamp=str(day.date())
        cross={};counts_at={}
        for rank_name,metric in rank_fields.items():
            vals={ticker:float(pf.at[day,metric]) for ticker,pf in price_data.items()
                  if day in pf.index and pd.notna(pf.at[day,metric]) and np.isfinite(float(pf.at[day,metric]))}
            counts_at[rank_name]=len(vals);counts[rank_name].append(len(vals))
            if vals:
                ranks=pd.Series(vals,dtype=float).rank(method="average",pct=True)
                cross[rank_name]={ticker:float(value) for ticker,value in ranks.items()}
            else:cross[rank_name]={}
        macro=market_rows[stamp]
        for ticker,frame in frames.items():
            if day not in frame.index:continue
            pf=price_data[ticker];flow=flow_data[ticker]
            flow_values={k:float(flow.at[day,k]) if day in flow.index and pd.notna(flow.at[day,k]) else np.nan for k in flow.columns}
            row={k:float(v) if np.isscalar(v) and not isinstance(v,str) and pd.notna(v) else np.nan for k,v in macro.items() if k not in ("market_regime","_benchmark_return_120d_pct")}
            for rank_name,metric in rank_fields.items():
                row[rank_name]=cross[rank_name].get(ticker,np.nan)
                row[rank_name+"_sample_n"]=counts_at[rank_name]
            index_return20=macro.get("market_index_return_20d_pct",np.nan)
            row["stock_excess_return_20d_pct"]=float(pf.at[day,"ret20"]-index_return20) if day in pf.index and np.isfinite(index_return20) and pd.notna(pf.at[day,"ret20"]) else np.nan
            index_return60=macro.get("market_index_return_60d_pct",np.nan)
            row["stock_excess_return_60d_pct"]=float(pf.at[day,"ret60"]-index_return60) if day in pf.index and np.isfinite(index_return60) and pd.notna(pf.at[day,"ret60"]) else np.nan
            # Benchmark 120-day return is computed with the same as-of rule.
            index_return120=macro.get("_benchmark_return_120d_pct",np.nan)
            row["stock_excess_return_120d_pct"]=float(pf.at[day,"ret120"]-index_return120) if day in pf.index and np.isfinite(index_return120) and pd.notna(pf.at[day,"ret120"]) else np.nan
            # Keep producer names identical to the fixed model schema. These
            # aliases were previously left under abbreviated names, silently
            # turning three inputs into center-valued constants downstream.
            feature_names={"high52_dist":"high52_distance_pct",
                           "pv_corr20":"price_volume_corr20",
                           "gap_pct":"signal_gap_pct",
                           "intraday_strength20":"intraday_strength20",
                           "turnover_ratio20":"turnover_ratio20",
                           "signed_volume_flow5":"signed_volume_flow5"}
            for source_name,model_name in feature_names.items():
                row[model_name]=float(pf.at[day,source_name]) if day in pf.index and pd.notna(pf.at[day,source_name]) else np.nan
            row.update(flow_values);row["market_regime"]=macro["market_regime"]
            features[ticker][stamp]=row
    rank_report={name:{"mean_available_n":round(float(np.mean(values)),2) if values else 0,
                       "min_available_n":int(min(values)) if values else 0,
                       "max_available_n":int(max(values)) if values else 0,
                       "dates_with_any_rankable_data":sum(v>0 for v in values)}
                 for name,values in counts.items()}
    return {"features":features,"regime_by_date":{d:r["market_regime"] for d,r in market_rows.items()},
            "market_rows":market_rows,"cross_sectional_rank_counts":rank_report,
            "candidate_date_count":len(union)}
