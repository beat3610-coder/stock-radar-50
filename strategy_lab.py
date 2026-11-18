"""Chronological research lab for entry strategies.

HISTORICAL DATA USED HERE HAS BEEN SEEN THROUGH 2026-11; the 2026-04+
holdout is a reproducible regression diagnostic, not a pristine independent
holdout. Future unseen paper records are mandatory before live certification.

No optimization over holdout. Market-specific models are chosen ONLY from
development trades, checked on a fixed validation window, then reported on
the diagnostic holdout WITHOUT tuning to the holdout.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from engine import compute
from providers import get_prices
from universe import get_universe
from audit_300 import simulate_exit, metrics
from portfolio_audit import replay
from trade_execution import entry_quality
from empirical_edge import walkforward_filter, walkforward_regime_filter
from predictive_model import walkforward_model_filter, walkforward_model_policies, POLICIES
from three_round_trade_model import three_round_audit
from profit_selector51 import discover as discover_51
from regime_selector60 import report as regime60_report
from feature_selector60 import audit as feature60_audit
from net_edge_model import audit as net_edge_audit
from trend_guard_model import audit as trend_guard_audit
from market_breadth_gate import audit as market_breadth_audit
from broad_candidates import broad_candidates_from_computed
from online_edge import score_stream, POLICIES as ONLINE_POLICIES

TRAIN_END="2025-11-01"
VALID_END="2026-04-01"
HOLDOUT_END="2026-11-01"
COST_PCT=0.35
MIN_TRAIN_TRADES=20
MIN_VALID_TRADES=20
LOOKBACK=220
STEP=2
HOLD=10
PARAMS={
 "trend_breakout":{"rr":2.0,"stop_atr":1.8},
 "trend_pullback":{"rr":1.7,"stop_atr":1.7},
 "momentum_reentry":{"rr":1.8,"stop_atr":1.8},
 # Development-only exit hypotheses: the signal is unchanged; exits become
 # moderately closer while the stop keeps room for ordinary ATR volatility.
 "trend_pullback_balanced":{"rr":1.55,"stop_atr":1.9},
 "momentum_reentry_balanced":{"rr":1.55,"stop_atr":1.9},
}
def _finite(*values):
    return all(v is not None and np.isfinite(v) for v in values)

def qualifying(row,prev,market,variant,prior_high):
    # Unknown strategy and incomplete derived trend features are *non-signals*,
    # never exceptions that can interrupt a 300-stock run.
    if variant not in PARAMS or "MA200_UP20" not in row.index:
        return False
    p=float(row.Close)
    atr=float(row.ATR)
    if not _finite(row.MA20,row.MA60,row.MA200,row.RSI,row.MACDH,row.ADX,row.PDI,row.MDI,atr,row.VOL20):
        return False
    if atr<=0 or row.VOL20<=0:return False
    volatility=atr/p
    if volatility<0.006 or volatility>0.075:return False
    # Reduce regime-chasing and late entries: the long average must be rising
    # over 20 sessions, not merely below today's price. Filter weak directional
    # momentum and extended entries without looking past the signal candle.
    if not _finite(row.MA200,row.MA60,row.MA20,row.PDI,row.MDI,row.ADX):
        return False
    if not (bool(row.MA200_UP20) and p>row.MA200 and row.MA60>row.MA200 and row.PDI>row.MDI and row.ADX>=18):
        return False
    if p/float(row.MA20)>1.08:
        return False
    # Match identical signal logic for the original and alternative exits.
    signal_variant=variant.removesuffix("_balanced")
    # Determine buying conditions strictly from signal-day and prior bars.
    if signal_variant=="trend_breakout":
        return (p>row.MA20>row.MA60>row.MA200 and p>prior_high
            and 50<=row.RSI<=76 and row.MACDH>0
            and row.Volume/row.VOL20>=1.2)
    if signal_variant=="trend_pullback":
        return (p>row.MA60>row.MA200 and p>row.MA20
            and 43<=row.RSI<=68 and
            float(prev.Close)<=float(prev.MA20)*1.018
            and p>float(prev.Close) and row.MACDH>float(prev.MACDH)
            and row.PDI>row.MDI)
    if signal_variant=="momentum_reentry":
        return (p>row.MA60>row.MA200 and
            45<=row.RSI<=72 and row.MACDH>0
            and float(prev.MACDH)<=0 and p>row.MA20 and row.PDI>row.MDI)
    return False

def assess_executable_entry(open_price, stop, take, cost_pct=COST_PCT):
    """Compatibility wrapper around the shared execution gate."""
    return entry_quality(open_price,stop,take,cost_pct=cost_pct)

def candidates_from_frame(frame,ticker,market):
    x=compute(frame)
    result={key:[] for key in PARAMS}
    for i in range(LOOKBACK,len(x)-HOLD,STEP):
        date=str(pd.Timestamp(x.index[i]).date())
        if date>=HOLDOUT_END:break
        r=x.iloc[i];prev=x.iloc[i-1]
        prior_high=float(x.High.iloc[i-20:i].max())
        close=float(r.Close)
        op=float(x.Open.iloc[i+1])
        atr=float(r.ATR)
        if not(_finite(close,op,atr) and op>0 and close>0):continue
        if abs(op/close-1)>0.025:continue
        for name,param in PARAMS.items():
            if not qualifying(r,prev,market,name,prior_high):continue
            # Stop and target are fixed at signal close. No future prices.
            stop=max(close*(1-0.095),close-param["stop_atr"]*atr)
            stop=min(stop,close*0.98)
            risk=op-stop
            # Entry is executed at next day's OPEN, not the signal close.
            # Re-check actual executable net reward/risk after gap and costs:
            # the old rule could approve a 2.5% gap-up despite <1R net payout.
            take=close+param["rr"]*(close-stop)
            execution_check=entry_quality(op,stop,take,reference=close)
            if not execution_check["accepted"]:
                continue
            outcome=simulate_exit(x,i,op,stop,take,max_days=HOLD,cost_pct=COST_PCT)
            if not outcome:continue
            trade={"market":market,"ticker":ticker,"strategy":name,
                "signal_date":date,"entry_date":str(pd.Timestamp(x.index[i+1]).date()),
                "exit_date":str(pd.Timestamp(x.index[outcome["exit_index"]]).date()),
                "entry":op,"stop":stop,"take":take,
                "entry_net_rr":execution_check["net_rr"],
                "entry_net_upside_pct":execution_check["net_upside_pct"],
                "score":float(r.RSI),
                "signal_atr_pct":round(100*atr/close,3),
                "signal_ma20_ma60_pct":round(100*(float(r.MA20)/float(r.MA60)-1),3),
                "signal_rsi":float(r.RSI),
                "signal_adx":float(r.ADX),
                "signal_volume_ratio":float(r.Volume/r.VOL20),
                "signal_5d_return_pct":100*(close/float(x.Close.iloc[i-5])-1),
                "entry_gap_pct":100*(op/close-1),
                **outcome}
            result[name].append(trade)
    # prevent same ticker stacking into its own previous trade
    for name in result:
        chosen=[];last_exit=""
        for t in sorted(result[name],key=lambda z:z["entry_date"]):
            if t["entry_date"]<=last_exit:continue
            chosen.append(t);last_exit=t["exit_date"]
        result[name]=chosen
    return x,result

def _interval(trades,part):
    if part=="train":return [t for t in trades if t["entry_date"]<TRAIN_END and t["exit_date"]<TRAIN_END]
    if part=="validate":return [t for t in trades if TRAIN_END<=t["entry_date"]<VALID_END and t["exit_date"]<VALID_END]
    return [t for t in trades if VALID_END<=t["entry_date"]<HOLDOUT_END and t["exit_date"]<HOLDOUT_END]

def train_choice(choices):
    """Development only. Don't reward single lucky huge wins."""
    scores={}
    for name,trades in choices.items():
        m=metrics(_interval(trades,"train"))
        n=m["trades"]
        pf=m["profit_factor"]
        avg=m["mean_net_pct"]
        # Baseline is 'no trade', not guaranteed positive expectancy.
        if n<MIN_TRAIN_TRADES or pf is None or avg is None or pf<1.15 or avg<=0:
            scores[name]=None
        else:
            # Penalize fragile small samples; cap one-off large returns.
            scores[name]=min(avg,3.0)*math.sqrt(n/(n+50))*min(pf,3)/3
    chosen=max((k for k,v in scores.items() if v is not None),
               key=lambda k:scores[k],default=None)
    return chosen,scores

def run(per_market=30):
    u=get_universe(include_us=False)
    res={"created_utc":datetime.now(timezone.utc).isoformat(),
        "type":"research_lab_v8_online_streaming",
        "method":"train-only selection, fixed validation screening, untouched-in-this-run diagnostic holdout",
        "train_end":TRAIN_END,"validation_end":VALID_END,"holdout_end":HOLDOUT_END,
        "research_target_win_rate_pct":70,
        "selection_requires":{"train_trades":MIN_TRAIN_TRADES,"validation_trades":MIN_VALID_TRADES,
                              "train_pf":1.15,"validation_pf":1.30,"validation_min_win_rate_pct":50,"validation_mean_net_pct":0.25},
        "caveats":["Data from 2026 has been examined during earlier development; not truly independent",
                   "Current constituents create survivorship bias",
                   "Historical tickers' listing/delisting state not reconstructed",
                   "OHLC stop-first, next-day open, costs assumed 0.35%",
                   "Market-level capital has no live currency conversion",
                   "Strategy selection done on training only; validation used only as rejection gate",
                   "2026 historical holdout was already examined; revised filters are development hypotheses, not new independent evidence",
                   "Never approve live trading without future independently observed paper results"],
        "markets":{}}
    # 30 KOSPI + 30 KOSDAQ stocks, two TIGER ETFs, and SOXL only.
    for market in ("KR","KQ","US"):
        target_count=1 if market=="US" else per_market+(2 if market=="KR" else 0)
        items=(u["US"][-1:] if market=="US" else
               u[market][:per_market]+(u["KR"][-2:] if market=="KR" else []))
        if len(items)<target_count:
            res.setdefault("input_failures",{})[market]={
                "requested":target_count,"available":len(items),
                "error":u.get(market+"_error"),
                "source":u.get(market+"_source"),
                "status":"INSUFFICIENT_UNIVERSE_NOT_A_BACKTEST"}
            print("LAB_INPUT_FAILURE "+market+" "+json.dumps(res["input_failures"][market],ensure_ascii=False),flush=True)
        frames={}
        variants={k:[] for k in PARAMS}
        broad_training_pool=[]
        failures=[]
        for idx,(ticker,_,_,_) in enumerate(items,1):
            try:
                raw,_=get_prices(ticker,market)
                x,trades=candidates_from_frame(raw,ticker,market)
                frames[ticker]=raw
                broad_training_pool.extend(broad_candidates_from_computed(x,ticker,market,
                     end_date=HOLDOUT_END,cost_pct=COST_PCT))
                for k in variants:variants[k].extend(trades[k])
            except Exception as exc:
                failures.append({"ticker":ticker,"error":str(exc)[:160]})
            if idx%5==0:
                print(f"LAB_PROGRESS {market} {idx}/{len(items)} loaded={len(frames)}",flush=True)
        if len(frames)<target_count:
            res.setdefault("price_failures",{})[market]={
                "requested":target_count,"loaded":len(frames),
                "errors":failures[:5],"status":"INCOMPLETE_PRICE_DATA_NO_PERFORMANCE_CLAIM"}
            print("LAB_PRICE_FAILURE "+market+" "+json.dumps(res["price_failures"][market],ensure_ascii=False),flush=True)
        # Compare the original three technical variants with prospective,
        # cost-adjusted probability/expectancy filters. Every calibration uses
        # ONLY prior trades that were fully exited before its signal date.
        edge_diagnostics={}
        for original in tuple(PARAMS):
            filtered,report=walkforward_filter(variants[original])
            variants[original+"_prob_edge"]=filtered
            high_confidence,high_confidence_report=walkforward_regime_filter(variants[original])
            variants[original+"_regime_edge"]=high_confidence
            ml_filtered,ml_report=walkforward_model_filter(variants[original])
            variants[original+"_ml_edge"]=ml_filtered
            edge_diagnostics[original]={**report,"regime_filter":high_confidence_report,
                                        "predictive_model":ml_report}
        # Three distinct prequential probability challengers; log diagnostics,
        # never choose a live model using already-inspected historical outcomes.
        three_round_report=three_round_audit(broad_training_pool,market)
        target51_report=discover_51(broad_training_pool,market)
        target60_report=regime60_report(broad_training_pool,market)
        feature60_report=feature60_audit(broad_training_pool,market)
        net_edge_report=net_edge_audit(broad_training_pool,market)
        trend_guard_report=trend_guard_audit(broad_training_pool,market)
        market_breadth_report=market_breadth_audit(broad_training_pool,frames,market)
        print('LAB_MARKET_BREADTH '+market+' '+json.dumps(market_breadth_report,ensure_ascii=False),flush=True)
        print('LAB_TREND_GUARD '+market+' '+json.dumps(trend_guard_report,ensure_ascii=False),flush=True)
        print('LAB_NET_EDGE '+market+' '+json.dumps(net_edge_report,ensure_ascii=False),flush=True)
        print('LAB_FEATURE60 '+market+' '+json.dumps(feature60_report,ensure_ascii=False),flush=True)
        print('LAB_TARGET60 '+market+' '+json.dumps(target60_report,ensure_ascii=False),flush=True)
        print('LAB_TARGET51 '+market+' '+json.dumps(target51_report,ensure_ascii=False),flush=True)
        print('LAB_THREE_ROUNDS '+market+' '+json.dumps(three_round_report,ensure_ascii=False),flush=True)
        broad_policy_trades,broad_model_report=walkforward_model_policies(broad_training_pool)
        for policy_name,policy_trades in broad_policy_trades.items():
            variants["broad_market_ml_"+policy_name]=policy_trades
        edge_diagnostics["broad_market_ml"]={"predictive_model":broad_model_report,
            "predeclared_policies":POLICIES,
            "base_population":len(broad_training_pool),
            "population_train":metrics(_interval(broad_training_pool,"train")),
            "population_validation":metrics(_interval(broad_training_pool,"validate")),
            "population_holdout_diagnostic":metrics(_interval(broad_training_pool,"test")),
            "policy_train":{n:metrics(_interval(v,"train")) for n,v in broad_policy_trades.items()},
            "policy_validation":{n:metrics(_interval(v,"validate")) for n,v in broad_policy_trades.items()}}
        # Independent O(1)-update, bounded-state regime model. All trading
        # outcomes are available ONLY after their historical exit date.
        online_trades,online_report=score_stream(broad_training_pool,market)
        for online_name,online_candidates in online_trades.items():
            variants[online_name]=online_candidates
        edge_diagnostics["online_edge"]={"algorithm":"disk-backed market-specific sequential Beta shrinkage",
            "filters":ONLINE_POLICIES,"diagnostics":online_report,
            "policy_train":{n:metrics(_interval(v,"train")) for n,v in online_trades.items()},
            "policy_validation":{n:metrics(_interval(v,"validate")) for n,v in online_trades.items()}}
        print(f"LAB_ONLINE {market} population={online_report['population']} scored={online_report['scored']} "
              f"accepted={online_report['by_policy']}",flush=True)
        leader,scores=train_choice(variants)
        train_metrics={k:metrics(_interval(v,"train")) for k,v in variants.items()}
        validation_metrics={k:metrics(_interval(v,"validate")) for k,v in variants.items()}
        test_metrics={k:metrics(_interval(v,"test")) for k,v in variants.items()}
        accepted=False
        if leader:
            vm=validation_metrics[leader]
            accepted=(vm["trades"]>=MIN_VALID_TRADES and
                      vm["win_rate_pct"] is not None and vm["win_rate_pct"]>=50 and
                      vm["mean_net_pct"] is not None and vm["mean_net_pct"]>0.25 and
                      vm["profit_factor"] is not None and vm["profit_factor"]>=1.30)
        # The candidate is always diagnosed in the known retrospective period,
        # even if the validation rejection gate failed. That prevents a "0%
        # drawdown" no-trade record from hiding a losing strategy. Rejection
        # still blocks live approval.
        selected=leader if accepted else None
        selected_trades=_interval(variants[selected],"test") if selected else []
        # Rejected strategy is still scored for transparent diagnostics but
        # NEVER replayed as if it were authorized portfolio trading.
        rejected_diagnostic=_interval(variants[leader],"test") if leader and not accepted else []
        holdout_measures=metrics(selected_trades)
        # Diagnose actual loss mechanisms instead of reporting only win rate.
        loss_trades=[t for t in selected_trades if t.get("net_pct",0)<=0]
        exit_counts={}
        loss_exit_counts={}
        for trade in selected_trades:
            label=str(trade.get("exit_reason","unknown"))
            exit_counts[label]=exit_counts.get(label,0)+1
            if trade.get("net_pct",0)<=0:
                loss_exit_counts[label]=loss_exit_counts.get(label,0)+1
        train_win=(train_metrics.get(selected) or {}).get("win_rate_pct") if selected else None
        val_win=(validation_metrics.get(selected) or {}).get("win_rate_pct") if selected else None
        hold_win=holdout_measures.get("win_rate_pct")
        win_rate_drift=(round(hold_win-train_win,2) if
                        train_win is not None and hold_win is not None else None)
        diagnosis={
            "completed_trades":len(selected_trades),
            "losing_trades":len(loss_trades),
            "exit_reason_counts":exit_counts,
            "loss_exit_reason_counts":loss_exit_counts,
            "training_win_rate_pct":train_win,
            "validation_win_rate_pct":val_win,
            "diagnostic_holdout_win_rate_pct":hold_win,
            "train_to_holdout_win_rate_change_pp":win_rate_drift,
            "not_truly_independent":True,
            "warning":"Already-inspected dates: diagnostic only; changes require new future evaluation",
        }
        base=replay(selected_trades,frames,start=VALID_END,
                    initial_cash=10_000_000 if market in ("KR","KQ") else 100_000)
        stress=replay(selected_trades,frames,start=VALID_END,
                    initial_cash=10_000_000 if market in ("KR","KQ") else 100_000,
                    cost=0.007)
        # Fail-closed deployment screen. Never retune parameters from these
        # previously inspected test-period results.
        research_eligible=bool(
            selected and len(frames)==target_count
            and holdout_measures["trades"]>=30
            and holdout_measures["win_rate_pct"] is not None
            and holdout_measures["win_rate_pct"]>=55
            and holdout_measures["mean_net_pct"]>0
            and holdout_measures["profit_factor"] is not None
            and holdout_measures["profit_factor"]>=1.3
            and base.get("total_return_pct",0)>0
            and stress.get("total_return_pct",0)>0
            and base.get("max_drawdown_pct",100)<=12)
        def compact(r):return {k:v for k,v in r.items() if k not in ("daily_equity","executions")}
        res["markets"][market]={
            "requested":target_count,"loaded":len(frames),"errors":failures,
            "training_selected":leader,"training_scores":scores,
            "validation_accepted":accepted,
            "edge_calibration":edge_diagnostics,
            "three_round_probability":three_round_report,
            "target51_research":target51_report,
            "target60_regime":target60_report,
            "target60_features":feature60_report,
            "net_edge_research":net_edge_report,
            "trend_guard_research":trend_guard_report,
            "market_breadth_research":market_breadth_report,
            "evaluation_policy":"diagnostic_heldout_even_if_validation_rejected; never live approve",
            "selected_for_diagnostic":selected,
            "rejected_candidate":leader if not accepted else None,
            "rejected_candidate_diagnostic":metrics(rejected_diagnostic),
            "research_eligible":research_eligible,
            "live_approved":False,
            "train":train_metrics,"validation":validation_metrics,
            "holdout_all_variants_diagnostic_only":test_metrics,
            "holdout_selected_trades":len(selected_trades),
            "holdout_selected_metrics":metrics(selected_trades),
            "loss_diagnosis":diagnosis,
            "portfolio":compact(base),"stress_portfolio":compact(stress),
            "status":"NOT_LIVE_APPROVED"
        }
        print(f"LAB_MARKET {market} "+json.dumps({
            "selected":selected,"train":train_metrics.get(leader),
            "validation":validation_metrics.get(leader),
            "holdout":metrics(selected_trades),"account_return":base.get("total_return_pct"),
            "stress_return":stress.get("total_return_pct"),
            "loaded":len(frames)},default=str),flush=True)
    res["release_status"]="NOT_LIVE_APPROVED"
    res["data_integrity_status"]=("FAIL" if res.get("input_failures") or res.get("price_failures") else "PASS")
    return res

if __name__=="__main__":
    report=run()
    Path("data").mkdir(exist_ok=True)
    Path("data/strategy-lab-latest.json").write_text(
         json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print("LAB_RESULT "+json.dumps(report,ensure_ascii=False,default=str),flush=True)
