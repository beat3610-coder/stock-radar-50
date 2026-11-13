"""300-stock, no-lookahead, next-open execution audit. Not a profit guarantee."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from engine import compute, evaluate
from providers import get_prices
from universe import get_universe
from portfolio_audit import replay
from certification import assess

BUY = {"연구용 매수 후보·실거래 미승인", "연구용 후보·위험조건 미충족"}
HOLD_DAYS = 10
COST_PCT = 0.35  # assumed round trip including commission/slippage; taxes may differ
WARMUP = 270
STEP = 10
FIXED_HOLDOUT_START = '2026-04-01'  # fixed in advance of strategy comparisons

def simulate_exit(frame, signal_index, entry_price, stop, target, max_days=HOLD_DAYS,
                  cost_pct=COST_PCT):
    """Signal at t close; fill at t+1 open. Stops win ambiguous same-day bars."""
    if signal_index+1 >= len(frame):
        return None
    if not (np.isfinite(entry_price) and np.isfinite(stop) and np.isfinite(target)
            and 0 < stop < entry_price < target):
        return None
    end = min(signal_index + max_days, len(frame)-1)
    if end < signal_index + max_days:
        return None  # no truncated, artificially favorable trades
    exit_price = float(frame.Close.iloc[end])
    reason = "time"
    exit_index = end
    for j in range(signal_index+1, end+1):
        row = frame.iloc[j]
        op, lo, hi = float(row.Open), float(row.Low), float(row.High)
        if op <= stop:
            exit_price, reason, exit_index = op, "stop_gap", j
            break
        if op >= target:
            exit_price, reason, exit_index = op, "take_gap", j
            break
        if lo <= stop:  # conservative when stop and target both hit same day
            exit_price, reason, exit_index = stop, "stop", j
            break
        if hi >= target:
            exit_price, reason, exit_index = target, "take", j
            break
    return {"exit_index":exit_index, "exit_price":exit_price,
            "exit_reason":reason, "net_pct":100*(exit_price/entry_price-1)-cost_pct}

def audit_symbol(frame, market, ticker):
    x = compute(frame)
    rows = []
    # A trade requires a fully observed follow-up period; signals are spaced,
    # and we skip decisions while an earlier trade remains open.
    blocked_through = -1
    for t in range(WARMUP, len(x)-HOLD_DAYS, STEP):
        if t <= blocked_through:
            continue
        info = evaluate(x.iloc[:t+1], market=market, flow=None)
        if info["signal"] not in BUY:
            continue
        next_open = float(x.Open.iloc[t+1])
        reference = float(info["price"])
        stop = info.get("stop")
        target = info.get("trade_target")
        if stop is None or target is None or next_open <= 0:
            continue
        # Do not reuse a plan after large overnight price gaps. Recheck risk
        # and reward at the actual executable next-day opening price.
        if abs(next_open/reference-1) > 0.025:
            continue
        risk = (next_open-float(stop))/next_open
        reward = (float(target)-next_open)/next_open
        if not (0 < risk <= 0.12 and reward >= 0.02 and reward/risk >= 1.2):
            continue
        outcome = simulate_exit(x, t, next_open, float(stop), float(target))
        if outcome is None:
            continue
        blocked_through = outcome["exit_index"]
        rows.append({"market":market,"ticker":ticker,
                     "signal_date":str(pd.Timestamp(x.index[t]).date()),
                     "entry_date":str(pd.Timestamp(x.index[t+1]).date()),
                     "entry":next_open,"stop":float(stop),"take":float(target),
                     "exit_date":str(pd.Timestamp(x.index[outcome["exit_index"]]).date()),
                     "score":info.get("score"), **outcome})
    return rows

def metrics(trades):
    # Stable historical ordering is necessary even for a diagnostic trade sequence.
    # Overlapping positions still make this different from a capital-weighted portfolio.
    trades = sorted(trades, key=lambda r:(str(r.get("entry_date","")), str(r.get("ticker",""))))
    pnl = np.array([r["net_pct"] for r in trades], dtype=float)
    if not len(pnl):
        return {"trades":0,"win_rate_pct":None,"mean_net_pct":None,
                "profit_factor":None,"max_drawdown_pct":None}
    # This is a trade-sequence diagnostic, not a real synchronized portfolio.
    gains = pnl[pnl>0].sum()
    losses = -pnl[pnl<0].sum()
    equity = np.cumprod(1+pnl/100)
    highwater = np.maximum.accumulate(np.r_[1.,equity])
    dd = 100*(1-equity/highwater[1:])
    return {"trades":int(len(pnl)), "win_rate_pct":round(float((pnl>0).mean()*100),2),
            "mean_net_pct":round(float(pnl.mean()),3),
            "profit_factor":round(float(gains/losses),3) if losses>0 else None,
            "max_drawdown_pct":round(float(dd.max()),2)}

def run(per_market=100):
    universe = get_universe()
    result = {"created_utc":datetime.now(timezone.utc).isoformat(),
              "status":"NOT_VALIDATED", "scope":"Current constituents only: survivorship bias",
              "assumptions":{"next_session_open":True,"max_holding_days":HOLD_DAYS,
                             "cost_round_trip_pct":COST_PCT,
                             "same_day_stop_target":"stop_first",
                             "flow_history":"unavailable, excluded",
                             "selection":"every 10th trading session, no overlapping positions"},
              "universe_sources":{k:universe.get(k+"_source") for k in ("KR","KQ","US")},
              "markets":{}, "errors":[], "trades":[],
              "portfolios":{}, "historical_data_previously_inspected":True,
              "survivorship_bias":True}
    for market in ("KR","KQ","US"):
        # Remove mandatory ETFs; audit exactly 100 ordinary companies per market.
        extras = {"KR":{"472150","0262B0"},"US":{"SOXL"},"KQ":set()}[market]
        items = [i for i in universe[market] if i[0] not in extras][:per_market]
        trades = []
        frames = {}
        successes = 0
        for ticker,name,_,_ in items:
            try:
                raw,source = get_prices(ticker,market)
                rows = audit_symbol(raw,market,ticker)
                frames[ticker]=raw
                for row in rows: row["provider"] = source
                trades.extend(rows)
                successes += 1
            except Exception as exc:
                result["errors"].append({"market":market,"ticker":ticker,
                                          "reason":str(exc)[:250]})
        # The holdout boundary must NOT depend on which dates traded.
        cut = FIXED_HOLDOUT_START
        train = [r for r in trades if r["entry_date"]<cut]
        test = [r for r in trades if r["entry_date"]>=cut]
        m = metrics(test)
        result["markets"][market] = {"requested":per_market,"listed":len(items),
             "loaded":successes,"date_split":cut,
             "train":metrics(train),"holdout":m,
             "status":"PASS" if (len(items)==per_market and successes==per_market
                     and m["trades"]>=30 and m["win_rate_pct"]>=55
                     and m["mean_net_pct"]>0 and m["profit_factor"] is not None
                     and m["profit_factor"]>=1.3) else "FAIL_OR_INSUFFICIENT"}
        result["trades"].extend(trades)
        portfolio=replay(test,frames,start=cut,
                         initial_cash=10_000_000.0 if market in ("KR","KQ") else 100_000.0)
        portfolio["currency"]="KRW" if market in ("KR","KQ") else "USD"
        # Detail lives in the audit JSON. Summary never substitutes for broker fills.
        result["portfolios"][market]=portfolio
    result["status"] = ("PASS_WITH_LIMITATIONS"
                        if all(result["markets"][m]["status"]=="PASS"
                               for m in ("KR","KQ","US"))
                        else "FAIL_OR_INSUFFICIENT")
    portfolio_checks=[result["portfolios"][m] for m in ("KR","KQ","US")]
    result["certification"]=assess(result,portfolio={
        "valid":all(p.get("valid") for p in portfolio_checks),
        "max_drawdown_pct":max((p.get("max_drawdown_pct") or 0) for p in portfolio_checks),
        "benchmark_verified":False},paper=None)
    return result

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="data/three-hundred-audit.json")
    parser.add_argument("--per-market",type=int,default=100,choices=[20,100])
    args=parser.parse_args()
    report=run(per_market=args.per_market)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("trades",)},
                     ensure_ascii=False,indent=2))
    # Deliberately do not exit 1 on an unprofitable strategy: data should be
    # retained and inspected rather than CI hiding a negative outcome.
