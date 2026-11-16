"""STOCK RADAR investment-readiness controls.

This is a release *rejection* mechanism, not a proof of profitable returns.
Historical market data consulted during development are not pristine holdouts.
A live approval additionally requires a forward paper-trading history that is
not available simply by re-running the historical backtest.
"""
from math import sqrt

MIN_SYMBOLS_PER_MARKET=100
MIN_TRADES_PER_MARKET=30
MIN_WIN_RATE_PCT=60.0
MIN_PAPER_DAYS=90
MIN_PAPER_TRADES=60
MAX_PORTFOLIO_DRAWDOWN_PCT=12.0
MAX_TRADE_RISK_FRACTION=0.01
MAX_POSITION_FRACTION=0.20
MAX_CONCURRENT_POSITIONS=5

def wilson_lower(wins,n,z=1.96):
    """Approximate conservative 95% lower bound on a binary win rate."""
    if n<=0:return None
    p=wins/n
    return (p+z*z/(2*n)-z*sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)

def assess(report, *, portfolio=None, paper=None):
    """Separate historical research quality from authorization for real trading.
    No historical backtest on previously inspected dates can ever grant LIVE.
    """
    reasons=[]
    markets=report.get("markets",{})
    for market in ("KR","KQ","US"):
        r=markets.get(market)
        if not r:
            reasons.append(market+": missing")
            continue
        if r.get("requested",0)<MIN_SYMBOLS_PER_MARKET or r.get("loaded",0)<MIN_SYMBOLS_PER_MARKET:
            reasons.append(market+": fewer than 100 successfully priced symbols")
        h=r.get("holdout_selected_metrics") or r.get("holdout") or {}
        n=h.get("trades") or 0
        if n<MIN_TRADES_PER_MARKET:
            reasons.append(market+": insufficient holdout trades ("+str(n)+")")
            continue
        wr=h.get("win_rate_pct")
        pf=h.get("profit_factor")
        net=h.get("mean_net_pct")
        if wr is None or wr<MIN_WIN_RATE_PCT:
            reasons.append(market+": win rate below 60%")
        if pf is None or pf<1.3:reasons.append(market+": profit factor below 1.30")
        if net is None or net<=0:reasons.append(market+": nonpositive mean net return")
        lower=wilson_lower(round(n*wr/100),n) if wr is not None else None
        if lower is None or lower<0.45:
            reasons.append(market+": 95% Wilson win-rate lower bound below 45%")
    # Evaluate market-specific portfolios instead of silently assuming an
    # unprovided combined cross-currency portfolio is valid.
    for key in ("KR","KQ","US"):
        item=markets.get(key) or {}
        actual=item.get("portfolio") or {}
        stress=item.get("stress_portfolio") or {}
        if not actual.get("valid") or actual.get("total_return_pct") is None or actual["total_return_pct"]<=0:
            reasons.append(key+": nonpositive or invalid portfolio return")
        if actual.get("max_drawdown_pct") is None or actual["max_drawdown_pct"]>MAX_PORTFOLIO_DRAWDOWN_PCT:
            reasons.append(key+": maximum drawdown exceeds 12% or missing")
        if stress.get("total_return_pct") is None or stress["total_return_pct"]<=0:
            reasons.append(key+": cost stress return not positive")
    p=portfolio or {}
    if p and (not p.get("valid") or p.get("max_drawdown_pct") is None or p["max_drawdown_pct"]>MAX_PORTFOLIO_DRAWDOWN_PCT):
        reasons.append("combined portfolio invalid or drawdown exceeds 12%")
    if not (p.get("benchmark_verified") or report.get("benchmark_verified")):
        reasons.append("market benchmark comparison unavailable")
    if report.get("scope")=="Current constituents only: survivorship bias" or report.get("survivorship_bias",True):
        reasons.append("historical constituent/survivorship-bias controls missing")
    if report.get("historical_data_previously_inspected",True):
        reasons.append("historical test interval previously inspected during strategy development")
    if any("Holdout dates were already inspected" in str(v) for v in report.get("limitations",[])):
        reasons.append("diagnostic holdout already inspected during development")
    paper=paper or {}
    if not(paper.get("verified") and paper.get("calendar_days",0)>=MIN_PAPER_DAYS
           and paper.get("completed_trades",0)>=MIN_PAPER_TRADES
           and paper.get("net_profit_after_costs",0)>0):
        reasons.append("unobserved or insufficient 90-day/60-trade forward paper record")
    if paper.get("market_data_redistribution_rights_verified") is not True:
        reasons.append("commercial market-data rights unverified")
    if paper.get("regulatory_review_completed") is not True:
        reasons.append("financial promotion/regulatory review unverified")
    if paper.get("subscription_operations_verified") is not True:
        reasons.append("membership/payment/refund operations unverified")
    return {"status":"LIVE_APPROVED" if not reasons else "NOT_APPROVED",
            "reasons":reasons,
            "guardrails":{"max_trade_risk_pct":1,"max_position_pct":20,
                          "max_concurrent_positions":5,"max_portfolio_mdd_pct":12},
            "notes":"Approval requires independently verified prospective fills; historical trading is not investment certification."}
