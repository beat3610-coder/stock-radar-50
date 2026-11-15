"""Run a user-authorized, one-shot 300-symbol historical strategy backtest on Render.

Opt-in only: STOCK_AUDIT_ON_START=1. App UI remains responsive, output is logged.
A result is never labelled independent: the holdout dates have been inspected.
"""
import json
import logging
import os
import threading
import time
_thread = None
_lock = threading.Lock()
log = logging.getLogger("stock_radar.lab300")
def _run():
    started=time.monotonic()
    try:
        from strategy_lab import run
        log.warning("LAB300_START requested=100_each markets=KR,KQ,US cost_roundtrip_pct=0.35 retrospective_only=yes")
        report=run(per_market=100)
        summary={"created_utc":report["created_utc"],"release_status":report["release_status"],
                 "elapsed_seconds":round(time.monotonic()-started,1),"markets":{}}
        for market,m in report["markets"].items():
            summary["markets"][market]={
                "loaded":m["loaded"],"requested":m["requested"],
                "training_selected":m["training_selected"],
                "validation_accepted":m["validation_accepted"],
                "selected_for_diagnostic":m["selected_for_diagnostic"],
                "holdout":m["holdout_selected_metrics"],
                "portfolio":m["portfolio"],
                "stress_portfolio":m["stress_portfolio"],
                "loss_diagnosis":m.get("loss_diagnosis"),
                "errors":m["errors"][:8]
            }
            log.warning("LAB300_MARKET %s %s",market,json.dumps(summary["markets"][market],ensure_ascii=False,default=str))
        log.warning("LAB300_RESULT %s",json.dumps(summary,ensure_ascii=False,default=str))
        with open("/tmp/stock-radar-lab300.json","w",encoding="utf-8") as outfile:
            json.dump(report,outfile,ensure_ascii=False,indent=2,default=str)
        log.warning("LAB300_COMPLETE elapsed_seconds=%s",summary["elapsed_seconds"])
    except Exception:
        log.exception("LAB300_FATAL")
def ensure_started():
    global _thread
    if os.environ.get("STOCK_AUDIT_ON_START")!="1":
        return
    with _lock:
        if _thread is None:
            _thread=threading.Thread(target=_run,name="stock-lab300",daemon=True)
            _thread.start()
