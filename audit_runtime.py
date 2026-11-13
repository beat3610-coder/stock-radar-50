"""Run a one-shot market audit in the existing Render process without blocking UI."""
import json
import logging
import os
import threading
import traceback
_thread = None
_lock = threading.Lock()
log = logging.getLogger("stock_radar.audit_60")
def _run():
    try:
        from audit_300 import run
        log.warning("AUDIT60_START market_data=live 20_per_market")
        report = run(per_market=20)
        summary = {
            "status": report["status"],
            "markets": report["markets"],
            "errors": report["errors"],
            "trade_count": len(report["trades"]),
        }
        log.warning("AUDIT60_RESULT %s", json.dumps(summary,ensure_ascii=False,default=str))
        path = "/tmp/stock-radar-audit-60.json"
        with open(path,"w",encoding="utf-8") as f:
            json.dump(report,f,ensure_ascii=False,indent=2,default=str)
    except Exception as e:
        log.exception("AUDIT60_FATAL %s",e)
def ensure_started():
    global _thread
    if os.environ.get("STOCK_AUDIT_ON_START")!="1":
        return
    with _lock:
        if _thread is None:
            _thread=threading.Thread(target=_run,name="stock-audit-60",daemon=True)
            _thread.start()
