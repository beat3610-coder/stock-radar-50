"""Separate, opt-in, one-shot 60-stock audit process on the hosted server."""
import json,sys,time,traceback
print("AUDIT60_WORKER_START",flush=True)
time.sleep(18)  # let Streamlit bind its service port first
try:
    from audit_300 import run
    result=run(per_market=20)
    import os
    os.makedirs("/tmp",exist_ok=True)
    with open("/tmp/stock-radar-audit-60.json","w",encoding="utf-8") as out:
        json.dump(result,out,ensure_ascii=False,default=str)
    print("AUDIT60_RESULT "+json.dumps({
        "status":result["status"],
        "markets":result["markets"],
        "error_count":len(result["errors"]),
        "first_errors":result["errors"][:12],
        "trade_count":len(result["trades"])},ensure_ascii=False,default=str),flush=True)
except Exception:
    print("AUDIT60_FATAL "+traceback.format_exc(),flush=True)
