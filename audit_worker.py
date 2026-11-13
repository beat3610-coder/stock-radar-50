"""Separate, opt-in, one-shot 60-stock audit process on the hosted server."""
import json,sys,time,traceback
print("AUDIT60_WORKER_START",flush=True)
time.sleep(18)  # let Streamlit bind its service port first
try:
    import subprocess
    check=subprocess.run([sys.executable,"-m","unittest","-v","test_risk_guard.py","test_audit_300.py","test_portfolio_audit.py"],capture_output=True,text=True,timeout=120)
    print("AUDIT60_TESTS exit="+str(check.returncode)+" "+(check.stderr+check.stdout)[-3500:],flush=True)
    if check.returncode: raise RuntimeError("risk and execution unit tests failed")
    from audit_300 import run
    result=run(per_market=20)
    import os
    os.makedirs("/tmp",exist_ok=True)
    with open("/tmp/stock-radar-audit-60.json","w",encoding="utf-8") as out:
        json.dump(result,out,ensure_ascii=False,default=str)
    public_summary={
        "status":result["status"],
        "markets":result["markets"],
        "portfolios":{m:{k:v for k,v in p.items() if k not in ("daily_equity","executions")}
                      for m,p in result["portfolios"].items()},
        "certification":result["certification"],
        "error_count":len(result["errors"]),
        "errors":result["errors"][:12],
        "candidate_trades":len(result["trades"])
    }
    print("LIVE_AUDIT_RESULT "+json.dumps(public_summary,ensure_ascii=False,default=str),flush=True)
except Exception:
    print("AUDIT60_FATAL "+traceback.format_exc(),flush=True)
