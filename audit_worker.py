"""Separate, opt-in, one-shot 60-stock audit process on the hosted server."""
import json,sys,time,traceback
print("AUDIT60_WORKER_START",flush=True)
time.sleep(18)  # let Streamlit bind its service port first
try:
    import subprocess
    check=subprocess.run([sys.executable,"-m","unittest","-v","test_risk_guard.py","test_audit_300.py","test_portfolio_audit.py","test_strategy_lab.py"],capture_output=True,text=True,timeout=120)
    print("AUDIT60_TESTS exit="+str(check.returncode)+" "+(check.stderr+check.stdout)[-3500:],flush=True)
    if check.returncode: raise RuntimeError("risk and execution unit tests failed")
    from strategy_lab import run
    report=run(per_market=20)
    print("LAB_RESULT "+json.dumps(report,ensure_ascii=False,default=str),flush=True)
except Exception:
    print("AUDIT60_FATAL "+traceback.format_exc(),flush=True)
