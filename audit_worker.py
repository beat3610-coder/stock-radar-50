"""Separate, opt-in, one-shot 60-stock audit process on the hosted server."""
import json,sys,time,traceback
print("AUDIT60_WORKER_START",flush=True)
time.sleep(18)  # let Streamlit bind its service port first
try:
    import subprocess
    check=subprocess.run([sys.executable,"-m","unittest","-v","test_risk_guard.py","test_audit_300.py"],capture_output=True,text=True,timeout=120)
    print("AUDIT60_TESTS exit="+str(check.returncode)+" "+(check.stderr+check.stdout)[-3500:],flush=True)
    if check.returncode: raise RuntimeError("risk and execution unit tests failed")
    from diagnose_filters import run
    result=run(per_market=20)
    import os
    os.makedirs("/tmp",exist_ok=True)
    with open("/tmp/stock-radar-audit-60.json","w",encoding="utf-8") as out:
        json.dump(result,out,ensure_ascii=False,default=str)
    print("COMPARE_RESULT "+json.dumps(result,ensure_ascii=False,default=str),flush=True)
except Exception:
    print("AUDIT60_FATAL "+traceback.format_exc(),flush=True)
