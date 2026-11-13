"""Separate, opt-in, one-shot 60-stock audit process on the hosted server."""
import json,sys,time,traceback
print("AUDIT60_WORKER_START",flush=True)
time.sleep(18)  # let Streamlit bind its service port first
try:
    import subprocess
    check=subprocess.run([sys.executable,"-m","unittest","-v","test_risk_guard.py","test_audit_300.py","test_portfolio_audit.py","test_strategy_lab.py","test_signal_playbook.py","test_position_decision.py"],capture_output=True,text=True,timeout=120)
    print("AUDIT60_TESTS exit="+str(check.returncode)+" "+(check.stderr+check.stdout)[-3500:],flush=True)
    if check.returncode: raise RuntimeError("risk and execution unit tests failed")
    compiled=subprocess.run([sys.executable,"-m","py_compile","app.py","strategy_lab.py",
                             "signal_playbook.py","portfolio_audit.py"],
                            capture_output=True,text=True,timeout=45)
    print("APP_COMPILE_CHECK exit="+str(compiled.returncode)+
          " "+compiled.stderr[-600:],flush=True)
    if compiled.returncode:raise RuntimeError("app/module source compilation failed")
    import os
    if os.environ.get("STOCK_AUDIT_TEST_ONLY")=="1":
        print("APP_SMOKE_ONLY_PASSED",flush=True)
        sys.exit(0)
    from strategy_lab import run
    report=run(per_market=100)
    import os
    os.makedirs("/tmp",exist_ok=True)
    with open("/tmp/strategy-lab-300.json","w",encoding="utf-8") as outfile:
        json.dump(report,outfile,ensure_ascii=False,default=str)
    brief={"created_utc":report["created_utc"],"train_end":report["train_end"],
           "validation_end":report["validation_end"],"holdout_end":report["holdout_end"],
           "markets":{}}
    for m,r in report["markets"].items():
        brief["markets"][m]={
            "requested":r["requested"],"loaded":r["loaded"],"error_count":len(r["errors"]),
            "errors":r["errors"][:5],
            "train_selected":r["training_selected"],"selected":r["selected_for_diagnostic"],
            "train":r["train"].get(r["training_selected"]),
            "validation":r["validation"].get(r["training_selected"]),
            "holdout":r["holdout_selected_metrics"],
            "holdout_variants":r["holdout_all_variants_diagnostic_only"],
            "portfolio":r["portfolio"],"stress_portfolio":r["stress_portfolio"]}
    print("LAB300_SUMMARY "+json.dumps(brief,ensure_ascii=False,default=str),flush=True)
except Exception:
    print("AUDIT60_FATAL "+traceback.format_exc(),flush=True)
