"""Opt-in, separate-process historical audit, never importing market libraries
inside Streamlit initialization. Disabled unless STOCK_AUDIT_ON_START=1."""
import os
import sys
if os.environ.get("STOCK_AUDIT_ON_START")=="1" and "streamlit" in os.path.basename(sys.argv[0]).lower():
    try:
        import subprocess
        lock="/tmp/stock-radar-audit-60.started"
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        os.close(fd)
        subprocess.Popen([sys.executable,"-u","audit_worker.py"],
                         cwd=os.getcwd(),close_fds=True)
        print("AUDIT60_SEPARATE_PROCESS_STARTED",flush=True)
    except FileExistsError:
        pass
    except Exception as exc:
        print("AUDIT60_PROCESS_START_ERROR "+repr(exc),flush=True)
