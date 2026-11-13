"""Optional hosted one-shot audit bootstrap.
Python imports sitecustomize on interpreter initialization when repo cwd is in
PYTHONPATH. Disabled unless STOCK_AUDIT_ON_START=1. Only run for Streamlit.
"""
import os
import sys
if os.environ.get("STOCK_AUDIT_ON_START")=="1" and "streamlit" in os.path.basename(sys.argv[0]).lower():
    try:
        from audit_runtime import ensure_started
        ensure_started()
        print("AUDIT60_BOOTSTRAP thread launched",flush=True)
    except Exception as exc:
        print("AUDIT60_BOOTSTRAP_ERROR "+repr(exc),flush=True)
