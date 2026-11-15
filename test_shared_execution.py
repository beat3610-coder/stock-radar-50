"""Shared execution gate: deterministic, offline regression checks."""
from trade_execution import entry_quality

def run():
    tests=[
        (100,94.9,108.67,100,True),
        (102.5,94.9,108.67,100,False),
        (99,101,110,None,False),
        (float("nan"),94.9,108.67,None,False),
        (100,94.9,108.67,0,False),
        (103,96,112,100,False),
    ]
    for entry,stop,target,reference,expected in tests:
        r=entry_quality(entry,stop,target,reference=reference)
        assert r["accepted"] is expected,(entry,r)
    assert entry_quality(100,94.9,108.67,reference=100)["net_rr"]==1.527
    assert entry_quality(102.5,94.9,108.67,reference=100)["net_rr"]==0.730
    print("PASS: six shared execution scenarios and numeric R/R")

if __name__=="__main__":
    run()
