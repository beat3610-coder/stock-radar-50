"""Executable entry filters: prevent gap-driven reward/risk deterioration."""
from strategy_lab import assess_executable_entry

def run():
    # Previously: signal-close plan passed but opening 2.5% higher paid <1R.
    late=assess_executable_entry(102.5,94.9,108.67)
    assert not late["accepted"],late
    assert late["net_rr"]<1.0,late
    # At signal close, equivalent plan has >=1.3R net, up >3%, risk <7%.
    normal=assess_executable_entry(100,94.9,108.67)
    assert normal["accepted"],normal
    assert normal["net_rr"]>=1.3,normal
    # Invalid/reversed prices fail closed.
    assert not assess_executable_entry(100,105,110)["accepted"]
    assert not assess_executable_entry(100,None,110)["accepted"]
    print("PASS executable next-open RR, gap risk, invalid data")

if __name__=="__main__":
    run()
