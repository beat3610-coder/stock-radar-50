"""300-ticker universe selection and conservative daily OHLC execution checks."""
from unittest.mock import patch
import pandas as pd
from trade_strategy_audit import pick_300,execute_path,execute_variant
symbols=lambda market:[(f"{market}{i:03d}",f"{market} stock {i}",market,"Sector") for i in range(100)]
fake={"KR":symbols("KR"),"KQ":symbols("KQ"),"US":symbols("US"),
      "KR_source":"test","KQ_source":"test","US_source":"test"}
with patch("trade_strategy_audit.get_universe",return_value=fake):
    items,universe=pick_300()
assert len(items)==300
assert [sum(x[0]==market for x in items) for market in ("KR","KQ","US")]==[100,100,100]
assert len({(m,c) for m,c,_ in items})==300
# Same-bar stops and takes -> stop first. Next open is the entry.
d=pd.DataFrame({"Open":[100,100,100],"High":[101,111,101],
               "Low":[99,89,98],"Close":[100,100,100]})
out=execute_path(d,0,100,5,8)
assert out["exit_reason"]=="stop" and abs(out["net_return_pct"]+5.35)<1e-8,out
# Gap below stop exits at actual open, never at optimistic stop trigger.
d.loc[1,"Open"]=88
out=execute_path(d,0,100,5,8)
assert out["exit_reason"]=="gap_stop" and abs(out["net_return_pct"]+12.35)<1e-8,out
print("PASS 300-company selection and conservative execution")

# No exit variant may use future price data beyond the 10-session window.
bars=pd.DataFrame({"Open":[100]*13,"High":[101]*13,"Low":[99]*13,"Close":[100]*13})
for strategy in ("TAKE_STOP","HOLD_10","STOP_10","HALF_TRAIL"):
    value=execute_variant(bars,0,100,95,108,strategy)
    assert abs(value+0.35)<1e-9,(strategy,value)
assert execute_variant(d,0,100,95,108,"STOP_10")<=-5.35
print("Frozen alternative exit simulations PASS")
