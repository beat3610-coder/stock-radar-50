"""51%-target trade selector. Select ON TRAIN ONLY, veto on validation.
Retrospective results are research, not unseen live evidence.
"""
from itertools import combinations
from math import sqrt, isfinite
TRAIN="2025-11-01"
VALID="2026-04-01"
END="2026-11-01"
BOUNDS={
 "signal_rsi":[(">=",35),(">=",45),(">=",50),("<=",45),("<=",55),("<=",65)],
 "signal_adx":[(">=",20),(">=",25),("<=",20),("<=",30)],
 "signal_atr_pct":[("<=",2.5),("<=",4),(">=",2),(">=",3)],
 "signal_ma20_ma60_pct":[(">=",0),(">=",2),("<=",0),("<=",-2)],
 "signal_volume_ratio":[(">=",1),(">=",1.5),("<=",1),("<=",2)],
 "signal_5d_return_pct":[(">=",0),(">=",2),("<=",0),("<=",-3)],
 "entry_gap_pct":[("<=",1),(">=",-1),("<=",0),(">=",0)],
 "entry_net_rr":[(">=",1.5),(">=",1.8)],
 "entry_net_upside_pct":[(">=",4),(">=",6)]}
def stats(rows):
 n=len(rows)
 if n==0:return {"n":0,"win_pct":None,"mean_net_pct":None,"profit_factor":None}
 pnl=[r["net_pct"] for r in rows]
 wins=sum(v>0 for v in pnl);gains=sum(v for v in pnl if v>0)
 losses=-sum(v for v in pnl if v<0)
 return {"n":n,"win_pct":round(100*wins/n,3),
  "mean_net_pct":round(sum(pnl)/n,4),
  "profit_factor":round(gains/losses,3) if losses>0 else None}
def obey(r,rule):
 for key,op,cut in rule:
  try:v=float(r[key])
  except (TypeError,ValueError,KeyError):return False
  if not isfinite(v) or (op==">=" and v<cut) or (op=="<=" and v>cut):return False
 return True
def wilson(w,n):
 z=1.28;p=w/n
 return (p+z*z/(2*n)-z*sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
def discover(trades,market=None):
 clean=[]
 for row in trades:
  if market and row.get("market")!=market:continue
  sig=str(row.get("signal_date",""))[:10];end=str(row.get("exit_date",""))[:10]
  try:pnl=float(row["net_pct"])
  except (KeyError,TypeError,ValueError):continue
  if len(sig)!=10 or len(end)!=10 or end<sig or not isfinite(pnl):continue
  clean.append({**row,"net_pct":pnl,"signal_date":sig,"exit_date":end})
 windows={"train":[r for r in clean if r["signal_date"]<TRAIN and r["exit_date"]<TRAIN],
  "validation":[r for r in clean if TRAIN<=r["signal_date"] and r["exit_date"]<VALID],
  "diagnostic":[r for r in clean if VALID<=r["signal_date"] and r["exit_date"]<END]}
 conditions=[(k,op,v) for k,bounds in BOUNDS.items() for op,v in bounds]
 rules=[()]+[(p,) for p in conditions]+[pair for pair in combinations(conditions,2) if pair[0][0]!=pair[1][0]]
 qualified=[]
 for rule in rules:
  picked=[r for r in windows["train"] if obey(r,rule)]
  s=stats(picked)
  if s["n"]<45 or s["mean_net_pct"]<=0:continue
  lower=wilson(sum(r["net_pct"]>0 for r in picked),s["n"])
  qualified.append((lower-0.005*len(rule)+0.001*min(s["mean_net_pct"],5),rule))
 result={"market":market,"target_win_pct":51,"candidates_tested":len(rules),
  "baseline":{k:stats(v) for k,v in windows.items()},
  "selected":None,"status":"NO_TRAINING_CANDIDATE",
  "eligible_for_live":False}
 if not qualified:return result
 qualified.sort(key=lambda z:(-z[0],len(z[1]),str(z[1])))
 rule=qualified[0][1]
 sample={k:stats([r for r in rows if obey(r,rule)]) for k,rows in windows.items()}
 v=sample["validation"]
 passed=(v["n"]>=30 and v["win_pct"]>=51 and v["mean_net_pct"]>0 and
  v["profit_factor"] is not None and v["profit_factor"]>1)
 result.update({"status":"VALIDATION_PASS_RESEARCH_ONLY" if passed else "VALIDATION_REJECTED",
  "selected":{"conditions":[{"feature":k,"op":o,"threshold":t} for k,o,t in rule],
  "train":sample["train"],"validation":v,"diagnostic_research_only":sample["diagnostic"],
  "validation_pass":passed}})
 return result
