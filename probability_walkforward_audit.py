"""Leakage-safe, expanding-window audit for STOCK RADAR 50 probabilities.

CSV columns: date,market,actual,p_up,p_down,p_flat.
actual: up/down/flat. Predictions must be archived *before* outcomes
are known; retrospective scoring is explicitly research-only.
No claims of automatic trading or certified probabilities.
"""
import argparse
import json
import numpy as np
import pandas as pd

CLASSES = ("up", "down", "flat")
P_COLS = ("p_up", "p_down", "p_flat")

def score(y, p):
    t = np.eye(3)[y]
    brier = float(np.mean(np.sum((p-t)**2, axis=1)))
    accuracy = float(np.mean(p.argmax(axis=1) == y))
    recalls = [float(np.mean(p[y == c].argmax(axis=1) == c)) if np.any(y == c) else None
               for c in range(3)]
    return {"n": len(y), "accuracy_pct": round(100*accuracy, 3),
            "multiclass_brier": round(brier, 6),
            "recall_by_class": dict(zip(CLASSES, recalls))}

def transform(p, prior, temperature, shrink):
    # Softmax temperature and smoothed historical prior; no outcomes from
    # the date being predicted may be included in 'prior'.
    logits = np.log(np.clip(p, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    q = exp / exp.sum(axis=1, keepdims=True)
    return (1-shrink)*q + shrink*prior[None, :]

def audit(df, min_train_dates=100, min_test_dates=30, refit_every=20):
    required = set(("date", "market", "actual", *P_COLS))
    missing = required-set(df.columns)
    if missing: raise ValueError("Missing required fields: "+str(sorted(missing)))
    df = df.copy()
    df["date"] = pd.to_datetime(df.date, errors="raise").dt.normalize()
    if df[list(P_COLS)].isna().any().any(): raise ValueError("Missing prediction probability")
    if not df.actual.isin(CLASSES).all(): raise ValueError("Unknown outcome label")
    probs = df[list(P_COLS)].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1),1,atol=1e-4):
        raise ValueError("Probabilities must be finite, within [0,1], and sum to 1")
    if df.duplicated(["market","date", "symbol"]).any() if "symbol" in df else False:
        raise ValueError("Duplicate symbol-market-date rows")
    outputs = {}
    for market, group in df.groupby("market"):
        group = group.sort_values("date", kind="stable")
        days = group.date.drop_duplicates().to_numpy()
        if len(days) < min_train_dates + min_test_dates:
            outputs[str(market)] = {"status": "INSUFFICIENT_DATA", "dates": len(days)}
            continue
        p = group[list(P_COLS)].to_numpy(float)
        y = pd.Categorical(group.actual, categories=CLASSES).codes
        date_vals = group.date.to_numpy()
        predictions = np.full_like(p, np.nan)
        selected = None
        for k in range(min_train_dates, len(days)):
            day = days[k]
            past_idx = date_vals < day
            now_idx = date_vals == day
            if (k-min_train_dates) % refit_every == 0:
                py = y[past_idx]
                pp = p[past_idx]
                prior = (np.bincount(py, minlength=3)+1)/(len(py)+3)
                truth = np.eye(3)[py]
                best = None
                for temp in (0.8,1.0,1.25,1.5,2.0):
                    for shrink in (0.0,0.1,0.25,0.4):
                        pred = transform(pp, prior, temp, shrink)
                        loss = float(np.mean(np.sum((pred-truth)**2, axis=1)))
                        # Conservative tie break: prefer minimal distortion.
                        candidate = (loss, abs(temp-1)+shrink, temp, shrink)
                        if best is None or candidate < best: best = candidate
                selected = (prior, best[2], best[3])
            predictions[now_idx] = transform(p[now_idx], *selected)
        test_idx = date_vals >= days[min_train_dates]
        yt = y[test_idx]
        base = score(yt, p[test_idx])
        challenger = score(yt, predictions[test_idx])
        # No automatic promotion on retroactively inspected periods.
        outputs[str(market)] = {
            "status": "RESEARCH_ONLY_NOT_LIVE_APPROVED",
            "start": str(days[min_train_dates])[:10],
            "end": str(days[-1])[:10],
            "baseline": base,
            "calibrated": challenger,
            "accuracy_change_pp": round(challenger["accuracy_pct"]-base["accuracy_pct"],3),
            "brier_improvement": round(base["multiclass_brier"]-challenger["multiclass_brier"],6),
            "improved_both": (challenger["accuracy_pct"]>base["accuracy_pct"]
                              and challenger["multiclass_brier"]<base["multiclass_brier"]),
            "note": "Expanding past-only fit; historical outputs are research-only. No promotion without prospective archived signals."
        }
    return outputs

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("csv", help="Archived daily probability predictions with realized labels")
    cli.add_argument("--output", default="probability-audit.json")
    args = cli.parse_args()
    result = audit(pd.read_csv(args.csv))
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
