"""Reliability under shift at matched clean cost, for raw, clean-calibrated and shift-calibrated exits.

For every run and exit policy, sweep the threshold, record clean GFLOPs and the behaviour on corrupted test data,
then interpolate at fixed clean costs. Writes results/study/matched_cost.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_study import CONFIGS, discover, load  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.corruptions import CORRUPTIONS  # noqa: E402
from src.exits import THRESH, exit_scores, fit_temperature, route  # noqa: E402

TARGETS = {"CNN": (0.5, 0.6, 0.7, 0.8), "ViT": (0.3, 0.4, 0.5)}
SEVS = ("3", "5")
METRICS = ("early_frac", "early_acc", "acc")


def curve(st, costs, shifted, k):
    """Rows of (clean cost, early share, early-exit accuracy, accuracy) over the threshold grid."""
    rows = []
    for t in THRESH:
        clean = costs[route(st, t)].mean()
        ex = [route(s, t) for s, _ in shifted]
        early = [e < k - 1 for e in ex]
        ef = np.mean([e.mean() for e in early])
        ea = np.nanmean([c[e_, np.arange(len(e_))][m].mean() if m.any() else np.nan
                         for (s, c), e_, m in zip(shifted, ex, early)])
        acc = np.mean([c[e_, np.arange(len(e_))].mean() for (s, c), e_ in zip(shifted, ex)])
        rows.append((clean, ef, ea, acc))
    return np.array(rows)


def at(rows, target):
    r = rows[np.argsort(rows[:, 0])]
    return [float(np.interp(target, r[:, 0], r[:, j])) for j in (1, 2, 3)]


def run_curves(r):
    d = r["dir"]
    costs = np.array([c / 1e9 for c in r["flops"]["cumulative"]])
    lv, lt = load(d, "val"), load(d, "test")
    yv = np.load(os.path.join(d, "logits", "val_labels.npy"))
    yt = np.load(os.path.join(d, "logits", "test_labels.npy"))
    k = lt.shape[0]
    temps = r["temperature"]
    out = {}
    for sev in SEVS:
        logits = {c: load(d, f"{c}_{sev}") for c in CORRUPTIONS}
        correct = {c: lg.argmax(-1).numpy() == yt for c, lg in logits.items()}
        res = {}
        for policy, tt in (("raw", [1.0] * k), ("ts", temps)):
            st = exit_scores(lt, "max_prob", True, tt)
            shifted = [(exit_scores(logits[c], "max_prob", True, tt), correct[c]) for c in CORRUPTIONS]
            res[policy] = curve(st, costs, shifted, k)
        if r["val_dir"]:
            folds = []
            for c in CORRUPTIONS:
                pooled = [load(r["val_dir"], f"val_{o}_{s}") for o in CORRUPTIONS if o != c for s in ("1", "2", "3", "4", "5")]
                lp = torch.cat([lv] + pooled, 1)
                yp = torch.from_numpy(np.concatenate([yv] * (1 + len(pooled))))
                loco = [fit_temperature(lp[i], yp) for i in range(k)]
                st = exit_scores(lt, "max_prob", True, loco)
                folds.append(curve(st, costs, [(exit_scores(logits[c], "max_prob", True, loco), correct[c])], k))
            res["loco"] = folds
        out[sev] = res
    return out


def summarize(curves_by_seed, targets):
    out = {}
    for policy in curves_by_seed[0]:
        out[policy] = {}
        for t in targets:
            vals = []
            for cv in curves_by_seed:
                c = cv[policy]
                vals.append(np.mean([at(f, t) for f in c], 0) if isinstance(c, list) else at(c, t))
            vals = np.array(vals)
            out[policy][str(t)] = {m: [float(np.nanmean(vals[:, j])), float(np.nanstd(vals[:, j]))]
                                   for j, m in enumerate(METRICS)}
    return out


def main():
    runs = discover("runs")
    result = {}
    for cfg, rs in sorted(runs.items()):
        arch = CONFIGS[cfg][0]
        if arch not in TARGETS:
            continue
        per_seed = [run_curves(r) for r in rs]
        result[cfg] = {"n": len(rs), **{sev: summarize([p[sev] for p in per_seed], TARGETS[arch]) for sev in SEVS}}
        mid = str(TARGETS[arch][1])
        print(cfg, {p: (round(100 * v[mid]["early_frac"][0], 1), round(100 * v[mid]["early_acc"][0], 1))
                    for p, v in result[cfg]["5"].items()}, flush=True)
    with open(os.path.join("results", "study", "matched_cost.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
