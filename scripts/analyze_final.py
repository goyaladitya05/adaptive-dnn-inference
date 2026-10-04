"""Final analysis over seeds: operating points, per-exit thresholds, ablations, calibration, corruptions, latency.

Reads runs/final-*/<run>/ and runs/latency/<run>/latency.json, writes results/final/.
"""
import argparse
import glob
import itertools
import json
import os
import re
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import AQUA, BLUE, EXIT_RAMP, INK, INK2, ORANGE, POLICY_STYLE, VIOLET, plt, savefig  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.corruptions import CORRUPTIONS  # noqa: E402
from src.exits import (POLICIES, exit_scores, pick_threshold, reliability, route, routing_stats,  # noqa: E402
                       sweep, tune_per_exit)

SEVERITIES = ("1", "2", "3", "4", "5")
TOL, TOL_RELAXED = 0.0, 0.01
NAMES = {"resnet18": "ResNet-18 (full)", "ee_resnet18_kd": "EE-ResNet-18",
         "ee_resnet18_ce": "EE-ResNet-18, no distillation", "ee_resnet18_pool": "EE-ResNet-18, pooling heads",
         "ee_vit_tiny": "EE-ViT-Tiny"}
YELLOW_SAFE = POLICY_STYLE["Entropy + TS"][0]
CNAMES = {"gaussian_noise": "Gaussian noise", "shot_noise": "Shot noise", "impulse_noise": "Impulse noise",
          "gaussian_blur": "Gaussian blur", "defocus_blur": "Defocus blur", "contrast": "Contrast"}


def ms(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    return [float(np.nanmean(v)), float(np.nanstd(v))] if len(v) else [float("nan"), float("nan")]


def load(d, name):
    return torch.from_numpy(np.load(os.path.join(d, "logits", f"{name}.npy")).astype(np.float32))


def labels(d, split):
    return np.load(os.path.join(d, "logits", f"{split}_labels.npy"))


def discover(root):
    runs = {}
    for m in sorted(glob.glob(os.path.join(root, "final-*", "*", "metrics.json"))):
        d = os.path.dirname(m)
        name = os.path.basename(d)
        cfg, seed = re.match(r"(.+)_s(\d+)$", name).groups()
        with open(m) as f:
            metrics = json.load(f)
        with open(os.path.join(d, "history.json")) as f:
            metrics["history"] = json.load(f)
        metrics.update(dir=d, name=name, cfg=cfg, seed=int(seed))
        lat = os.path.join(root, "latency", name, "latency.json")
        if os.path.exists(lat):
            with open(lat) as f:
                metrics["latency"] = json.load(f)
        runs.setdefault(cfg, []).append(metrics)
    return runs


def subset_costs(seg, head, exits):
    """Cumulative cost of stopping at each exit in `exits` when only those heads are evaluated."""
    return [sum(seg[:e + 1]) + sum(head[j] for j in exits if j <= e) for e in exits]


def analyze_run(r, ref_cost, match_acc):
    d, k = r["dir"], len(r["acc"]["test"])
    out = {"seed": r["seed"], "test_acc": r["acc"]["test"], "val_acc": r["acc"]["val"],
           "gflops": [c / 1e9 for c in r["flops"]["cumulative"]], "params": r["params"],
           "temperature": r["temperature"], "ece_test": r["ece_test"]}
    if k == 1:
        out["corrupt_acc"] = {c: {s: r["corrupt_acc"][c][s][0] for s in SEVERITIES} for c in CORRUPTIONS}
        return out
    costs, temps = out["gflops"], r["temperature"]
    lv, lt = load(d, "val"), load(d, "test")
    yv, yt = labels(d, "val"), labels(d, "test")
    cv, ct = lv.argmax(-1).numpy() == yv, lt.argmax(-1).numpy() == yt
    target, relaxed = out["val_acc"][-1] - TOL, out["val_acc"][-1] - TOL_RELAXED
    match_acc = match_acc if match_acc is not None else out["test_acc"][-1]
    budgets = [round(f * ref_cost, 4) for f in (0.35, 0.5, 0.65, 0.8)]
    out["budgets"] = budgets
    out["policies"], out["curves"] = {}, {}
    for policy, cal, label in POLICIES:
        sv, st = exit_scores(lv, policy, cal, temps), exit_scores(lt, policy, cal, temps)
        sw_v, sw_t = sweep(sv, cv, costs), sweep(st, ct, costs)
        th = pick_threshold(sw_v, target)
        res = {"thresh": th, "test": routing_stats(route(st, th), ct, costs),
               "acc_at": [max([x["acc"] for x in sw_t if x["cost"] <= b + 1e-9], default=float("nan")) for b in budgets]}
        th_k, _ = tune_per_exit(sv, cv, costs, target, th)
        res["per_exit"] = {"thresh": th_k, "test": routing_stats(route(st, th_k), ct, costs)}
        th_r = pick_threshold(sw_v, relaxed)
        res["relaxed"] = {"thresh": th_r, "test": routing_stats(route(st, th_r), ct, costs)}
        res["match_cost"] = min([x["cost"] for x in sw_t if x["acc"] >= match_acc], default=None)
        out["policies"][label] = res
        out["curves"][label] = [(x["cost"], x["acc"]) for x in sw_t]
    first = np.where(ct.any(0), ct.argmax(0), k - 1)
    out["oracle"] = {"acc": float(ct.any(0).mean()), "cost": float(np.asarray(costs)[first].mean())}
    out["overthinking"] = {"destructive": float((ct[:-1].any(0) & ~ct[-1]).mean()),
                           "early_correct": [float(ct[i].mean()) for i in range(k)]}
    seg, head = [f / 1e9 for f in r["flops"]["segment"]], [f / 1e9 for f in r["flops"]["head"]]
    out["subsets"] = {}
    for n in range(k):
        for s in itertools.combinations(range(k - 1), n):
            exits = [*s, k - 1]
            c = subset_costs(seg, head, exits)
            if n:
                sv = exit_scores(lv[exits], "max_prob", False, [1.0] * len(exits))
                st = exit_scores(lt[exits], "max_prob", False, [1.0] * len(exits))
                th = pick_threshold(sweep(sv, cv[exits], c), target)
            else:
                st, th = np.zeros((0, lt.shape[1])), 2.0
            out["subsets"]["+".join(f"E{e + 1}" for e in exits)] = routing_stats(route(st, th), ct[exits], c)
    out["corrupt"], out["corrupt_ece"] = {}, {}
    for c in CORRUPTIONS:
        out["corrupt"][c], out["corrupt_ece"][c] = {}, {}
        for s in SEVERITIES:
            lg = load(d, f"{c}_{s}")
            cc = lg.argmax(-1).numpy() == yt
            row = {"final_acc": float(cc[-1].mean())}
            for policy, cal, label in POLICIES:
                ex = route(exit_scores(lg, policy, cal, temps), out["policies"][label]["thresh"])
                m = routing_stats(ex, cc, costs)
                early = ex < k - 1
                m["early_frac"] = float(early.mean())
                m["early_acc"] = float(cc[ex, np.arange(len(ex))][early].mean()) if early.any() else None
                row[label] = m
            out["corrupt"][c][s] = row
            yt_t = torch.from_numpy(yt)
            out["corrupt_ece"][c][s] = {"raw": [reliability(lg[i], yt_t)[1] for i in range(k)],
                                        "calibrated": [reliability(lg[i], yt_t, temps[i])[1] for i in range(k)]}
    return out


def aggregate(per_seed):
    """Mean and std over seeds of every numeric leaf that all seeds share."""
    first = per_seed[0]
    if isinstance(first, dict):
        return {key: aggregate([p[key] for p in per_seed]) for key in first if all(key in p for p in per_seed)}
    if isinstance(first, list) and first and not isinstance(first[0], (list, tuple, dict)):
        return [ms([p[i] for p in per_seed]) for i in range(len(first))]
    if isinstance(first, (int, float)) or first is None:
        return ms(per_seed)
    return first


def mean_curve(curves, grid):
    ys = []
    for cur in curves:
        ys.append([max([a for c, a in cur if c <= b + 1e-9], default=np.nan) for b in grid])
    ys = np.asarray(ys)
    return np.nanmean(ys, 0), np.nanstd(ys, 0)


def fig_tradeoff(A, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    base = A["resnet18"]["mean"]
    for ax, cfg in zip(axes, ("ee_resnet18_kd", "ee_vit_tiny")):
        if cfg not in A:
            ax.set_visible(False)
            continue
        seeds, m = A[cfg]["seeds"], A[cfg]["mean"]
        lo = min(x[0] for x in seeds[0]["curves"]["Max-prob"])
        hi = max(x[0] for x in seeds[0]["curves"]["Max-prob"])
        grid = np.linspace(lo, hi, 160)
        for label, (color, ls) in POLICY_STYLE.items():
            mu, sd = mean_curve([s["curves"][label] for s in seeds], grid)
            ax.plot(grid, 100 * mu, color=color, ls=ls, label=label)
            ax.fill_between(grid, 100 * (mu - sd), 100 * (mu + sd), color=color, alpha=0.12, lw=0)
            op = m["policies"][label]["test"]
            ax.plot(op["cost"][0], 100 * op["acc"][0], "o", color=color, ms=6, mec="white", mew=1.2, zorder=5)
        g, acc = m["gflops"], m["test_acc"]
        ax.errorbar([x[0] for x in g], [100 * a[0] for a in acc], yerr=[100 * a[1] for a in acc], fmt="s",
                    color=INK2, ms=4.5, capsize=2, label="Single exit (forced)")
        for i, (c, a) in enumerate(zip(g, acc)):
            ax.annotate(f"E{i + 1}", (c[0], 100 * a[0]), textcoords="offset points", xytext=(5, -10), fontsize=7,
                        color=INK2)
        if cfg == "ee_resnet18_kd":
            ax.errorbar(base["gflops"][0][0], 100 * base["test_acc"][0][0], yerr=100 * base["test_acc"][0][1],
                        fmt="*", color=INK, ms=11, capsize=2, label="ResNet-18 full inference")
        ax.set_title(f"{NAMES[cfg]} (mean of {len(seeds)} seeds)")
        ax.set_xlabel("Mean GFLOPs per image (test)")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.legend(fontsize=7, loc="lower right")
    savefig(fig, out, "tradeoff.png")


def fig_ablation(A, out):
    cfgs = [c for c in ("ee_resnet18_kd", "ee_resnet18_ce", "ee_resnet18_pool") if c in A]
    colors = {"ee_resnet18_kd": BLUE, "ee_resnet18_ce": ORANGE, "ee_resnet18_pool": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    w = 0.8 / len(cfgs)
    for j, cfg in enumerate(cfgs):
        acc = A[cfg]["mean"]["test_acc"]
        x = np.arange(len(acc)) + (j - (len(cfgs) - 1) / 2) * w
        axes[0].bar(x, [100 * a[0] for a in acc], w * 0.92, yerr=[100 * a[1] for a in acc], capsize=2,
                    color=colors[cfg], label=NAMES[cfg])
    axes[0].set_xticks(range(4), [f"Exit {i + 1}" for i in range(4)])
    axes[0].set_ylim(40, None)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("Accuracy of each exit")
    axes[0].legend(fontsize=7, loc="lower right")
    base = A["resnet18"]["mean"]
    for cfg in cfgs:
        seeds = A[cfg]["seeds"]
        grid = np.linspace(min(x[0] for x in seeds[0]["curves"]["Max-prob"]),
                           max(x[0] for x in seeds[0]["curves"]["Max-prob"]), 160)
        mu, _ = mean_curve([s["curves"]["Max-prob"] for s in seeds], grid)
        axes[1].plot(grid, 100 * mu, color=colors[cfg], label=NAMES[cfg])
    axes[1].plot(base["gflops"][0][0], 100 * base["test_acc"][0][0], "*", color=INK, ms=11, label="ResNet-18 full")
    axes[1].set_xlabel("Mean GFLOPs per image (test)")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_title("Max-prob exiting")
    axes[1].set_ylim(60, None)
    axes[1].legend(fontsize=7, loc="lower right")
    savefig(fig, out, "ablation_training_heads.png")


def fig_calibration(A, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    for ax, cfg in zip(axes, ("ee_resnet18_kd", "ee_vit_tiny")):
        if cfg not in A:
            continue
        m = A[cfg]["mean"]
        raw, cal = m["ece_test"]["raw"], m["ece_test"]["calibrated"]
        x = np.arange(len(raw))
        ax.bar(x - 0.18, [100 * v[0] for v in raw], 0.34, yerr=[100 * v[1] for v in raw], capsize=2, color=BLUE,
               label="Raw softmax")
        ax.bar(x + 0.18, [100 * v[0] for v in cal], 0.34, yerr=[100 * v[1] for v in cal], capsize=2, color=ORANGE,
               label="Temperature scaled")
        for i, t in enumerate(m["temperature"]):
            ax.annotate(f"T={t[0]:.2f}", (i, 100 * max(raw[i][0] + raw[i][1], cal[i][0] + cal[i][1])),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"Exit {i + 1}" for i in x])
        ax.set_ylabel("ECE on test (%)")
        ax.set_title(NAMES[cfg])
        ax.legend(fontsize=7.5)
    savefig(fig, out, "calibration_ece.png")


def fig_thresholds(A, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.1))
    for ax, cfg in zip(axes, ("ee_resnet18_kd", "ee_vit_tiny")):
        if cfg not in A:
            continue
        P = A[cfg]["mean"]["policies"]
        names = list(P)
        x = np.arange(len(names))
        g = [P[lab]["test"]["cost"] for lab in names]
        pe = [P[lab]["per_exit"]["test"]["cost"] for lab in names]
        ax.bar(x - 0.18, [v[0] for v in g], 0.34, yerr=[v[1] for v in g], capsize=2, color=BLUE,
               label="One global threshold")
        ax.bar(x + 0.18, [v[0] for v in pe], 0.34, yerr=[v[1] for v in pe], capsize=2, color=ORANGE,
               label="Per-exit thresholds")
        ax.set_xticks(x, names, fontsize=7.5)
        ax.set_ylabel("Mean GFLOPs at selected point")
        ax.set_title(f"{NAMES[cfg]}: cost at the validation-selected point", fontsize=9)
        ax.legend(fontsize=7)
    savefig(fig, out, "threshold_tuning.png")


def _corr_mean(seeds, fn):
    return [np.mean([np.nanmean([fn(s, c, sev) for c in CORRUPTIONS]) for s in seeds]) for sev in SEVERITIES]


def fig_robustness(A, out):
    lines = [("resnet18", None, INK, "-", "*"), ("ee_resnet18_kd", "Max-prob", BLUE, "-", "o"),
             ("ee_resnet18_kd", "Entropy + TS", YELLOW_SAFE, ":", "o"), ("ee_vit_tiny", "Max-prob", VIOLET, "--", "^")]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
    for ax, c in zip(axes.flat, CORRUPTIONS):
        for cfg, label, color, ls, mk in lines:
            if cfg not in A:
                continue
            seeds = A[cfg]["seeds"]
            if label is None:
                ys = [np.mean([s["corrupt_acc"][c][sev] for s in seeds]) for sev in SEVERITIES]
                clean = np.mean([s["test_acc"][0] for s in seeds])
                name = NAMES[cfg]
            else:
                ys = [np.mean([s["corrupt"][c][sev][label]["acc"] for s in seeds]) for sev in SEVERITIES]
                clean = np.mean([s["policies"][label]["test"]["acc"] for s in seeds])
                name = f"{NAMES[cfg]}, {label}"
            ax.plot(range(6), [100 * clean, *[100 * y for y in ys]], marker=mk, color=color, ls=ls, ms=4, label=name)
        ax.set_title(CNAMES[c])
    for ax in axes[1]:
        ax.set_xlabel("Severity (0 = clean)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 accuracy (%)")
    axes[0, 0].legend(fontsize=6.5)
    savefig(fig, out, "robustness_per_corruption.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for cfg, mk in (("ee_resnet18_kd", "o"), ("ee_vit_tiny", "^")):
        if cfg not in A:
            continue
        seeds = A[cfg]["seeds"]
        for label in ("Max-prob", "Entropy + TS"):
            color, ls = POLICY_STYLE[label]
            clean = [np.mean([s["policies"][label]["test"][key] for s in seeds]) for key in ("acc", "cost")]
            acc = _corr_mean(seeds, lambda s, c, v: s["corrupt"][c][v][label]["acc"])
            cost = _corr_mean(seeds, lambda s, c, v: s["corrupt"][c][v][label]["cost"])
            early = _corr_mean(seeds, lambda s, c, v: s["corrupt"][c][v][label]["early_acc"] or np.nan)
            name = f"{NAMES[cfg]}, {label}"
            axes[0].plot(range(6), [100 * clean[0], *[100 * a for a in acc]], marker=mk, color=color, ls=ls, label=name)
            axes[1].plot(range(6), [clean[1], *cost], marker=mk, color=color, ls=ls, label=name)
            axes[2].plot(range(1, 6), [100 * e for e in early], marker=mk, color=color, ls=ls, label=name)
    seeds = A["resnet18"]["seeds"]
    base = [np.mean([s["test_acc"][0] for s in seeds])] + _corr_mean(seeds, lambda s, c, v: s["corrupt_acc"][c][v])
    axes[0].plot(range(6), [100 * b for b in base], "*-", color=INK, ms=7, label="ResNet-18 full")
    axes[1].axhline(A["resnet18"]["mean"]["gflops"][0][0], color=INK, lw=0.9, ls="--")
    axes[1].text(0, A["resnet18"]["mean"]["gflops"][0][0], " ResNet-18 full", va="bottom", fontsize=7, color=INK2)
    axes[0].set(xlabel="Severity (0 = clean)", ylabel="Top-1 accuracy (%)", title="Accuracy, mean over 6 corruptions")
    axes[1].set(xlabel="Severity (0 = clean)", ylabel="Mean GFLOPs per image", title="Compute spent")
    axes[2].set(xlabel="Severity", ylabel="Accuracy of early-exited images (%)", title="Reliability of early exits")
    axes[0].legend(fontsize=6.5)
    savefig(fig, out, "robustness_summary.png")


def fig_exit_shift(A, out):
    if "ee_resnet18_kd" not in A:
        return
    seeds = A["ee_resnet18_kd"]["seeds"]
    rows = []
    for label in ("Max-prob", "Entropy + TS"):
        rows.append((f"{label}: clean", np.mean([s["policies"][label]["test"]["exit_frac"] for s in seeds], 0)))
        for sev in ("1", "3", "5"):
            frac = np.mean([[s["corrupt"][c][sev][label]["exit_frac"] for c in CORRUPTIONS] for s in seeds], (0, 1))
            rows.append((f"{label}: severity {sev}", frac))
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(rows) + 1.0))
    for j, (lab, frac) in enumerate(rows):
        left = 0
        for i, f in enumerate(frac):
            ax.barh(j, 100 * f, left=left, color=EXIT_RAMP[i], edgecolor="white", linewidth=1,
                    label=f"Exit {i + 1}" if j == 0 else None)
            if f > 0.06:
                ax.text(left + 50 * f, j, f"{100 * f:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if i > 0 else INK)
            left += 100 * f
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of test images (%), EE-ResNet-18, mean over corruptions and seeds")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    savefig(fig, out, "exit_shift.png")


def fig_corruption_ece(A, out):
    if "ee_resnet18_kd" not in A:
        return
    seeds = A["ee_resnet18_kd"]["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), sharey=True)
    for ax, key, title in ((axes[0], "raw", "Raw softmax"), (axes[1], "calibrated", "Temperature scaled (clean val)")):
        for i in range(4):
            clean = np.mean([s["ece_test"][key][i] for s in seeds])
            ys = _corr_mean(seeds, lambda s, c, v: s["corrupt_ece"][c][v][key][i])
            ax.plot(range(6), [100 * clean, *[100 * y for y in ys]], "o-", color=EXIT_RAMP[i], label=f"Exit {i + 1}")
        ax.set_title(title)
        ax.set_xlabel("Severity (0 = clean)")
    axes[0].set_ylabel("ECE (%), mean over corruptions")
    axes[0].legend(fontsize=7.5)
    savefig(fig, out, "corruption_ece.png")


def latency_rows(A):
    """Measured latency of each seed-0 model at its validation-selected max-prob point."""
    rows = []
    for cfg in ("resnet18", "ee_resnet18_kd", "ee_resnet18_pool", "ee_resnet18_ce", "ee_vit_tiny"):
        lat = next((s.get("latency") for s in A.get(cfg, {}).get("raw", []) if s["seed"] == 0), None)
        if not lat:
            continue
        for dev in ("gpu", "cpu1", "cpu4"):
            if dev not in lat:
                continue
            row = {"cfg": cfg, "device": dev, "device_name": lat[dev]["device"], "full_ms": lat[dev]["forced_ms"][-1],
                   "forced_ms": lat[dev]["forced_ms"]}
            for a in lat[dev]["adaptive"]:
                key = {"strict": "ops", "relaxed": "ops_relaxed"}.get(a.get("point"))
                if key:
                    row.setdefault(key, {})[a["label"]] = a
            rows.append(row)
        if "gpu" in lat and "batched" in lat["gpu"]:
            b = lat["gpu"]["batched"]
            row = {"cfg": cfg, "device": "gpu_batched", "full_tp": b["full"]}
            for a in b["adaptive"]:
                key = {"strict": "ops", "relaxed": "ops_relaxed"}.get(a.get("point"))
                if key:
                    row.setdefault(key, {})[a["label"]] = a
            rows.append(row)
    return rows


def fig_latency(rows, out):
    if not rows:
        return
    items = [("resnet18", None, "ResNet-18, full", INK2), ("ee_resnet18_kd", None, "EE-ResNet-18, full path", "#86b6ef"),
             ("ee_resnet18_kd", "Max-prob", "EE-ResNet-18, max-prob", BLUE),
             ("ee_resnet18_pool", "Max-prob", "EE-ResNet-18 pooling heads, max-prob", AQUA),
             ("ee_vit_tiny", None, "EE-ViT-Tiny, full path", "#9085e9"),
             ("ee_vit_tiny", "Max-prob", "EE-ViT-Tiny, max-prob", VIOLET)]
    devs = [("gpu", "GPU (T4), batch 1"), ("cpu1", "CPU, 1 thread, batch 1"), ("cpu4", "CPU, 4 threads, batch 1")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
    for ax, (dev, title) in zip(axes, devs):
        rs = {r["cfg"]: r for r in rows if r["device"] == dev}
        if "resnet18" not in rs:
            continue
        base = rs["resnet18"]["full_ms"]
        bars = []
        for cfg, label, name, color in items:
            if cfg in rs:
                v = rs[cfg]["full_ms"] if label is None else rs[cfg].get("ops", {}).get(label, {}).get("ms")
                if v:
                    bars.append((name, v, color))
        y = np.arange(len(bars))
        ax.barh(y, [b[1] for b in bars], 0.7, color=[b[2] for b in bars])
        for i, b in enumerate(bars):
            ax.text(b[1], i, f" {b[1]:.1f} ms, {base / b[1]:.2f}x", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(y, [b[0] for b in bars], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(b[1] for b in bars) * 1.45)
        ax.set_xlabel("ms per image")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    savefig(fig, out, "latency.png")


def fig_training(A, out):
    cfgs = [c for c in NAMES if c in A]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(2.6 * len(cfgs), 2.6), sharey=True)
    for ax, cfg in zip(np.atleast_1d(axes), cfgs):
        h = next(s for s in A[cfg]["raw"] if s["seed"] == 0)["history"]
        k = len(h[0]["val_acc"])
        for i in range(k):
            ax.plot([x["epoch"] for x in h], [100 * x["val_acc"][i] for x in h], color=INK if k == 1 else EXIT_RAMP[i],
                    lw=1.2, label="Final exit" if k == 1 else f"Exit {i + 1}")
        ax.set_title(NAMES[cfg].replace(", ", ",\n"), fontsize=8.5)
        ax.set_xlabel("Epoch")
    np.atleast_1d(axes)[0].set_ylabel("Validation accuracy (%)")
    np.atleast_1d(axes)[-1].legend(fontsize=6.5, loc="lower right")
    savefig(fig, out, "training_curves.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/final")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    runs = discover(args.runs)
    print({k: [r["seed"] for r in v] for k, v in runs.items()}, flush=True)
    base_cost = runs["resnet18"][0]["flops"]["cumulative"][0] / 1e9
    base_acc = float(np.mean([r["acc"]["test"][0] for r in runs["resnet18"]]))
    A = {}
    for cfg, rs in runs.items():
        vit = cfg == "ee_vit_tiny"
        ref = rs[0]["flops"]["cumulative"][-1] / 1e9 if vit else base_cost
        seeds = [analyze_run(r, ref, None if vit else base_acc) for r in rs]
        A[cfg] = {"raw": rs, "seeds": seeds, "mean": aggregate(seeds), "n": len(seeds)}
        print("analyzed", cfg, flush=True)
    lat = latency_rows(A)
    for name in ("dataset_samples.png", "corruption_examples.png"):
        src = os.path.join("results", "interim", "figures", name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, "figures", name))
    fig_training(A, args.out)
    fig_tradeoff(A, args.out)
    fig_ablation(A, args.out)
    fig_calibration(A, args.out)
    fig_thresholds(A, args.out)
    fig_robustness(A, args.out)
    fig_exit_shift(A, args.out)
    fig_corruption_ece(A, args.out)
    fig_latency(lat, args.out)
    summary = {cfg: {"n": v["n"], "mean": {k: x for k, x in v["mean"].items() if k != "curves"}}
               for cfg, v in A.items()}
    summary["latency"] = lat
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
