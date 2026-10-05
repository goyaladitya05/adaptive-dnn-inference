"""Follow-up study: distillation dose-response, architecture vs training recipe, and calibration vs routing.

Reads runs/final-*/ and runs/study-*/ (re-evaluated corrupted validation logits live in reeval_<run>/),
writes results/study/.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import AQUA, BLUE, INK2, ORANGE, VIOLET, plt, savefig  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.corruptions import CORRUPTIONS  # noqa: E402
from src.exits import (exit_scores, fit_temperature, pick_threshold, reliability, route,  # noqa: E402
                       routing_stats, sweep, tune_per_exit)

SEV = ("1", "2", "3", "4", "5")
# config -> (architecture, recipe, distillation weight)
CONFIGS = {
    "ee_resnet18_ce": ("CNN", "plain", 0.0), "ee_resnet18_kd25": ("CNN", "plain", 0.25),
    "ee_resnet18_kd": ("CNN", "plain", 0.5), "ee_resnet18_kd75": ("CNN", "plain", 0.75),
    "ee_resnet18_strong": ("CNN", "strong", 0.5), "ee_resnet18_ls": ("CNN", "label smoothing", 0.5),
    "ee_resnet18_mix": ("CNN", "mixup", 0.5), "ee_resnet18_ra": ("CNN", "RandAugment", 0.5),
    "ee_resnet18_pool": ("CNN pool", "plain", 0.5),
    "ee_vit_tiny": ("ViT", "strong", 0.5), "ee_vit_tiny_ce": ("ViT", "strong", 0.0),
    "ee_vit_tiny_plain": ("ViT", "plain", 0.5),
}


def load(d, name):
    return torch.from_numpy(np.load(os.path.join(d, "logits", f"{name}.npy")).astype(np.float32))


def discover(root):
    runs = {}
    reeval = {os.path.basename(d)[len("reeval_"):]: d for d in glob.glob(os.path.join(root, "*", "reeval_*"))}
    for m in sorted(glob.glob(os.path.join(root, "final-*", "*", "metrics.json"))
                    + glob.glob(os.path.join(root, "study-*", "*", "metrics.json"))):
        d = os.path.dirname(m)
        name = os.path.basename(d)
        cfg, seed = re.match(r"(.+)_s(\d+)$", name).groups()
        if cfg not in CONFIGS:
            continue
        with open(m) as f:
            r = json.load(f)
        val_dir = d if os.path.exists(os.path.join(d, "logits", "val_contrast_5.npy")) else reeval.get(name)
        r.update(dir=d, val_dir=val_dir, name=name, cfg=cfg, seed=int(seed))
        runs.setdefault(cfg, []).append(r)
    return runs


def auroc(score, correct):
    """Probability that a correct prediction scores higher than a wrong one."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = correct.sum(), (~correct).sum()
    return float((ranks[correct].sum() - pos * (pos + 1) / 2) / max(pos * neg, 1))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def strict_point(sv, cv, st, ct, costs, target, per_exit=False):
    th = pick_threshold(sweep(sv, cv, costs), target)
    if per_exit:
        th, _ = tune_per_exit(sv, cv, costs, target, th)
    return th, routing_stats(route(st, th), ct, costs)


def shift_metrics(lg, y, temps, th, policy, cal, costs):
    k = lg.shape[0]
    cc = lg.argmax(-1).numpy() == y
    ex = route(exit_scores(lg, policy, cal, temps), th)
    m = routing_stats(ex, cc, costs)
    early = ex < k - 1
    m["early_frac"] = float(early.mean())
    m["early_acc"] = float(cc[ex, np.arange(len(ex))][early].mean()) if early.any() else float("nan")
    return m


def analyze_run(r):
    d, k = r["dir"], len(r["acc"]["test"])
    costs = [c / 1e9 for c in r["flops"]["cumulative"]]
    lv, lt = load(d, "val"), load(d, "test")
    yv, yt = np.load(os.path.join(d, "logits", "val_labels.npy")), np.load(os.path.join(d, "logits", "test_labels.npy"))
    cv, ct = lv.argmax(-1).numpy() == yv, lt.argmax(-1).numpy() == yt
    temps = r["temperature"]
    target = float(cv[-1].mean())  # from the stored fp16 logits, so the final exit always meets it
    out = {"seed": r["seed"], "test_acc": r["acc"]["test"], "temperature": temps,
           "log_t_early": float(np.mean(np.log(temps[:-1]))), "ece_raw": r["ece_test"]["raw"],
           "ece_ts": r["ece_test"]["calibrated"]}
    pts = {}
    for label, policy, cal, per in (("mp", "max_prob", False, False), ("mp_ts", "max_prob", True, False),
                                    ("ent_ts", "entropy", True, False), ("mp_pe", "max_prob", False, True),
                                    ("mp_ts_pe", "max_prob", True, True)):
        sv, st = exit_scores(lv, policy, cal, temps), exit_scores(lt, policy, cal, temps)
        th, res = strict_point(sv, cv, st, ct, costs, target, per)
        pts[label] = {"thresh": th, **res}
    out["points"] = pts
    out["ts_cost_effect"] = pts["mp_ts"]["cost"] - pts["mp"]["cost"]
    raw = [torch.softmax(lt[i], -1).amax(-1).numpy() for i in range(k)]
    ts = [torch.softmax(lt[i] / temps[i], -1).amax(-1).numpy() for i in range(k)]
    out["auroc_raw"] = [auroc(raw[i], ct[i]) for i in range(k)]
    out["auroc_ts"] = [auroc(ts[i], ct[i]) for i in range(k)]
    out["spearman_raw_ts"] = [spearman(raw[i], ts[i]) for i in range(k)]
    out["agree_final_clean"] = [float((lt[i].argmax(-1) == lt[-1].argmax(-1)).float().mean()) for i in range(k - 1)]

    shift = {key: {s: [] for s in SEV} for key in ("mp", "mp_ts", "ent_ts", "mp_loco")}
    conf_gap = {s: [] for s in SEV}
    agree = {s: [] for s in SEV}
    ece_shift = {key: {s: [] for s in SEV} for key in ("raw", "ts", "loco")}
    clean_loco = []
    for ci, c in enumerate(CORRUPTIONS):
        loco = None
        if r["val_dir"]:
            pooled = [load(r["val_dir"], f"val_{o}_{s}") for o in CORRUPTIONS if o != c for s in SEV]
            lp = torch.cat([lv] + pooled, 1)
            yp = np.concatenate([yv] * (1 + len(pooled)))
            loco = [fit_temperature(lp[i], torch.from_numpy(yp)) for i in range(k)]
            sv = exit_scores(lv, "max_prob", True, loco)
            th_loco = pick_threshold(sweep(sv, cv, costs), target)
            clean_loco.append(routing_stats(route(exit_scores(lt, "max_prob", True, loco), th_loco), ct, costs))
        for s in SEV:
            lg = load(d, f"{c}_{s}")
            for key, policy, cal, tt in (("mp", "max_prob", False, temps), ("mp_ts", "max_prob", True, temps),
                                         ("ent_ts", "entropy", True, temps)):
                shift[key][s].append(shift_metrics(lg, yt, tt, pts[key]["thresh"], policy, cal, costs))
            if loco:
                shift["mp_loco"][s].append(shift_metrics(lg, yt, loco, th_loco, "max_prob", True, costs))
            p1 = torch.softmax(lg[0], -1)
            conf_gap[s].append(float(p1.amax(-1).mean() - (p1.argmax(-1).numpy() == yt).mean()))
            agree[s].append(float((lg[0].argmax(-1) == lg[-1].argmax(-1)).float().mean()))
            yt_t = torch.from_numpy(yt)
            ece_shift["raw"][s].append([reliability(lg[i], yt_t)[1] for i in range(k)])
            ece_shift["ts"][s].append([reliability(lg[i], yt_t, temps[i])[1] for i in range(k)])
            if loco:
                ece_shift["loco"][s].append([reliability(lg[i], yt_t, loco[i])[1] for i in range(k)])
    avg = lambda rows, key: float(np.nanmean([x[key] for x in rows]))  # noqa: E731
    out["shift"] = {key: {s: {m: avg(v[s], m) for m in ("acc", "cost", "early_frac", "early_acc")} for s in SEV}
                    for key, v in shift.items() if v["1"]}
    if clean_loco:
        out["clean_loco"] = {m: float(np.mean([x[m] for x in clean_loco])) for m in ("acc", "cost")}
    out["conf_gap_exit1"] = {s: float(np.mean(v)) for s, v in conf_gap.items()}
    out["agree_exit1_final"] = {s: float(np.mean(v)) for s, v in agree.items()}
    out["ece_shift"] = {key: {s: np.mean(v[s], 0).tolist() for s in SEV} for key, v in ece_shift.items() if v["1"]}
    return out


def fig_dose(A, out):
    alphas = [(c, CONFIGS[c][2]) for c in ("ee_resnet18_ce", "ee_resnet18_kd25", "ee_resnet18_kd", "ee_resnet18_kd75")
              if c in A]
    if len(alphas) < 2:
        return
    panels = [("Exit 1 accuracy (%)", lambda R: [100 * r["test_acc"][0] for r in R]),
              ("Clean cost at main point (GFLOPs)", lambda R: [r["points"]["mp"]["cost"] for r in R]),
              ("Early exits at severity 5 (%)", lambda R: [100 * r["shift"]["mp"]["5"]["early_frac"] for r in R]),
              ("Accuracy of early exits, severity 5 (%)", lambda R: [100 * r["shift"]["mp"]["5"]["early_acc"] for r in R]),
              ("Exit 1 confidence minus accuracy, sev. 5", lambda R: [100 * r["conf_gap_exit1"]["5"] for r in R])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.8))
    for ax, (title, fn) in zip(axes, panels):
        xs = [a for _, a in alphas]
        ms = [np.mean(fn(A[c])) for c, _ in alphas]
        ss = [np.std(fn(A[c])) for c, _ in alphas]
        ax.errorbar(xs, ms, yerr=ss, fmt="o-", color=BLUE, capsize=3)
        for (c, a), m in zip(alphas, ms):
            ax.annotate(f"n={len(A[c])}", (a, m), textcoords="offset points", xytext=(4, 6), fontsize=6.5, color=INK2)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Distillation weight alpha")
        ax.set_xticks(xs)
    savefig(fig, out, "lead1_distillation_dose.png")


def fig_temperature_scatter(A, out):
    style = {"CNN": (BLUE, "o"), "ViT": (VIOLET, "^"), "CNN pool": (AQUA, "s")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for cfg, runs in A.items():
        arch, recipe, alpha = CONFIGS[cfg]
        color, mk = style[arch]
        face = "white" if recipe == "plain" else color
        x = [r["log_t_early"] for r in runs]
        axes[0].scatter(x, [r["ts_cost_effect"] for r in runs], marker=mk, s=40, edgecolor=color, facecolor=face,
                        linewidth=1.4, label=f"{arch}, {recipe}, alpha {alpha}")
        y = [100 * r["shift"]["mp"]["5"]["early_acc"] for r in runs]
        axes[1].scatter(x, y, marker=mk, s=40, edgecolor=color, facecolor=face, linewidth=1.4)
    for ax in axes:
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_xlabel("Mean log temperature of early exits (> 0: overconfident)")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("Cost change from temperature scaling (GFLOPs)")
    axes[0].set_title("Lead 3: effect of calibration on routing cost", fontsize=9)
    axes[1].set_ylabel("Accuracy of early exits at severity 5 (%)")
    axes[1].set_title("Lead 2: confidence state vs reliability under shift", fontsize=9)
    axes[0].legend(fontsize=6, loc="best")
    savefig(fig, out, "lead23_temperature.png")


def fig_two_by_two(A, out):
    cells = [("ee_resnet18_kd", "CNN, plain"), ("ee_resnet18_strong", "CNN, strong"),
             ("ee_vit_tiny_plain", "ViT, plain"), ("ee_vit_tiny", "ViT, strong")]
    cells = [(c, n) for c, n in cells if c in A]
    if len(cells) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, (cfg, name) in enumerate(cells):
        runs = A[cfg]
        color = [BLUE, ORANGE, VIOLET, AQUA][i]
        for ax, key in zip(axes[:2], ("early_frac", "early_acc")):
            ys = [[100 * r["shift"]["mp"][s][key] for s in SEV] for r in runs]
            ax.errorbar(range(1, 6), np.mean(ys, 0), yerr=np.std(ys, 0), fmt="o-", color=color, capsize=2, label=name)
        ys = [[100 * r["shift"]["mp"][s]["acc"] for s in SEV] for r in runs]
        axes[2].errorbar(range(1, 6), np.mean(ys, 0), yerr=np.std(ys, 0), fmt="o-", color=color, capsize=2, label=name)
    axes[0].set(title="Images exiting early (max-prob)", ylabel="%", xlabel="Severity")
    axes[1].set(title="Accuracy of early-exited images", ylabel="%", xlabel="Severity")
    axes[2].set(title="Overall accuracy", ylabel="%", xlabel="Severity")
    axes[2].legend(fontsize=7)
    savefig(fig, out, "lead2_architecture_recipe.png")


def fig_loco(A, out):
    cfgs = [c for c in A if "mp_loco" in A[c]["seeds"][0]["shift"]]
    if not cfgs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    x = np.arange(len(cfgs))
    for j, (key, label, color) in enumerate((("mp", "Raw", BLUE), ("mp_ts", "T from clean val", ORANGE),
                                            ("mp_loco", "T from other corruptions", AQUA))):
        ys = [[100 * r["shift"][key]["5"]["early_acc"] for r in A[c]["seeds"]] for c in cfgs]
        axes[0].bar(x + (j - 1) * 0.27, [np.mean(v) for v in ys], 0.26, yerr=[np.std(v) for v in ys], capsize=2,
                    color=color, label=label)
        ys = [[r["shift"][key]["5"]["cost"] for r in A[c]["seeds"]] for c in cfgs]
        axes[1].bar(x + (j - 1) * 0.27, [np.mean(v) for v in ys], 0.26, yerr=[np.std(v) for v in ys], capsize=2,
                    color=color, label=label)
    for ax in axes:
        ax.set_xticks(x, [c.replace("ee_", "").replace("_", " ") for c in cfgs], fontsize=6.5, rotation=20)
    axes[0].set(ylabel="Accuracy of early exits, sev. 5 (%)", title="Reliability of early exits")
    axes[1].set(ylabel="GFLOPs at severity 5", title="Computation spent")
    axes[0].legend(fontsize=7)
    savefig(fig, out, "lead3_shift_temperature.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/study")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    runs = discover(args.runs)
    print({c: sorted(r["seed"] for r in v) for c, v in runs.items()}, flush=True)
    A = {}
    for cfg, rs in runs.items():
        seeds = [analyze_run(r) for r in rs]
        A[cfg] = {"seeds": seeds, "n": len(seeds), "arch": CONFIGS[cfg][0], "recipe": CONFIGS[cfg][1],
                  "alpha": CONFIGS[cfg][2]}
        print("analyzed", cfg, len(seeds), flush=True)
    fig_dose({c: v["seeds"] for c, v in A.items()}, args.out)
    fig_temperature_scatter({c: v["seeds"] for c, v in A.items()}, args.out)
    fig_two_by_two({c: v["seeds"] for c, v in A.items()}, args.out)
    fig_loco(A, args.out)
    xs = [r["log_t_early"] for v in A.values() for r in v["seeds"] if v["arch"] != "CNN pool"]
    ys = [r["ts_cost_effect"] for v in A.values() for r in v["seeds"] if v["arch"] != "CNN pool"]
    zs = [r["shift"]["mp"]["5"]["early_acc"] for v in A.values() for r in v["seeds"] if v["arch"] != "CNN pool"]
    summary = {"configs": A, "correlations": {
        "n_runs": len(xs),
        "spearman_logT_vs_ts_cost_effect": spearman(np.array(xs), np.array(ys)) if len(xs) > 2 else None,
        "spearman_logT_vs_early_acc_sev5": spearman(np.array(xs), np.array(zs)) if len(xs) > 2 else None}}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary["correlations"]), flush=True)


if __name__ == "__main__":
    main()
