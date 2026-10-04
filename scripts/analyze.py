"""Offline analysis of the Kaggle runs: threshold sweeps, calibration, corruptions, latency, figures.

Reads runs/<job>/<model>/{metrics.json,history.json,logits/*.npy}, writes results/.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.corruptions import CORRUPTIONS, corrupt  # noqa: E402
from src.exits import SCORES, reliability  # noqa: E402

RUNS = {"resnet18": "runs/resnets/resnet18", "ee_resnet18": "runs/resnets/ee_resnet18",
        "ee_vit_tiny": "runs/vit/ee_vit_tiny"}
NAMES = {"resnet18": "ResNet-18 (full)", "ee_resnet18": "EE-ResNet-18", "ee_vit_tiny": "EE-ViT-Tiny"}
POLICIES = [("max_prob", False, "Max-prob"), ("entropy", False, "Entropy"),
            ("max_prob", True, "Max-prob + TS"), ("entropy", True, "Entropy + TS")]
SEVERITIES = (1, 2, 3, 4, 5)
TOL = 0.01

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POLICY_STYLE = {"Max-prob": (BLUE, "-"), "Entropy": (AQUA, "--"), "Max-prob + TS": (ORANGE, "-."),
                "Entropy + TS": (YELLOW, ":")}
EXIT_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
    "font.family": "DejaVu Sans",
})


def load_run(name):
    d = RUNS[name]
    with open(os.path.join(d, "metrics.json")) as f:
        m = json.load(f)
    with open(os.path.join(d, "history.json")) as f:
        m["history"] = json.load(f)
    m["dir"] = d
    return m


def logits(run, split):
    return torch.from_numpy(np.load(os.path.join(run["dir"], "logits", f"{split}.npy")).astype(np.float32))


def labels(run, split):
    return torch.from_numpy(np.load(os.path.join(run["dir"], "logits", f"{split}_labels.npy")))


def scores_for(lg, policy, calibrated, temps):
    k = lg.shape[0]
    return np.stack([SCORES[policy](lg[i], temps[i] if calibrated else 1.0).numpy() for i in range(k - 1)])


def route(scores, th):
    """First exit whose score clears the threshold, else the last exit."""
    hit = np.vstack([scores >= th, np.ones((1, scores.shape[1]), bool)])
    return hit.argmax(0)


def evaluate_routing(exit_at, correct, costs):
    k = correct.shape[0]
    return {"acc": float(correct[exit_at, np.arange(correct.shape[1])].mean()),
            "cost": float(np.asarray(costs)[exit_at].mean()),
            "exit_frac": np.bincount(exit_at, minlength=k).astype(float).__truediv__(len(exit_at)).tolist()}


THRESH = np.unique(np.concatenate([np.linspace(0, 0.9, 91), np.linspace(0.9, 1.0, 201), [1.01]]))


def sweep(lg, y, costs, policy, calibrated, temps):
    s = scores_for(lg, policy, calibrated, temps)
    correct = lg.argmax(-1).eq(y).numpy()
    return [dict(thresh=float(t), **evaluate_routing(route(s, t), correct, costs)) for t in THRESH]


def pick_threshold(val_sweep, target):
    ok = [r for r in val_sweep if r["acc"] >= target]
    return min(ok, key=lambda r: (r["cost"], -r["acc"]))["thresh"]


def acc_at_budget(sw, budget):
    ok = [r["acc"] for r in sw if r["cost"] <= budget + 1e-9]
    return max(ok) if ok else float("nan")


def oracle(lg, y, costs):
    correct = lg.argmax(-1).eq(y).numpy()
    k = correct.shape[0]
    first = np.where(correct.any(0), correct.argmax(0), k - 1)
    return {"acc": float(correct.any(0).mean()), "cost": float(np.asarray(costs)[first].mean())}


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gflops(run):
    return [c / 1e9 for c in run["flops"]["cumulative"]]


def analyze_policies(runs, out, summary):
    base = runs["resnet18"]
    base_acc = base["acc"]["test"][0]
    base_cost = gflops(base)[0]
    summary["baseline"] = {"test_acc": base_acc, "gflops": base_cost, "params": base["params"]}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, name in zip(axes, ("ee_resnet18", "ee_vit_tiny")):
        if name not in runs:
            ax.set_visible(False)
            continue
        r = runs[name]
        costs, temps = gflops(r), r["temperature"]
        lv, yv, lt, yt = logits(r, "val"), labels(r, "val"), logits(r, "test"), labels(r, "test")
        target = r["acc"]["val"][-1] - TOL
        s = summary.setdefault(name, {})
        s.update(params=r["params"], gflops=costs, test_acc=r["acc"]["test"], val_acc=r["acc"]["val"],
                 temperature=temps, ece_test=r["ece_test"], oracle=oracle(lt, yt, costs), policies={})
        for policy, cal, label in POLICIES:
            sw_v = sweep(lv, yv, costs, policy, cal, temps)
            sw_t = sweep(lt, yt, costs, policy, cal, temps)
            th = pick_threshold(sw_v, target)
            op = next(x for x in sw_t if x["thresh"] == th)
            s["policies"][label] = {
                "policy": policy, "calibrated": cal, "thresh": th, "test": op,
                "acc_at": {f"{b:.2f}": acc_at_budget(sw_t, b) for b in (0.4, 0.55, 0.7, 0.85)},
                "fixed": {f"{t}": next(x for x in sw_t if abs(x["thresh"] - t) < 1e-9)
                          for t in (0.5, 0.8, 0.9, 0.95)},
            }
            pts = sorted((x["cost"], x["acc"]) for x in sw_t)
            color, ls = POLICY_STYLE[label]
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=color, ls=ls, label=label)
            ax.plot(op["cost"], 100 * op["acc"], "o", color=color, ms=6, mec="white", mew=1.2, zorder=5)
        ax.plot(costs, [100 * a for a in r["acc"]["test"]], "s", color=INK2, ms=5, label="Single exit (forced)")
        for i, (c, a) in enumerate(zip(costs, r["acc"]["test"])):
            ax.annotate(f"E{i + 1}", (c, 100 * a), textcoords="offset points", xytext=(5, -10), fontsize=7, color=INK2)
        ax.plot(base_cost, 100 * base_acc, "*", color=INK, ms=11, label="ResNet-18 full inference")
        ax.set_title(NAMES[name])
        ax.set_xlabel("Mean GFLOPs per image (test)")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.legend(fontsize=7.5, loc="lower right")
    savefig(fig, out, "accuracy_vs_compute.png")


def analyze_calibration(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, name in zip(axes, ("ee_resnet18", "ee_vit_tiny")):
        if name not in runs:
            ax.set_visible(False)
            continue
        e = runs[name]["ece_test"]
        x = np.arange(len(e["raw"]))
        ax.bar(x - 0.18, [100 * v for v in e["raw"]], 0.34, color=BLUE, label="Raw softmax")
        ax.bar(x + 0.18, [100 * v for v in e["calibrated"]], 0.34, color=ORANGE, label="Temperature scaled")
        for i, t in enumerate(runs[name]["temperature"]):
            ax.annotate(f"T={t:.2f}", (i, 100 * max(e["raw"][i], e["calibrated"][i])), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"Exit {i + 1}" for i in x])
        ax.set_ylabel("ECE on test (%)")
        ax.set_title(NAMES[name])
        ax.legend(fontsize=7.5)
    savefig(fig, out, "calibration_ece.png")

    r = runs["ee_resnet18"]
    lt, yt = logits(r, "test"), labels(r, "test")
    k = lt.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(2.4 * k, 2.6), sharey=True)
    for i, ax in enumerate(axes):
        for t, color, lab in ((1.0, BLUE, "Raw"), (r["temperature"][i], ORANGE, "TS")):
            rows, e = reliability(lt[i], yt, t)
            ax.plot([c for c, _, _ in rows], [a for _, a, _ in rows], "o-", color=color, ms=3.5,
                    label=f"{lab} (ECE {100 * e:.1f}%)")
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax.set_title(f"Exit {i + 1}")
        ax.set_xlabel("Confidence")
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("Accuracy")
    savefig(fig, out, "reliability_ee_resnet18.png")


def analyze_exits(summary, out):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    rows = []
    for name in ("ee_resnet18", "ee_vit_tiny"):
        if name in summary:
            for label, p in summary[name]["policies"].items():
                rows.append((f"{NAMES[name]}\n{label}", p["test"]["exit_frac"]))
    for j, (lab, frac) in enumerate(rows):
        left = 0
        for i, f in enumerate(frac):
            ax.barh(j, 100 * f, left=left, color=EXIT_RAMP[i], edgecolor="white", linewidth=1,
                    label=f"Exit {i + 1}" if j == 0 else None)
            if f > 0.06:
                ax.text(left + 50 * f, j, f"{100 * f:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if i > 0 else INK)
            left += 100 * f
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Share of test images (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    savefig(fig, out, "exit_distribution.png")


def analyze_corruptions(runs, out, summary):
    res = {}
    for name in ("resnet18", "ee_resnet18", "ee_vit_tiny"):
        if name not in runs or "corrupt_acc" not in runs[name]:
            continue
        r = runs[name]
        res[name] = {"final_acc": {c: {s: r["corrupt_acc"][c][str(s)][-1] for s in SEVERITIES} for c in CORRUPTIONS}}
        if name == "resnet18":
            continue
        costs, temps = gflops(r), r["temperature"]
        res[name]["policies"] = {}
        for policy, cal, label in POLICIES:
            th = summary[name]["policies"][label]["thresh"]
            per = {}
            for c in CORRUPTIONS:
                per[c] = {}
                for s in SEVERITIES:
                    lg = logits(r, f"{c}_{s}")
                    y = labels(r, "test")
                    correct = lg.argmax(-1).eq(y).numpy()
                    ex = route(scores_for(lg, policy, cal, temps), th)
                    m = evaluate_routing(ex, correct, costs)
                    early = ex < lg.shape[0] - 1
                    m["early_acc"] = float(correct[ex, np.arange(len(ex))][early].mean()) if early.any() else float("nan")
                    m["early_frac"] = float(early.mean())
                    per[c][s] = m
            res[name]["policies"][label] = per
        clean_logits = logits(r, "test")
        res[name]["ece"] = {}
        for c in CORRUPTIONS:
            res[name]["ece"][c] = {}
            for s in SEVERITIES:
                lg = logits(r, f"{c}_{s}")
                y = labels(r, "test")
                res[name]["ece"][c][s] = {
                    "raw": [reliability(lg[k], y, 1.0)[1] for k in range(lg.shape[0])],
                    "calibrated": [reliability(lg[k], y, temps[k])[1] for k in range(lg.shape[0])]}
        del clean_logits
    summary["corruptions"] = res

    def mean_over(fn):
        return [np.mean([fn(c, s) for c in CORRUPTIONS]) for s in SEVERITIES]

    clean = {n: summary["baseline"]["test_acc"] if n == "resnet18" else summary[n]["test_acc"][-1] for n in res}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    sev = [0, *SEVERITIES]
    ax = axes[0]
    ax.plot(sev, [100 * clean["resnet18"], *[100 * v for v in mean_over(lambda c, s: res["resnet18"]["final_acc"][c][s])]],
            "*-", color=INK, ms=7, label="ResNet-18 full")
    for name in ("ee_resnet18", "ee_vit_tiny"):
        if name not in res:
            continue
        for label in ("Max-prob", "Entropy + TS"):
            color, ls = POLICY_STYLE[label]
            p = res[name]["policies"][label]
            clean_acc = summary[name]["policies"][label]["test"]["acc"]
            mk = "o" if name == "ee_resnet18" else "^"
            ax.plot(sev, [100 * clean_acc, *[100 * v for v in mean_over(lambda c, s: p[c][s]["acc"])]], marker=mk,
                    color=color, ls=ls, label=f"{NAMES[name]}, {label}")
    ax.set_xlabel("Severity (0 = clean)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy, mean over 6 corruptions")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    for name in ("ee_resnet18", "ee_vit_tiny"):
        if name not in res:
            continue
        for label in ("Max-prob", "Entropy + TS"):
            color, ls = POLICY_STYLE[label]
            p = res[name]["policies"][label]
            clean_cost = summary[name]["policies"][label]["test"]["cost"]
            mk = "o" if name == "ee_resnet18" else "^"
            ax.plot(sev, [clean_cost, *mean_over(lambda c, s: p[c][s]["cost"])], marker=mk, color=color, ls=ls,
                    label=f"{NAMES[name]}, {label}")
    ax.axhline(summary["baseline"]["gflops"], color=INK, lw=0.9, ls="--")
    ax.text(0, summary["baseline"]["gflops"], " ResNet-18 full", va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("Severity (0 = clean)")
    ax.set_ylabel("Mean GFLOPs per image")
    ax.set_title("Compute spent, mean over 6 corruptions")

    ax = axes[2]
    for name in ("ee_resnet18", "ee_vit_tiny"):
        if name not in res:
            continue
        for label in ("Max-prob", "Entropy + TS"):
            color, ls = POLICY_STYLE[label]
            p = res[name]["policies"][label]
            mk = "o" if name == "ee_resnet18" else "^"
            ax.plot(list(SEVERITIES), [100 * v for v in mean_over(lambda c, s: p[c][s]["early_acc"])], marker=mk,
                    color=color, ls=ls, label=f"{NAMES[name]}, {label}")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy of early-exited images (%)")
    ax.set_title("Reliability of early exits")
    savefig(fig, out, "corruption_robustness.png")

    if "ee_resnet18" in res:
        r = res["ee_resnet18"]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), sharey=True)
        for ax, key, title in ((axes[0], "raw", "Raw softmax"), (axes[1], "calibrated", "Temperature scaled (clean val)")):
            k = len(r["ece"]["gaussian_noise"][1][key])
            for i in range(k):
                ax.plot(list(SEVERITIES), [100 * v for v in mean_over(lambda c, s: r["ece"][c][s][key][i])], "o-",
                        color=EXIT_RAMP[i], label=f"Exit {i + 1}")
            ax.set_title(title)
            ax.set_xlabel("Severity")
        axes[0].set_ylabel("ECE (%), mean over corruptions")
        axes[0].legend(fontsize=7.5)
        savefig(fig, out, "corruption_ece_ee_resnet18.png")


def analyze_latency(runs, out, summary):
    lat = {}
    base = runs["resnet18"].get("latency")
    if not base:
        return
    for name in ("resnet18", "ee_resnet18", "ee_vit_tiny"):
        if name in runs and "latency" in runs[name]:
            lat[name] = runs[name]["latency"]
    summary["latency"] = lat
    r = runs["ee_resnet18"]
    if "latency" not in r:
        return
    base_flops = summary["baseline"]["gflops"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    for ax, dev in zip(axes, ("cpu", "gpu")):
        if dev not in lat["resnet18"]:
            continue
        base_ms = lat["resnet18"][dev]["forced_ms"][0]
        for name, mk in (("ee_resnet18", "o"), ("ee_vit_tiny", "^")):
            if name not in lat or dev not in lat[name]:
                continue
            for label, (policy, cal) in (("Max-prob", ("max_prob", False)), ("Entropy + TS", ("entropy", True))):
                rows = [a for a in lat[name][dev]["adaptive"] if a["policy"] == policy and a["calibrated"] == cal]
                color, ls = POLICY_STYLE[label]
                ax.plot([a["thresh"] for a in rows], [base_ms / a["ms"] for a in rows], marker=mk, color=color, ls=ls,
                        label=f"{NAMES[name]}, {label} (measured)")
                if name == "ee_resnet18":
                    costs = gflops(runs[name])
                    flop_sp = [base_flops / float(np.dot(a["exit_frac"], costs)) for a in rows]
                    ax.plot([a["thresh"] for a in rows], flop_sp, marker=mk, color=color, ls=ls, alpha=0.35,
                            label=f"{NAMES[name]}, {label} (FLOPs ratio)")
        ax.axhline(1.0, color=INK, lw=0.9, ls="--")
        ax.set_title(f"{dev.upper()}, batch 1: {lat['resnet18'][dev]['device']}", fontsize=8.5)
        ax.set_xlabel("Exit threshold")
    axes[0].set_ylabel("Speed-up over ResNet-18 full")
    axes[1].legend(fontsize=6.5, loc="upper left")
    savefig(fig, out, "latency_speedup.png")


def plot_training(runs, out):
    names = [n for n in ("resnet18", "ee_resnet18", "ee_vit_tiny") if n in runs]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 2.8), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        h = runs[name]["history"]
        ep = [x["epoch"] for x in h]
        k = len(h[0]["val_acc"])
        for i in range(k):
            color = INK if k == 1 else EXIT_RAMP[i]
            ax.plot(ep, [100 * x["val_acc"][i] for x in h], color=color, label="Final exit" if k == 1 else f"Exit {i + 1}")
        ax.set_title(NAMES[name])
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7, loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("Validation accuracy (%)")
    savefig(fig, out, "training_curves.png")


def plot_dataset(out, data_root):
    from src.data import load_cifar100
    from torchvision.datasets import CIFAR100
    classes = CIFAR100(data_root, train=False, download=False).classes
    d = load_cifar100(data_root)
    x, y = d["test"]
    rng = np.random.default_rng(3)
    idx = rng.choice(len(y), 16, replace=False)
    fig, axes = plt.subplots(2, 8, figsize=(9, 2.7))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(x[i], interpolation="nearest")
        ax.set_title(classes[y[i]].replace("_", " "), fontsize=7)
        ax.axis("off")
    savefig(fig, out, "dataset_samples.png")

    img = torch.from_numpy(x[idx[:1]]).permute(0, 3, 1, 2).float() / 255
    sevs = (1, 3, 5)
    fig, axes = plt.subplots(len(sevs), len(CORRUPTIONS) + 1, figsize=(8.6, 4.0))
    for r, s in enumerate(sevs):
        axes[r, 0].imshow(img[0].permute(1, 2, 0).numpy(), interpolation="nearest")
        axes[r, 0].set_ylabel(f"severity {s}", fontsize=7.5)
        for c, name in enumerate(CORRUPTIONS, 1):
            xc = corrupt(name, s, torch.Generator().manual_seed(s))(img)
            axes[r, c].imshow(xc[0].permute(1, 2, 0).numpy(), interpolation="nearest")
            if r == 0:
                axes[r, c].set_title(name.replace("_", " "), fontsize=7.5)
        if r == 0:
            axes[r, 0].set_title("clean", fontsize=7.5)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    savefig(fig, out, "corruption_examples.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--data", default="data")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    runs = {n: load_run(n) for n in RUNS if os.path.exists(os.path.join(RUNS[n], "metrics.json"))}
    print("runs:", list(runs))
    summary = {}
    plot_dataset(args.out, args.data)
    plot_training(runs, args.out)
    analyze_policies(runs, args.out, summary)
    analyze_calibration(runs, args.out)
    analyze_exits(summary, args.out)
    analyze_corruptions(runs, args.out, summary)
    analyze_latency(runs, args.out, summary)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
