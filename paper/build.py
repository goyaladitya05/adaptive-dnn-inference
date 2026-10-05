"""Generate the paper's tables, number macros and figures from results/study and results/final, then compile.

    python paper/build.py            # needs a `tectonic` binary on PATH or in $TECTONIC to compile
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze import AQUA, BLUE, INK2, ORANGE, VIOLET, plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
SEV = ("1", "2", "3", "4", "5")
ROWS = [  # config, label, macro prefix
    ("ee_resnet18_ce", r"CNN, plain, $\alpha$=0", "cnnce"),
    ("ee_resnet18_kd25", r"CNN, plain, $\alpha$=0.25", "cnnkdq"),
    ("ee_resnet18_kd", r"CNN, plain, $\alpha$=0.5", "cnnkd"),
    ("ee_resnet18_kd75", r"CNN, plain, $\alpha$=0.75", "cnnkdt"),
    ("ee_resnet18_ls", r"CNN, +label smoothing", "cnnls"),
    ("ee_resnet18_mix", r"CNN, +mixup/CutMix", "cnnmix"),
    ("ee_resnet18_ra", r"CNN, +RandAugment", "cnnra"),
    ("ee_resnet18_strong", r"CNN, strong", "cnnstrong"),
    ("ee_vit_tiny_plain", r"ViT, plain", "vitplain"),
    ("ee_vit_tiny_ce", r"ViT, strong, $\alpha$=0", "vitce"),
    ("ee_vit_tiny", r"ViT, strong", "vitkd"),
]


def load():
    with open(os.path.join(ROOT, "results", "study", "summary.json")) as f:
        study = json.load(f)
    with open(os.path.join(ROOT, "results", "final", "summary.json")) as f:
        final = json.load(f)
    return study, final


def vals(runs, fn):
    return np.array([fn(r) for r in runs], dtype=float)


def fmt(v, scale=1.0, d=1, pm=True):
    v = v * scale
    m, s = np.nanmean(v), np.nanstd(v)
    if np.isnan(m):
        return "--"
    return f"{m:.{d}f}" + (f"\\,$\\pm$\\,{s:.{d}f}" if pm and len(v) > 1 else "")


def macros(study, final):
    C = study["configs"]
    out = {}
    for cfg, _, key in ROWS:
        if cfg not in C:
            continue
        R = C[cfg]["seeds"]
        g = lambda fn, scale=1.0, d=1: fmt(vals(R, fn), scale, d, pm=False)  # noqa: E731
        out[key + "final"] = g(lambda r: r["test_acc"][-1], 100)
        out[key + "first"] = g(lambda r: r["test_acc"][0], 100)
        out[key + "logt"] = g(lambda r: r["log_t_early"], 1, 2)
        out[key + "temp"] = g(lambda r: np.exp(r["log_t_early"]), 1, 2)
        out[key + "cost"] = g(lambda r: r["points"]["mp"]["cost"], 1, 2)
        out[key + "costpe"] = g(lambda r: r["points"]["mp_pe"]["cost"], 1, 2)
        out[key + "costts"] = g(lambda r: r["points"]["mp_ts"]["cost"], 1, 2)
        out[key + "early"] = g(lambda r: r["shift"]["mp"]["5"]["early_frac"], 100, 0)
        out[key + "prec"] = g(lambda r: r["shift"]["mp"]["5"]["early_acc"], 100, 0)
        out[key + "gap"] = g(lambda r: r["conf_gap_exit1"]["5"], 100, 0)
        if "mp_loco" in R[0]["shift"]:
            out[key + "prects"] = g(lambda r: r["shift"]["mp_ts"]["5"]["early_acc"], 100, 0)
            out[key + "precloco"] = g(lambda r: r["shift"]["mp_loco"]["5"]["early_acc"], 100, 0)
            out[key + "earlyloco"] = g(lambda r: r["shift"]["mp_loco"]["5"]["early_frac"], 100, 0)
            out[key + "eceraw"] = g(lambda r: r["ece_shift"]["raw"]["5"][-1], 100, 0)
            out[key + "ecets"] = g(lambda r: r["ece_shift"]["ts"]["5"][-1], 100, 0)
            out[key + "eceloco"] = g(lambda r: r["ece_shift"]["loco"]["5"][-1], 100, 0)
            if "clean_loco" in R[0]:
                out[key + "costloco"] = g(lambda r: r["clean_loco"]["cost"], 1, 2)
    for _, _, key in ROWS:
        for suffix in ("final", "first", "logt", "temp", "cost", "costpe", "costts", "early", "prec", "gap", "prects",
                       "precloco", "earlyloco", "eceraw", "ecets", "eceloco", "costloco"):
            out.setdefault(key + suffix, "--")
    corr = study["correlations"]
    out["nruns"] = str(corr["n_runs"])
    out["rhoprec"] = f"{corr['spearman_logT_vs_early_acc_sev5']:.2f}"
    out["rhots"] = f"{corr['spearman_logT_vs_ts_cost_effect']:.2f}"
    rho = [x for c in C.values() for r in c["seeds"] for x in r["spearman_raw_ts"]]
    out["minrankcorr"] = f"{min(rho):.2f}"
    b = final["resnet18"]["mean"]
    out["basegflops"] = f"{b['gflops'][0][0]:.2f}"
    out["baseacc"] = f"{100 * b['test_acc'][0][0]:.1f}"
    lat = {(r["cfg"], r["device"]): r for r in final.get("latency", [])}
    if ("resnet18", "cpu1") in lat and ("ee_resnet18_kd", "cpu1") in lat:
        for dev, key in (("cpu1", "cpu"), ("gpu", "gpu")):
            out["speed" + key] = f"{lat[('resnet18', dev)]['full_ms'] / lat[('ee_resnet18_kd', dev)]['ops']['Max-prob']['ms']:.2f}"
    with open(os.path.join(HERE, "numbers.tex"), "w") as f:
        for k, v in out.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return out


def table_main(study):
    C = study["configs"]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Configuration & $n$ & $\overline{\log T}$ & Final acc. & Exit-1 acc. & Clean GFLOPs & Early (\%) & Early acc. (\%) \\",
             r"\midrule"]
    for cfg, label, _ in ROWS:
        if cfg not in C:
            continue
        R = C[cfg]["seeds"]
        lines.append(" & ".join([
            label, str(len(R)), fmt(vals(R, lambda r: r["log_t_early"]), 1, 2),
            fmt(vals(R, lambda r: r["test_acc"][-1]), 100), fmt(vals(R, lambda r: r["test_acc"][0]), 100),
            fmt(vals(R, lambda r: r["points"]["mp"]["cost"]), 1, 2),
            fmt(vals(R, lambda r: r["shift"]["mp"]["5"]["early_frac"]), 100, 0),
            fmt(vals(R, lambda r: r["shift"]["mp"]["5"]["early_acc"]), 100, 0)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(HERE, "table_main.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def table_calibration(study):
    C = study["configs"]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Exit 1, raw vs.\ TS} & \multicolumn{4}{c}{Clean GFLOPs at the strict point} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
             r"Configuration & $\rho_s$ & AUROC & Raw, global & Raw, per-exit & TS, global & TS, per-exit \\",
             r"\midrule"]
    for cfg, label, _ in ROWS:
        if cfg not in C or C[cfg]["n"] < 3:
            continue
        R = C[cfg]["seeds"]
        lines.append(" & ".join([
            label, fmt(vals(R, lambda r: r["spearman_raw_ts"][0]), 1, 3, pm=False),
            fmt(vals(R, lambda r: r["auroc_raw"][0]), 1, 3, pm=False) + r" / " +
            fmt(vals(R, lambda r: r["auroc_ts"][0]), 1, 3, pm=False),
            *[fmt(vals(R, lambda r, k=k: r["points"][k]["cost"]), 1, 2) for k in ("mp", "mp_pe", "mp_ts", "mp_ts_pe")]])
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(HERE, "table_calibration.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def table_shift_calibration(study):
    C = study["configs"]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Early-exit acc., sev.\ 5 (\%)} & \multicolumn{3}{c}{Final-exit ECE, sev.\ 5 (\%)} & Clean \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Configuration & Raw & TS & TS-LOCO & Raw & TS & TS-LOCO & GFLOPs \\",
             r"\midrule"]
    for cfg, label, _ in ROWS:
        if cfg not in C or C[cfg]["n"] < 3 or "mp_loco" not in C[cfg]["seeds"][0]["shift"]:
            continue
        R = C[cfg]["seeds"]
        cells = [label]
        cells += [fmt(vals(R, lambda r, k=k: r["shift"][k]["5"]["early_acc"]), 100, 0) for k in ("mp", "mp_ts", "mp_loco")]
        cells += [fmt(vals(R, lambda r, k=k: r["ece_shift"][k]["5"][-1]), 100, 0) for k in ("raw", "ts", "loco")]
        cells.append(fmt(vals(R, lambda r: r["points"]["mp"]["cost"]), 1, 2, pm=False) + r" / " +
                     (fmt(vals(R, lambda r: r["clean_loco"]["cost"]), 1, 2, pm=False) if "clean_loco" in R[0] else "--"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(HERE, "table_shift.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def fig_two_by_two(study):
    C = study["configs"]
    cells = [("ee_resnet18_kd", "CNN, plain", BLUE, "-", "o"), ("ee_resnet18_strong", "CNN, strong", ORANGE, "-", "s"),
             ("ee_vit_tiny_plain", "ViT, plain", VIOLET, "--", "^"), ("ee_vit_tiny", "ViT, strong", AQUA, "--", "D")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.9))
    for cfg, name, color, ls, mk in cells:
        if cfg not in C:
            continue
        R = C[cfg]["seeds"]
        for ax, key in zip(axes, ("early_frac", "early_acc", "acc")):
            ys = np.array([[100 * r["shift"]["mp"][s][key] for s in SEV] for r in R])
            ax.errorbar(range(1, 6), np.nanmean(ys, 0), yerr=np.nanstd(ys, 0), color=color, ls=ls, marker=mk, ms=3.5,
                        capsize=1.5, lw=1.2, label=name)
    for ax, t in zip(axes, ("Inputs exiting early (%)", "Accuracy of early exits (%)", "Overall accuracy (%)")):
        ax.set_title(t)
        ax.set_xlabel("Corruption severity")
        ax.set_xticks(range(1, 6))
    axes[2].legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(HERE, "figures", "two_by_two.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_confidence(study):
    C = study["configs"]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0))
    arch_style = {"CNN": (BLUE, "o"), "ViT": (VIOLET, "^")}
    for cfg, v in C.items():
        if v["arch"] not in arch_style:
            continue
        color, mk = arch_style[v["arch"]]
        face = color if v["recipe"] == "strong" else "white"
        R = v["seeds"]
        axes[0].scatter([r["log_t_early"] for r in R], [100 * r["shift"]["mp"]["5"]["early_acc"] for r in R], marker=mk,
                        s=18, edgecolor=color, facecolor=face, linewidth=1.0)
    axes[0].axvline(0, color=INK2, lw=0.7, ls="--")
    axes[0].set(xlabel=r"Mean $\log T$ of early exits", ylabel="Early-exit acc., sev. 5 (%)",
                title=r"Confidence state vs. reliability")
    axes[0].scatter([], [], marker="o", edgecolor=BLUE, facecolor="white", label="CNN, plain-type")
    axes[0].scatter([], [], marker="o", edgecolor=BLUE, facecolor=BLUE, label="CNN, strong")
    axes[0].scatter([], [], marker="^", edgecolor=VIOLET, facecolor="white", label="ViT, plain")
    axes[0].scatter([], [], marker="^", edgecolor=VIOLET, facecolor=VIOLET, label="ViT, strong")
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    alphas = [(c, C[c]["alpha"]) for c in ("ee_resnet18_ce", "ee_resnet18_kd25", "ee_resnet18_kd", "ee_resnet18_kd75")
              if c in C]
    for ax, fn, title in ((axes[1], lambda r: r["points"]["mp"]["cost"], "Clean GFLOPs (strict point)"),
                          (axes[2], lambda r: 100 * r["shift"]["mp"]["5"]["early_frac"], "Early exits at sev. 5 (%)")):
        xs = [a for _, a in alphas]
        ax.errorbar(xs, [np.mean(vals(C[c]["seeds"], fn)) for c, _ in alphas],
                    yerr=[np.std(vals(C[c]["seeds"], fn)) for c, _ in alphas], color=BLUE, marker="o", ms=3.5,
                    capsize=1.5, lw=1.2, label="CNN")
        vit = [(c, C[c]["alpha"]) for c in ("ee_vit_tiny_ce", "ee_vit_tiny") if c in C]
        ax.errorbar([a for _, a in vit], [np.mean(vals(C[c]["seeds"], fn)) for c, _ in vit],
                    yerr=[np.std(vals(C[c]["seeds"], fn)) for c, _ in vit], color=VIOLET, marker="^", ms=3.5,
                    capsize=1.5, lw=1.2, ls="--", label="ViT")
        ax.set(xlabel=r"Distillation weight $\alpha$", title=title)
        ax.set_xticks([0, 0.25, 0.5, 0.75])
    axes[2].legend(frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(HERE, "figures", "confidence.pdf"), bbox_inches="tight")
    plt.close(fig)


def compile_pdf():
    exe = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not exe:
        print("tectonic not found; skipped compilation")
        return
    subprocess.run([exe, "main.tex"], cwd=HERE, check=True)


def main():
    study, final = load()
    m = macros(study, final)
    table_main(study)
    table_calibration(study)
    table_shift_calibration(study)
    fig_two_by_two(study)
    fig_confidence(study)
    print(f"{len(m)} macros, tables and figures written")
    if os.path.exists(os.path.join(HERE, "main.tex")):
        compile_pdf()


if __name__ == "__main__":
    main()
