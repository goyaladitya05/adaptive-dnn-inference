"""Build notebooks/adaptive_inference.ipynb: the whole project in one Colab notebook.

The project code is inlined from src/ and scripts/ so the notebook always matches what was run.
"""
import os
import re

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/goyaladitya05/adaptive-dnn-inference"


def source(path, rename_main=None, renames=(), start=None, stop=None):
    """File contents without package imports, path hacks and the __main__ block."""
    with open(os.path.join(ROOT, path)) as f:
        s = f.read()
    if start:
        s = s[s.index(start):]
    if stop:
        s = s[:s.index(stop)]
    if 'if __name__ == "__main__":' in s:
        s = s[:s.index('if __name__ == "__main__":')]
    s = re.sub(r"^from (\.|src\.|analyze\b)[^\n(]*\([^)]*\)[^\n]*\n", "", s, flags=re.M)
    s = re.sub(r"^from (\.|src\.|analyze\b)[^\n]*\n", "", s, flags=re.M)
    s = re.sub(r"^sys\.path\.insert[^\n]*\n", "", s, flags=re.M)
    s = re.sub(r'^matplotlib\.use\("Agg"\)\n', "", s, flags=re.M)
    if rename_main:
        s = s.replace("def main(", f"def {rename_main}(")
        s = s.replace("def parse_args(", f"def {rename_main}_args(").replace("= parse_args()", f"= {rename_main}_args()")
    for a, b in renames:
        s = re.sub(rf"\b{a}\b", b, s)
    return s.strip() + "\n"


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md(f"""# Adaptive Deep Neural Network Inference Using Confidence-Based Early Exits

ICT-4442 Deep Learning Project, Team 15: Aditya Goyal, Raunak Panshikar, Arth Mishra, Kartikey Goyal.

Early-exit networks attach classifiers at several depths of a backbone. At inference an image stops at the first
exit whose prediction is confident enough, so easy images use less computation. This notebook contains the whole
project, end to end, on CIFAR-100:

1. **Data**: stratified train/val split, GPU augmentation, six CIFAR-100-C corruptions at five severities.
2. **Models**: ResNet-18 (Model 1), early-exit ResNet-18 used by the confidence rule (Model 2) and the
   calibrated / entropy rule (Model 3), and an early-exit ViT-Tiny (Model 4).
3. **Training** with self-distillation from the final exit into the early exits.
4. **Exit rules**: max softmax probability, entropy, per-exit temperature scaling, thresholds chosen on validation.
5. **Evaluation**: accuracy vs FLOPs, calibration, real batch-1 latency (GPU and CPU), robustness under corruption.
6. **Results** of our full-scale runs (3 seeds, ablations, follow-up study), loaded from the repository.

Code repository: {REPO}

**Runtime**: Colab with a GPU (Runtime > Change runtime type > T4 GPU). `MODE = "fast"` trains for 30 epochs
(about 1 hour for the three models); `MODE = "full"` uses the 200-epoch settings of the report (about 3.5 hours,
longer than a free Colab session may last, so mount Google Drive: finished runs are skipped when you re-run)."""),
         md("## 0. Setup"),
         code("""import os, sys, json, glob, shutil, subprocess, time, math, copy, re, itertools, platform, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

MODE = os.environ.get("NB_MODE", "fast")      # "fast", "full", or "smoke" (CPU check, a few minutes)
USE_DRIVE = False                             # True: keep runs in Google Drive so they survive disconnects
SEEDS = [0]                                   # the report uses [0, 1, 2]

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/adaptive-dnn-inference"
else:
    WORK = os.path.abspath(os.environ.get("NB_WORK", "work"))
DATA = os.path.join(WORK, "data")
RUNS_DIR = os.path.join(WORK, "runs")             # runs/final-colab/<model>_s<seed>/, as the analysis expects
RESULTS_DIR = os.path.join(WORK, "results")
for d in (DATA, RUNS_DIR, os.path.join(RESULTS_DIR, "figures")):
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() and MODE != "smoke" else "cpu"
print("torch", torch.__version__, "| device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")


def cli(fn, *args):
    \"\"\"Call one of the project's command-line entry points with the given arguments.\"\"\"
    sys.argv = [fn.__name__, *map(str, args)]
    return fn()"""),
         md("""## 1. Data pipeline

CIFAR-100 is downloaded with torchvision. 5,000 training images (50 per class, seed 0) form a validation set used
only for fitting temperatures and choosing thresholds; the official 10,000 test images are used for reporting. The
whole uint8 dataset is kept on the GPU and augmented there (random crop with reflect padding, horizontal flip;
RandAugment for the ViT), which avoids the CPU data-loader bottleneck."""),
         code(source("src/data.py")),
         md("""### Corruptions

Six corruptions of the CIFAR-100-C benchmark (Hendrycks and Dietterich, 2019), re-implemented on the GPU with the
original severity constants and re-quantised to 8 bits."""),
         code(source("src/corruptions.py")),
         code("""data = load_cifar100(DATA)
if MODE == "smoke":   # tiny splits so the whole notebook can be checked on a CPU
    data = {"train": tuple(a[:512] for a in data["train"]), "val": tuple(a[:500] for a in data["val"]),
            "test": tuple(a[:500] for a in data["test"])}
    _full = load_cifar100
    load_cifar100 = lambda root="data", val_size=5000, seed=0: data  # noqa: E731
for name, (x, y) in data.items():
    print(f"{name:5s} {len(y):6d} images, per-class min/max {np.bincount(y, minlength=100).min()}/{np.bincount(y, minlength=100).max()}")

from torchvision.datasets import CIFAR100
classes = CIFAR100(DATA, train=False, download=False).classes
x, y = data["test"]
fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for ax, i in zip(axes.flat, np.random.default_rng(3).choice(len(y), 16, replace=False)):
    ax.imshow(x[i]); ax.set_title(classes[y[i]].replace("_", " "), fontsize=7); ax.axis("off")
plt.show()

img = torch.from_numpy(x[:1]).permute(0, 3, 1, 2).float() / 255
fig, axes = plt.subplots(3, len(CORRUPTIONS) + 1, figsize=(10, 4.5))
for r, s in enumerate((1, 3, 5)):
    axes[r, 0].imshow(img[0].permute(1, 2, 0)); axes[r, 0].set_ylabel(f"severity {s}")
    for c, name in enumerate(CORRUPTIONS, 1):
        axes[r, c].imshow(corrupt(name, s, torch.Generator().manual_seed(s))(img)[0].permute(1, 2, 0))
        if r == 0: axes[r, c].set_title(name.replace("_", " "), fontsize=8)
    if r == 0: axes[r, 0].set_title("clean", fontsize=8)
for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
plt.show()"""),
         md("""## 2. Models

Every model is a backbone split into segments with a classifier head after each segment (a plain network is one
segment and one head). EE-ResNet-18 has exits after its four stages; the heads of stages 1 to 3 use strided 3x3
convolutions down to 4x4 before pooling. EE-ViT-Tiny (patch 4, width 192, 12 blocks) has exits after blocks 3, 6,
9 and 12. FLOPs are counted with PyTorch's FlopCounterMode, including every head an image passes."""),
         code(source("src/models/base.py") + "\n\n" + source("src/models/resnet.py") + "\n\n" +
              source("src/models/vit.py") + "\n\n" + source("src/models/__init__.py") + "\n\n" + source("src/flops.py")),
         code("""for name in MODELS:
    m = build_model(name)
    c = exit_costs(m)
    print(f"{name:17s} {count_params(m) / 1e6:5.2f} M params | GFLOPs to reach each exit:",
          [round(f / 1e9, 3) for f in c["cumulative"]])"""),
         md("""## 3. Exit rules, calibration and early-exit inference

* **Max-prob** (Model 2): exit when the largest softmax probability is at least a threshold.
* **Entropy**: exit when 1 - H(p)/log(100) is at least a threshold.
* **Temperature scaling** (Model 3): divide each exit's logits by a temperature fitted on validation data.
* Thresholds are chosen on validation: the cheapest one that loses no accuracy relative to the model's own final
  exit (one global threshold, or one per exit by coordinate descent).
* `adaptive_forward` is real early-exit inference: images that exit are removed from the batch, so the saving is
  real computation."""),
         code(source("src/exits.py")),
         md("""## 4. Training

All exits are trained jointly with an equally weighted loss. With self-distillation weight alpha, each early exit
minimises (1 - alpha) CE + alpha tau^2 KL(final exit || exit) at temperature tau = 3, with the final exit
detached. ResNets use SGD; the ViT uses AdamW with label smoothing, mixup/CutMix and RandAugment. Mixed precision
(fp16) is used on the GPU."""),
         code(source("src/train.py", rename_main="train_main")),
         md("## 5. Evaluation code\n\nLogits of every exit are stored for the validation set, the clean test set and "
            "all 30 corrupted test sets, so every exit rule can be analysed offline. Latency is measured with real "
            "early-exit inference, one image at a time, on the GPU and on the CPU."),
         code(source("src/latency.py", rename_main="latency_main")),
         code(source("src/evaluate.py", rename_main="evaluate_main",
                     renames=[("latency_suite", "eval_latency_suite"), ("throughput_suite", "eval_throughput_suite")])),
         md("""## 6. Train and evaluate

The three models of the report, with the settings of the final runs. A run whose `metrics.json` exists is
skipped, so with Google Drive mounted you can re-run this cell after a disconnect. Add
`("ee_resnet18_ce", "ee_resnet18", RESNET)` for the no-distillation ablation or
`("ee_resnet18_pool", "ee_resnet18_pool", RESNET + KD)` for the pooling-head ablation."""),
         code("""EPOCHS = {"full": 200, "fast": 30, "smoke": 1}[MODE]
RESNET = ["--epochs", EPOCHS, "--bs", 128, "--opt", "sgd", "--lr", 0.1, "--wd", 5e-4, "--warmup", 2 if EPOCHS > 2 else 0]
VIT = ["--epochs", EPOCHS, "--bs", 256, "--opt", "adamw", "--lr", 1e-3, "--wd", 0.05, "--warmup", 5 if EPOCHS > 5 else 0,
       "--label-smoothing", 0.1, "--mix", "--clip", 1.0, "--randaug"]
KD = ["--distill", 0.5, "--kd-temp", 3]
JOBS = [("resnet18", "resnet18", RESNET),              # Model 1
        ("ee_resnet18_kd", "ee_resnet18", RESNET + KD),  # Models 2 and 3
        ("ee_vit_tiny", "ee_vit_tiny", VIT + KD)]        # Model 4
SMOKE = ["--bs", 64, "--limit", 256] if MODE == "smoke" else []

for seed in SEEDS:
    for name, model, args in JOBS:
        out = os.path.join(RUNS_DIR, "final-colab", f"{name}_s{seed}")
        if os.path.exists(os.path.join(out, "metrics.json")):
            print("done:", out); continue
        t0 = time.time()
        cli(train_main, "--model", model, "--seed", seed, "--out", out, "--data", DATA, "--device", DEVICE, *args, *SMOKE)
        cli(evaluate_main, "--run", out, "--data", DATA, "--device", DEVICE, "--no-latency")
        print(f"{name} seed {seed}: {(time.time() - t0) / 60:.1f} min")"""),
         md("""## 7. Latency

Time per image when images arrive one at a time (batch 1) on the GPU and on one and four CPU threads, for every
seed-0 model at its validation-selected thresholds, plus GPU throughput with batches of 256."""),
         code("""cli(latency_main, "--runs", os.path.join(RUNS_DIR, "final-colab"), "--only", "_s0$", "--out", os.path.join(RUNS_DIR, "latency"),
    "--data", DATA, "--device", DEVICE, *(["--gpu-images", 50, "--cpu-images", 20, "--threads", 1] if MODE == "smoke" else []))"""),
         md("""## 8. Analysis

Threshold sweeps for the four exit rules, operating points chosen on validation (global and per-exit thresholds),
calibration, overthinking, latency and robustness under the 30 corruption settings, averaged over seeds."""),
         code("import matplotlib\n\n" + source("scripts/analyze.py", start="BLUE, ORANGE", stop="def load_run(",
                     renames=[("GRID", "GRID_COLOR")]) + "\n\n" +
              source("scripts/analyze.py", start="def savefig(", stop="def gflops(")),
         code(source("scripts/analyze_final.py", rename_main="analyze_final_main",
                     renames=[("SEVERITIES", "SEV_LABELS")])),
         code("""cli(analyze_final_main, "--runs", RUNS_DIR, "--out", RESULTS_DIR)
S = json.load(open(os.path.join(RESULTS_DIR, "summary.json")))
b = S["resnet18"]["mean"]
print(f"ResNet-18: {100 * b['test_acc'][0][0]:.2f}% test accuracy, {b['gflops'][0][0]:.3f} GFLOPs")
for cfg in ("ee_resnet18_kd", "ee_vit_tiny"):
    if cfg not in S: continue
    m = S[cfg]["mean"]
    ref = b["gflops"][0][0] if cfg != "ee_vit_tiny" else m["gflops"][-1][0]
    print(f"\\n{cfg}: per-exit accuracy", [round(100 * a[0], 2) for a in m["test_acc"]])
    for label, p in m["policies"].items():
        t = p["test"]
        print(f"  {label:14s} {100 * t['acc'][0]:6.2f}%  {t['cost'][0]:.3f} GFLOPs ({100 * (1 - t['cost'][0] / ref):.0f}% saved)"
              f"  exits {[round(100 * f[0]) for f in t['exit_frac']]}")
for fig_name in ("training_curves", "tradeoff", "calibration_ece", "threshold_tuning", "latency",
                 "robustness_summary", "exit_shift", "corruption_ece"):
    path = os.path.join(RESULTS_DIR, "figures", fig_name + ".png")
    if os.path.exists(path):
        display(Markdown(f"**{fig_name.replace('_', ' ')}**")); display(Image(path))"""),
         md(f"""## 9. Our full-scale results

The cells above reproduce the pipeline on one seed. Our reported numbers come from 200-epoch runs with three seeds
on Kaggle T4 GPUs, two ablations, and a follow-up study of 20 more runs. Their summaries and figures are in the
repository, loaded below."""),
         code(f"""REPO_DIR = os.path.join(WORK, "repo")
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}", REPO_DIR], check=True)
FINAL = json.load(open(os.path.join(REPO_DIR, "results/final/summary.json")))
import pandas as pd
rows = []
names = {{"resnet18": "ResNet-18", "ee_resnet18_kd": "EE-ResNet-18", "ee_resnet18_ce": "EE-ResNet-18, no distillation",
         "ee_resnet18_pool": "EE-ResNet-18, pooling heads", "ee_vit_tiny": "EE-ViT-Tiny"}}
for cfg, label in names.items():
    m = FINAL[cfg]["mean"]
    accs = [f"{{100 * a[0]:.2f}} ± {{100 * a[1]:.2f}}" for a in m["test_acc"]]
    rows.append([label, FINAL[cfg]["n"], *(["-"] * (4 - len(accs)) + accs)])
display(pd.DataFrame(rows, columns=["Model", "Seeds", "Exit 1", "Exit 2", "Exit 3", "Final exit"]).set_index("Model"))

ref = FINAL["resnet18"]["mean"]["gflops"][0][0]
rows = []
for label, p in FINAL["ee_resnet18_kd"]["mean"]["policies"].items():
    for kind, t in (("global", p["test"]), ("per-exit", p["per_exit"]["test"])):
        rows.append([label, kind, f"{{100 * t['acc'][0]:.2f}}", f"{{t['cost'][0]:.3f}}", f"{{100 * (1 - t['cost'][0] / ref):.0f}}%"])
display(Markdown("**EE-ResNet-18 operating points (no accuracy loss on validation), mean of 3 seeds**"))
display(pd.DataFrame(rows, columns=["Exit rule", "Thresholds", "Test acc. (%)", "GFLOPs", "Saved"]))
for fig_name in ("tradeoff", "ablation_training_heads", "calibration_ece", "latency", "robustness_summary", "exit_shift"):
    display(Markdown(f"**{{fig_name.replace('_', ' ')}}**"))
    display(Image(os.path.join(REPO_DIR, "results/final/figures", fig_name + ".png")))"""),
         md("""### Follow-up study: what makes early exits reliable under corruption?

We crossed architecture (CNN, ViT) with training recipe (plain, strong = label smoothing + mixup/CutMix +
RandAugment), varied the distillation weight (0 to 0.75) and ablated the recipe components, with three seeds per
main configuration. The decisive comparison is at **matched clean computation**: at the same clean GFLOPs, how
many severely corrupted images exit early, and how many of those early exits are correct."""),
         code("""MC = json.load(open(os.path.join(REPO_DIR, "results/study/matched_cost.json")))
rows = []
for cfg, v in MC.items():
    budget = "0.6" if "resnet" in cfg else "0.4"
    for pol, label in (("raw", "raw"), ("ts", "TS (clean)"), ("loco", "TS (other corruptions)")):
        if pol in v["5"]:
            r = v["5"][pol][budget]
            rows.append([cfg.replace("ee_", ""), label, budget, f"{100 * r['early_frac'][0]:.1f}", f"{100 * r['early_acc'][0]:.1f}"])
display(pd.DataFrame(rows, columns=["Configuration", "Calibration", "Clean GFLOPs", "Early exits at sev. 5 (%)",
                                    "Early-exit accuracy (%)"]))
for fig_name in ("lead2_architecture_recipe", "lead23_temperature"):
    display(Image(os.path.join(REPO_DIR, "results/study/figures", fig_name + ".png")))"""),
         md("""## 10. Conclusions

* **Early exits save computation.** With thresholds chosen so that no accuracy is lost on validation, EE-ResNet-18
  needs about half the FLOPs of ResNet-18 (51% fewer, 55% with per-exit thresholds), and 65% of test images stop
  at the first exit. Its final exit matches ResNet-18 (76.1%).
* **FLOPs are not latency.** The saving is a 1.9x speed-up on one CPU thread, but only 1.06x on a T4 GPU at
  batch 1, where per-exit synchronisation dominates; batching with exited images removed recovers 1.37x.
* **Calibration.** CNN exits are overconfident and ViT exits (trained with label smoothing and mixup)
  underconfident; temperature scaling brings every exit to at most about 4% ECE, but barely changes how exits rank
  images, so per-exit thresholds are the better tool for efficiency.
* **Under corruption**, confident exits keep stopping images they misclassify (Mehra et al., 2022, observed the
  same for multi-exit CNNs). At matched clean computation, calibration (temperature scaling, distillation strength,
  label smoothing, mixup) moves a model only a few points along its cost and reliability curve, while
  augmentation-based robustness (RandAugment, the strong recipe) roughly doubles the accuracy of early exits under
  severe corruption: **calibration moves a model along the curve; robustness moves the curve.**""")]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
nbf.write(nb, os.path.join(ROOT, "notebooks", "adaptive_inference.ipynb"))
print("wrote notebooks/adaptive_inference.ipynb with", len(cells), "cells")
