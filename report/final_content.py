"""Text and tables of the final report. Numbers come from results/final/summary.json."""
import content as C

NAMES = {"resnet18": "ResNet-18 (Model 1)", "ee_resnet18_kd": "EE-ResNet-18 (Models 2, 3)",
         "ee_resnet18_ce": "EE-ResNet-18, no distillation", "ee_resnet18_pool": "EE-ResNet-18, pooling heads",
         "ee_vit_tiny": "EE-ViT-Tiny (Model 4)"}
POLICIES = ["Max-prob", "Entropy", "Max-prob + TS", "Entropy + TS"]
SEV = ("1", "2", "3", "4", "5")

CODE_KD = '''
def multi_exit_loss(outs, ya, yb, lam, ls, alpha=0.0, tau=3.0):
    """Mean over exits; early exits also match the final exit (self-distillation)."""
    teacher = F.softmax(outs[-1].detach().float() / tau, -1)
    total = ce(outs[-1])
    for o in outs[:-1]:
        loss = ce(o)
        if alpha:
            kd = F.kl_div(F.log_softmax(o.float() / tau, -1), teacher,
                          reduction="batchmean") * tau ** 2
            loss = (1 - alpha) * loss + alpha * kd
        total = total + loss
    return total / len(outs)
'''

CODE_TUNE = '''
def tune_per_exit(scores, correct, costs, target, init, grid=THRESH, passes=3):
    """Coordinate descent over one threshold per exit: min cost s.t. accuracy >= target."""
    th = np.full(scores.shape[0], float(init))
    best = routing_stats(route(scores, th), correct, costs)
    for _ in range(passes):
        for k in range(len(th)):
            for t in grid:
                cand = th.copy(); cand[k] = t
                r = routing_stats(route(scores, cand), correct, costs)
                if r["acc"] >= target and (r["cost"], -r["acc"]) < (best["cost"], -best["acc"]):
                    th, best = cand, r
    return th.tolist(), best
'''


def pm(v, scale=100, d=2):
    """mean +- std from a [mean, std] pair."""
    m, s = v
    return f"{scale * m:.{d}f} +- {scale * s:.{d}f}" if s > 0 else f"{scale * m:.{d}f}"


def mean(v, scale=100, d=1):
    return f"{scale * v[0]:.{d}f}"


def M(S, cfg):
    return S[cfg]["mean"] if cfg in S else None


def narrative(S):
    return {}


def write_body(R, S):
    T = narrative(S)
    R.para("Team No.: 15", bold=True, after=0)
    for name, reg in C.TEAM:
        R.para(f"{name} ({reg})", after=0)
    R.para(f"Code repository: {C.REPO}", before=4)
    abstract(R, T)
    introduction(R, S)
    C.literature(R)
    C.dataset(R, final=True)
    methodology(R, S)
    setup(R, S)
    results(R, S, T)
    discussion(R, T)
    conclusion(R, T)
    references(R)
    appendix(R)


def abstract(R, T):
    R.h1("Abstract")
    R.body(T.get("abstract", ""))


def introduction(R, S):
    R.h1("1. Introduction")
    R.body(
        "Deep neural networks normally execute every layer for every input, although many inputs can be classified "
        "correctly by a fraction of the network. Early-exit networks attach intermediate classifiers to a backbone and "
        "let an input stop at the first classifier whose prediction is confident enough, so that easy inputs use less "
        "computation and difficult inputs receive the full network. This is attractive for latency-sensitive and "
        "resource-constrained deployment, but it raises three practical questions that this project studies on "
        "CIFAR-100: how much computation can be saved for a given loss in accuracy, whether the confidence used for "
        "routing can be trusted (in particular after calibration and under corrupted inputs), and whether savings in "
        "FLOPs translate into measured latency.")
    R.h2("1.1 Objectives")
    R.bullets([
        "Develop an adaptive image-classification network that dynamically selects inference depth for individual "
        "inputs.",
        "Quantify the trade-off between classification accuracy, average computation and inference latency for "
        "different exit strategies.",
        "Evaluate the reliability of confidence-based early exits under clean and corrupted input conditions.",
    ], style="List Number")
    R.h2("1.2 Contributions")
    R.bullets([
        "A common implementation of four models (ResNet-18, confidence-based and calibrated/entropy-based early-exit "
        "ResNet-18, early-exit ViT-Tiny) with real early-exit inference that removes exited images from the batch.",
        "A GPU data pipeline for CIFAR-100 with a re-implementation of six CIFAR-100-C corruptions at five severities.",
        "A comparison of four exit rules (max-prob and entropy, with and without temperature scaling), with global and "
        "per-exit thresholds chosen on validation data, averaged over three training seeds.",
        "Ablations of self-distillation, exit-head design and exit placement.",
        "Measured latency on a GPU and a CPU next to FLOP counts, and an analysis of how exit decisions behave under "
        "distribution shift.",
    ])
    R.h2("1.3 Team and Model Ownership")
    R.table(["Member", "Model / task owned"], [
        ("Aditya Goyal", "Model 2: Confidence-based early-exit ResNet-18"),
        ("Raunak Panshikar", "Model 1: Standard ResNet-18 baseline"),
        ("Arth Mishra", "Model 3: Calibrated / entropy-based early-exit ResNet-18"),
        ("Kartikey Goyal", "Model 4: Early-exit ViT-Tiny"),
    ], "Model-to-member assignment.", widths=[5.0, 11.6])


def methodology(R, S):
    R.h1("4. Methodology")
    R.h2("4.1 Multi-Exit Networks and Early-Exit Inference")
    R.body(
        "Every model is a backbone split into K segments with a classifier head after each segment; a plain network is "
        "the case K = 1. At inference, an image passes through segment k, head k produces logits and a score, and the "
        "image stops if the score clears a threshold. Our implementation removes exited images from the batch before "
        "the next segment, so savings are real and not only simulated. Stopping at exit k costs the backbone up to "
        "segment k plus every head evaluated so far.")
    R.code(C.CODE_EXIT)
    R.h2("4.2 Models")
    rows = []
    for cfg in ("resnet18", "ee_resnet18_kd", "ee_resnet18_pool", "ee_vit_tiny"):
        m = M(S, cfg)
        if m:
            g = [x[0] for x in m["gflops"]]
            rows.append((NAMES[cfg], f"{m['params'][0] / 1e6:.2f}", str(len(g)), " / ".join(f"{x:.3f}" for x in g)))
    R.table(["Model", "Params (M)", "Exits", "Cumulative GFLOPs at each exit"], rows,
            "Architectures used in the final experiments.", widths=[5.2, 2.2, 1.4, 7.8], align_cols={1, 2})
    R.bullets([
        "**Model 1, ResNet-18 [12].** CIFAR variant: 3x3 stem without max-pooling, four stages of two basic blocks "
        "(64 to 512 channels), global pooling and a linear classifier.",
        "**Models 2 and 3, EE-ResNet-18.** The same backbone with an exit after each stage. Heads for stages 1 to 3 use "
        "strided 3x3 convolutions (128 channels) down to 4x4, then pooling and a linear layer; the final exit is the "
        "standard classifier. Models 2 and 3 share one trained network and differ only in the exit rule.",
        "**Ablation, pooling heads.** Every exit is only global pooling and a linear layer, as in SDN [4]. Heads are "
        "almost free but see less processed features.",
        "**Model 4, EE-ViT-Tiny [13], [14].** Width 192, 12 blocks, 3 heads, 4x4 patches (64 tokens), exits after "
        "blocks 3, 6, 9 and 12, each using LayerNorm, token averaging and a linear layer.",
    ])
    R.h2("4.3 Training Objective with Self-Distillation")
    R.body(
        "All exits are trained jointly. The final exit minimises the cross-entropy with the labels. Each earlier exit "
        "minimises (1 - alpha) times its cross-entropy plus alpha times a distillation term, the KL divergence between "
        "the softened prediction of the final exit (temperature tau, gradient stopped) and its own softened prediction, "
        "scaled by tau squared. We use alpha = 0.5 and tau = 3, and average the losses over exits. Self-distillation "
        "lets shallow exits learn from the richer predictions of the deepest exit, as in LGViT [9]; an ablation trains "
        "the same network without it.")
    R.code(CODE_KD)
    R.h2("4.4 Exit Rules and Calibration")
    R.body(
        "**Max-prob** (Model 2) exits when the largest softmax probability is at least a threshold tau. **Entropy** "
        "exits when 1 - H(p)/log(100) is at least tau, so it uses the whole predictive distribution, as in BranchyNet "
        "[1]. For Model 3, each exit is first calibrated with **temperature scaling** [5]: a scalar T_k is fitted on the "
        "validation set by minimising negative log-likelihood and logits are divided by T_k. This keeps every "
        "prediction unchanged but places the confidences of all exits on a common, calibrated scale. Calibration is "
        "measured with the expected calibration error (ECE, 15 bins).")
    R.h2("4.5 Threshold Selection")
    R.body(
        "All thresholds are chosen on the validation set and then frozen. A **global threshold** is the cheapest single "
        "value whose validation accuracy is within one point of the same model's final exit. **Per-exit thresholds** "
        "start from the global value and are refined by coordinate descent, minimising validation cost subject to the "
        "same accuracy constraint. Under corruption, the thresholds chosen on clean validation data are kept, as they "
        "would be in deployment.")
    R.code(CODE_TUNE)
    R.h2("4.6 Metrics")
    R.bullets([
        "**Accuracy and compute:** top-1 test accuracy and mean FLOPs per image, counted with PyTorch's "
        "FlopCounterMode including every head an image passes.",
        "**Calibration:** ECE per exit, before and after temperature scaling.",
        "**Overthinking:** share of test images that some early exit classifies correctly but the final exit does not, "
        "and the oracle that stops each image at its earliest correct exit.",
        "**Latency:** mean time per image at batch 1 on a T4 GPU and on a CPU with 1 and 4 threads, and throughput "
        "with batches of 256, measured for all seed-0 models in one Kaggle session.",
        "**Robustness:** accuracy, mean compute, exit distribution and accuracy of early-exited images on 30 corrupted "
        "test sets, and ECE under corruption.",
    ])


def setup(R, S):
    R.h1("5. Experimental Setup")
    R.table(["Setting", "ResNet-18 family", "EE-ViT-Tiny"], [
        ("Optimiser", "SGD, momentum 0.9, Nesterov", "AdamW (0.9, 0.999)"),
        ("Learning rate", "0.1, 2 warmup epochs, cosine", "1e-3, 5 warmup epochs, cosine"),
        ("Weight decay", "5e-4 (not on BN / bias)", "0.05 (not on norm / bias / position)"),
        ("Batch size, epochs", "128, 200", "256, 200"),
        ("Augmentation", "random crop, flip", "crop, flip, RandAugment, mixup / CutMix"),
        ("Regularisation", "none", "label smoothing 0.1, drop path 0.1, gradient clip 1.0"),
        ("Self-distillation", "alpha 0.5, tau 3 (early-exit models)", "alpha 0.5, tau 3"),
        ("Seeds", "0, 1, 2 (ablations: seed 0)", "0, 1, 2"),
        ("Hardware", "Kaggle, 2 x Tesla T4, fp16 mixed precision", "Kaggle, 2 x Tesla T4, fp16 mixed precision"),
    ], "Final training configuration.", widths=[3.4, 6.4, 6.8])
    R.body(
        "Eleven training runs were executed in two Kaggle sessions, each running one queue of runs per T4 GPU: three "
        "seeds each of ResNet-18, EE-ResNet-18 with self-distillation and EE-ViT-Tiny, plus EE-ResNet-18 without "
        "distillation and with pooling heads (seed 0). After training, each run stores the logits of every exit for "
        "the validation set, the clean test set and the 30 corrupted test sets, so that all exit rules, thresholds and "
        "ablations are evaluated offline from the same predictions. Latency was measured afterwards in a separate "
        "session that loads the seed-0 checkpoints, so all models were timed on the same idle machine. Results are "
        "reported as mean +- standard deviation over seeds where three seeds are available.")


def _op_rows(m, base_cost, per_exit=False):
    rows = []
    for label in POLICIES:
        p = m["policies"][label]
        t = (p["per_exit"] if per_exit else p)["test"]
        rows.append((label + (", per-exit" if per_exit else ""), pm(t["acc"]), pm(t["cost"], 1, 3),
                     f"{100 * (1 - t['cost'][0] / base_cost):.1f}",
                     " / ".join(mean(f, d=0) for f in t["exit_frac"])))
    return rows


def results(R, S, T):
    base = M(S, "resnet18")
    ee, vt = M(S, "ee_resnet18_kd"), M(S, "ee_vit_tiny")
    bc = base["gflops"][0][0]
    R.h1("6. Results")
    R.h2("6.1 Training")
    R.figure("training_curves.png", "Validation accuracy of every exit during training (seed 0).", width=16.5)
    R.body(T.get("training", ""))

    R.h2("6.2 Accuracy of Each Exit")
    rows = []
    for cfg in ("resnet18", "ee_resnet18_kd", "ee_resnet18_ce", "ee_resnet18_pool", "ee_vit_tiny"):
        m = M(S, cfg)
        if not m:
            continue
        acc = [pm(a) for a in m["test_acc"]]
        acc = ["-"] * (4 - len(acc)) + acc
        rows.append((NAMES[cfg], str(S[cfg]["n"]), *acc))
    R.table(["Model", "Seeds", "Exit 1", "Exit 2", "Exit 3", "Final exit"], rows,
            "Top-1 test accuracy (%) when every image is forced through a given exit.",
            widths=[4.8, 1.2, 2.6, 2.6, 2.6, 2.8], align_cols={1, 2, 3, 4, 5}, size=8.5)
    R.body(T.get("accuracy", ""))

    R.h2("6.3 Accuracy versus Computation")
    R.figure("tradeoff.png", "Test accuracy against mean GFLOPs per image while sweeping a global threshold (mean "
                             "and standard deviation over seeds). Dots are the thresholds selected on validation.",
             width=16.5)
    hdr = ["Exit rule", "Test acc. (%)", "Mean GFLOPs", "FLOPs saved vs ResNet-18 (%)", "Exits 1/2/3/4 (%)"]
    w = [3.8, 2.9, 2.6, 3.3, 4.0]
    R.table(hdr, _op_rows(ee, bc) + _op_rows(ee, bc, True),
            f"EE-ResNet-18: operating points selected on validation (ResNet-18: {pm(base['test_acc'][0])}% at "
            f"{bc:.3f} GFLOPs).", widths=w, align_cols={1, 2, 3, 4}, size=8.5)
    if vt:
        vc = vt["gflops"][-1][0]
        rows = _op_rows(vt, vc) + _op_rows(vt, vc, True)
        R.table(["Exit rule", "Test acc. (%)", "Mean GFLOPs", "FLOPs saved vs full ViT (%)", "Exits 1/2/3/4 (%)"],
                rows, f"EE-ViT-Tiny: operating points selected on validation (final exit {pm(vt['test_acc'][-1])}% at "
                      f"{vc:.3f} GFLOPs).", widths=w, align_cols={1, 2, 3, 4}, size=8.5)
    R.body(T.get("tradeoff", ""))
    R.figure("threshold_tuning.png", "Mean GFLOPs at the selected operating point with one global threshold and with "
                                     "per-exit thresholds.", width=16)
    R.body(T.get("thresholds", ""))
    rows = []
    for name, m, ref in (("EE-ResNet-18", ee, bc), ("EE-ViT-Tiny", vt, vt["gflops"][-1][0] if vt else 1)):
        if not m:
            continue
        for label in POLICIES:
            rows.append((name, label, *[pm(a) for a in m["policies"][label]["acc_at"]]))
    if ee:
        R.table(["Model", "Exit rule", "35% budget", "50% budget", "65% budget", "80% budget"], rows,
                "Best test accuracy (%) reachable within a mean compute budget, as a share of ResNet-18 FLOPs (CNN) or "
                "of the full ViT (transformer).", widths=[2.8, 2.9, 2.7, 2.7, 2.7, 2.8], align_cols={2, 3, 4, 5}, size=8)

    R.h2("6.4 Ablations")
    R.figure("ablation_training_heads.png", "Effect of self-distillation and of the exit-head design: accuracy of "
                                            "each exit (left) and max-prob accuracy-compute curves (right).", width=16)
    rows = []
    for cfg in ("ee_resnet18_kd", "ee_resnet18_ce", "ee_resnet18_pool"):
        m = M(S, cfg)
        if not m:
            continue
        p = m["policies"]["Max-prob"]["test"]
        rows.append((NAMES[cfg], pm(m["test_acc"][0]), pm(m["test_acc"][-1]), f"{m['gflops'][-1][0]:.3f}",
                     pm(p["acc"]), pm(p["cost"], 1, 3), f"{100 * m['ece_test']['raw'][-1][0]:.1f}"))
    R.table(["Variant", "Exit 1 acc.", "Final acc.", "Full-path GFLOPs", "Max-prob acc.", "Max-prob GFLOPs",
             "Final ECE (%)"], rows, "Ablation of training objective and head design (max-prob at the selected "
                                     "threshold).", widths=[4.4, 2.0, 2.0, 2.1, 2.1, 2.1, 1.9], align_cols={1, 2, 3, 4, 5, 6},
            size=8)
    rows = [(k, pm(v["acc"]), pm(v["cost"], 1, 3), f"{100 * (1 - v['cost'][0] / bc):.1f}")
            for k, v in sorted(ee["subsets"].items(), key=lambda kv: kv[1]["cost"][0])]
    R.table(["Exits used", "Test acc. (%)", "Mean GFLOPs", "FLOPs saved (%)"], rows,
            "Exit placement: EE-ResNet-18 using only a subset of its trained exits (max-prob, threshold selected on "
            "validation; unused heads are not evaluated).", widths=[4.0, 3.6, 3.4, 3.4], align_cols={1, 2, 3}, size=8.5)
    R.body(T.get("ablation", ""))

    R.h2("6.5 Calibration")
    R.figure("calibration_ece.png", "ECE of each exit before and after temperature scaling (mean over seeds; "
                                    "temperatures above the bars).", width=16)
    R.body(T.get("calibration", ""))

    R.h2("6.6 Overthinking")
    rows = []
    for cfg in ("ee_resnet18_kd", "ee_resnet18_ce", "ee_resnet18_pool", "ee_vit_tiny"):
        m = M(S, cfg)
        if m:
            rows.append((NAMES[cfg], pm(m["test_acc"][-1]), pm(m["oracle"]["acc"]), pm(m["oracle"]["cost"], 1, 3),
                         pm(m["overthinking"]["destructive"])))
    R.table(["Model", "Final exit acc. (%)", "Oracle acc. (%)", "Oracle GFLOPs", "Correct early, wrong at end (%)"],
            rows, "Overthinking: an oracle that stops each image at its earliest correct exit.",
            widths=[4.8, 2.8, 2.8, 2.6, 3.6], align_cols={1, 2, 3, 4}, size=8.5)
    R.body(T.get("overthinking", ""))

    R.h2("6.7 Measured Latency")
    R.figure("latency.png", "Measured time per image at batch 1 (seed-0 models, one machine) and speed-up over "
                            "full ResNet-18 on the same device.", width=16.5)
    lat = S.get("latency", [])
    if lat:
        rows = []
        for r in lat:
            if r["device"] == "gpu_batched":
                continue
            ops = r.get("ops", {})
            rows.append((NAMES[r["cfg"]].split(" (")[0], r["device"].upper().replace("CPU", "CPU-"),
                         " / ".join(f"{x:.2f}" for x in r["forced_ms"]),
                         f"{ops['Max-prob']['ms']:.2f}" if "Max-prob" in ops else "-",
                         f"{ops['Entropy + TS']['ms']:.2f}" if "Entropy + TS" in ops else "-"))
        R.table(["Model", "Device", "Forced exit 1/2/3/4 (ms)", "Max-prob (ms)", "Entropy + TS (ms)"], rows,
                "Batch-1 latency per image: forced exits and adaptive inference at the selected thresholds. CPU-1 and "
                "CPU-4 use 1 and 4 threads.", widths=[4.0, 1.7, 5.2, 2.6, 2.9], align_cols={1, 3, 4}, size=8)
        rows = []
        for r in lat:
            if r["device"] == "gpu_batched":
                ops = r.get("ops", {})
                rows.append((NAMES[r["cfg"]].split(" (")[0], f"{r['full_tp']:.0f}",
                             f"{ops['Max-prob']['img_per_s']:.0f}" if "Max-prob" in ops else "-",
                             f"{ops['Entropy + TS']['img_per_s']:.0f}" if "Entropy + TS" in ops else "-"))
        R.table(["Model", "Full network (img/s)", "Max-prob (img/s)", "Entropy + TS (img/s)"], rows,
                "GPU throughput with batches of 256; exited images are removed from the batch.",
                widths=[5.0, 3.8, 3.8, 3.8], align_cols={1, 2, 3}, size=8.5)
    R.body(T.get("latency", ""))

    R.h2("6.8 Robustness under Corruptions")
    R.figure("robustness_per_corruption.png", "Accuracy against corruption severity for each corruption (mean over "
                                              "seeds; thresholds chosen on clean validation data).", width=16.5)
    R.figure("robustness_summary.png", "Mean over the six corruptions: accuracy (left), computation (middle) and "
                                       "accuracy of the images that exit early (right).", width=16.5)
    rows = []
    for c in C.CORRUPTIONS:
        for sev in ("3", "5"):
            row = [C.CORR_NAMES[c], sev, mean(base["corrupt_acc"][c][sev])]
            for label in ("Max-prob", "Entropy + TS"):
                x = ee["corrupt"][c][sev][label]
                row += [mean(x["acc"]), f"{x['cost'][0]:.2f}", mean(x["early_acc"]) if x["early_acc"][0] == x["early_acc"][0] else "-"]
            rows.append(row)
    R.table(["Corruption", "Sev.", "ResNet-18 acc.", "MP acc.", "MP GFLOPs", "MP early acc.", "E+TS acc.",
             "E+TS GFLOPs", "E+TS early acc."], rows,
            "Accuracy (%), mean GFLOPs and accuracy of early-exited images (%) for EE-ResNet-18 with max-prob (MP) "
            "and entropy with temperature scaling (E+TS), severities 3 and 5, mean over seeds.",
            widths=[2.6, 0.9, 1.9, 1.6, 1.8, 1.8, 1.6, 1.8, 1.8], align_cols=set(range(1, 9)), size=7.5)
    R.figure("exit_shift.png", "Where EE-ResNet-18 images exit on clean and corrupted inputs.", width=14.5)
    R.figure("corruption_ece.png", "ECE of the EE-ResNet-18 exits under corruption, raw and with temperatures fitted "
                                   "on clean validation data.", width=15)
    R.body(T.get("robustness", ""))


def discussion(R, T):
    R.h1("7. Discussion")
    for key, title in (("obj1", "7.1 Adaptive depth (Objective 1)"), ("obj2", "7.2 Accuracy, computation and latency "
                       "(Objective 2)"), ("obj3", "7.3 Reliability under corruption (Objective 3)"),
                       ("literature", "7.4 Relation to Prior Work"), ("limitations", "7.5 Limitations")):
        R.h2(title)
        R.body(T.get(key, ""))


def conclusion(R, T):
    R.h1("8. Conclusion and Future Work")
    R.body(T.get("conclusion", ""))
    R.bullets(T.get("future", []))


def references(R):
    R.h1("References")
    for i, r in enumerate(C.REFS, 1):
        R.para(f"[{i}] {r}", size=9.5, after=2)


def appendix(R):
    R.h1("Appendix A: Reproducing the Results")
    R.body("All code is in the repository listed on the first page. Training and evaluation run on Kaggle GPUs; "
           "analysis and report generation run on a CPU.")
    R.code('''
pip install -r requirements.txt
python kaggle/launch.py push final-a      # 2 x T4: one queue of runs per GPU
python kaggle/launch.py push final-b
python kaggle/launch.py push latency      # loads seed-0 checkpoints of both sessions
python kaggle/launch.py fetch final-a && python kaggle/launch.py fetch final-b
python kaggle/launch.py fetch latency
python scripts/analyze_final.py           # results/final/
python report/build_report.py final       # report/Final_Report.docx and .pdf
''')
    R.table(["Path", "Contents"], [
        ("src/data.py, src/corruptions.py", "CIFAR-100 split, GPU augmentation, RandAugment, six corruptions"),
        ("src/models/", "MultiExitNet, ResNet-18 variants, ViT-Tiny"),
        ("src/train.py", "multi-exit training with self-distillation"),
        ("src/exits.py", "exit scores, temperature scaling, ECE, routing, threshold selection and tuning"),
        ("src/evaluate.py, src/latency.py", "logits on clean and corrupted data, latency benchmark"),
        ("kaggle/", "Kaggle job template and launcher"),
        ("scripts/analyze_final.py", "final analysis and figures"),
        ("report/", "report generators and outputs"),
    ], "Repository layout.", widths=[5.4, 11.2], size=8.5)
