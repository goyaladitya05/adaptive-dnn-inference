"""Text and tables of the interim report. Numbers in the results sections come from results/summary.json."""
import numpy as np

TEAM = [("Aditya Goyal", "230953554"), ("Raunak Panshikar", "230953326"), ("Arth Mishra", "230911568"),
        ("Kartikey Goyal", "230911536")]
MONTH = "OCTOBER 2026"
REPO = "https://github.com/goyaladitya05/adaptive-dnn-inference"
CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "gaussian_blur", "defocus_blur", "contrast"]
CORR_NAMES = {"gaussian_noise": "Gaussian noise", "shot_noise": "Shot noise", "impulse_noise": "Impulse noise",
              "gaussian_blur": "Gaussian blur", "defocus_blur": "Defocus blur", "contrast": "Contrast"}

LIT = [
    ("[1]", "Teerapittayanon et al., ICPR 2016 (BranchyNet)",
     "Side-branch classifiers on LeNet, AlexNet and ResNet-110, trained jointly; a sample exits at the first branch whose softmax entropy is below a tuned threshold.",
     "MNIST, CIFAR-10",
     "B-ResNet-110 (CIFAR-10): 1.9x measured speed-up on CPU and GPU, accuracy 79.2% vs 80.7%; B-LeNet: 5.4x CPU speed-up, 94% of samples exit early.",
     "Origin of the entropy exit criterion used in Model 3; one of few works reporting measured latency rather than FLOPs."),
    ("[2]", "Huang et al., ICLR 2018 (MSDNet)",
     "Multi-scale dense network keeping coarse and fine features throughout, with intermediate classifiers; exits when the maximum softmax probability exceeds a threshold.",
     "CIFAR-10, CIFAR-100, ImageNet",
     "CIFAR-100 budgeted batch: matches ResNet-110 accuracy with about 1/10 of its FLOPs; ImageNet anytime: about 4-8% higher accuracy than ResNet/DenseNet ensembles at low budgets.",
     "Standard CIFAR-100 multi-exit baseline; shows intermediate classifiers can reduce a ResNet's final accuracy (up to 7%), a risk for our ResNet-18."),
    ("[3]", "Wang et al., ECCV 2018 (SkipNet)",
     "Lightweight gates skip residual blocks per input; trained with supervised pre-training followed by hybrid reinforcement learning.",
     "CIFAR-10, CIFAR-100, SVHN, ImageNet",
     "Computation reduced by 30-90% at preserved accuracy; for the deepest ResNets, savings of 37% (CIFAR-100), 50% (CIFAR-10), 86% (SVHN), 30% (ImageNet).",
     "Layer-skipping alternative to early exiting; positions our exit-based design among dynamic-depth methods for ResNets."),
    ("[4]", "Kaya et al., ICML 2019 (SDN)",
     "Internal classifiers added to VGG-16, ResNet-56, WRN-32-4 and MobileNet; exits on maximum softmax confidence; characterises network \"overthinking\".",
     "CIFAR-10, CIFAR-100, Tiny ImageNet",
     "Confidence exits cut average inference cost (FLOPs) by over 50% on CIFAR-10/100 while preserving accuracy; VGG-16-SDN on CIFAR-100: 72.5% at <=50% cost vs 70.9%.",
     "Closest precedent for Model 2: the same max-softmax exit rule on CIFAR-100 ResNets."),
    ("[5]", "Guo et al., ICML 2017",
     "Study of calibration in modern networks; compares histogram binning, isotonic regression, BBQ, and matrix, vector and temperature scaling as post-hoc fixes.",
     "CIFAR-10/100, ImageNet, SVHN, CUB, Cars; 20 News, Reuters, SST",
     "Modern networks are poorly calibrated; temperature scaling reduces ResNet-110 CIFAR-100 ECE from 16.53% to 1.26% without changing accuracy.",
     "Provides ECE and temperature scaling, applied per exit in Model 3 so that one threshold means the same at every exit."),
    ("[6]", "Wu et al., CVPR 2018 (BlockDrop)",
     "A policy network trained with reinforcement learning decides in one shot which residual blocks of a pretrained ResNet to execute per image.",
     "CIFAR-10, CIFAR-100, ImageNet",
     "ResNet-101 on ImageNet: 20% average speed-up (in FLOPs) at unchanged 76.4% top-1; ResNet-110 on CIFAR-100: 73.7% using 55% of blocks.",
     "Its sequential-policy variant ran slower than the full ResNet despite fewer blocks: FLOP savings need not imply wall-clock savings."),
    ("[7]", "Hendrycks and Dietterich, ICLR 2019",
     "Robustness benchmark of 15 algorithmic corruptions (noise, blur, weather, digital) at five severities, summarised by mean Corruption Error (mCE).",
     "ImageNet-C/P, CIFAR-10-C, Tiny ImageNet-C (CIFAR-100-C in the same repository)",
     "Relative corruption robustness barely changed from AlexNet to ResNet; ResNet-50 mCE 76.7%, MSDNet 73.6%.",
     "Defines the corruption types and severity constants we re-implement to stress-test exit policies."),
    ("[8]", "Wolczyk et al., NeurIPS 2021 (ZTW)",
     "Cascade connections pass each internal classifier's output to the next; a weighted geometric ensemble of past predictions feeds the confidence-based exit decision.",
     "CIFAR-10, CIFAR-100, Tiny ImageNet, ImageNet",
     "ResNet-56 on CIFAR-100 at 50% of base FLOPs: 62.1% accuracy vs 57.2% (SDN) and 53.5% (PABEE).",
     "Strong confidence-exit baseline on CIFAR-100; shows discarded early predictions carry information an exit policy can reuse."),
    ("[9]", "Xu et al., ACM MM 2023 (LGViT)",
     "Early-exit ViTs with heterogeneous heads (local perception, global aggregation), two-stage training with self-distillation, max-softmax exits.",
     "CIFAR-100, Food-101, ImageNet-1K",
     "About 1.8x average speed-up for about 2% accuracy loss; ViT-B/16 on CIFAR-100: 88.5% vs 90.8% at 1.87x (computed from exit layers).",
     "Closest vision-transformer precedent for Model 4; its speed-up is derived from exit depth, not measured latency."),
    ("[10]", "Meronen et al., WACV 2024",
     "Post-hoc last-layer Laplace approximation at each exit plus model-internal ensembling and tuned temperature, for uncertainty-aware exits in MSDNet.",
     "CIFAR-100, ImageNet, Caltech-256",
     "CIFAR-100 (small MSDNet, averaged over budgets): ECE falls from 0.182 to 0.017 and top-1 rises from 69.25% to 69.84%.",
     "Direct evidence that early-exit confidences are overconfident and that calibrating them helps exit decisions; motivates Model 3."),
]

REFS = [
    'S. Teerapittayanon, B. McDanel, and H. T. Kung, "BranchyNet: Fast inference via early exiting from deep neural networks," in Proc. 23rd Int. Conf. Pattern Recognit. (ICPR), 2016, pp. 2464-2469, doi: 10.1109/ICPR.2016.7900006.',
    'G. Huang, D. Chen, T. Li, F. Wu, L. van der Maaten, and K. Q. Weinberger, "Multi-scale dense networks for resource efficient image classification," in Proc. Int. Conf. Learn. Represent. (ICLR), 2018, arXiv:1703.09844.',
    'X. Wang, F. Yu, Z.-Y. Dou, T. Darrell, and J. E. Gonzalez, "SkipNet: Learning dynamic routing in convolutional networks," in Proc. Eur. Conf. Comput. Vis. (ECCV), Springer, 2018, pp. 420-436, doi: 10.1007/978-3-030-01261-8_25.',
    'Y. Kaya, S. Hong, and T. Dumitras, "Shallow-deep networks: Understanding and mitigating network overthinking," in Proc. 36th Int. Conf. Mach. Learn. (ICML), PMLR vol. 97, 2019, pp. 3301-3310.',
    'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. 34th Int. Conf. Mach. Learn. (ICML), PMLR vol. 70, 2017, pp. 1321-1330.',
    'Z. Wu, T. Nagarajan, A. Kumar, S. Rennie, L. S. Davis, K. Grauman, and R. Feris, "BlockDrop: Dynamic inference paths in residual networks," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 8817-8826, doi: 10.1109/CVPR.2018.00919.',
    'D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to common corruptions and perturbations," in Proc. Int. Conf. Learn. Represent. (ICLR), 2019, arXiv:1903.12261.',
    'M. Wolczyk, B. Wojcik, K. Balazy, I. T. Podolak, J. Tabor, M. Smieja, and T. Trzcinski, "Zero time waste: Recycling predictions in early exit neural networks," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, 2021, pp. 2516-2528.',
    'G. Xu, J. Hao, L. Shen, H. Hu, Y. Luo, H. Lin, and J. Shen, "LGViT: Dynamic early exiting for accelerating vision transformer," in Proc. 31st ACM Int. Conf. Multimedia (MM), 2023, pp. 9103-9114, doi: 10.1145/3581783.3611762.',
    'L. Meronen, M. Trapp, A. Pilzer, L. Yang, and A. Solin, "Fixing overconfidence in dynamic neural networks," in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2024, pp. 2668-2678, doi: 10.1109/WACV57701.2024.00266.',
    'A. Krizhevsky, "Learning multiple layers of features from tiny images," Tech. Rep., Univ. of Toronto, 2009.',
    'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770-778.',
    'A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.',
    'H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, "Training data-efficient image transformers and distillation through attention," in Proc. 38th Int. Conf. Mach. Learn. (ICML), PMLR vol. 139, 2021, pp. 10347-10357.',
    'H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.',
    'S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Regularization strategy to train strong classifiers with localizable features," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 6023-6032.',
]

CODE_SPLIT = '''
def load_cifar100(root="data", val_size=5000, seed=0):
    tr = CIFAR100(root, train=True, download=True)
    te = CIFAR100(root, train=False, download=True)
    x, y = tr.data, np.asarray(tr.targets)
    rng = np.random.default_rng(seed)
    per_class = val_size // NUM_CLASSES                  # 50 images per class
    val_idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per_class]
                              for c in range(NUM_CLASSES)])
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return {"train": (x[mask], y[mask]),
            "val": (x[val_idx], y[val_idx]),
            "test": (te.data, np.asarray(te.targets))}
'''

CODE_AUG = '''
def augment(x, pad=4):
    """Random crop with reflect padding and horizontal flip, on the GPU."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n, 1), device=x.device) + arange(h)
    j = torch.randint(0, 2 * pad + 1, (n, 1), device=x.device) + arange(w)
    x = x[arange(n)[:, None, None, None], arange(3)[None, :, None, None],
          i[:, None, :, None], j[:, None, None, :]]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)

class GPULoader:                    # the whole uint8 split lives on the GPU
    def __iter__(self):
        idx = torch.randperm(n) if self.train else torch.arange(n)
        for k in range(0, n, self.bs):
            b = idx[k:k + self.bs]
            xb = self.x[b].float().div_(255)
            if self.train:
                xb = augment(xb)
            if self.transform is not None:   # corruptions, Section 3.3
                xb = self.transform(xb)
            yield normalize(xb), self.y[b]
'''

CODE_CORR = '''
def gaussian_noise(x, s, g):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][s - 1]
    return (x + torch.randn(x.shape, generator=g, device=x.device) * c).clamp(0, 1)

def gaussian_blur(x, s, g):
    c = [0.4, 0.6, 0.7, 0.8, 1.0][s - 1]
    k = _gauss1d(c, int(4 * c + 0.5), x.device)
    return _filter(x, torch.outer(k, k), "replicate").clamp(0, 1)

def corrupt(name, severity, generator):   # re-quantize to 8 bits like stored CIFAR-C
    fn = CORRUPTIONS[name]
    return lambda x: torch.round(fn(x, severity, generator) * 255) / 255
'''

CODE_EXIT = '''
@torch.no_grad()
def adaptive_forward(model, x, score, thresh, temps=None):
    """Real early-exit inference: exited samples are dropped before the next segment."""
    alive, h = torch.arange(len(x)), x
    for k in range(model.num_exits):
        h = model.segments[k](h)
        logits = model.heads[k](h)
        if k == model.num_exits - 1:
            preds[alive] = logits.argmax(1); break
        done = score(logits, temps[k]) >= thresh      # max_prob or 1 - H(p)/log C
        preds[alive[done]] = logits[done].argmax(1); exit_at[alive[done]] = k
        alive, h = alive[~done], h[~done]
'''


def _mean_corr(S, model, policy, sev, key):
    pol = S["corruptions"][model]["policies"][policy]
    vals = [pol[c][sev][key] for c in pol]
    return float(np.nanmean(vals))


def _ece_corr(S, model, sev, key):
    e = S["corruptions"][model]["ece"]
    return np.mean([e[c][sev][key] for c in e], 0)


def narrative(S):
    b = S["baseline"]
    ee, vt = S.get("ee_resnet18"), S.get("ee_vit_tiny")
    lat = S.get("latency", {})
    T = {}
    a, g = ee["test_acc"], ee["gflops"]
    rel = [x / b["gflops"] for x in g]
    T["accuracy"] = (
        f"The baseline ResNet-18 reaches {pct(b['test_acc'], 2)}% test accuracy. The final exit of EE-ResNet-18 "
        f"reaches {pct(a[-1], 2)}%, {100 * (a[-1] - b['test_acc']):+.2f} points relative to the baseline, so jointly "
        f"training the early heads did not harm the full network in our setting, possibly because the extra "
        f"supervision at intermediate depths acts as a regulariser; this contrasts with the degradation of up to 7% "
        f"reported for naive intermediate classifiers in [2]. The early exits are strong: exit 1 reaches "
        f"{pct(a[0])}% with {pct(rel[0], 0)}% of the baseline FLOPs, exit 2 {pct(a[1])}% with {pct(rel[1], 0)}%, and "
        f"exit 3 already matches the baseline ({pct(a[2])}% with {pct(rel[2], 0)}% of its FLOPs).")
    if vt:
        T["accuracy"] += (
            f" EE-ViT-Tiny, trained from scratch, reaches {pct(vt['test_acc'][-1])}% at its final exit and "
            f"{pct(vt['test_acc'][0])}%, {pct(vt['test_acc'][1])}% and {pct(vt['test_acc'][2])}% at exits 1 to 3. "
            "It is clearly weaker than the CNN, as expected for a vision transformer without pre-training on a small "
            "32x32 dataset and a short schedule [13], [14]; closing this gap is part of the remaining work.")

    P = ee["policies"]
    mp, mpt, ent, entt = P["Max-prob"], P["Max-prob + TS"], P["Entropy"], P["Entropy + TS"]
    o = ee["oracle"]
    T["tradeoff"] = (
        f"Figure 4 and the tables above contain the central result. With the threshold selected on validation, "
        f"confidence-based exiting (Model 2) matches the accuracy of full ResNet-18 ({pct(mp['test']['acc'], 2)}% "
        f"versus {pct(b['test_acc'], 2)}%) while spending on average {mp['test']['cost']:.3f} GFLOPs instead of "
        f"{b['gflops']:.3f}, a {pct(1 - mp['test']['cost'] / b['gflops'])}% reduction; "
        f"{pct(mp['test']['exit_frac'][0], 0)}% of test images leave at the first exit and only "
        f"{pct(mp['test']['exit_frac'][-1], 0)}% need the full network. The curves saturate early: beyond about 0.7 "
        f"GFLOPs additional computation brings no accuracy gain. An oracle that stops every image at its earliest "
        f"correct exit would reach {pct(o['acc'])}% accuracy with only {o['cost']:.2f} GFLOPs. Many images are "
        f"therefore classified correctly by an early exit but wrongly by the final one (overthinking [4]), and a "
        f"better exit rule has considerable headroom.\n"
        f"The calibrated rules (Model 3) are slightly less efficient on clean data: for similar accuracy, max-prob "
        f"with temperature scaling needs {mpt['test']['cost']:.3f} GFLOPs, entropy {ent['test']['cost']:.3f} and "
        f"entropy with temperature scaling {entt['test']['cost']:.3f} GFLOPs, and at a budget of 0.55 GFLOPs raw "
        f"max-prob is ahead by up to {100 * (mp['acc_at']['0.55'] - entt['acc_at']['0.55']):.1f} points. Temperature "
        f"scaling preserves the ranking of images within an exit and only changes how confidences compare across "
        f"exits. Since deeper exits receive larger temperatures (Section 5.3), they become less willing to stop "
        f"images, which pushes images to the costly final exit. Calibration makes confidences faithful, but this is "
        f"not the same objective as minimising cost at fixed accuracy; tuning per-exit thresholds directly for the "
        f"budget is a natural next step.")
    if vt:
        vp, vpt = vt["policies"]["Max-prob"]["test"], vt["policies"]["Max-prob + TS"]["test"]
        T["tradeoff"] += (
            f"\nFor EE-ViT-Tiny, max-prob exiting stays within one point of its final exit "
            f"({pct(vp['acc'], 2)}% versus {pct(vt['test_acc'][-1], 2)}%) with {vp['cost']:.3f} of its "
            f"{vt['gflops'][-1]:.3f} GFLOPs ({pct(1 - vp['cost'] / vt['gflops'][-1])}% fewer), and exit 2, after only "
            f"six of twelve blocks, is already within {100 * (vt['test_acc'][-1] - vt['test_acc'][1]):.1f} points of the "
            f"final exit. Here temperature scaling gives the cheapest operating point ({vpt['cost']:.3f} GFLOPs at "
            f"{pct(vpt['acc'], 2)}%), the opposite of the CNN: the ViT exits are underconfident (temperatures below 1, "
            f"Section 5.3), so scaling sharpens their confidences and lets more images stop early. Calibration therefore "
            f"helps routing when exits are underconfident and costs computation when deeper exits are overconfident.")

    er, ec = ee["ece_test"]["raw"], ee["ece_test"]["calibrated"]
    t = ee["temperature"]
    T["calibration"] = (
        f"Raw calibration error grows with depth: exit 1 is nearly calibrated (ECE {pct(er[0])}%), while the final "
        f"exit has an ECE of {pct(er[-1])}%, the typical overconfidence of a deep network trained to near-zero "
        f"training loss [5]. The fitted temperatures increase accordingly from {t[0]:.2f} to {t[-1]:.2f}. "
        f"Temperature scaling reduces the ECE of every exit, to between {pct(min(ec))}% and {pct(max(ec))}%, and the "
        f"reliability curves of the deep exits move from below the diagonal (overconfident) close to it.")
    if vt:
        vr, vc = vt["ece_test"]["raw"], vt["ece_test"]["calibrated"]
        vtt = vt["temperature"]
        T["calibration"] += (
            f" EE-ViT-Tiny behaves differently: trained with label smoothing and mixup, its raw ECE is "
            f"{pct(min(vr))}% to {pct(max(vr))}% and its temperatures range from {min(vtt):.2f} to {max(vtt):.2f} "
            f"(below 1 means the exit was underconfident); after scaling its ECE is {pct(min(vc))}% to {pct(max(vc))}%.")

    T["budget"] = (
        f"At 0.40 GFLOPs (36% of the baseline cost) EE-ResNet-18 still reaches about {pct(mp['acc_at']['0.40'], 0)}%, "
        f"and at 0.70 GFLOPs (63%) every rule exceeds full ResNet-18.")

    if lat:
        bl, el = lat["resnet18"], lat["ee_resnet18"]
        def row(dev, th, pol="max_prob", cal=False):
            return next(x for x in el[dev]["adaptive"] if x["policy"] == pol and x["calibrated"] == cal and x["thresh"] == th)
        c8, g6, g9 = row("cpu", 0.8), row("gpu", 0.6), row("gpu", 0.9)
        flop8 = b["gflops"] / float(np.dot(c8["exit_frac"], g))
        bt = el["gpu"]["batched"]
        t8 = next(x for x in bt["adaptive"] if x["policy"] == "max_prob" and not x["calibrated"] and x["thresh"] == 0.8)
        base_tp = bl["gpu"]["batched"]["full"]
        T["latency"] = (
            f"Measured latency is more nuanced than FLOPs. On one CPU thread, latency follows FLOPs closely: stopping "
            f"at exit 1 takes {el['cpu']['forced_ms'][0]:.1f} ms against {bl['cpu']['forced_ms'][0]:.1f} ms for full "
            f"ResNet-18, and adaptive inference with max-prob at tau = 0.8 needs {c8['ms']:.1f} ms per image, a "
            f"{bl['cpu']['forced_ms'][0] / c8['ms']:.2f}x speed-up, close to the {flop8:.2f}x predicted by FLOPs. The "
            f"full path of the early-exit model is {pct(el['cpu']['forced_ms'][-1] / bl['cpu']['forced_ms'][0] - 1, 0)}% "
            f"slower than the baseline, which is the price paid by hard images. On the T4 GPU at batch 1 the picture "
            f"changes: execution is dominated by kernel launches and by the GPU-to-CPU synchronisation needed after "
            f"every exit to decide whether to stop. Full ResNet-18 takes {bl['gpu']['forced_ms'][0]:.2f} ms and exit 1 "
            f"alone {el['gpu']['forced_ms'][0]:.2f} ms, but the full early-exit path takes "
            f"{el['gpu']['forced_ms'][-1]:.2f} ms, so adaptive inference is at best on par with the baseline "
            f"({bl['gpu']['forced_ms'][0] / g6['ms']:.2f}x at tau = 0.6) and slower at higher thresholds "
            f"({bl['gpu']['forced_ms'][0] / g9['ms']:.2f}x at tau = 0.9). With batches of 256, where exited images are "
            f"removed from the batch, the GPU benefits again: throughput rises from {base_tp:.0f} images/s for "
            f"ResNet-18 to {t8['img_per_s']:.0f} images/s at tau = 0.8 ({t8['img_per_s'] / base_tp:.2f}x). Early exits "
            f"therefore pay off on CPUs and edge devices and in batched serving, while FLOP counts overstate the "
            f"benefit for single images on a GPU, in line with the timing observation of BlockDrop [6].")
        if "ee_vit_tiny" in lat:
            vl = lat["ee_vit_tiny"]
            v8 = next(x for x in vl["cpu"]["adaptive"] if x["policy"] == "max_prob" and not x["calibrated"] and x["thresh"] == 0.8)
            T["latency"] += (
                f" EE-ViT-Tiny shows the same pattern more strongly. Its full path is cheaper than ResNet-18 on the CPU "
                f"({vl['cpu']['forced_ms'][-1]:.1f} ms) and adaptive inference at tau = 0.8 brings it to {v8['ms']:.1f} ms, "
                f"but on the GPU at batch 1 its twelve blocks launch many small kernels and the full path takes "
                f"{vl['gpu']['forced_ms'][-1]:.2f} ms, {vl['gpu']['forced_ms'][-1] / bl['gpu']['forced_ms'][0]:.1f}x "
                f"slower than ResNet-18 despite needing fewer FLOPs.")

    if "corruptions" in S:
        m = "ee_resnet18"
        acc = {s: _mean_corr(S, m, "Max-prob", s, "acc") for s in ("1", "3", "5")}
        cost5 = {k: _mean_corr(S, m, k, "5", "cost") for k in ("Max-prob", "Entropy + TS")}
        ef5 = {k: _mean_corr(S, m, k, "5", "early_frac") for k in ("Max-prob", "Entropy + TS")}
        ea5 = {k: _mean_corr(S, m, k, "5", "early_acc") for k in ("Max-prob", "Entropy + TS")}
        imp = S["corruptions"][m]["policies"]["Max-prob"]["impulse_noise"]["5"]
        e1, e5 = _ece_corr(S, m, "1", "calibrated"), _ece_corr(S, m, "5", "calibrated")
        r5 = _ece_corr(S, m, "5", "raw")
        T["robustness"] = (
            f"With thresholds fixed on clean validation data, the early-exit model keeps the accuracy of full ResNet-18 "
            f"under corruption: averaged over the six corruptions, max-prob exiting reaches {pct(acc['1'])}%, "
            f"{pct(acc['3'])}% and {pct(acc['5'])}% at severities 1, 3 and 5, within about one point of the baseline. "
            f"The model also spends more computation on corrupted images, since fewer of them clear the threshold early: "
            f"mean cost rises from {mp['test']['cost']:.2f} GFLOPs on clean data to {cost5['Max-prob']:.2f} GFLOPs at "
            f"severity 5. However, the raw max-prob rule stays overconfident under shift. At severity 5 it still exits "
            f"{pct(ef5['Max-prob'], 0)}% of images early, and only {pct(ea5['Max-prob'], 0)}% of those early decisions are "
            f"correct. The worst case is impulse noise at severity 5, where accuracy collapses to {pct(imp['acc'])}% while "
            f"{pct(imp['early_frac'], 0)}% of images exit early and the model spends less computation than on clean data "
            f"({imp['cost']:.2f} GFLOPs). The calibrated entropy rule is more cautious: it exits "
            f"{pct(ef5['Entropy + TS'], 0)}% of images early at severity 5, of which {pct(ea5['Entropy + TS'], 0)}% are "
            f"correct, at the cost of more computation ({cost5['Entropy + TS']:.2f} GFLOPs). Temperatures fitted on clean "
            f"data transfer only partly: averaged over corruptions, the ECE of the final exit is {pct(r5[-1])}% with raw "
            f"logits and still {pct(e5[-1])}% after scaling at severity 5 (Figure 10), compared with "
            f"{pct(ee['ece_test']['calibrated'][-1])}% on clean data.")

    T["findings"] = [
        f"**Compute.** Confidence-based EE-ResNet-18 matches full ResNet-18 accuracy ({pct(mp['test']['acc'], 2)}% vs "
        f"{pct(b['test_acc'], 2)}%) with {pct(1 - mp['test']['cost'] / b['gflops'], 0)}% fewer FLOPs; its final exit is "
        f"{100 * (a[-1] - b['test_acc']):+.2f} points above the baseline, and exit 1 alone gives {pct(a[0])}% at "
        f"{pct(rel[0], 0)}% of the cost.",
        f"**Headroom.** An oracle exit would reach {pct(o['acc'])}% at {o['cost']:.2f} GFLOPs, so better exit rules "
        f"(and reducing overthinking) can still help.",
        f"**Calibration.** Deeper exits are increasingly overconfident (ECE {pct(er[0])}% at exit 1 to {pct(er[-1])}% at "
        f"the final exit); temperature scaling lowers ECE to at most {pct(max(ec))}%.",
        "**Exit rule.** On clean data, calibration does not improve the accuracy-compute trade-off of a single "
        "global threshold; under corruption, the calibrated entropy rule makes early decisions far more reliable, "
        "at the price of extra computation.",
    ]
    if lat:
        T["findings"].append(
            f"**Latency.** FLOP savings become real speed-ups on a CPU ({bl['cpu']['forced_ms'][0] / c8['ms']:.2f}x at "
            f"tau = 0.8) and in batched GPU inference ({t8['img_per_s'] / base_tp:.2f}x), but not for single images on a "
            f"GPU, where per-exit synchronisation dominates.")
    if vt:
        T["findings"].append(
            f"**Transformer.** Early exiting also works for ViT-Tiny ({pct(1 - vt['policies']['Max-prob']['test']['cost'] / vt['gflops'][-1], 0)}% "
            f"fewer FLOPs within one point of its final exit), but its from-scratch accuracy "
            f"({pct(vt['test_acc'][-1])}%) needs improvement before a fair comparison with the CNN.")
    return T


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def write_body(R, S):
    R.para("PART B: INTERIM REPORT", bold=True, size=14, align="center", after=8)
    R.para("Team No. and Names (with Registration Numbers):", bold=True, after=2)
    R.para("Team No.: __________", after=0)
    for name, reg in TEAM:
        R.para(f"{name} ({reg})", after=0)
    R.para("Title of the Project:", bold=True, before=6, after=2)
    R.para("Adaptive Deep Neural Network Inference Using Confidence-Based Early Exits")
    R.para("Code repository:", bold=True, before=4, after=2)
    R.para(REPO)

    overview(R, S)
    literature(R)
    dataset(R)
    models(R, S)
    results(R, S)
    plan(R, S)
    references(R)


def overview(R, S):
    R.h1("1. Project Overview and Interim Status")
    R.body(
        "Standard deep networks spend the same computation on every input, although many images can be classified "
        "correctly by a fraction of the network. This project builds early-exit networks for CIFAR-100: classifiers "
        "are attached at several depths of a backbone, and at inference time an image leaves the network at the first "
        "exit whose prediction satisfies a confidence criterion. We compare four models: (1) a standard ResNet-18 with "
        "full inference, (2) a confidence-based early-exit ResNet-18 using the maximum softmax probability, (3) a "
        "calibrated and entropy-based variant that applies per-exit temperature scaling before thresholding, and (4) an "
        "early-exit ViT-Tiny. For each model we measure accuracy, average computation (FLOPs), real batch-1 latency on "
        "a GPU and on a CPU, calibration, and the behaviour of the exit policies under noise, blur and contrast "
        "corruptions.")
    R.body(
        "At this interim stage, the data pipeline (including corrupted test sets) and all four models are implemented "
        "and trained once with preliminary, untuned settings on Kaggle T4 GPUs. Because Models 2 and 3 deliberately "
        "share one trained network and differ only in the exit rule, three training runs cover all four models. "
        "Table 1 summarises the status of each deliverable.")
    rows = [
        ("Literature review (10 papers)", "Complete", "Section 2, Table 2"),
        ("Dataset acquisition and preprocessing", "Complete", "Section 3; src/data.py, src/corruptions.py"),
        ("Model 1: ResNet-18 baseline", "Implemented, trained", "Sections 4.2, 5"),
        ("Model 2: Confidence EE-ResNet-18", "Implemented, trained", "Sections 4.3, 5"),
        ("Model 3: Calibrated / entropy EE-ResNet-18", "Implemented, evaluated", "Sections 4.4, 5"),
        ("Model 4: EE-ViT-Tiny", "Implemented, trained (preliminary)", "Sections 4.5, 5"),
        ("Latency benchmark (GPU and CPU, batch 1)", "Implemented, first measurements", "Section 5.5"),
        ("Corruption robustness study", "Pipeline complete, preliminary results", "Section 5.6"),
        ("Risks and plan to final submission", "Complete", "Section 6"),
    ]
    R.table(["Deliverable", "Status", "Where"], rows, "Interim status of project deliverables.", widths=[6.4, 4.6, 5.6])


def literature(R):
    R.h1("2. Literature Review")
    R.body(
        "We reviewed ten papers covering early-exit CNNs [1], [2], [4], [8], input-dependent routing [3], [6], "
        "calibration [5], [10], corruption robustness [7] and early exiting in vision transformers [9]. Table 2 "
        "summarises each paper; all quantitative results were checked against the original papers.")
    R.table(["#", "Paper", "Method", "Dataset(s)", "Key result", "Relevance to this project"],
            [list(r) for r in LIT], "Summary of reviewed literature.", widths=[1.0, 2.3, 3.4, 2.2, 4.0, 3.7], size=7.5)
    R.h2("2.1 Discussion")
    R.body(
        "**Early-exit networks.** BranchyNet [1] introduced jointly trained side branches and an entropy threshold, "
        "reporting a measured 1.9x speed-up for a branchy ResNet-110 on CIFAR-10 at a cost of 1.5 points of accuracy. "
        "MSDNet [2] showed that naive intermediate classifiers can lower a ResNet's final accuracy by up to 7% on "
        "CIFAR-100, and addressed this with a multi-scale dense architecture that matches ResNet-110 with about one "
        "tenth of the computation. Shallow-Deep Networks [4] attached internal classifiers to off-the-shelf networks, "
        "identified overthinking (correct intermediate predictions that later become wrong), and cut average FLOPs by "
        "more than half with a max-softmax threshold. Zero Time Waste [8] reuses the predictions of exits that did not "
        "fire through cascade connections and ensembling, improving the accuracy of confidence-based exits on CIFAR-100 "
        "at a fixed budget.")
    R.body(
        "**Dynamic routing.** SkipNet [3] and BlockDrop [6] make depth input-dependent by skipping residual blocks "
        "using learned gates or a policy network. They report 30-90% and 20% FLOP reductions respectively, but "
        "BlockDrop's own timing shows that a sequential decision variant can run slower than the unmodified network, "
        "an early warning that FLOP counts and wall-clock time can disagree.")
    R.body(
        "**Calibration and robustness.** Guo et al. [5] showed that modern networks are overconfident and that a single "
        "temperature fitted on validation data almost removes the calibration error without changing predictions. "
        "Meronen et al. [10] found the same overconfidence in early-exit networks and showed that calibrating each exit "
        "improves both ECE and accuracy at a given budget. Hendrycks and Dietterich [7] defined the corruption benchmark "
        "(15 corruption types, 5 severities) that we follow. LGViT [9] brought early exits to vision transformers, "
        "showing that simple heads on shallow transformer blocks are weak and require specialised heads and "
        "self-distillation.")
    R.h2("2.2 Research Gap")
    R.body(
        "Most early-exit work decides with raw softmax confidence or entropy [1], [2], [4], [8], [9] without an explicit "
        "calibration step, although networks, and early exits in particular, are overconfident [5], [10]. Efficiency is "
        "mostly reported as FLOPs or exit depth [2]-[4], [8]-[10], although decision overhead can erase theoretical "
        "savings [6]. Finally, none of the reviewed early-exit methods studies how exit thresholds, exit distributions "
        "and calibration behave when inputs are corrupted [7]. This project addresses all three points on a common "
        "CIFAR-100 setup: it compares raw and calibrated exit rules, reports measured latency next to FLOPs, and "
        "evaluates every exit policy under noise, blur and contrast corruptions.")


def dataset(R):
    R.h1("3. Dataset Acquisition and Preprocessing Pipeline")
    R.h2("3.1 Dataset")
    R.body(
        "CIFAR-100 [11] contains 60,000 RGB images of 32x32 pixels in 100 classes (600 per class), split into 50,000 "
        "training and 10,000 test images. It is downloaded programmatically through torchvision, so every run (local or "
        "on Kaggle) obtains identical data. We hold out a stratified validation set of 5,000 training images (50 per "
        "class, fixed seed 0) that is used for model selection, fitting temperatures and choosing exit thresholds. The "
        "official test set is used only for reporting. Figure 1 shows random test samples.")
    R.figure("dataset_samples.png", "Random CIFAR-100 test images with their class labels.", width=15.5)
    R.table(["Split", "Images", "Per class", "Used for"], [
        ("Train", "45,000", "450", "Gradient updates"),
        ("Validation", "5,000", "50", "Monitoring, temperature fitting, threshold selection"),
        ("Test (clean)", "10,000", "100", "Final accuracy, compute, calibration, latency"),
        ("Test (corrupted)", "6 x 5 x 10,000", "100", "Robustness of exit policies (Section 3.3)"),
    ], "Data splits.", widths=[3.0, 3.0, 2.0, 8.6], align_cols={1, 2})
    R.h2("3.2 Preprocessing and Augmentation")
    R.body(
        "Kaggle machines provide two T4 GPUs but only four CPU cores, so a standard CPU data loader with per-image "
        "PIL transforms becomes the bottleneck for small 32x32 images. Instead, the full uint8 dataset (about 150 MB) "
        "is copied to GPU memory once, and batches are sampled, augmented and normalised on the GPU. Training batches "
        "use the standard light CIFAR augmentation: random 32x32 crops from a 4-pixel reflect-padded image and random "
        "horizontal flips. Images are scaled to [0, 1] and normalised with the per-channel CIFAR-100 training mean "
        "(0.507, 0.487, 0.441) and standard deviation (0.267, 0.256, 0.276), which we verified on our training split. "
        "The ViT-Tiny additionally uses mixup [15] or CutMix [16] on every batch and label smoothing of 0.1, since "
        "transformers trained from scratch on small datasets need stronger regularisation. Evaluation uses no "
        "augmentation.")
    R.code(CODE_SPLIT)
    R.code(CODE_AUG)
    R.h2("3.3 Corrupted Test Sets")
    R.body(
        "To study whether exit decisions stay reliable under distribution shift, we re-implemented six corruptions of "
        "the CIFAR-100-C benchmark [7] in PyTorch so they run on the GPU: Gaussian, shot (Poisson) and impulse "
        "(salt-and-pepper) noise, Gaussian and defocus blur, and contrast reduction, each at five severities with the "
        "constants of the original CIFAR-C generation code. Corrupted images are re-quantised to 8 bits, as in the "
        "stored benchmark, and every corruption and severity uses a fixed random seed, so all models see identical "
        "corrupted images. This gives 30 corrupted copies of the test set (300,000 images) without storing any extra "
        "data. Figure 2 shows one test image under each corruption.")
    R.figure("corruption_examples.png", "One test image under the six corruptions at severities 1, 3 and 5.", width=15)
    R.code(CODE_CORR)


def models(R, S):
    R.h1("4. Models and Training Setup")
    R.h2("4.1 Common Multi-Exit Design")
    R.body(
        "All models share one abstraction: a backbone split into K segments with a classifier head after each segment. "
        "A plain network is the special case K = 1. Training minimises the equally weighted mean of the cross-entropy "
        "losses of all exits, so a single run trains every exit. At inference, the image passes through segment k, head k "
        "produces logits, and the image exits if its score clears a threshold; otherwise it continues. Our "
        "implementation removes exited images from the batch before the next segment, so savings are real and not only "
        "simulated. Stopping at exit k costs the backbone up to segment k plus every head evaluated so far.")
    R.code(CODE_EXIT)
    b = S["baseline"]
    ee = S.get("ee_resnet18")
    vt = S.get("ee_vit_tiny")
    R.h2("4.2 Model 1: ResNet-18 Baseline")
    R.body(
        f"We use the CIFAR variant of ResNet-18 [12]: a 3x3 convolution stem without max-pooling followed by four "
        f"stages of two basic residual blocks (64, 128, 256 and 512 channels; spatial sizes 32, 16, 8 and 4), global "
        f"average pooling and a linear classifier. It has {b['params'] / 1e6:.2f} M parameters and needs "
        f"{b['gflops']:.2f} GFLOPs per image. Every image uses the full network.")
    R.h2("4.3 Model 2: Confidence-Based Early-Exit ResNet-18")
    R.body(
        "The same backbone receives an exit after each of the four stages. Exit heads for stages 1 to 3 use strided "
        "3x3 convolutions (128 channels, batch normalisation, ReLU) to reduce the feature map to 4x4, followed by "
        "global average pooling and a linear layer; the final exit is the standard pooled classifier. Deeper exits "
        "need fewer downsampling steps, so heads become cheaper with depth. An image exits at exit k when the maximum "
        "softmax probability of head k is at least a threshold tau, the rule of SDN [4] and MSDNet [2].")
    if ee:
        rows = []
        for k in range(len(ee["gflops"])):
            rows.append((f"Exit {k + 1}", f"after stage {k + 1}", f"{ee['gflops'][k]:.3f}",
                         pct(ee["gflops"][k] / b["gflops"]) + "%"))
        R.table(["Exit", "Position", "Cumulative GFLOPs", "Relative to ResNet-18"], rows,
                f"Cost of stopping at each exit of EE-ResNet-18 ({ee['params'] / 1e6:.2f} M parameters). Heads add "
                f"{pct(ee['gflops'][-1] / b['gflops'] - 1)}% to the cost of a full pass.",
                widths=[2.5, 3.5, 4.5, 4.5], align_cols={2, 3})
    R.h2("4.4 Model 3: Calibrated / Entropy-Based Early-Exit ResNet-18")
    R.body(
        "Model 3 reuses the trained network of Model 2 and changes only the exit rule, which isolates the effect of the "
        "decision criterion from training noise. First, each exit k is calibrated with temperature scaling [5]: a "
        "scalar T_k is fitted on the validation set by minimising negative log-likelihood, and logits are divided by "
        "T_k before the softmax. This does not change any prediction, but it puts the confidences of different exits on "
        "a common, calibrated scale, so a single threshold means the same thing at every exit. Second, as an "
        "alternative to the maximum probability, we use the normalised entropy score 1 - H(p)/log(100), which uses the "
        "whole predictive distribution, following BranchyNet [1]. We evaluate four rules: max-prob and entropy, each "
        "with raw and temperature-scaled (TS) logits.")
    R.h2("4.5 Model 4: Early-Exit ViT-Tiny")
    vit_cost = f" It has {vt['params'] / 1e6:.2f} M parameters and needs {vt['gflops'][-1]:.2f} GFLOPs for a full pass." if vt else ""
    R.body(
        "The transformer follows the ViT-Tiny configuration [13], [14] (embedding width 192, 12 blocks, 3 attention "
        "heads, MLP ratio 4) adapted to 32x32 inputs with 4x4 patches, giving 64 tokens and learned position "
        "embeddings. Instead of a class token, each exit applies LayerNorm, averages the tokens and uses a linear "
        "classifier. Exits are placed after blocks 3, 6, 9 and 12, so exit costs grow linearly with depth and heads are "
        "almost free. Stochastic depth with rate up to 0.1 is used during training." + vit_cost)
    R.h2("4.6 Training Configuration")
    R.table(["Setting", "ResNet-18 and EE-ResNet-18", "EE-ViT-Tiny"], [
        ("Optimiser", "SGD, momentum 0.9, Nesterov", "AdamW (0.9, 0.999)"),
        ("Learning rate", "0.1, 2 warmup epochs, cosine decay", "1e-3, 5 warmup epochs, cosine decay"),
        ("Weight decay", "5e-4 (not on BN / bias)", "0.05 (not on norm / bias / position)"),
        ("Batch size, epochs", "128, 80", "256, 100"),
        ("Augmentation", "random crop, flip", "random crop, flip, mixup / CutMix"),
        ("Regularisation", "none", "label smoothing 0.1, drop path 0.1, gradient clip 1.0"),
        ("Loss", "mean cross-entropy over exits", "mean cross-entropy over exits"),
        ("Precision, hardware", "fp16 mixed precision, Kaggle T4", "fp16 mixed precision, Kaggle T4"),
    ], "Preliminary training settings (not yet tuned).", widths=[3.4, 6.4, 6.8])
    R.body(
        "The baseline and EE-ResNet-18 were trained in parallel on the two T4 GPUs of one Kaggle session and the ViT in "
        "a second session. Each Kaggle job clones the repository at a given commit, downloads the data, trains, and then "
        "runs the evaluation script, which stores the logits of every exit for the validation set, the clean test set "
        "and all 30 corrupted test sets. All exit policies are then analysed offline from these logits, so new "
        "thresholds or rules need no further GPU time.")
    R.h2("4.7 Evaluation Protocol")
    R.bullets([
        "**Accuracy and compute.** Top-1 test accuracy and mean FLOPs per image, counted with PyTorch's FlopCounterMode "
        "including all heads that an image passes.",
        "**Threshold selection.** For each rule, the threshold is chosen on the validation set as the cheapest one "
        "whose validation accuracy is within 1 point of the same model's final exit; test numbers are reported at that "
        "threshold. Full accuracy and compute curves over all thresholds are also reported.",
        "**Calibration.** Expected calibration error (ECE, 15 bins) per exit, before and after temperature scaling.",
        "**Latency.** Mean time per image when images arrive one at a time (batch 1), using real early-exit inference, "
        "on a Kaggle T4 GPU and on one CPU thread, after warm-up.",
        "**Robustness.** Accuracy, mean compute and accuracy of early-exited images on the 30 corrupted test sets, "
        "using thresholds selected on clean validation data.",
    ])


def _policy_rows(m, b):
    rows = []
    for label, p in m["policies"].items():
        t = p["test"]
        rows.append((label, f"{p['thresh']:.3f}", pct(t["acc"], 2), f"{t['cost']:.3f}",
                     pct(1 - t["cost"] / b["gflops"]), " / ".join(pct(f, 0) for f in t["exit_frac"])))
    return rows


def results(R, S):
    T = narrative(S)
    b = S["baseline"]
    ee, vt = S.get("ee_resnet18"), S.get("ee_vit_tiny")
    R.h1("5. Preliminary Results")
    R.body(
        "All numbers below come from a single training run per model with untuned hyperparameters and should be "
        "read as preliminary. Figure 3 shows the validation accuracy of every exit during training.")
    R.figure("training_curves.png", "Validation accuracy during training (one curve per exit).", width=16)

    R.h2("5.1 Accuracy of Each Model and Exit")
    rows = [("ResNet-18 (Model 1)", f"{b['params'] / 1e6:.2f}", f"{b['gflops']:.2f}", "-", "-", "-",
             pct(b["test_acc"], 2))]
    for name, m in (("EE-ResNet-18 (Models 2, 3)", ee), ("EE-ViT-Tiny (Model 4)", vt)):
        if m:
            accs = [pct(a, 2) for a in m["test_acc"]]
            rows.append((name, f"{m['params'] / 1e6:.2f}", f"{m['gflops'][-1]:.2f}", *accs))
    R.table(["Model", "Params (M)", "GFLOPs (full)", "Exit 1", "Exit 2", "Exit 3", "Final exit"], rows,
            "Top-1 test accuracy (%) of every exit when all images are forced through that exit.",
            widths=[4.6, 1.9, 2.2, 1.9, 1.9, 1.9, 2.2], align_cols={1, 2, 3, 4, 5, 6})
    R.body(T.get("accuracy", ""))

    R.h2("5.2 Accuracy versus Computation")
    R.figure("accuracy_vs_compute.png",
             "Test accuracy against mean GFLOPs per image while sweeping the exit threshold. Dots mark the thresholds "
             "selected on validation; squares are single exits; the star is ResNet-18 with full inference (left).", width=16.5)
    for name, m in (("EE-ResNet-18", ee), ("EE-ViT-Tiny", vt)):
        if m:
            R.table(["Exit rule", "Threshold", "Test acc. (%)", "Mean GFLOPs", "FLOPs saved vs ResNet-18 (%)",
                     "Exits 1/2/3/4 (%)"], _policy_rows(m, b),
                    f"{name}: operating points selected on validation (within 1 point of the final exit).",
                    widths=[3.0, 1.9, 2.2, 2.2, 3.4, 3.9], align_cols={1, 2, 3, 4, 5})
    R.body(T.get("tradeoff", ""))
    R.figure("exit_distribution.png", "Share of test images leaving at each exit at the selected thresholds.", width=14)

    R.h2("5.3 Calibration of the Exits")
    R.figure("calibration_ece.png", "Expected calibration error of each exit before and after temperature scaling "
                                    "(temperatures fitted on validation).", width=16)
    R.figure("reliability_ee_resnet18.png", "Reliability diagrams of the EE-ResNet-18 exits on the test set.", width=16.5)
    R.body(T.get("calibration", ""))

    R.h2("5.4 Accuracy at Fixed Compute Budgets")
    for name, m in (("EE-ResNet-18", ee), ("EE-ViT-Tiny", vt)):
        if m:
            budgets = list(next(iter(m["policies"].values()))["acc_at"].keys())
            rows = [(label, *[pct(p["acc_at"][k], 2) for k in budgets]) for label, p in m["policies"].items()]
            R.table(["Exit rule", *[f"<= {k} GFLOPs" for k in budgets]], rows,
                    f"{name}: best test accuracy (%) reachable within a mean compute budget.",
                    widths=[4.2, *[3.1] * len(budgets)], align_cols=set(range(1, len(budgets) + 1)))
    R.body(T.get("budget", ""))

    R.h2("5.5 Measured Latency")
    lat = S.get("latency", {})
    if lat:
        rows = []
        for name, label in (("resnet18", "ResNet-18"), ("ee_resnet18", "EE-ResNet-18"), ("ee_vit_tiny", "EE-ViT-Tiny")):
            if name in lat:
                for dev in ("gpu", "cpu"):
                    if dev in lat[name]:
                        f = lat[name][dev]["forced_ms"]
                        cells = [f"{v:.2f}" for v in f] if len(f) > 1 else ["-", "-", "-", f"{f[0]:.2f}"]
                        rows.append((label, dev.upper(), *cells))
        R.table(["Model", "Device", "Exit 1", "Exit 2", "Exit 3", "Final exit"], rows,
                "Batch-1 latency (ms per image) when every image is forced to stop at a given exit. GPU: "
                f"{lat['resnet18']['gpu']['device'] if 'gpu' in lat['resnet18'] else '-'}; CPU: "
                f"{lat['resnet18']['cpu']['device']}.", widths=[3.6, 1.8, 2.6, 2.6, 2.6, 2.6], align_cols={2, 3, 4, 5})
        rows = []
        for name, label in (("ee_resnet18", "EE-ResNet-18"), ("ee_vit_tiny", "EE-ViT-Tiny")):
            if name not in lat:
                continue
            for a_gpu, a_cpu in zip(lat[name]["gpu"]["adaptive"], lat[name]["cpu"]["adaptive"]):
                rule = {("max_prob", False): "Max-prob", ("max_prob", True): "Max-prob + TS",
                        ("entropy", True): "Entropy + TS"}[(a_gpu["policy"], a_gpu["calibrated"])]
                if a_gpu["thresh"] not in (0.8, 0.9):
                    continue
                rows.append((label, rule, f"{a_gpu['thresh']:.2f}", f"{a_gpu['ms']:.2f}",
                             f"{lat['resnet18']['gpu']['forced_ms'][0] / a_gpu['ms']:.2f}x", f"{a_cpu['ms']:.2f}",
                             f"{lat['resnet18']['cpu']['forced_ms'][0] / a_cpu['ms']:.2f}x"))
        R.table(["Model", "Exit rule", "Threshold", "GPU ms", "GPU speed-up", "CPU ms", "CPU speed-up"], rows,
                "Measured batch-1 latency of real early-exit inference and speed-up over full ResNet-18 "
                "(GPU on the first 2,000 and CPU on the first 300 test images).",
                widths=[2.9, 2.7, 1.8, 1.8, 2.4, 1.8, 2.4], align_cols={2, 3, 4, 5, 6})
        R.figure("latency_speedup.png", "Speed-up over full ResNet-18 at batch 1 as a function of the threshold: "
                                         "measured (solid markers) and predicted from FLOPs (faded).", width=16)
    R.body(T.get("latency", ""))

    R.h2("5.6 Robustness under Corruptions (Preliminary)")
    R.figure("corruption_robustness.png", "Mean over six corruptions against severity, with thresholds chosen on "
                                          "clean validation data: accuracy (left), compute (middle) and accuracy of "
                                          "the images that exit early (right).", width=16.5)
    corr = S.get("corruptions", {})
    if corr and ee:
        rows = []
        for c in CORRUPTIONS:
            row = [CORR_NAMES[c], pct(corr["resnet18"]["final_acc"][c]["3"])]
            for label in ("Max-prob", "Entropy + TS"):
                m = corr["ee_resnet18"]["policies"][label][c]["3"]
                row += [pct(m["acc"]), f"{m['cost']:.2f}"]
            rows.append(row)
        R.table(["Corruption", "ResNet-18 acc.", "Max-prob acc.", "Max-prob GFLOPs", "Entropy+TS acc.",
                 "Entropy+TS GFLOPs"], rows, "Severity 3: accuracy (%) of full ResNet-18 and of EE-ResNet-18 with "
                                             "two exit rules, and mean GFLOPs spent by the early-exit model.",
                widths=[3.2, 2.5, 2.5, 2.8, 2.7, 2.9], align_cols={1, 2, 3, 4, 5})
    R.figure("corruption_ece_ee_resnet18.png", "ECE of the EE-ResNet-18 exits under corruption, with raw logits and "
                                               "with temperatures fitted on clean validation data.", width=15)
    R.body(T.get("robustness", ""))

    R.h2("5.7 Summary of Preliminary Findings")
    R.bullets(T.get("findings", []))


def plan(R, S):
    R.h1("6. Risks and Plan for the Remaining Work")
    R.h2("6.1 Risks and Mitigation")
    R.table(["Risk", "Impact", "Mitigation"], [
        ("Early-exit heads reduce the accuracy of the final exit (shared backbone must serve shallow and deep heads) [2].",
         "Medium", "Compare EE final exit against Model 1; try exit-loss weighting, gradient scaling for early heads, "
                   "and self-distillation from the final exit."),
        ("Shallow exits are weak, so few images exit early and savings are limited.", "High",
         "Stronger heads at early exits, self-distillation, longer training; report accuracy at fixed budgets."),
        ("ViT-Tiny trained from scratch on 32x32 CIFAR-100 underperforms the CNN, making the comparison unfair.", "Medium",
         "Longer schedule and stronger augmentation; optionally fine-tune an ImageNet-pretrained ViT-Tiny and report "
         "its cost separately."),
        ("FLOP savings do not translate into GPU latency at batch 1 (kernel launch and per-exit synchronisation "
         "overhead).", "High", "Report GPU and CPU separately; reduce synchronisation; study batched inference with "
                               "exited samples removed; discuss where early exits pay off."),
        ("Temperatures fitted on clean data do not transfer to corrupted inputs.", "Medium",
         "Measure ECE under corruption; compare per-exit thresholds and corruption-aware calibration."),
        ("Single runs give noisy comparisons between exit rules.", "Medium",
         "Repeat final experiments with three seeds and report mean and standard deviation."),
        ("Kaggle GPU quota (about 30 h per week) and 12 h session limit.", "Low",
         "Two models per session on 2 x T4; checkpoints every 10 epochs; logits stored so analysis is offline."),
        ("Latency noise on shared cloud machines.", "Low", "Warm-up, many images, fixed thread count, repeated runs."),
    ], "Risks for the remaining work.", widths=[6.6, 1.6, 8.4], size=8.5)
    R.h2("6.2 Remaining Work")
    R.bullets([
        "**Model 1 (Raunak Panshikar).** Train the baseline with a longer schedule (200 epochs) and three seeds; "
        "extend the latency study to several CPU thread counts and batch sizes.",
        "**Model 2 (Aditya Goyal).** Improve early-exit accuracy with exit-loss weighting and self-distillation; "
        "evaluate per-exit thresholds and the overthinking rate (images correct early but wrong at the end).",
        "**Model 3 (Arth Mishra).** Compare temperature scaling with vector scaling and a patience-based rule; fit "
        "calibration under corruption and study whether calibration chosen on clean data remains valid.",
        "**Model 4 (Kartikey Goyal).** Retrain ViT-Tiny with a longer schedule and stronger augmentation; try "
        "distillation from the final exit to early exits as in LGViT [9]; measure latency.",
        "**Joint.** Complete the corruption study (all severities, optionally the official CIFAR-100-C files for a "
        "cross-check), final comparison of the four models, report and presentation.",
    ])
    R.h2("6.3 Timeline to Final Submission")
    R.table(["Sl. No.", "Work element / milestone", "Expected completion"], [
        ("1", "Literature review, dataset preparation and preprocessing", "Done"),
        ("2", "Baseline ResNet-18 and confidence-based early-exit implementation", "Done"),
        ("3", "Calibrated early-exit and ViT-Tiny implementation", "Done"),
        ("4", "Preliminary evaluation and interim report", "Done (04/10/2026)"),
        ("5", "Tuned training of all models (longer schedules, distillation, 3 seeds)", "11/10/2026"),
        ("6", "Corruption experiments, threshold tuning and latency analysis", "17/10/2026"),
        ("7", "Ablations (head design, number of exits) and final comparison", "23/10/2026"),
        ("8", "Final report and presentation preparation", "29/10/2026"),
        ("9", "Final submission", "31/10/2026"),
    ], "Updated work plan.", widths=[1.6, 10.6, 4.4], align_cols={0, 2})


def references(R):
    R.h1("References")
    for i, r in enumerate(REFS, 1):
        p = R.para(f"[{i}] {r}", size=9.5, after=2)
        p.paragraph_format.left_indent = R.doc.styles["Normal"].paragraph_format.left_indent
