"""Latency of trained runs measured on one idle machine: batch-1 GPU and CPU, batched GPU throughput."""
import argparse
import copy
import glob
import json
import os
import platform
import re
import time

import numpy as np
import torch

from .data import GPULoader, load_cifar100, to_device
from .exits import POLICIES, SCORES, adaptive_forward, exit_scores, pick_threshold, sweep
from .models import build_model

GRID = (0.6, 0.8, 0.9, 0.95)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def batch1_latency(model, xs, dev, **kw):
    """Mean milliseconds per image when images arrive one at a time."""
    for i in range(min(20, len(xs))):
        adaptive_forward(model, xs[i:i + 1], **kw)
    _sync(dev)
    preds, exits = [], []
    t0 = time.perf_counter()
    for i in range(len(xs)):
        p, e = adaptive_forward(model, xs[i:i + 1], **kw)
        preds.append(p)
        exits.append(e)
    _sync(dev)
    ms = (time.perf_counter() - t0) * 1000 / len(xs)
    return ms, torch.cat(preds).cpu(), torch.cat(exits).cpu()


def throughput(model, xs, dev, bs=256, **kw):
    for i in range(0, min(len(xs), 4 * bs), bs):
        adaptive_forward(model, xs[i:i + bs], **kw)
    _sync(dev)
    t0 = time.perf_counter()
    for i in range(0, len(xs), bs):
        adaptive_forward(model, xs[i:i + bs], **kw)
    _sync(dev)
    return len(xs) / (time.perf_counter() - t0)


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def operating_points(run, metrics, tol=0.01):
    """Thresholds selected on validation for each policy, with the same rule as the analysis."""
    lv = torch.from_numpy(np.load(os.path.join(run, "logits", "val.npy")).astype(np.float32))
    correct = lv.argmax(-1).numpy() == np.load(os.path.join(run, "logits", "val_labels.npy"))
    costs = metrics["flops"]["cumulative"]
    target = metrics["acc"]["val"][-1] - tol
    return {label: pick_threshold(sweep(exit_scores(lv, p, cal, metrics["temperature"]), correct, costs), target)
            for p, cal, label in POLICIES}


def configs(metrics, ops):
    rows = [{"label": label, "policy": p, "calibrated": cal, "thresh": ops[label]} for p, cal, label in POLICIES]
    rows += [{"label": "Max-prob", "policy": "max_prob", "calibrated": False, "thresh": t} for t in GRID]
    return rows


def _kw(row, temps):
    return {"score": SCORES[row["policy"]], "thresh": row["thresh"], "temps": temps if row["calibrated"] else None}


def latency_suite(model, xs, ys, dev, rows, temps):
    k = model.num_exits
    res = {"forced_ms": [batch1_latency(model, xs, dev, score=SCORES["max_prob"], thresh=2.0, force_exit=i)[0]
                         for i in range(k)], "adaptive": []}
    for row in rows:
        ms, preds, exits = batch1_latency(model, xs, dev, **_kw(row, temps))
        res["adaptive"].append({**row, "ms": ms, "acc": preds.eq(ys.cpu()).float().mean().item(),
                                "exit_frac": torch.bincount(exits, minlength=k).div(len(exits)).tolist()})
    return res


def throughput_suite(model, xs, dev, rows, temps):
    res = {"full": throughput(model, xs, dev, score=SCORES["max_prob"], thresh=2.0), "adaptive": []}
    for row in rows:
        res["adaptive"].append({**row, "img_per_s": throughput(model, xs, dev, **_kw(row, temps))})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="globs matching run directories")
    p.add_argument("--only", default=".*", help="regex on the run directory name")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-images", type=int, default=2000)
    p.add_argument("--cpu-images", type=int, default=500)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    args = p.parse_args()
    dev = torch.device(args.device)
    runs = sorted({os.path.dirname(m) for g in args.runs for m in glob.glob(os.path.join(g, "metrics.json"))})
    runs = [r for r in runs if re.search(args.only, os.path.basename(r))]
    print("runs:", runs, flush=True)
    data = load_cifar100(args.data)
    x, y = to_device(data["test"], dev)
    xs = next(iter(GPULoader(x, y, len(y))))[0]
    for run in runs:
        with open(os.path.join(run, "metrics.json")) as f:
            metrics = json.load(f)
        ck = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
        model = build_model(ck["model"])
        model.load_state_dict(ck["state_dict"])
        model.eval().to(dev)
        temps = metrics["temperature"]
        ops = operating_points(run, metrics) if model.num_exits > 1 else {}
        rows = configs(metrics, ops) if model.num_exits > 1 else []
        out = {"run": os.path.basename(run), "model": ck["model"], "ops": ops}
        if dev.type == "cuda":
            n = args.gpu_images
            out["gpu"] = latency_suite(model, xs[:n], y[:n], dev, rows, temps)
            out["gpu"]["batched"] = throughput_suite(model, xs, dev, rows, temps)
            out["gpu"]["device"] = torch.cuda.get_device_name(dev)
        cpu_model = copy.deepcopy(model).cpu()
        n = args.cpu_images
        for t in args.threads:
            torch.set_num_threads(t)
            out[f"cpu{t}"] = latency_suite(cpu_model, xs[:n].cpu(), y[:n].cpu(), torch.device("cpu"), rows, temps)
            out[f"cpu{t}"]["device"] = f"{cpu_name()} ({t} thread{'s' if t > 1 else ''})"
        d = os.path.join(args.out, os.path.basename(run))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "latency.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(out["run"], {k: v["forced_ms"] for k, v in out.items() if isinstance(v, dict) and "forced_ms" in v},
              flush=True)


if __name__ == "__main__":
    main()
