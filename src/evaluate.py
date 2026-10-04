import argparse
import copy
import json
import os
import platform
import time

import numpy as np
import torch

from .corruptions import CORRUPTIONS, corrupt
from .data import GPULoader, load_cifar100, to_device
from .exits import SCORES, adaptive_forward, ece, fit_temperature
from .flops import count_params, exit_costs
from .models import build_model

SEVERITIES = (1, 2, 3, 4, 5)
LAT_THRESH = (0.6, 0.8, 0.9, 0.95)
LAT_POLICIES = (("max_prob", False), ("max_prob", True), ("entropy", True))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training output dir containing model.pt")
    p.add_argument("--data", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-images", type=int, default=2000)
    p.add_argument("--cpu-images", type=int, default=300)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--no-corrupt", action="store_true")
    p.add_argument("--no-latency", action="store_true")
    return p.parse_args()


@torch.no_grad()
def predict(model, x, y, transform=None):
    """Logits of every exit as a [K, N, C] float16 CPU tensor."""
    return torch.cat([torch.stack(model(xb)).half().cpu() for xb, _ in GPULoader(x, y, 500, transform=transform)], 1)


def per_exit_acc(logits, labels):
    return logits.argmax(-1).eq(labels).float().mean(1).tolist()


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


def _policies(temps):
    for name, calibrated in LAT_POLICIES:
        for th in LAT_THRESH:
            yield {"policy": name, "calibrated": calibrated, "thresh": th}, \
                {"score": SCORES[name], "thresh": th, "temps": temps if calibrated else None}


def latency_suite(model, xs, ys, dev, temps):
    """Batch-1 latency of every forced exit and of each threshold policy."""
    k_total = model.num_exits
    res = {"forced_ms": [], "adaptive": []}
    for k in range(k_total):
        res["forced_ms"].append(batch1_latency(model, xs, dev, score=SCORES["max_prob"], thresh=2.0, force_exit=k)[0])
    if k_total > 1:
        for row, kw in _policies(temps):
            ms, preds, exits = batch1_latency(model, xs, dev, **kw)
            row.update(ms=ms, acc=preds.eq(ys.cpu()).float().mean().item(),
                       exit_frac=torch.bincount(exits, minlength=k_total).div(len(exits)).tolist())
            res["adaptive"].append(row)
    return res


def throughput_suite(model, xs, dev, temps):
    """Images per second with batches of 256, exited samples removed from the batch."""
    res = {"full": throughput(model, xs, dev, score=SCORES["max_prob"], thresh=2.0), "adaptive": []}
    if model.num_exits > 1:
        for row, kw in _policies(temps):
            row["img_per_s"] = throughput(model, xs, dev, **kw)
            res["adaptive"].append(row)
    return res


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def main():
    args = parse_args()
    dev = torch.device(args.device)
    ck = torch.load(os.path.join(args.run, "model.pt"), map_location="cpu")
    model = build_model(ck["model"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out = {"model": ck["model"], "train_args": ck["args"], "params": count_params(model),
           "flops": exit_costs(model), "commit": os.environ.get("GIT_COMMIT", "")}
    model.to(dev)

    data = load_cifar100(args.data)
    logit_dir = os.path.join(args.run, "logits")
    os.makedirs(logit_dir, exist_ok=True)
    splits = {}
    for split in ("val", "test"):
        x, y = to_device(data[split], dev)
        logits = predict(model, x, y)
        splits[split] = (x, y, logits)
        np.save(os.path.join(logit_dir, f"{split}.npy"), logits.numpy())
        np.save(os.path.join(logit_dir, f"{split}_labels.npy"), y.cpu().numpy())
    yv, yt = splits["val"][1].cpu(), splits["test"][1].cpu()
    lv, lt = splits["val"][2], splits["test"][2]
    out["acc"] = {"val": per_exit_acc(lv, yv), "test": per_exit_acc(lt, yt)}
    temps = [fit_temperature(lv[k], yv) for k in range(model.num_exits)]
    out["temperature"] = temps
    out["ece_test"] = {"raw": [ece(lt[k], yt) for k in range(len(temps))],
                       "calibrated": [ece(lt[k], yt, temps[k]) for k in range(len(temps))]}
    print(json.dumps({k: out[k] for k in ("acc", "temperature", "ece_test")}), flush=True)

    if not args.no_corrupt:
        xt = splits["test"][0]
        out["corrupt_acc"] = {}
        for ci, name in enumerate(CORRUPTIONS):
            out["corrupt_acc"][name] = {}
            for s in SEVERITIES:
                g = torch.Generator(device=dev).manual_seed(1000 * ci + s)
                logits = predict(model, xt, splits["test"][1], transform=corrupt(name, s, g))
                np.save(os.path.join(logit_dir, f"{name}_{s}.npy"), logits.numpy())
                out["corrupt_acc"][name][s] = per_exit_acc(logits, yt)
            print(name, json.dumps(out["corrupt_acc"][name]), flush=True)

    if not args.no_latency:
        x, y = splits["test"][0], splits["test"][1]
        xs = next(iter(GPULoader(x, y, len(y))))[0]
        lat = {}
        if dev.type == "cuda":
            lat["gpu"] = latency_suite(model, xs[:args.gpu_images], y[:args.gpu_images], dev, temps)
            lat["gpu"]["batched"] = throughput_suite(model, xs, dev, temps)
            lat["gpu"]["device"] = torch.cuda.get_device_name(dev)
        torch.set_num_threads(args.cpu_threads)
        cpu_model = copy.deepcopy(model).cpu()
        n = args.cpu_images
        lat["cpu"] = latency_suite(cpu_model, xs[:n].cpu(), y[:n].cpu(), torch.device("cpu"), temps)
        lat["cpu"]["device"] = f"{cpu_name()} ({args.cpu_threads} thread)"
        out["latency"] = lat

    with open(os.path.join(args.run, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(args.run, "metrics.json"), flush=True)


if __name__ == "__main__":
    main()
