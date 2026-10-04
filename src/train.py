import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import ChunkRandAugment, GPULoader, load_cifar100, to_device
from .models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--opt", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=2, help="warmup epochs")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mix", action="store_true", help="mixup or cutmix on every batch")
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--distill", type=float, default=0.0, help="weight of self-distillation from the final exit")
    p.add_argument("--kd-temp", type=float, default=3.0)
    p.add_argument("--randaug", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="train on the first N images (smoke tests)")
    return p.parse_args()


def mix_batch(x, y):
    perm = torch.randperm(x.shape[0], device=x.device)
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(0.8, 0.8))
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    lam = float(np.random.beta(1.0, 1.0))
    h, w = x.shape[2:]
    rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    cy, cx = np.random.randint(h), np.random.randint(w)
    y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, h), max(cx - rw // 2, 0), min(cx + rw // 2, w)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    return x, y, y[perm], 1 - (y1 - y0) * (x1 - x0) / (h * w)


def multi_exit_loss(outs, ya, yb, lam, ls, alpha=0.0, tau=3.0):
    """Equal-weight average over exits; early exits optionally also match the final exit (self-distillation)."""
    def ce(o):
        loss = lam * F.cross_entropy(o, ya, label_smoothing=ls)
        if lam < 1:
            loss = loss + (1 - lam) * F.cross_entropy(o, yb, label_smoothing=ls)
        return loss

    teacher = F.softmax(outs[-1].detach().float() / tau, -1)
    total = ce(outs[-1])
    for o in outs[:-1]:
        loss = ce(o)
        if alpha:
            kd = F.kl_div(F.log_softmax(o.float() / tau, -1), teacher, reduction="batchmean") * tau ** 2
            loss = (1 - alpha) * loss + alpha * kd
        total = total + loss
    return total / len(outs)


@torch.no_grad()
def evaluate(model, loader, amp):
    model.eval()
    correct, n = None, 0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            outs = model(x)
        c = torch.stack([o.argmax(1).eq(y).sum() for o in outs])
        correct = c if correct is None else correct + c
        n += len(y)
    return (correct.float() / n).tolist()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    amp = dev.type == "cuda"
    torch.backends.cudnn.benchmark = True

    data = load_cifar100(args.data)
    xtr, ytr = to_device(data["train"], dev)
    if args.limit:
        xtr, ytr = xtr[:args.limit], ytr[:args.limit]
    xva, yva = to_device(data["val"], dev)
    train_loader = GPULoader(xtr, ytr, args.bs, train=True, transform=ChunkRandAugment() if args.randaug else None)
    val_loader = GPULoader(xva, yva, 500)

    model = build_model(args.model).to(dev)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "pos" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or "pos" in n]
    groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
    if args.opt == "sgd":
        opt = torch.optim.SGD(groups, lr=args.lr, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
    steps = args.epochs * len(train_loader)
    warm = args.warmup * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = []
    for epoch in range(args.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in train_loader:
            ya, yb, lam = y, y, 1.0
            if args.mix:
                x, ya, yb, lam = mix_batch(x, y)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = multi_exit_loss(model(x), ya, yb, lam, args.label_smoothing, args.distill, args.kd_temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        val_acc = evaluate(model, val_loader, amp)
        rec = {"epoch": epoch + 1, "loss": tot / nb, "val_acc": val_acc, "lr": sched.get_last_lr()[0],
               "time": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            torch.save({"model": args.model, "args": vars(args), "state_dict": model.state_dict()},
                       os.path.join(args.out, "model.pt"))
            with open(os.path.join(args.out, "history.json"), "w") as f:
                json.dump(history, f)


if __name__ == "__main__":
    main()
