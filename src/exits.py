"""Exit criteria, temperature scaling, calibration metrics and early-exit inference."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def max_prob(logits, t=1.0):
    return F.softmax(logits.float() / t, -1).amax(-1)


def neg_entropy(logits, t=1.0):
    """1 - H(p) / log(C), so that larger means more confident, as with max_prob."""
    logp = F.log_softmax(logits.float() / t, -1)
    h = -(logp.exp() * logp).sum(-1)
    return 1 - h / math.log(logits.shape[-1])


SCORES = {"max_prob": max_prob, "entropy": neg_entropy}


def fit_temperature(logits, labels, iters=200):
    """Single temperature minimizing validation NLL (Guo et al., 2017)."""
    logits = logits.float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def reliability(logits, labels, t=1.0, bins=15):
    """Per-bin (confidence, accuracy, fraction) and the expected calibration error."""
    conf, pred = F.softmax(logits.float() / t, -1).max(-1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            frac = m.float().mean().item()
            c, a = conf[m].mean().item(), correct[m].mean().item()
            ece += frac * abs(c - a)
            rows.append((c, a, frac))
    return rows, ece


def ece(logits, labels, t=1.0, bins=15):
    return reliability(logits, labels, t, bins)[1]


def assign_exits(logits, score, thresh, temps=None):
    """Offline routing. logits: [K, N, C]. Returns the exit index taken by each sample."""
    k_total, n, _ = logits.shape
    temps = temps or [1.0] * k_total
    exit_at = torch.full((n,), k_total - 1, dtype=torch.long)
    pending = torch.ones(n, dtype=torch.bool)
    for k in range(k_total - 1):
        hit = pending & (score(logits[k], temps[k]) >= thresh)
        exit_at[hit] = k
        pending &= ~hit
    return exit_at


def simulate(logits, labels, costs, score, thresh, temps=None):
    """Accuracy, mean cost and exit histogram of a threshold policy, computed from stored logits."""
    exit_at = assign_exits(logits, score, thresh, temps)
    n = labels.shape[0]
    pred = logits.argmax(-1)[exit_at, torch.arange(n)]
    costs = torch.as_tensor(costs, dtype=torch.float64)
    return {
        "thresh": float(thresh),
        "acc": pred.eq(labels).float().mean().item(),
        "cost": costs[exit_at].mean().item(),
        "exit_frac": torch.bincount(exit_at, minlength=logits.shape[0]).div(n).tolist(),
    }


@torch.no_grad()
def adaptive_forward(model, x, score, thresh, temps=None, force_exit=None):
    """Real early-exit inference: samples that exit are dropped from the batch before the next segment."""
    k_total = model.num_exits
    temps = temps or [1.0] * k_total
    last = k_total - 1 if force_exit is None else force_exit
    n = x.shape[0]
    preds = torch.empty(n, dtype=torch.long, device=x.device)
    exit_at = torch.full((n,), last, dtype=torch.long, device=x.device)
    alive = torch.arange(n, device=x.device)
    h = x
    for k in range(last + 1):
        h = model.segments[k](h)
        logits = model.heads[k](h)
        if k == last:
            preds[alive] = logits.argmax(1)
            break
        done = score(logits, temps[k]) >= thresh
        if force_exit is not None:
            done = torch.zeros_like(done)
        if done.any():
            preds[alive[done]] = logits[done].argmax(1)
            exit_at[alive[done]] = k
            keep = ~done
            if not keep.any():
                break
            alive, h = alive[keep], h[keep]
    return preds, exit_at


THRESH = np.unique(np.concatenate([np.linspace(0, 0.9, 91), np.linspace(0.9, 1.0, 201), [1.01]]))
POLICIES = [("max_prob", False, "Max-prob"), ("entropy", False, "Entropy"),
            ("max_prob", True, "Max-prob + TS"), ("entropy", True, "Entropy + TS")]


def exit_scores(logits, policy, calibrated, temps):
    """Scores of the non-final exits as a [K-1, N] array."""
    k = logits.shape[0]
    return np.stack([SCORES[policy](logits[i], temps[i] if calibrated else 1.0).numpy() for i in range(k - 1)])


def route(scores, th):
    """First exit whose score clears its threshold (one value, or one per exit), else the last exit."""
    th = np.asarray(th, dtype=float)
    if th.ndim:
        th = th[:, None]
    hit = np.vstack([scores >= th, np.ones((1, scores.shape[1]), bool)])
    return hit.argmax(0)


def routing_stats(exit_at, correct, costs):
    k, n = correct.shape
    return {"acc": float(correct[exit_at, np.arange(n)].mean()), "cost": float(np.asarray(costs)[exit_at].mean()),
            "exit_frac": (np.bincount(exit_at, minlength=k) / n).tolist()}


def sweep(scores, correct, costs, grid=THRESH):
    return [dict(thresh=float(t), **routing_stats(route(scores, t), correct, costs)) for t in grid]


def pick_threshold(rows, target):
    """Cheapest threshold whose accuracy reaches the target."""
    ok = [r for r in rows if r["acc"] >= target]
    return min(ok, key=lambda r: (r["cost"], -r["acc"]))["thresh"]


def tune_per_exit(scores, correct, costs, target, init, grid=THRESH, passes=3):
    """Coordinate descent over one threshold per exit, minimising cost subject to accuracy >= target."""
    th = np.full(scores.shape[0], float(init))
    best = routing_stats(route(scores, th), correct, costs)
    for _ in range(passes):
        for k in range(len(th)):
            for t in grid:
                cand = th.copy()
                cand[k] = t
                r = routing_stats(route(scores, cand), correct, costs)
                if r["acc"] >= target and (r["cost"], -r["acc"]) < (best["cost"], -best["acc"]):
                    th, best = cand, r
    return th.tolist(), best
