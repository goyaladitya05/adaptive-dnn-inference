"""Exit criteria, temperature scaling, calibration metrics and early-exit inference."""
import math

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
