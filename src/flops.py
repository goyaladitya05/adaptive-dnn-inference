import torch
from torch.utils.flop_counter import FlopCounterMode

from .models.vit import Attention


def _count(module, x):
    with FlopCounterMode(display=False) as fc:
        out = module(x)
    return fc.get_total_flops(), out


@torch.no_grad()
def exit_costs(model, img_size=32):
    """FLOPs per segment and per head, and the cumulative cost of stopping at each exit.

    Stopping at exit k means running segments 0..k and evaluating heads 0..k.
    """
    model = model.eval().cpu()
    h = torch.zeros(1, 3, img_size, img_size)
    seg, head = [], []
    Attention.manual = True
    try:
        for s, hd in zip(model.segments, model.heads):
            f, h = _count(s, h)
            seg.append(f)
            head.append(_count(hd, h)[0])
    finally:
        Attention.manual = False
    cum, total = [], 0
    for s, hd in zip(seg, head):
        total += s + hd
        cum.append(total)
    return {"segment": seg, "head": head, "cumulative": cum}


def count_params(model):
    return sum(p.numel() for p in model.parameters())
