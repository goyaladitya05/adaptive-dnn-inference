import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2

NUM_CLASSES = 100
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def load_cifar100(root="data", val_size=5000, seed=0):
    """Returns uint8 NHWC arrays for a stratified train/val split and the official test set."""
    tr = CIFAR100(root, train=True, download=True)
    te = CIFAR100(root, train=False, download=True)
    x, y = tr.data, np.asarray(tr.targets)
    rng = np.random.default_rng(seed)
    per_class = val_size // NUM_CLASSES
    val_idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per_class] for c in range(NUM_CLASSES)])
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return {
        "train": (x[mask], y[mask]),
        "val": (x[val_idx], y[val_idx]),
        "test": (te.data, np.asarray(te.targets)),
    }


def to_device(split, device):
    x, y = split
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device), torch.from_numpy(y).long().to(device)


def normalize(x):
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x, pad=4):
    """Random crop with reflect padding and horizontal flip, done on the GPU."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n, 1), device=x.device) + torch.arange(h, device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n, 1), device=x.device) + torch.arange(w, device=x.device)
    x = x[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
          i[:, None, :, None], j[:, None, None, :]]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


class ChunkRandAugment:
    """RandAugment on the GPU; each chunk of the batch draws its own operations."""

    def __init__(self, chunk=32, num_ops=2, magnitude=9):
        self.op = v2.RandAugment(num_ops=num_ops, magnitude=magnitude)
        self.chunk = chunk

    def __call__(self, x):
        u8 = (x * 255).round().to(torch.uint8)
        return torch.cat([self.op(c) for c in u8.split(self.chunk)]).float().div_(255)


class GPULoader:
    """Keeps the whole uint8 split on the device and yields normalized float batches."""

    def __init__(self, x, y, batch_size, train=False, transform=None):
        self.x, self.y, self.bs, self.train, self.transform = x, y, batch_size, train, transform

    def __len__(self):
        return (len(self.y) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, device=self.y.device) if self.train else torch.arange(n, device=self.y.device)
        for k in range(0, n, self.bs):
            b = idx[k:k + self.bs]
            xb = self.x[b].float().div_(255)
            if self.train:
                xb = augment(xb)
            if self.transform is not None:
                xb = self.transform(xb)
            yield normalize(xb), self.y[b]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download CIFAR-100 and print split statistics")
    p.add_argument("--root", default="data")
    args = p.parse_args()
    d = load_cifar100(args.root)
    for name, (x, y) in d.items():
        counts = np.bincount(y, minlength=NUM_CLASSES)
        print(f"{name:5s} images={len(y):6d} shape={x.shape[1:]} per-class min/max={counts.min()}/{counts.max()}")
    x = d["train"][0].reshape(-1, 3) / 255.0
    print("train mean", x.mean(0).round(4), "std", x.std(0).round(4))
