"""GPU re-implementation of a subset of the CIFAR-10/100-C corruptions (Hendrycks and Dietterich, 2019).

Inputs are float NCHW tensors in [0, 1]. Severity constants follow the CIFAR-C generation code.
"""
import torch
import torch.nn.functional as F


def gaussian_noise(x, s, g):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][s - 1]
    return (x + torch.randn(x.shape, generator=g, device=x.device) * c).clamp(0, 1)


def shot_noise(x, s, g):
    c = [500, 250, 100, 75, 50][s - 1]
    return (torch.poisson(x * c, generator=g) / c).clamp(0, 1)


def impulse_noise(x, s, g):
    c = [0.01, 0.02, 0.03, 0.05, 0.07][s - 1]
    u = torch.rand(x.shape, generator=g, device=x.device)
    x = x.masked_fill(u < c / 2, 0.0)
    return x.masked_fill((u >= c / 2) & (u < c), 1.0)


def _filter(x, k, mode):
    kh, kw = k.shape
    x = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode=mode)
    return F.conv2d(x, k.expand(x.shape[1], 1, kh, kw), groups=x.shape[1])


def _gauss1d(sigma, radius, device):
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x, s, g):
    c = [0.4, 0.6, 0.7, 0.8, 1.0][s - 1]
    k = _gauss1d(c, int(4 * c + 0.5), x.device)
    return _filter(x, torch.outer(k, k), "replicate").clamp(0, 1)


def defocus_blur(x, s, g):
    radius, alias = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1.0, 0.2), (1.5, 0.1)][s - 1]
    t = torch.arange(-8, 9, device=x.device, dtype=torch.float32)
    disk = ((t[:, None] ** 2 + t[None, :] ** 2) <= radius ** 2).float()
    disk = disk / disk.sum()
    g3 = _gauss1d(alias, 1, x.device)
    disk = _filter(disk[None, None], torch.outer(g3, g3), "reflect")[0, 0]
    return _filter(x, disk, "reflect").clamp(0, 1)


def contrast(x, s, g):
    c = [0.75, 0.5, 0.4, 0.3, 0.15][s - 1]
    m = x.mean(dim=(2, 3), keepdim=True)
    return ((x - m) * c + m).clamp(0, 1)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "gaussian_blur": gaussian_blur,
    "defocus_blur": defocus_blur,
    "contrast": contrast,
}


def corrupt(name, severity, generator):
    """Returns a transform that applies the corruption and re-quantizes to 8 bits like the stored CIFAR-C."""
    fn = CORRUPTIONS[name]
    return lambda x: torch.round(fn(x, severity, generator) * 255) / 255
