import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MultiExitNet


class Attention(nn.Module):
    manual = False  # explicit matmuls so FlopCounterMode sees the attention core

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if self.manual:
            x = (q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5).softmax(-1) @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.drop_path = drop_path

    def _drop(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path).to(x.dtype)
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        x = x + self._drop(self.attn(self.norm1(x)))
        return x + self._drop(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, (img_size // patch) ** 2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2) + self.pos


class TokenHead(nn.Module):
    """LayerNorm, mean over tokens, linear classifier."""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(self.norm(x).mean(1))


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.zeros_(m.bias)


def ee_vit_tiny(num_classes=100, img_size=32, patch=4, dim=192, depth=12, heads=3, exits=(3, 6, 9, 12), drop_path=0.1):
    """ViT-Tiny (patch 4 for 32x32 inputs) with exits after blocks 3, 6, 9 and 12."""
    blocks = [Block(dim, heads, drop_path=drop_path * i / (depth - 1)) for i in range(depth)]
    segments, start = [], 0
    for e in exits:
        segments.append(nn.Sequential(*blocks[start:e]))
        start = e
    segments[0] = nn.Sequential(PatchEmbed(img_size, patch, dim), *segments[0])
    model = MultiExitNet(segments, [TokenHead(dim, num_classes) for _ in exits])
    model.apply(_init)
    return model
