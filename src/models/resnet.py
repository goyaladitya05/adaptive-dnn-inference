import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MultiExitNet


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class PoolHead(nn.Module):
    def __init__(self, cin, num_classes):
        super().__init__()
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class ConvHead(nn.Module):
    """Strided 3x3 convs down to 4x4 resolution, then a pooled linear classifier."""

    def __init__(self, cin, size, num_classes, width=128):
        super().__init__()
        layers = []
        while size > 4:
            layers += [nn.Conv2d(cin, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            cin, size = width, size // 2
        self.features = nn.Sequential(*layers)
        self.pool = PoolHead(cin, num_classes)

    def forward(self, x):
        return self.pool(self.features(x))


def _stem():
    return nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))


def _stages():
    def stage(cin, cout, stride):
        return nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1))

    return [stage(64, 64, 1), stage(64, 128, 2), stage(128, 256, 2), stage(256, 512, 2)]


def resnet18(num_classes=100):
    """CIFAR ResNet-18: 3x3 stem, no max-pool, single exit."""
    return MultiExitNet([nn.Sequential(_stem(), *_stages())], [PoolHead(512, num_classes)])


def ee_resnet18(num_classes=100):
    """ResNet-18 with exits after each of the four residual stages."""
    s = _stages()
    segments = [nn.Sequential(_stem(), s[0]), s[1], s[2], s[3]]
    heads = [
        ConvHead(64, 32, num_classes),
        ConvHead(128, 16, num_classes),
        ConvHead(256, 8, num_classes),
        PoolHead(512, num_classes),
    ]
    return MultiExitNet(segments, heads)
