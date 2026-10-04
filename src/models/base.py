import torch.nn as nn


class MultiExitNet(nn.Module):
    """Backbone split into segments with one classifier head after each segment.

    A plain network is the special case with a single segment and a single head.
    """

    def __init__(self, segments, heads):
        super().__init__()
        assert len(segments) == len(heads)
        self.segments = nn.ModuleList(segments)
        self.heads = nn.ModuleList(heads)

    @property
    def num_exits(self):
        return len(self.heads)

    def forward(self, x):
        outs = []
        for seg, head in zip(self.segments, self.heads):
            x = seg(x)
            outs.append(head(x))
        return outs
