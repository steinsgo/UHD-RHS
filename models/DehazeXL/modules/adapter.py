import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    """1x1 -> DW3x3 -> 1x1 with tanh gate (residual, same shape in/out).

    channels: input/output channels (C)
    r: reduction ratio for bottleneck (e.g., 1/16)
    """

    def __init__(self, channels: int, r: float = 1 / 16):
        super().__init__()
        mid = max(1, int(channels * r))
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, groups=mid, bias=False)
        self.expand = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        g = self.gate(x)
        y = self.reduce(x)
        y = self.dw(y)
        y = self.expand(y)
        # gated residual
        y = self.norm(identity + g * y)
        return y


def make_stage_adapters(channels_per_stage, r: float = 1 / 16) -> nn.ModuleList:
    """Create one adapter per encoder stage to keep parameter cost small.

    channels_per_stage: list[int] aligned with encoder feature channels.
    """
    return nn.ModuleList([BottleneckAdapter(c, r=r) for c in channels_per_stage])

