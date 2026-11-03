"""SqueezeNet 1.1 features as torchvision numbers them, tapped at seven depths."""

import torch
from torch import Tensor, nn

from sensecraft.nets.base import ConvReLU, TapSequence, conv_relu

TAPS = [1, 4, 7, 9, 10, 11, 12]
CHANNELS = [64, 128, 256, 384, 384, 512, 512]


class Fire(nn.Module):
    """A 1x1 squeeze, then 1x1 and 3x3 expands concatenated on channels."""

    def __init__(self, in_channels: int, squeeze: int, expand: int):
        super().__init__()
        self.squeeze = ConvReLU(in_channels, squeeze, kernel_size=1)
        self.expand1x1 = ConvReLU(squeeze, expand, kernel_size=1)
        self.expand3x3 = ConvReLU(squeeze, expand, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)


def squeezenet() -> TapSequence:
    """SqueezeNet 1.1 up to its last fire module; weights come from `sensecraft.weights`."""
    pool = nn.MaxPool2d(3, 2, ceil_mode=True)
    layers = [
        *conv_relu(3, 64, first=True, kernel_size=3, stride=2),
        pool,
        Fire(64, 16, 64),
        Fire(128, 16, 64),
        pool,
        Fire(128, 32, 128),
        Fire(256, 32, 128),
        pool,
        Fire(256, 48, 192),
        Fire(384, 48, 192),
        Fire(384, 64, 256),
        Fire(512, 64, 256),
    ]
    return TapSequence(layers, TAPS)
