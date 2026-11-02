"""VGG16 features as torchvision numbers them, tapped after each block's last ReLU."""

from torch import Tensor, nn

from sensecraft.kernels import max_pool2x2
from sensecraft.nets.base import TapSequence, conv_relu


class MaxPool2x2(nn.Module):
    """2x2 max-pool with stride 2 that keeps no indices for backward."""

    def forward(self, x: Tensor) -> Tensor:
        return max_pool2x2(x)


# Channels per conv, "pool" between blocks
LAYOUT = [
    64,
    64,
    "pool",
    128,
    128,
    "pool",
    256,
    256,
    256,
    "pool",
    512,
    512,
    512,
    "pool",
]
LAYOUT += [512, 512, 512]
TAPS = [3, 8, 15, 22, 29]
CHANNELS = [64, 128, 256, 512, 512]


def vgg16() -> TapSequence:
    """VGG16 up to its last ReLU; weights come from `sensecraft.weights`."""
    layers: list[nn.Module] = []
    channels = 3
    for item in LAYOUT:
        if item == "pool":
            layers.append(MaxPool2x2())
            continue
        layers += conv_relu(channels, item, kernel_size=3, padding=1)
        channels = item
    return TapSequence(layers, TAPS)
