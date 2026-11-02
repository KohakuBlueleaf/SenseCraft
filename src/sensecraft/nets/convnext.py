"""ConvNeXt as one flat run of layers: stem conv and norm, blocks, and norm and conv between stages.

The flat numbering is the one the 0.3.13 losses tapped: torchvision's `features` and the DINOv3
stages both unrolled into single layers. Torchvision ConvNeXt and DINOv3 ConvNeXt share it; only
the sizes, and where DINOv3 keeps them, its `config.json`, differ.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sensecraft.nets.base import TapSequence


@dataclass(frozen=True)
class ConvNeXtConfig:
    """The fields of a ConvNeXt `config.json` the net needs."""

    hidden_sizes: list[int]
    depths: list[int]
    layer_norm_eps: float
    hidden_act: str

    @classmethod
    def of(cls, config: dict) -> "ConvNeXtConfig":
        """The config from a parsed `config.json`."""
        return cls(**{name: config[name] for name in cls.__dataclass_fields__})

    @property
    def layer_count(self) -> int:
        """Flat layers in the whole net: a conv and a norm per stage, then its blocks."""
        return sum(self.depths) + 2 * len(self.depths)


# Torchvision's ImageNet ConvNeXts, as `torchvision.models.convnext_<size>` builds them
TORCHVISION = {
    "tiny": ConvNeXtConfig([96, 192, 384, 768], [3, 3, 9, 3], 1e-6, "gelu"),
    "small": ConvNeXtConfig([96, 192, 384, 768], [3, 3, 27, 3], 1e-6, "gelu"),
    "base": ConvNeXtConfig([128, 256, 512, 1024], [3, 3, 27, 3], 1e-6, "gelu"),
    "large": ConvNeXtConfig([192, 384, 768, 1536], [3, 3, 27, 3], 1e-6, "gelu"),
}

ACTIVATIONS = {"gelu": F.gelu}


class ChannelNorm(nn.LayerNorm):
    """LayerNorm over the channels of a channel-first map, through channels-last as upstream does."""

    def forward(self, x: Tensor) -> Tensor:
        x = F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return x.permute(0, 3, 1, 2)


class Block(nn.Module):
    """Depthwise 7x7, then channels-last norm, up, activation, down and layer scale, plus the input."""

    def __init__(self, channels: int, config: ConvNeXtConfig):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.layer_norm = nn.LayerNorm(channels, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Linear(channels, 4 * channels)
        self.pointwise_conv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.ones(channels))
        self.act = ACTIVATIONS[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        branch = self.depthwise_conv(x).permute(0, 2, 3, 1)
        branch = self.pointwise_conv2(
            self.act(self.pointwise_conv1(self.layer_norm(branch)))
        )
        return x + (branch * self.gamma).permute(0, 3, 1, 2)


def convnext(config: ConvNeXtConfig, taps: list[int]) -> TapSequence:
    """ConvNeXt up to its deepest tap; weights come from `sensecraft.weights`."""
    eps = config.layer_norm_eps
    first = config.hidden_sizes[0]
    layers: list[nn.Module] = [
        nn.Conv2d(3, first, kernel_size=4, stride=4),
        ChannelNorm(first, eps=eps),
    ]
    previous = first
    for stage, (channels, depth) in enumerate(
        zip(config.hidden_sizes, config.depths, strict=True)
    ):
        # Every stage past the stem opens with a norm and a strided conv
        if stage > 0:
            layers += [
                ChannelNorm(previous, eps=eps),
                nn.Conv2d(previous, channels, kernel_size=2, stride=2),
            ]
        layers += [Block(channels, config) for _ in range(depth)]
        previous = channels
    return TapSequence(layers, taps)
