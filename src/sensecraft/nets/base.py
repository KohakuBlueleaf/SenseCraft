"""Feature nets: frozen, they return the features at their taps and know nothing about losses."""

import torch
from torch import Tensor, nn

from sensecraft.kernels import frozen_conv_relu


class FeatureNet(nn.Module):
    """A frozen net; `forward` returns the features at its taps, shallow first."""

    def freeze(self) -> "FeatureNet":
        """Eval mode with every parameter held fixed."""
        self.requires_grad_(False)
        return self.eval()

    @property
    def dtype(self) -> torch.dtype:
        """The dtype the net computes in: that of its weights."""
        return next(self.parameters()).dtype

    def train(self, mode: bool = True) -> "FeatureNet":
        # Frozen nets never leave eval mode, whatever the module holding them does
        return super().train(False)


class TapSequence(FeatureNet):
    """Layers run in order; the outputs of the layers at `taps` are the features."""

    def __init__(self, layers: list[nn.Module], taps: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(layers[: max(taps) + 1])
        self.taps = set(taps)

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in self.taps:
                features.append(x)
        return features


class ConvReLU(nn.Conv2d):
    """A conv followed by ReLU, run as one frozen op."""

    def forward(self, x: Tensor) -> Tensor:
        return frozen_conv_relu(x, self.weight, self.bias, self.stride, self.padding)


def conv_relu(in_channels: int, out_channels: int, **conv) -> list[nn.Module]:
    """A fused conv and ReLU, then a placeholder keeping torchvision's layer numbering."""
    return [ConvReLU(in_channels, out_channels, **conv), nn.Identity()]
