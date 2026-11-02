"""Ops the nets run through when a faster form exists, each with a plain PyTorch fallback."""

from sensecraft.kernels.conv_relu import frozen_conv_relu
from sensecraft.kernels.max_pool import max_pool2x2

__all__ = ["frozen_conv_relu", "max_pool2x2"]
