"""Ops the nets run through when a faster form exists, each with a plain PyTorch fallback."""

from sensecraft.kernels.conv_relu import frozen_conv_relu

__all__ = ["frozen_conv_relu"]
