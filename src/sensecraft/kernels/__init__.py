"""Ops the nets run through when a faster or more exact form exists, each with a plain fallback."""

from sensecraft.kernels.conv_relu import frozen_conv_relu
from sensecraft.kernels.max_pool import max_pool2x2
from sensecraft.kernels.split_input import compute_dtype, split_layer, wide

__all__ = ["compute_dtype", "frozen_conv_relu", "max_pool2x2", "split_layer", "wide"]
