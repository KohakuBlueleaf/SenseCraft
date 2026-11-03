"""The first layer of a feature net, keeping the low bits of its fp32 input.

A net computing in fp16, bf16 or TF32 rounds its input to that dtype's mantissa before the first
layer, so two images that differ by less than that spacing give the same features and the gradient
between them loses its direction. Here the input is split into a high part, the input rounded to the
compute dtype, and the remainder, scaled up so it keeps its own exponent. Both run through the layer
in the compute dtype and the outputs add in fp32. The layer is linear, so the sum is the layer of
the whole input, up to the rounding of the remainder.
"""

from collections.abc import Callable

import torch
from torch import Tensor

LOW_SCALE = (
    2.0**10
)  # lifts the remainder clear of fp16 subnormals; undone after the layer
TF32_DROPPED_BITS = 13  # fp32 keeps 23 mantissa bits, TF32 10


def wide(x: Tensor) -> Tensor:
    """`x` in at least fp32: fp16 and bf16 go to fp32, fp32 and fp64 stay."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def compute_dtype(x: Tensor, weight: Tensor) -> torch.dtype:
    """The dtype a layer computes in: the autocast dtype when autocast is on, else the weight's."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return weight.dtype


def tf32_round(x: Tensor) -> Tensor:
    """fp32 `x` rounded to the 10-bit mantissa TF32 multiplies with."""
    half = 1 << (TF32_DROPPED_BITS - 1)
    mask = ~((1 << TF32_DROPPED_BITS) - 1)
    return ((x.view(torch.int32) + half) & mask).view(torch.float32)


def split_layer(
    layer: Callable[[Tensor, Tensor, Tensor | None], Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """`layer(x, weight, bias)` in the compute dtype with `x` split in two; the result in fp32.

    `layer` must be linear in its input, such as a conv or a matmul. In fp64 it runs once as is.
    """
    dtype = compute_dtype(x, weight)
    if dtype == torch.float64:
        return layer(
            x.double(), weight.double(), None if bias is None else bias.double()
        )

    with torch.autocast(x.device.type, enabled=False):
        x = x.float()
        high = tf32_round(x) if dtype == torch.float32 else x.to(dtype).float()
        low = (x - high) * LOW_SCALE
        weight = weight.to(dtype)
        bias = None if bias is None else bias.to(dtype)
        out = layer(high.to(dtype), weight, bias).float()
        return out + layer(low.to(dtype), weight, None).float() / LOW_SCALE
