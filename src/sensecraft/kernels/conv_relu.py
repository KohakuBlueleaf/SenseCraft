"""Frozen conv, bias and ReLU as one op: cuDNN's fused forward, and an input-only backward.

The weights of a feature net never get gradients, so backward is the ReLU mask applied to the
incoming gradient, then the conv's input gradient. Outside cuDNN's dtypes the op runs as plain
`relu(conv2d(...))`.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

FUSED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class _FrozenConvReLU(torch.autograd.Function):
    """`relu(conv2d(x, weight, bias))` with gradient for `x` only."""

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        out = torch.cudnn_convolution_relu(x, weight, bias, stride, padding, (1, 1), 1)
        ctx.save_for_backward(x, weight, out)
        ctx.stride, ctx.padding = stride, padding
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, out = ctx.saved_tensors
        masked = grad_out * (out > 0)
        grad_x = torch.ops.aten.convolution_backward(
            masked,
            x,
            weight,
            None,
            ctx.stride,
            ctx.padding,
            (1, 1),
            False,
            (0, 0),
            1,
            (True, False, False),
        )[0]
        return grad_x, None, None, None, None


def frozen_conv_relu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple,
    padding: tuple,
    fused: bool = True,
) -> Tensor:
    """`relu(conv2d(x, weight, bias))` for frozen weights, fused on CUDA unless `fused` is off."""
    if torch.is_autocast_enabled(x.device.type):
        dtype = torch.get_autocast_dtype(x.device.type)
        x, weight = x.to(dtype), weight.to(dtype)
        bias = bias.to(dtype) if bias is not None else None
    plain = not fused or weight.requires_grad
    if plain or x.device.type != "cuda" or x.dtype not in FUSED_DTYPES:
        return F.relu(F.conv2d(x, weight, bias, stride, padding))
    with torch.autocast(x.device.type, enabled=False):
        return _FrozenConvReLU.apply(x, weight, bias, stride, padding)
