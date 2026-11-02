"""2x2 max-pool with stride 2 that keeps no indices.

Backward finds each window's maximum again by comparing the saved input with the pooled output,
taking the first maximum in row order as PyTorch does. A feature net already holds the pool
input for its ReLU, so the pool adds no memory, and under compile the comparison fuses with the
masks around it.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


class _MaxPool2x2(torch.autograd.Function):
    """`max_pool2d(x, 2, 2)` with a comparison backward."""

    @staticmethod
    def forward(ctx, x):
        out = F.max_pool2d(x, 2, 2)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, out = ctx.saved_tensors
        height, width = out.shape[-2:]

        # Windows as (N, C, h, 2, w, 2); odd trailing rows and columns never win
        windows = (
            x[..., : 2 * height, : 2 * width]
            .unflatten(2, (height, 2))
            .unflatten(4, (width, 2))
        )
        hits = windows == out[:, :, :, None, :, None]

        # Keep only the first maximum of each window, scanning rows then columns
        first = hits[:, :, :, 0, :, 0]
        second = hits[:, :, :, 0, :, 1] & ~first
        third = hits[:, :, :, 1, :, 0] & ~(first | second)
        fourth = hits[:, :, :, 1, :, 1] & ~(first | second | third)
        top = torch.stack((first, second), dim=-1)
        bottom = torch.stack((third, fourth), dim=-1)
        chosen = torch.stack((top, bottom), dim=3)

        grad = chosen * grad_out[:, :, :, None, :, None]
        grad = grad.flatten(4, 5).flatten(2, 3)
        if grad.shape[-2:] != x.shape[-2:]:
            grad = F.pad(
                grad, (0, x.shape[-1] - grad.shape[-1], 0, x.shape[-2] - grad.shape[-2])
            )
        # Match the input's layout, as PyTorch's own pool backward does
        layout = (
            torch.channels_last
            if x.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
        return grad.contiguous(memory_format=layout)


def max_pool2x2(x: Tensor) -> Tensor:
    """2x2 max-pool, stride 2, recomputing its argmax in backward."""
    if not x.requires_grad:
        return F.max_pool2d(x, 2, 2)
    return _MaxPool2x2.apply(x)
