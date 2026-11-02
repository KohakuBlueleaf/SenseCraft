"""The LPIPS distance on two feature lists: unit channel vectors, a learned 1x1 head, a mean.

It runs in fp32 whatever the features are: the channel sum of squares overflows fp16, and the
difference of two close unit vectors cancels to noise in bf16.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-10


def unit_channels(x: Tensor) -> Tensor:
    """`x` with each pixel's channel vector scaled to unit length."""
    return x / (x.square().sum(dim=1, keepdim=True).sqrt() + EPS)


def lpips_distance(
    fake: list[Tensor], real: list[Tensor], heads: list[Tensor]
) -> Tensor:
    """Mean over the batch of the summed per-tap distances; `heads` are `(1, C, 1, 1)`."""
    wide = torch.promote_types(fake[0].dtype, torch.float32)
    total = 0.0
    with torch.autocast(fake[0].device.type, enabled=False):
        for fake_tap, real_tap, head in zip(fake, real, heads, strict=True):
            fake_unit = unit_channels(fake_tap.to(wide))
            real_unit = unit_channels(real_tap.to(wide))
            squared = (fake_unit - real_unit).square()
            total = total + F.conv2d(squared, head.to(wide)).mean(dim=(2, 3))
    return total.mean()
