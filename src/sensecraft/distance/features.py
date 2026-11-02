"""Distances between two feature tensors, in fp32 whatever the features are."""

import torch
import torch.nn.functional as F
from torch import Tensor


def gram(x: Tensor) -> Tensor:
    """`(B, C, C)` gram matrix of `(B, N, C)` tokens, divided by the token count."""
    return torch.bmm(x.transpose(1, 2), x) / x.shape[1]


def feature_distance(
    fake: Tensor, real: Tensor, use_gram: bool, use_norm: bool, dim: int
) -> Tensor:
    """MSE of the features, or L1 of their gram matrices; `use_norm` unit-scales along `dim` first.

    Gram matrices are taken over `(B, N, C)` tokens; pass channel-first maps flattened.
    """
    wide = torch.promote_types(fake.dtype, torch.float32)
    with torch.autocast(fake.device.type, enabled=False):
        fake, real = fake.to(wide), real.to(wide)
        if use_norm:
            fake, real = F.normalize(fake, dim=dim), F.normalize(real, dim=dim)
        if use_gram:
            return F.l1_loss(gram(fake), gram(real))
        return F.mse_loss(fake, real)
