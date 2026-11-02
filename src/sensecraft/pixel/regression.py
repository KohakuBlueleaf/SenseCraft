"""Pointwise regression losses: input and target compared per value, then averaged."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..registry import register

Distance = Literal["l1", "mse", "charbonnier"]
DISTANCES = ("l1", "mse", "charbonnier")

# The eps every `loss_type="charbonnier"` option in the package uses
INNER_CHARBONNIER_EPS = 1e-6


def check_distance(kind: str) -> str:
    """`kind` if it names a distance in DISTANCES; raises otherwise."""
    if kind not in DISTANCES:
        raise ValueError(f"loss_type {kind!r} not in {DISTANCES}")
    return kind


def charbonnier_map(input: Tensor, target: Tensor, eps: float) -> Tensor:
    """sqrt((input - target)^2 + eps^2), per element."""
    return torch.sqrt((input - target) ** 2 + eps * eps)


def distance_map(kind: Distance, input: Tensor, target: Tensor) -> Tensor:
    """The per-element distance named by `kind`, unreduced."""
    match kind:
        case "l1":
            return F.l1_loss(input, target, reduction="none")
        case "mse":
            return F.mse_loss(input, target, reduction="none")
        case "charbonnier":
            return charbonnier_map(input, target, INNER_CHARBONNIER_EPS)


def distance(kind: Distance, input: Tensor, target: Tensor) -> Tensor:
    """The mean distance named by `kind`."""
    match kind:
        case "l1":
            return F.l1_loss(input, target)
        case "mse":
            return F.mse_loss(input, target)
        case "charbonnier":
            return charbonnier_map(input, target, INNER_CHARBONNIER_EPS).mean()


# Standard regression losses


@register("mse")
class MSELoss(nn.Module):
    """Mean squared error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)


@register("l1")
class L1Loss(nn.Module):
    """Mean absolute error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target)


@register("huber")
class HuberLoss(nn.Module):
    """Huber loss: quadratic below `delta`, linear above."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input, target, delta=self.delta)


@register("smooth_l1")
class SmoothL1Loss(nn.Module):
    """Smooth L1: Huber divided by `beta`."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, beta=self.beta)


@register("rmse")
class RMSELoss(nn.Module):
    """Root mean squared error, sqrt(mse + eps)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse = ((input - target) ** 2).mean()
        return torch.sqrt(mse + self.eps)


@register("mape")
class MAPELoss(nn.Module):
    """Mean absolute percentage error, |x - y| / |y|; unstable for targets near zero."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.abs(input - target) / (torch.abs(target) + self.eps)
        return loss.mean()


@register("smape")
class SMAPELoss(nn.Module):
    """Symmetric mean absolute percentage error, 2 |x - y| / (|x| + |y|)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        numerator = torch.abs(input - target)
        denominator = torch.abs(input) + torch.abs(target) + self.eps
        return (2.0 * numerator / denominator).mean()


@register("log_cosh")
class LogCoshLoss(nn.Module):
    """log(cosh(x - y)): L2 for small errors, L1 for large ones."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target

        # log(cosh(d)) = d + softplus(-2d) - log(2), which does not overflow
        loss = diff + F.softplus(-2.0 * diff) - math.log(2)
        return loss.mean()


@register("quantile")
class QuantileLoss(nn.Module):
    """Pinball loss for quantile q: q max(y - x, 0) + (1 - q) max(x - y, 0)."""

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = target - input
        loss = torch.max(self.quantile * diff, (self.quantile - 1) * diff)
        return loss.mean()


# Robust regression losses


@register("cauchy")
class CauchyLoss(nn.Module):
    """Cauchy (Lorentzian) loss, log(1 + ((x - y) / c)^2)."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = (input - target) / self.scale
        return torch.log1p(diff**2).mean()


@register("geman_mcclure")
class GemanMcClureLoss(nn.Module):
    """Geman-McClure loss, d^2 / (1 + d^2) with d = (x - y) / c; bounded by 1."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff_sq = ((input - target) / self.scale) ** 2
        return (diff_sq / (1 + diff_sq)).mean()


@register("welsch")
class WelschLoss(nn.Module):
    """Welsch (Leclerc) loss, 1 - exp(-d^2 / 2) with d = (x - y) / c."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff_sq = ((input - target) / self.scale) ** 2
        return (1 - torch.exp(-0.5 * diff_sq)).mean()


@register("tukey")
class TukeyBiweightLoss(nn.Module):
    """Tukey's biweight: (c^2 / 6)(1 - (1 - (d / c)^2)^3) for |d| <= c, else c^2 / 6."""

    def __init__(self, c: float = 4.685):
        super().__init__()
        self.c = c
        self.plateau = c**2 / 6

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        scaled = diff / self.c

        inside = self.plateau * (1 - (1 - scaled**2) ** 3)
        loss = torch.where(torch.abs(diff) <= self.c, inside, self.plateau)
        return loss.mean()


@register("wing")
class WingLoss(nn.Module):
    """Wing loss: w ln(1 + |d| / eps) below width w, |d| - offset above it."""

    def __init__(self, width: float = 5.0, curvature: float = 0.5):
        super().__init__()
        self.width = width
        self.curvature = curvature

        # Offset that joins the two pieces continuously at |d| = width
        self.offset = width - width * math.log(1 + width / curvature)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = torch.abs(input - target)
        loss = torch.where(
            diff < self.width,
            self.width * torch.log1p(diff / self.curvature),
            diff - self.offset,
        )
        return loss.mean()


@register("charbonnier")
class CharbonnierLoss(nn.Module):
    """Charbonnier loss, sqrt((x - y)^2 + eps^2): an L1 that is smooth at zero."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return charbonnier_map(input, target, self.eps).mean()


# Noise-aware loss


@register("gaussian_noise")
class GaussianNoiseLoss(nn.Module):
    """A distance to the target, optionally after adding Gaussian noise to the target.

    With `add_noise` the noise std is `sigma`, or drawn uniformly from `sigma_range` on
    every call when that is given. Without it the loss is the plain distance.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        sigma_range: tuple[float, float] | None = None,
        loss_type: Distance = "l1",
        add_noise: bool = False,
    ):
        super().__init__()
        self.sigma = sigma
        self.sigma_range = sigma_range
        self.loss_type = check_distance(loss_type)
        self.add_noise = add_noise

    def noise_std(self, target: Tensor) -> Tensor | float:
        """The noise std for this call; a drawn one stays on the device."""
        if self.sigma_range is None:
            return self.sigma
        low, high = self.sigma_range
        return torch.empty((), device=target.device).uniform_(low, high)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.add_noise:
            sigma = self.noise_std(target)
            target = target + torch.randn_like(target) * sigma
        return distance(self.loss_type, input, target)
