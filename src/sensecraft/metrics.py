"""Image metrics on device tensors: no gradient, no host reads; print the result to read it.

Ranges are given by `data_range`: 1.0 for [0, 1], 2.0 for [-1, 1].
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as _ms_ssim
from torch import Tensor

from sensecraft.perceptual import LPIPS
from sensecraft.signal.ssim import GaussianWindow, ssim_score

_WINDOWS: dict[tuple, GaussianWindow] = {}
_LPIPS: dict[tuple, LPIPS] = {}


def _per_sample(values: Tensor) -> Tensor:
    """Mean over everything but the batch dim."""
    return values.reshape(values.shape[0], -1).mean(dim=1)


def _window(size: int, sigma: float, device: torch.device) -> GaussianWindow:
    """A cached Gaussian window on `device`."""
    key = (size, sigma, device)
    if key not in _WINDOWS:
        _WINDOWS[key] = GaussianWindow(size, sigma).to(device)
    return _WINDOWS[key]


def as_db(score: Tensor) -> Tensor:
    """A similarity in [0, 1] as decibels: -10 log10(1 - score)."""
    return -10 * torch.log10(1 - score.clamp(max=1.0 - 1e-7))


@torch.no_grad()
def psnr(
    input: Tensor, target: Tensor, data_range: float = 1.0, per_sample: bool = False
) -> Tensor:
    """PSNR in dB, the batch mean unless `per_sample`."""
    mse = _per_sample(F.mse_loss(input, target, reduction="none")).clamp(min=1e-10)
    values = 10 * torch.log10(data_range**2 / mse)
    return values if per_sample else values.mean()


@torch.no_grad()
def ssim(
    input: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    db: bool = False,
) -> Tensor:
    """Mean SSIM of `(B, C, H, W)` images, or its dB form."""
    window = _window(win_size, win_sigma, input.device)
    score = ssim_score(input.float(), target.float(), window, data_range)
    return as_db(score) if db else score


@torch.no_grad()
def ms_ssim(
    input: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    weights: list[float] | None = None,
    db: bool = False,
) -> Tensor:
    """Mean MS-SSIM of `(B, C, H, W)` images of side 161 or more, or its dB form."""
    window = _window(win_size, win_sigma, input.device)(input.shape[1])
    score = _ms_ssim(
        input.float(),
        target.float(),
        data_range=data_range,
        win=window,
        weights=weights,
    )
    return as_db(score) if db else score


@torch.no_grad()
def rmse(input: Tensor, target: Tensor, per_sample: bool = False) -> Tensor:
    """Root mean squared error, the batch mean unless `per_sample`."""
    values = _per_sample(F.mse_loss(input, target, reduction="none")).sqrt()
    return values if per_sample else values.mean()


@torch.no_grad()
def mae(input: Tensor, target: Tensor, per_sample: bool = False) -> Tensor:
    """Mean absolute error, the batch mean unless `per_sample`."""
    values = _per_sample(F.l1_loss(input, target, reduction="none"))
    return values if per_sample else values.mean()


@torch.no_grad()
def mape(
    input: Tensor, target: Tensor, eps: float = 1e-8, per_sample: bool = False
) -> Tensor:
    """Mean absolute error relative to `target`, the batch mean unless `per_sample`."""
    values = _per_sample(((input - target) / (target + eps)).abs())
    return values if per_sample else values.mean()


@torch.no_grad()
def lpips(input: Tensor, target: Tensor, net: str = "alex") -> Tensor:
    """LPIPS of images in [-1, 1]; the net is built once per device."""
    key = (net, input.device)
    if key not in _LPIPS:
        _LPIPS[key] = LPIPS(net).to(input.device)
    return _LPIPS[key](input, target)
