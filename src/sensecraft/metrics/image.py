"""
Image quality metrics.

Provides standard image quality metrics for evaluation:
- PSNR: Peak Signal-to-Noise Ratio (dB)
- SSIM: Structural Similarity Index (0-1 or dB)
- MSSSIM: Multi-Scale SSIM (0-1 or dB)
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- LPIPSMetric: Learned Perceptual similarity

Both functional API and class-based API are provided.

Functional API (recommended for simple use):
    >>> from sensecraft.metrics import psnr, ssim, lpips
    >>>
    >>> print(f"PSNR: {psnr(pred, target):.2f} dB")
    >>> print(f"SSIM: {ssim(pred, target):.4f}")
    >>> print(f"LPIPS: {lpips(pred, target):.4f}")  # auto-loads model

Class-based API (for repeated use with same settings):
    >>> from sensecraft.metrics import PSNR, SSIM
    >>>
    >>> psnr_metric = PSNR(data_range=1.0)
    >>> ssim_metric = SSIM(data_range=1.0, as_db=True)

For SSIM/MS-SSIM, dB scale is computed as: -10 * log10(1 - ssim_value)
This gives values like ~15-25 dB for typical quality ranges.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _ssim_fn, ms_ssim as _ms_ssim_fn

    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False
    _ssim_fn = None
    _ms_ssim_fn = None

# Import our own LPIPS implementation
from ..loss.lpips import LPIPS as _LPIPS

# Global cache for LPIPS models (keyed by net type and device)
_LPIPS_CACHE: Dict[str, Any] = {}


def _get_lpips_model(net: str, device: torch.device):
    """Get or create a cached LPIPS model for the given net type and device."""
    cache_key = f"{net}_{device}"
    if cache_key not in _LPIPS_CACHE:
        model = _LPIPS(net=net, verbose=False)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _LPIPS_CACHE[cache_key] = model

    return _LPIPS_CACHE[cache_key]


# =============================================================================
# Functional API
# =============================================================================


def psnr(
    input: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute PSNR (Peak Signal-to-Noise Ratio).

    Args:
        input: Predicted tensor (B, C, H, W) or (B, C, T, H, W)
        target: Target tensor, same shape as input
        data_range: Dynamic range (max - min). Use 1.0 for [0,1], 2.0 for [-1,1], 255.0 for [0,255]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        PSNR value in dB (higher is better)
    """
    # Ensure contiguous tensors
    input = input.contiguous()
    target = target.contiguous()

    mse = F.mse_loss(input, target, reduction="none")
    mse = mse.reshape(mse.shape[0], -1).mean(dim=1)
    mse = torch.clamp(mse, min=1e-10)

    psnr_val = 10 * torch.log10((data_range**2) / mse)

    if reduction == "mean":
        return psnr_val.mean()
    elif reduction == "sum":
        return psnr_val.sum()
    return psnr_val


def ssim(
    input: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    as_db: bool = False,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Compute SSIM (Structural Similarity Index).

    Args:
        input: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        data_range: Dynamic range. Use 1.0 for [0,1], 2.0 for [-1,1]
        as_db: If True, return in dB scale: -10 * log10(1 - ssim)
        win_size: Window size
        win_sigma: Gaussian window sigma
        K1, K2: SSIM constants

    Returns:
        SSIM value (0-1, higher is better) or dB if as_db=True
    """
    if not HAS_PYTORCH_MSSSIM:
        raise ImportError(
            "pytorch_msssim is required for SSIM. Install with: pip install pytorch-msssim"
        )

    ssim_val = _ssim_fn(
        input,
        target,
        data_range=data_range,
        size_average=True,
        win_size=win_size,
        win_sigma=win_sigma,
        K=(K1, K2),
    )

    if as_db:
        ssim_val = torch.clamp(ssim_val, max=1.0 - 1e-7)
        return -10 * torch.log10(1 - ssim_val)

    return ssim_val


def ms_ssim(
    input: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    as_db: bool = False,
    win_size: int = 11,
    win_sigma: float = 1.5,
    weights: Optional[list] = None,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Compute MS-SSIM (Multi-Scale Structural Similarity Index).

    Args:
        input: Predicted tensor (B, C, H, W), min size ~160x160 for default 5 scales
        target: Target tensor (B, C, H, W)
        data_range: Dynamic range. Use 1.0 for [0,1], 2.0 for [-1,1]
        as_db: If True, return in dB scale: -10 * log10(1 - ms_ssim)
        win_size: Window size
        win_sigma: Gaussian window sigma
        weights: Weights for each scale (default: 5 scales)
        K1, K2: SSIM constants

    Returns:
        MS-SSIM value (0-1, higher is better) or dB if as_db=True
    """
    if not HAS_PYTORCH_MSSSIM:
        raise ImportError(
            "pytorch_msssim is required for MS-SSIM. Install with: pip install pytorch-msssim"
        )

    ms_ssim_val = _ms_ssim_fn(
        input,
        target,
        data_range=data_range,
        size_average=True,
        win_size=win_size,
        win_sigma=win_sigma,
        weights=weights,
        K=(K1, K2),
    )

    if as_db:
        ms_ssim_val = torch.clamp(ms_ssim_val, max=1.0 - 1e-7)
        return -10 * torch.log10(1 - ms_ssim_val)

    return ms_ssim_val


def rmse(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute RMSE (Root Mean Squared Error).

    Args:
        input: Predicted tensor
        target: Target tensor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        RMSE value (lower is better)
    """
    input = input.contiguous()
    target = target.contiguous()

    mse = F.mse_loss(input, target, reduction="none")
    mse = mse.reshape(mse.shape[0], -1).mean(dim=1)
    rmse_val = torch.sqrt(mse)

    if reduction == "mean":
        return rmse_val.mean()
    elif reduction == "sum":
        return rmse_val.sum()
    return rmse_val


def mae(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute MAE (Mean Absolute Error).

    Args:
        input: Predicted tensor
        target: Target tensor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MAE value (lower is better)
    """
    input = input.contiguous()
    target = target.contiguous()

    ae = F.l1_loss(input, target, reduction="none")
    ae = ae.reshape(ae.shape[0], -1).mean(dim=1)

    if reduction == "mean":
        return ae.mean()
    elif reduction == "sum":
        return ae.sum()
    return ae


def mape(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute MAPE (Mean Absolute Percentage Error).

    MAPE = mean(|(input - target) / target|)

    This gives the average percentage error, making it scale-invariant.

    Args:
        input: Predicted tensor
        target: Target tensor
        reduction: 'mean', 'sum', or 'none'
        eps: Small value to avoid division by zero

    Returns:
        MAPE value (lower is better, 0 = perfect, 1 = 100% average error)
    """
    input = input.contiguous()
    target = target.contiguous()

    # Compute |(input - target) / target| element-wise
    relative_error = torch.abs((input - target) / (target + eps))
    mape_val = relative_error.reshape(relative_error.shape[0], -1).mean(dim=1)

    if reduction == "mean":
        return mape_val.mean()
    elif reduction == "sum":
        return mape_val.sum()
    return mape_val


def lpips(
    input: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
) -> torch.Tensor:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity).

    The LPIPS model is automatically cached and moved to the input device.

    Args:
        input: Predicted tensor (B, C, H, W), should be in [-1, 1]
        target: Target tensor (B, C, H, W), should be in [-1, 1]
        net: Network type ("alex", "vgg", "squeeze")

    Returns:
        LPIPS distance (lower is better, 0 = identical)
    """
    model = _get_lpips_model(net, input.device)
    return model(input, target).mean()


# =============================================================================
# Class-based API
# =============================================================================


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio metric (class-based wrapper).

    See `psnr()` function for details.
    """

    def __init__(self, data_range: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr(input, target, self.data_range, self.reduction)


class SSIM(nn.Module):
    """Structural Similarity Index metric (class-based wrapper).

    See `ssim()` function for details.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        as_db: bool = False,
        win_size: int = 11,
        win_sigma: float = 1.5,
        K1: float = 0.01,
        K2: float = 0.03,
    ):
        super().__init__()
        if not HAS_PYTORCH_MSSSIM:
            raise ImportError(
                "pytorch_msssim is required for SSIM. Install with: pip install pytorch-msssim"
            )
        self.data_range = data_range
        self.as_db = as_db
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.K1 = K1
        self.K2 = K2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ssim(
            input,
            target,
            self.data_range,
            self.as_db,
            self.win_size,
            self.win_sigma,
            self.K1,
            self.K2,
        )


class MSSSIM(nn.Module):
    """Multi-Scale Structural Similarity Index metric (class-based wrapper).

    See `ms_ssim()` function for details.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        as_db: bool = False,
        win_size: int = 11,
        win_sigma: float = 1.5,
        weights: Optional[list] = None,
        K1: float = 0.01,
        K2: float = 0.03,
    ):
        super().__init__()
        if not HAS_PYTORCH_MSSSIM:
            raise ImportError(
                "pytorch_msssim is required for MS-SSIM. Install with: pip install pytorch-msssim"
            )
        self.data_range = data_range
        self.as_db = as_db
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.weights = weights
        self.K1 = K1
        self.K2 = K2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ms_ssim(
            input,
            target,
            self.data_range,
            self.as_db,
            self.win_size,
            self.win_sigma,
            self.weights,
            self.K1,
            self.K2,
        )


class RMSE(nn.Module):
    """Root Mean Squared Error metric (class-based wrapper).

    See `rmse()` function for details.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return rmse(input, target, self.reduction)


class MAE(nn.Module):
    """Mean Absolute Error metric (class-based wrapper).

    See `mae()` function for details.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mae(input, target, self.reduction)


class MAPE(nn.Module):
    """Mean Absolute Percentage Error metric (class-based wrapper).

    See `mape()` function for details.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mape(input, target, self.reduction, self.eps)


class LPIPSMetric(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) metric.

    Lower is better (0 = identical, higher = more different).

    Note: This is a wrapper that provides the same interface as other metrics.
    The underlying LPIPS model returns a "distance" which is already a metric.

    Args:
        net: Network type ("alex", "vgg", "squeeze")
        version: LPIPS version ("0.0" or "0.1")

    Example:
        >>> lpips = LPIPSMetric(net="alex")
        >>> value = lpips(pred, target)  # e.g., 0.12
    """

    def __init__(
        self,
        net: str = "alex",
        version: str = "0.1",
    ):
        super().__init__()
        self.lpips = _LPIPS(net=net, version=version)
        # Freeze LPIPS model
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS distance.

        Args:
            input: Predicted tensor (B, C, H, W), should be in [-1, 1]
            target: Target tensor (B, C, H, W), should be in [-1, 1]

        Returns:
            LPIPS distance (lower is better)
        """
        return self.lpips(input, target).mean()
