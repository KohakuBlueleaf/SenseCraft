"""
SenseCraft Metrics Module.

Provides evaluation metrics for image quality assessment.
Unlike loss functions, these return the actual metric values (higher is better for most).

Functional API (recommended):
    >>> from sensecraft.metrics import psnr, ssim, lpips
    >>>
    >>> print(f"PSNR: {psnr(pred, target):.2f} dB")
    >>> print(f"SSIM: {ssim(pred, target):.4f}")
    >>> print(f"LPIPS: {lpips(pred, target):.4f}")  # auto-loads model

Class-based API:
    >>> from sensecraft.metrics import PSNR, SSIM
    >>>
    >>> psnr_metric = PSNR(data_range=1.0)
    >>> ssim_metric = SSIM(data_range=1.0, as_db=True)
    >>>
    >>> psnr_val = psnr_metric(pred, target)
    >>> ssim_val = ssim_metric(pred, target)

Available metrics:
- psnr / PSNR: Peak Signal-to-Noise Ratio (dB, higher is better)
- ssim / SSIM: Structural Similarity Index (0-1 or dB, higher is better)
- ms_ssim / MSSSIM: Multi-Scale SSIM (0-1 or dB, higher is better)
- rmse / RMSE: Root Mean Squared Error (lower is better)
- mae / MAE: Mean Absolute Error (lower is better)
- mape / MAPE: Mean Absolute Percentage Error (lower is better)
- lpips / LPIPSMetric: Learned Perceptual similarity (lower is better)
"""

from .image import (
    # Functional API
    psnr,
    ssim,
    ms_ssim,
    rmse,
    mae,
    mape,
    lpips,
    # Class-based API
    PSNR,
    SSIM,
    MSSSIM,
    RMSE,
    MAE,
    MAPE,
    LPIPSMetric,
)

__all__ = [
    # Functional API
    "psnr",
    "ssim",
    "ms_ssim",
    "rmse",
    "mae",
    "mape",
    "lpips",
    # Class-based API
    "PSNR",
    "SSIM",
    "MSSSIM",
    "RMSE",
    "MAE",
    "MAPE",
    "LPIPSMetric",
]
