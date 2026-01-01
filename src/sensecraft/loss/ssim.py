"""
SSIM and MS-SSIM loss wrappers.

Provides unified interface for SSIM-based losses, wrapping pytorch_msssim
with consistent API and loss (1 - similarity) formulation.

Example:
    >>> from sensecraft.loss import SSIMLoss, MSSSIMLoss
    >>>
    >>> ssim_loss = SSIMLoss(data_range=1.0)
    >>> loss = ssim_loss(pred, target)  # Returns 1 - SSIM
"""

import torch
import torch.nn as nn

from .config import register_loss, ValueRange

try:
    from pytorch_msssim import ssim, ms_ssim

    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False
    ssim = None
    ms_ssim = None


@register_loss("ssim", ValueRange.UNIT, is_2d_only=True)
class SSIMLoss(nn.Module):
    """SSIM Loss (1 - SSIM).

    Wraps pytorch_msssim.ssim with a loss formulation (lower is better).

    Args:
        data_range: Value range of input (1.0 for [0,1], 2.0 for [-1,1])
        size_average: If True, return scalar mean. If False, return per-sample.
        win_size: Window size for SSIM computation
        win_sigma: Gaussian window sigma
        K1: SSIM constant K1
        K2: SSIM constant K2
    """

    def __init__(
        self,
        data_range: float = 1.0,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        K1: float = 0.01,
        K2: float = 0.03,
    ):
        super().__init__()
        if not HAS_PYTORCH_MSSSIM:
            raise ImportError(
                "pytorch_msssim is required for SSIMLoss. "
                "Install with: pip install pytorch-msssim"
            )

        self.data_range = data_range
        self.size_average = size_average
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.K1 = K1
        self.K2 = K2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM).

        Args:
            input: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)

        Returns:
            SSIM loss value (1 - SSIM)
        """
        ssim_val = ssim(
            input,
            target,
            data_range=self.data_range,
            size_average=self.size_average,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            K=(self.K1, self.K2),
        )
        return 1 - ssim_val


@register_loss("ms_ssim", ValueRange.UNIT, is_2d_only=True)
class MSSSIMLoss(nn.Module):
    """MS-SSIM Loss (1 - MS-SSIM).

    Wraps pytorch_msssim.ms_ssim with a loss formulation (lower is better).

    Args:
        data_range: Value range of input (1.0 for [0,1], 2.0 for [-1,1])
        size_average: If True, return scalar mean. If False, return per-sample.
        win_size: Window size for SSIM computation
        win_sigma: Gaussian window sigma
        weights: Weights for each scale (default: 5 scales)
        K1: SSIM constant K1
        K2: SSIM constant K2
    """

    def __init__(
        self,
        data_range: float = 1.0,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        weights: list = None,
        K1: float = 0.01,
        K2: float = 0.03,
    ):
        super().__init__()
        if not HAS_PYTORCH_MSSSIM:
            raise ImportError(
                "pytorch_msssim is required for MSSSIMLoss. "
                "Install with: pip install pytorch-msssim"
            )

        self.data_range = data_range
        self.size_average = size_average
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.weights = weights
        self.K1 = K1
        self.K2 = K2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MS-SSIM loss (1 - MS-SSIM).

        Args:
            input: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)

        Returns:
            MS-SSIM loss value (1 - MS-SSIM)
        """
        ms_ssim_val = ms_ssim(
            input,
            target,
            data_range=self.data_range,
            size_average=self.size_average,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            weights=self.weights,
            K=(self.K1, self.K2),
        )
        return 1 - ms_ssim_val
