"""
General loss functions for image processing tasks.

This module provides various loss functions commonly used in image restoration,
super-resolution, and enhancement tasks.

Includes:
- CharbonnierLoss: Smooth L1 alternative, differentiable everywhere
- PatchFFTLoss: Frequency domain loss on image patches
- FFTLoss: Global frequency domain loss
- GaussianNoiseLoss: Noise-aware loss for denoising tasks

For edge and gradient-based losses, see the `edge` module.

Example:
    >>> from sensecraft.loss import CharbonnierLoss, PatchFFTLoss
    >>>
    >>> # Charbonnier loss (smooth L1)
    >>> loss_fn = CharbonnierLoss(eps=1e-6)
    >>> loss = loss_fn(predicted, target)
    >>>
    >>> # Patch FFT loss with log normalization
    >>> fft_loss = PatchFFTLoss(patch_size=8, norm_type=NormType.LOG1P)
    >>> loss = fft_loss(predicted, target)
"""

from enum import Enum
from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import register_loss, ValueRange


@register_loss("charbonnier")
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (differentiable variant of L1 loss).

    L(x, y) = sqrt((x - y)^2 + eps^2)

    This provides a smooth approximation to L1 loss that is differentiable
    everywhere, including at zero. The epsilon parameter controls the
    smoothness near zero.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initialize Charbonnier loss.

        Args:
            eps: Small constant for numerical stability and smoothness.
                 Larger values make the loss behave more like L2 near zero.
            reduction: Specifies the reduction to apply: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.eps = eps
        self.eps_sq = eps * eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss.

        Args:
            input: Predicted tensor
            target: Target tensor

        Returns:
            Charbonnier loss value
        """
        diff_sq = (input - target) ** 2
        loss = torch.sqrt(diff_sq + self.eps_sq)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class NormType(Enum):
    """Normalization types for FFT loss."""

    NONE = "none"
    L2 = "l2"
    LOG = "log"
    LOG1P = "log1p"


@register_loss("patch_fft", is_2d_only=True, requires_fp32=True)
class PatchFFTLoss(nn.Module):
    """Patch-based FFT loss for frequency domain comparison.

    Divides images into patches, computes FFT on each patch, and computes
    loss on the frequency domain representation.

    The FFT magnitudes can be very large, so normalization is recommended.
    """

    def __init__(
        self,
        patch_size: int = 8,
        loss_type: Literal["mse", "l1", "charbonnier"] = "mse",
        norm_type: NormType = NormType.LOG1P,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """Initialize PatchFFTLoss.

        Args:
            patch_size: Size of patches to extract (e.g., 8 for 8x8 patches)
            loss_type: Type of loss to use on FFT features ('mse', 'l1', 'charbonnier')
            norm_type: Normalization to apply to FFT real/imag parts
            eps: Small constant for numerical stability
            reduction: Specifies the reduction to apply: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.patch_size = patch_size
        self.loss_type = loss_type
        self.norm_type = norm_type
        self.eps = eps
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction="none")

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from image tensor.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Patches of shape (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        """
        B, C, H, W = x.shape
        ps = self.patch_size

        # Ensure dimensions are divisible by patch size
        H_pad = (ps - H % ps) % ps
        W_pad = (ps - W % ps) % ps
        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad), mode="reflect")

        B, C, H, W = x.shape

        # Reshape to extract patches
        # (B, C, H, W) -> (B, C, H//ps, ps, W//ps, ps)
        x = x.view(B, C, H // ps, ps, W // ps, ps)
        # (B, C, H//ps, ps, W//ps, ps) -> (B, C, H//ps, W//ps, ps, ps)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()

        return x

    def compute_fft(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D FFT on patches and extract real and imaginary parts.

        Args:
            patches: Tensor of shape (B, C, nh, nw, ps, ps)

        Returns:
            Tuple of (real, imag) tensors
        """
        # Compute 2D FFT on last two dimensions (patch spatial dims)
        fft = torch.fft.fft2(patches, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        # Extract real and imaginary parts separately
        real = fft_shifted.real
        imag = fft_shifted.imag

        # Apply normalization
        if self.norm_type == NormType.L2:
            # L2 normalize across spatial dimensions of each patch
            real_norm = torch.norm(real, dim=(-2, -1), keepdim=True) + self.eps
            imag_norm = torch.norm(imag, dim=(-2, -1), keepdim=True) + self.eps
            real = real / real_norm
            imag = imag / imag_norm
        elif self.norm_type == NormType.LOG:
            # For log, we need to handle negative values - use sign-preserving log
            real = torch.sign(real) * torch.log(torch.abs(real) + self.eps)
            imag = torch.sign(imag) * torch.log(torch.abs(imag) + self.eps)
        elif self.norm_type == NormType.LOG1P:
            # Sign-preserving log1p
            real = torch.sign(real) * torch.log1p(torch.abs(real))
            imag = torch.sign(imag) * torch.log1p(torch.abs(imag))

        return real, imag

    def compute_loss(
        self, input_feat: torch.Tensor, target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between two feature tensors.

        Args:
            input_feat: Input features
            target_feat: Target features

        Returns:
            Loss value (unreduced)
        """
        if self.loss_type == "mse":
            return F.mse_loss(input_feat, target_feat, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(input_feat, target_feat, reduction="none")
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_feat, target_feat)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute patch FFT loss.

        Args:
            input: Input tensor of shape (B, C, H, W)
            target: Target tensor of shape (B, C, H, W)

        Returns:
            FFT loss value
        """
        # Extract patches
        input_patches = self.extract_patches(input)
        target_patches = self.extract_patches(target)

        # Compute FFT - returns (real, imag) parts
        input_real, input_imag = self.compute_fft(input_patches)
        target_real, target_imag = self.compute_fft(target_patches)

        # Compute loss on real and imaginary parts separately
        real_loss = self.compute_loss(input_real, target_real)
        imag_loss = self.compute_loss(input_imag, target_imag)

        # Combine losses
        loss = real_loss + imag_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("fft", is_2d_only=True, requires_fp32=True)
class FFTLoss(nn.Module):
    """Global FFT loss (no patching, operates on full image).

    Computes FFT on the entire image and computes loss on frequency domain.
    """

    def __init__(
        self,
        loss_type: Literal["mse", "l1", "charbonnier"] = "mse",
        norm_type: NormType = NormType.LOG1P,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """Initialize FFTLoss.

        Args:
            loss_type: Type of loss to use on FFT features ('mse', 'l1', 'charbonnier')
            norm_type: Normalization to apply to FFT real/imag parts
            eps: Small constant for numerical stability
            reduction: Specifies the reduction to apply: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.loss_type = loss_type
        self.norm_type = norm_type
        self.eps = eps
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction="none")

    def compute_fft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D FFT and extract real and imaginary parts.

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tuple of (real, imag) tensors
        """
        # Compute 2D FFT on spatial dimensions
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        # Extract real and imaginary parts separately
        real = fft_shifted.real
        imag = fft_shifted.imag

        # Apply normalization
        if self.norm_type == NormType.L2:
            real_norm = torch.norm(real, dim=(-2, -1), keepdim=True) + self.eps
            imag_norm = torch.norm(imag, dim=(-2, -1), keepdim=True) + self.eps
            real = real / real_norm
            imag = imag / imag_norm
        elif self.norm_type == NormType.LOG:
            # Sign-preserving log
            real = torch.sign(real) * torch.log(torch.abs(real) + self.eps)
            imag = torch.sign(imag) * torch.log(torch.abs(imag) + self.eps)
        elif self.norm_type == NormType.LOG1P:
            # Sign-preserving log1p
            real = torch.sign(real) * torch.log1p(torch.abs(real))
            imag = torch.sign(imag) * torch.log1p(torch.abs(imag))

        return real, imag

    def compute_loss(
        self, input_feat: torch.Tensor, target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between two feature tensors."""
        if self.loss_type == "mse":
            return F.mse_loss(input_feat, target_feat, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(input_feat, target_feat, reduction="none")
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_feat, target_feat)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute FFT loss.

        Args:
            input: Input tensor of shape (B, C, H, W)
            target: Target tensor of shape (B, C, H, W)

        Returns:
            FFT loss value
        """
        # Compute FFT - returns (real, imag) parts
        input_real, input_imag = self.compute_fft(input)
        target_real, target_imag = self.compute_fft(target)

        # Compute loss on real and imaginary parts separately
        real_loss = self.compute_loss(input_real, target_real)
        imag_loss = self.compute_loss(input_imag, target_imag)

        # Combine losses
        loss = real_loss + imag_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("gaussian_noise")
class GaussianNoiseLoss(nn.Module):
    """Gaussian noise-aware loss for training denoising models.

    This loss adds Gaussian noise to the target and computes the loss between
    the prediction and the noisy target. This can help with training robust
    denoising models by exposing them to various noise levels.

    Can also be used as a regularizer to prevent overfitting to exact pixel values.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        sigma_range: Optional[Tuple[float, float]] = None,
        loss_type: Literal["mse", "l1", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize GaussianNoiseLoss.

        Args:
            sigma: Fixed standard deviation of Gaussian noise (used if sigma_range is None)
            sigma_range: If provided, sample sigma uniformly from this range (min, max)
            loss_type: Type of loss to use ('mse', 'l1', 'charbonnier')
            reduction: Specifies the reduction to apply: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.sigma = sigma
        self.sigma_range = sigma_range
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        add_noise_to_target: bool = False,
    ) -> torch.Tensor:
        """Compute noise-aware loss.

        Args:
            input: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            add_noise_to_target: If True, add noise to target before computing loss

        Returns:
            Loss value
        """
        if add_noise_to_target:
            # Sample or use fixed sigma
            if self.sigma_range is not None:
                sigma = (
                    torch.empty(1, device=target.device)
                    .uniform_(self.sigma_range[0], self.sigma_range[1])
                    .item()
                )
            else:
                sigma = self.sigma

            # Add noise to target
            noise = torch.randn_like(target) * sigma
            target = target + noise

        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(input, target, reduction=self.reduction)
        elif self.loss_type == "l1":
            loss = F.l1_loss(input, target, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            loss = self.charbonnier(input, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss
