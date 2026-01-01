"""
Video and 3D temporal loss functions.

This module provides loss functions for video processing tasks that consider
temporal consistency and 3D spatio-temporal features.

Includes:
- STSSIM: Spatio-Temporal SSIM for video quality
- SSIM3D: 3D SSIM using 3D Gaussian windows
- TSSIM: Temporal SSIM for frame-to-frame consistency
- FDBLoss: Frame Difference Boundary loss for temporal edges
- TemporalAccelerationLoss: Penalize acceleration (second-order temporal derivative)
- TemporalFFTLoss: FFT loss along temporal dimension
- PatchFFT3DLoss: 3D patch-based FFT loss (spatio-temporal)

Example:
    >>> from sensecraft.loss.video import STSSIM, TemporalFFTLoss
    >>>
    >>> # Spatio-temporal SSIM for video
    >>> stssim = STSSIM()
    >>> loss = stssim(pred_video, target_video)  # (B, T, C, H, W)
    >>>
    >>> # Temporal FFT loss
    >>> temp_fft = TemporalFFTLoss()
    >>> loss = temp_fft(pred_video, target_video)
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import register_loss, ValueRange
from .general import CharbonnierLoss, NormType


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    x = torch.arange(size, device=device).float() - (size - 1) / 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create 2D Gaussian kernel."""
    k1d = _gaussian_kernel_1d(size, sigma, device)
    return k1d[:, None] * k1d[None, :]


def _causal_temporal_kernel(
    size: int, decay: float, device: torch.device
) -> torch.Tensor:
    """Create one-sided (causal) temporal kernel with exponential decay.

    The kernel looks back in time: weights are for [t-K+1, ..., t-1, t].
    """
    weights = torch.exp(-torch.arange(size, 0, -1, device=device).float() * decay)
    return weights / weights.sum()


def _create_gaussian_window_2d(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian
        channels: Number of channels
        device: Device to create tensor on
        dtype: Data type of tensor

    Returns:
        2D Gaussian window tensor of shape (channels, 1, window_size, window_size)
    """
    coords = torch.arange(window_size, device=device, dtype=dtype or torch.float32)
    coords -= window_size // 2
    gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d = gauss_2d.view(1, 1, window_size, window_size)
    gauss_2d = gauss_2d.repeat(channels, 1, 1, 1)

    return gauss_2d


@register_loss("ssim3d", ValueRange.UNIT, is_3d_only=True, requires_fp32=True)
class SSIM3D(nn.Module):
    """3D-SSIM with causal temporal kernel.

    Computes SSIM using separable 3D local neighborhoods. Time is treated differently
    via a one-sided (causal) temporal kernel that only looks at past frames.

    This avoids numerical issues from full 3D convolution and provides
    better temporal consistency.

    Args:
        kernel_t: Temporal kernel size (causal window)
        kernel_s: Spatial kernel size
        sigma_s: Spatial Gaussian sigma
        temporal_decay: Decay rate for causal temporal kernel
        C1: SSIM constant (luminance)
        C2: SSIM constant (contrast)
    """

    def __init__(
        self,
        kernel_t: int = 5,
        kernel_s: int = 11,
        sigma_s: float = 1.5,
        temporal_decay: float = 0.5,
        C1: float = 0.01**2,
        C2: float = 0.03**2,
    ):
        super().__init__()
        self.kernel_t = kernel_t
        self.kernel_s = kernel_s
        self.sigma_s = sigma_s
        self.temporal_decay = temporal_decay
        self.C1 = C1
        self.C2 = C2

        # Precompute kernels (will be moved to device on first forward)
        self._spatial_kernel: torch.Tensor | None = None
        self._temporal_kernel: torch.Tensor | None = None

    def _get_kernels(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Get or create kernels on the correct device."""
        if self._spatial_kernel is None or self._spatial_kernel.device != device:
            self._spatial_kernel = _gaussian_kernel_2d(
                self.kernel_s, self.sigma_s, device
            )
            self._temporal_kernel = _causal_temporal_kernel(
                self.kernel_t, self.temporal_decay, device
            )
        return self._spatial_kernel, self._temporal_kernel

    def _separable_3d_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local mean using separable 3D convolution.

        Uses causal (one-sided) padding for temporal dimension.

        Args:
            x: (B, T, C, H, W) input tensor

        Returns:
            (B, T, C, H, W) local mean tensor
        """
        B, T, C, H, W = x.shape
        spatial_k, temporal_k = self._get_kernels(x.device)

        # Reshape for spatial conv: (B*T, C, H, W)
        x_flat = x.flatten(0, 1)

        # Create 2D spatial kernel for depthwise conv
        # Shape: (C, 1, Ks, Ks)
        spatial_kernel = spatial_k[None, None, :, :].expand(C, 1, -1, -1)

        # Apply spatial Gaussian (groups=C for depthwise)
        pad_s = self.kernel_s // 2
        x_spatial = F.conv2d(
            x_flat, spatial_kernel.to(x.dtype), padding=pad_s, groups=C
        )

        # Reshape back: (B, T, C, H, W)
        x_spatial = x_spatial.view(B, T, C, H, W)

        # Apply temporal convolution (causal padding)
        # Reshape: (B, C, H, W, T) for 1D conv over T
        x_t = x_spatial.permute(0, 2, 3, 4, 1).flatten(0, 3)  # (B*C*H*W, T)

        # Causal padding: pad only on the left (past)
        x_t = F.pad(x_t, (self.kernel_t - 1, 0), mode="replicate")

        # 1D conv
        temporal_kernel = temporal_k[None, None, :].to(x.dtype)  # (1, 1, Kt)
        x_t = F.conv1d(x_t[:, None, :], temporal_kernel)[:, 0]  # (B*C*H*W, T)

        # Reshape back: (B, T, C, H, W)
        x_t = x_t.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)

        return x_t

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D-SSIM loss.

        Args:
            input: (B, T, C, H, W) predicted video
            target: (B, T, C, H, W) target video

        Returns:
            3D-SSIM loss (1 - 3D-SSIM score)
        """
        # Normalize to [0, 1] if needed (input might be [-1, 1])
        if input.min() < 0:
            input = input * 0.5 + 0.5
            target = target * 0.5 + 0.5

        # Compute local statistics using separable convolution
        mu_x = self._separable_3d_mean(input)
        mu_y = self._separable_3d_mean(target)

        sigma_x_sq = self._separable_3d_mean(input * input) - mu_x * mu_x
        sigma_y_sq = self._separable_3d_mean(target * target) - mu_y * mu_y
        sigma_xy = self._separable_3d_mean(input * target) - mu_x * mu_y

        # Clamp variances to avoid numerical issues
        sigma_x_sq = sigma_x_sq.clamp(min=0)
        sigma_y_sq = sigma_y_sq.clamp(min=0)

        # SSIM formula
        numerator = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x**2 + mu_y**2 + self.C1) * (
            sigma_x_sq + sigma_y_sq + self.C2
        )

        ssim_map = numerator / denominator

        return 1 - ssim_map.mean()


@register_loss("stssim", ValueRange.UNIT, is_3d_only=True, requires_fp32=True)
class STSSIM(nn.Module):
    """Spatio-Temporal SSIM (ST-SSIM).

    Computes SSIM on three types of 2D slices:
    - X-Y planes (standard frame-wise SSIM)
    - X-T planes (horizontal scanline over time)
    - Y-T planes (vertical scanline over time)

    This captures both spatial quality and temporal consistency.
    Matches ref.py implementation.

    Args:
        alpha: Weight for X-Y (spatial) SSIM
        beta: Weight for X-T SSIM
        gamma: Weight for Y-T SSIM
        sample_slices: Number of X-T and Y-T slices to sample (None = all)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        gamma: float = 0.25,
        sample_slices: int | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sample_slices = sample_slices

        # Import ssim function
        try:
            from pytorch_msssim import ssim as ssim_fn

            self.ssim_fn = ssim_fn
        except ImportError:
            raise ImportError(
                "pytorch_msssim is required for STSSIM. "
                "Install with: pip install pytorch-msssim"
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ST-SSIM loss.

        Args:
            input: Predicted video (B, T, C, H, W), range [0, 1] or [-1, 1]
            target: Target video (B, T, C, H, W), same range as input

        Returns:
            ST-SSIM loss (1 - ST-SSIM score)
        """
        B, T, C, H, W = input.shape

        # Normalize to [0, 1] if needed
        if input.min() < 0:
            input = input * 0.5 + 0.5
            target = target * 0.5 + 0.5

        # X-Y SSIM (standard frame-wise)
        ssim_xy = self.ssim_fn(
            input.flatten(0, 1), target.flatten(0, 1), data_range=1.0, size_average=True
        )

        # X-T SSIM: for each row y, we have a (T, W) slice
        # Reshape: (B, T, C, H, W) -> (B, H, C, T, W) -> (B*H, C, T, W)
        input_xt = input.permute(0, 3, 2, 1, 4).reshape(B * H, C, T, W)
        target_xt = target.permute(0, 3, 2, 1, 4).reshape(B * H, C, T, W)

        if self.sample_slices is not None and self.sample_slices < H:
            indices = torch.randperm(B * H, device=input.device)[: self.sample_slices]
            input_xt = input_xt[indices]
            target_xt = target_xt[indices]

        ssim_xt = self.ssim_fn(input_xt, target_xt, data_range=1.0, size_average=True)

        # Y-T SSIM: for each column x, we have a (T, H) slice
        # Reshape: (B, T, C, H, W) -> (B, W, C, T, H) -> (B*W, C, T, H)
        input_yt = input.permute(0, 4, 2, 1, 3).reshape(B * W, C, T, H)
        target_yt = target.permute(0, 4, 2, 1, 3).reshape(B * W, C, T, H)

        if self.sample_slices is not None and self.sample_slices < W:
            indices = torch.randperm(B * W, device=input.device)[: self.sample_slices]
            input_yt = input_yt[indices]
            target_yt = target_yt[indices]

        ssim_yt = self.ssim_fn(input_yt, target_yt, data_range=1.0, size_average=True)

        # Combine
        st_ssim = self.alpha * ssim_xy + self.beta * ssim_xt + self.gamma * ssim_yt

        return 1 - st_ssim


@register_loss("tssim", ValueRange.UNIT, is_3d_only=True, requires_fp32=True)
class TSSIM(nn.Module):
    """T-SSIM: SSIM on frame differences.

    Captures temporal structure preservation beyond pixel-level differences.
    Matches ref.py implementation with proper normalization.

    Args:
        combine_with_spatial: If True, combine with spatial SSIM
        spatial_weight: Weight for spatial SSIM (if combined)
    """

    def __init__(
        self,
        combine_with_spatial: bool = False,
        spatial_weight: float = 0.7,
    ):
        super().__init__()
        self.combine_with_spatial = combine_with_spatial
        self.spatial_weight = spatial_weight

        # Import ssim function
        try:
            from pytorch_msssim import ssim as ssim_fn

            self.ssim_fn = ssim_fn
        except ImportError:
            raise ImportError(
                "pytorch_msssim is required for TSSIM. "
                "Install with: pip install pytorch-msssim"
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute T-SSIM loss.

        Args:
            input: Predicted video (B, T, C, H, W), range [-1, 1]
            target: Target video (B, T, C, H, W), range [-1, 1]

        Returns:
            T-SSIM loss (1 - T-SSIM score)
        """
        B, T, C, H, W = input.shape

        if T < 2:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Compute frame differences
        input_diff = input[:, 1:] - input[:, :-1]  # (B, T-1, C, H, W)
        target_diff = target[:, 1:] - target[:, :-1]

        # Normalize differences to [0, 1] range for SSIM
        # Original range is approximately [-2, 2], so (diff + 2) / 4
        input_diff_01 = (input_diff + 2) / 4
        target_diff_01 = (target_diff + 2) / 4

        # SSIM on differences
        ssim_temporal = self.ssim_fn(
            input_diff_01.flatten(0, 1),
            target_diff_01.flatten(0, 1),
            data_range=1.0,
            size_average=True,
        )

        if self.combine_with_spatial:
            # Spatial SSIM on frames
            input_01 = input * 0.5 + 0.5
            target_01 = target * 0.5 + 0.5
            ssim_spatial = self.ssim_fn(
                input_01.flatten(0, 1),
                target_01.flatten(0, 1),
                data_range=1.0,
                size_average=True,
            )
            combined = (
                self.spatial_weight * ssim_spatial
                + (1 - self.spatial_weight) * ssim_temporal
            )
            return 1 - combined

        return 1 - ssim_temporal


@register_loss("fdb", is_3d_only=True)
class FDBLoss(nn.Module):
    """Frame Difference Boundary Loss.

    Penalizes differences in temporal boundaries (frame transitions)
    between predicted and target videos. Useful for temporal edge consistency.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize FDB Loss.

        Args:
            loss_type: Type of loss to compute on frame differences
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute FDB loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            FDB loss value
        """
        B, T, C, H, W = input.shape

        if T < 2:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Frame differences
        input_diff = input[:, 1:] - input[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]

        if self.loss_type == "l1":
            return F.l1_loss(input_diff, target_diff, reduction=self.reduction)
        elif self.loss_type == "mse":
            return F.mse_loss(input_diff, target_diff, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_diff, target_diff)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


@register_loss("temporal_accel", is_3d_only=True)
class TemporalAccelerationLoss(nn.Module):
    """Temporal Acceleration Loss (second-order temporal derivative).

    Penalizes rapid changes in motion/velocity to encourage smooth
    temporal transitions. Useful for video generation/interpolation.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize Temporal Acceleration Loss.

        Args:
            loss_type: Type of loss to compute
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute temporal acceleration loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            Acceleration loss value
        """
        B, T, C, H, W = input.shape

        if T < 3:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Second-order temporal derivative (acceleration)
        # accel = f(t+1) - 2*f(t) + f(t-1)
        input_accel = input[:, 2:] - 2 * input[:, 1:-1] + input[:, :-2]
        target_accel = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]

        if self.loss_type == "l1":
            return F.l1_loss(input_accel, target_accel, reduction=self.reduction)
        elif self.loss_type == "mse":
            return F.mse_loss(input_accel, target_accel, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_accel, target_accel)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


@register_loss("temporal_fft", is_3d_only=True, requires_fp32=True)
class TemporalFFTLoss(nn.Module):
    """Temporal FFT Loss.

    Computes FFT along the temporal dimension and applies loss on
    the frequency domain representation. Useful for temporal consistency.

    Default behavior matches ref.py: uses rfft, amplitude-only mode, no normalization.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        norm_type: NormType = NormType.NONE,
        amplitude_only: bool = True,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """Initialize Temporal FFT Loss.

        Args:
            loss_type: Type of loss to compute
            norm_type: Normalization for FFT components (default: NONE)
            amplitude_only: If True, only compare amplitudes (default: True, matching ref.py)
            eps: Small constant for numerical stability
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.norm_type = norm_type
        self.amplitude_only = amplitude_only
        self.eps = eps
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute temporal FFT loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            Temporal FFT loss value
        """
        # Use rfft along temporal dimension (real FFT, matching ref.py)
        input_fft = torch.fft.rfft(input, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)

        if self.amplitude_only:
            # Compare amplitudes only (matching ref.py default)
            input_mag = input_fft.abs()
            target_mag = target_fft.abs()

            # Apply optional normalization to magnitudes
            if self.norm_type == NormType.L2:
                input_mag = input_mag / (
                    torch.norm(input_mag, dim=1, keepdim=True) + self.eps
                )
                target_mag = target_mag / (
                    torch.norm(target_mag, dim=1, keepdim=True) + self.eps
                )
            elif self.norm_type == NormType.LOG:
                input_mag = torch.log(input_mag + self.eps)
                target_mag = torch.log(target_mag + self.eps)
            elif self.norm_type == NormType.LOG1P:
                input_mag = torch.log1p(input_mag)
                target_mag = torch.log1p(target_mag)

            # Compute loss
            if self.loss_type == "mse":
                loss = F.mse_loss(input_mag, target_mag, reduction="none")
            elif self.loss_type == "l1":
                loss = F.l1_loss(input_mag, target_mag, reduction="none")
            elif self.loss_type == "charbonnier":
                loss = self.charbonnier(input_mag, target_mag)
        else:
            # Compare real and imaginary parts separately
            input_real, input_imag = input_fft.real, input_fft.imag
            target_real, target_imag = target_fft.real, target_fft.imag

            # Apply optional normalization
            if self.norm_type == NormType.L2:
                input_real = input_real / (
                    torch.norm(input_real, dim=1, keepdim=True) + self.eps
                )
                input_imag = input_imag / (
                    torch.norm(input_imag, dim=1, keepdim=True) + self.eps
                )
                target_real = target_real / (
                    torch.norm(target_real, dim=1, keepdim=True) + self.eps
                )
                target_imag = target_imag / (
                    torch.norm(target_imag, dim=1, keepdim=True) + self.eps
                )
            elif self.norm_type == NormType.LOG:
                input_real = torch.sign(input_real) * torch.log(
                    torch.abs(input_real) + self.eps
                )
                input_imag = torch.sign(input_imag) * torch.log(
                    torch.abs(input_imag) + self.eps
                )
                target_real = torch.sign(target_real) * torch.log(
                    torch.abs(target_real) + self.eps
                )
                target_imag = torch.sign(target_imag) * torch.log(
                    torch.abs(target_imag) + self.eps
                )
            elif self.norm_type == NormType.LOG1P:
                input_real = torch.sign(input_real) * torch.log1p(torch.abs(input_real))
                input_imag = torch.sign(input_imag) * torch.log1p(torch.abs(input_imag))
                target_real = torch.sign(target_real) * torch.log1p(
                    torch.abs(target_real)
                )
                target_imag = torch.sign(target_imag) * torch.log1p(
                    torch.abs(target_imag)
                )

            # Compute loss on real and imaginary parts
            if self.loss_type == "mse":
                loss = F.mse_loss(
                    input_real, target_real, reduction="none"
                ) + F.mse_loss(input_imag, target_imag, reduction="none")
            elif self.loss_type == "l1":
                loss = F.l1_loss(input_real, target_real, reduction="none") + F.l1_loss(
                    input_imag, target_imag, reduction="none"
                )
            elif self.loss_type == "charbonnier":
                loss = self.charbonnier(input_real, target_real) + self.charbonnier(
                    input_imag, target_imag
                )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("patch_fft_3d", is_3d_only=True, requires_fp32=True)
class PatchFFT3DLoss(nn.Module):
    """3D Patch-based FFT Loss for spatio-temporal frequency analysis.

    Divides video into 3D patches (T, H, W) and computes FFT on each patch.
    Useful for capturing local spatio-temporal frequency patterns.

    Default behavior matches ref.py: raw FFT without normalization, skips keyframe.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (8, 16, 16),
        loss_type: Literal["mse", "l1", "charbonnier"] = "mse",
        norm_type: NormType = NormType.NONE,
        skip_keyframe: bool = True,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """Initialize 3D Patch FFT Loss.

        Args:
            patch_size: Tuple of (temporal, height, width) patch sizes
            loss_type: Type of loss to compute
            norm_type: Normalization for FFT real/imag (default: NONE for raw FFT)
            skip_keyframe: If True, skip frame 0 (keyframe) before patching
            eps: Small constant for numerical stability
            reduction: Reduction method
        """
        super().__init__()
        self.patch_size = patch_size
        self.loss_type = loss_type
        self.norm_type = norm_type
        self.skip_keyframe = skip_keyframe
        self.eps = eps
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D patch FFT loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            3D patch FFT loss value
        """
        B, T, C, H, W = input.shape
        pt, ph, pw = self.patch_size

        # Skip keyframe if requested
        if self.skip_keyframe:
            input = input[:, 1:]
            target = target[:, 1:]
            T = T - 1

        # Check divisibility (matching ref.py behavior)
        assert T % pt == 0, f"Temporal dim {T} must be divisible by patch_t {pt}"
        assert H % ph == 0, f"Height {H} must be divisible by patch_h {ph}"
        assert W % pw == 0, f"Width {W} must be divisible by patch_w {pw}"

        # Number of patches in each dimension
        nt = T // pt
        nh = H // ph
        nw = W // pw

        # Reshape into 3D patches
        # (B, T, C, H, W) -> (B, nt, pt, C, nh, ph, nw, pw)
        input = input.view(B, nt, pt, C, nh, ph, nw, pw)
        target = target.view(B, nt, pt, C, nh, ph, nw, pw)

        # Rearrange to (B, C, nt, nh, nw, pt, ph, pw)
        input = input.permute(0, 3, 1, 4, 6, 2, 5, 7)
        target = target.permute(0, 3, 1, 4, 6, 2, 5, 7)

        # Flatten batch and patch count dimensions
        # -> (B * C * nt * nh * nw, pt, ph, pw)
        input = input.reshape(-1, pt, ph, pw)
        target = target.reshape(-1, pt, ph, pw)

        # Apply 3D FFT to each patch (no norm, matching ref.py)
        input_fft = torch.fft.fftn(input, dim=(1, 2, 3))
        target_fft = torch.fft.fftn(target, dim=(1, 2, 3))

        # Extract real and imaginary parts
        input_real, input_imag = input_fft.real, input_fft.imag
        target_real, target_imag = target_fft.real, target_fft.imag

        # Apply optional normalization
        if self.norm_type == NormType.L2:
            # Flatten for norm computation, then reshape back
            input_real_flat = input_real.flatten(1)
            input_imag_flat = input_imag.flatten(1)
            target_real_flat = target_real.flatten(1)
            target_imag_flat = target_imag.flatten(1)
            input_real = input_real / (
                input_real_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + self.eps
            )
            input_imag = input_imag / (
                input_imag_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + self.eps
            )
            target_real = target_real / (
                target_real_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + self.eps
            )
            target_imag = target_imag / (
                target_imag_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + self.eps
            )
        elif self.norm_type == NormType.LOG:
            input_real = torch.sign(input_real) * torch.log(
                torch.abs(input_real) + self.eps
            )
            input_imag = torch.sign(input_imag) * torch.log(
                torch.abs(input_imag) + self.eps
            )
            target_real = torch.sign(target_real) * torch.log(
                torch.abs(target_real) + self.eps
            )
            target_imag = torch.sign(target_imag) * torch.log(
                torch.abs(target_imag) + self.eps
            )
        elif self.norm_type == NormType.LOG1P:
            input_real = torch.sign(input_real) * torch.log1p(torch.abs(input_real))
            input_imag = torch.sign(input_imag) * torch.log1p(torch.abs(input_imag))
            target_real = torch.sign(target_real) * torch.log1p(torch.abs(target_real))
            target_imag = torch.sign(target_imag) * torch.log1p(torch.abs(target_imag))

        # Compute loss on real and imaginary parts separately
        if self.loss_type == "mse":
            real_loss = F.mse_loss(input_real, target_real, reduction="none")
            imag_loss = F.mse_loss(input_imag, target_imag, reduction="none")
        elif self.loss_type == "l1":
            real_loss = F.l1_loss(input_real, target_real, reduction="none")
            imag_loss = F.l1_loss(input_imag, target_imag, reduction="none")
        elif self.loss_type == "charbonnier":
            real_loss = self.charbonnier(input_real, target_real)
            imag_loss = self.charbonnier(input_imag, target_imag)

        loss = real_loss + imag_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss("temporal_gradient", is_3d_only=True)
class TemporalGradientLoss(nn.Module):
    """Temporal gradient loss (first-order temporal derivative).

    Computes loss on temporal gradients (frame differences) to ensure
    temporal consistency in video generation/reconstruction.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize Temporal Gradient Loss.

        Args:
            loss_type: Type of loss to compute
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute temporal gradient loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            Temporal gradient loss value
        """
        B, T, C, H, W = input.shape

        if T < 2:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Temporal gradients
        input_grad = input[:, 1:] - input[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]

        if self.loss_type == "l1":
            return F.l1_loss(input_grad, target_grad, reduction=self.reduction)
        elif self.loss_type == "mse":
            return F.mse_loss(input_grad, target_grad, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_grad, target_grad)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
