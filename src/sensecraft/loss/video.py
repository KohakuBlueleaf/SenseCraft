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

from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import register_loss, ValueRange
from .general import CharbonnierLoss, NormType


def _create_gaussian_window_3d(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Create a 3D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the Gaussian window (applied to each dimension)
        sigma: Standard deviation of the Gaussian
        channels: Number of channels
        device: Device to create tensor on
        dtype: Data type of tensor

    Returns:
        3D Gaussian window tensor of shape (channels, 1, window_size, window_size, window_size)
    """
    # Create 1D Gaussian
    coords = torch.arange(window_size, device=device, dtype=dtype or torch.float32)
    coords -= window_size // 2
    gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # Create 3D Gaussian via outer product
    gauss_3d = (
        gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    )
    gauss_3d = gauss_3d.view(1, 1, window_size, window_size, window_size)
    gauss_3d = gauss_3d.repeat(channels, 1, 1, 1, 1)

    return gauss_3d


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


@register_loss("ssim3d", ValueRange.UNIT, is_3d_only=True)
class SSIM3D(nn.Module):
    """3D SSIM loss using volumetric Gaussian windows.

    Computes SSIM in 3D space (T, H, W) for video quality assessment.
    Useful for evaluating spatio-temporal consistency.
    """

    def __init__(
        self,
        window_size: int = 7,
        sigma: float = 1.5,
        data_range: float = 1.0,
        K1: float = 0.01,
        K2: float = 0.03,
        reduction: str = "mean",
    ):
        """Initialize 3D SSIM loss.

        Args:
            window_size: Size of the 3D Gaussian window
            sigma: Standard deviation for Gaussian
            data_range: Dynamic range of pixel values (1.0 for [0,1], 2.0 for [-1,1])
            K1: Constant for luminance stability
            K2: Constant for contrast stability
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.K1 = K1
        self.K2 = K2
        self.reduction = reduction

        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2

        # Window will be created on first forward pass
        self._window = None
        self._channels = None

    def _get_window(
        self, channels: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create the Gaussian window."""
        if self._window is None or self._channels != channels:
            self._window = _create_gaussian_window_3d(
                self.window_size, self.sigma, channels, device, dtype
            )
            self._channels = channels
        return self._window.to(device=device, dtype=dtype)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D SSIM loss.

        Args:
            input: Predicted video tensor (B, T, C, H, W) or (B, C, T, H, W)
            target: Target video tensor (same shape as input)

        Returns:
            SSIM loss (1 - SSIM for minimization)
        """
        # Normalize to (B, C, T, H, W) format
        if input.dim() == 5 and input.shape[1] != input.shape[2]:
            # Assume (B, T, C, H, W) format, convert to (B, C, T, H, W)
            if input.shape[2] in [1, 3]:  # C is likely at position 2
                input = input.permute(0, 2, 1, 3, 4)
                target = target.permute(0, 2, 1, 3, 4)

        B, C, T, H, W = input.shape
        window = self._get_window(C, input.device, input.dtype)

        padding = self.window_size // 2

        # Compute means
        mu1 = F.conv3d(input, window, padding=padding, groups=C)
        mu2 = F.conv3d(target, window, padding=padding, groups=C)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = F.conv3d(input**2, window, padding=padding, groups=C) - mu1_sq
        sigma2_sq = F.conv3d(target**2, window, padding=padding, groups=C) - mu2_sq
        sigma12 = F.conv3d(input * target, window, padding=padding, groups=C) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        # Return 1 - SSIM for loss (minimization)
        loss = 1 - ssim_map

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


@register_loss("stssim", ValueRange.UNIT, is_3d_only=True)
class STSSIM(nn.Module):
    """Spatio-Temporal SSIM combining spatial and temporal SSIM.

    Computes SSIM on individual frames and between consecutive frames
    to ensure both spatial quality and temporal consistency.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        spatial_weight: float = 0.5,
        temporal_weight: float = 0.5,
        reduction: str = "mean",
    ):
        """Initialize ST-SSIM loss.

        Args:
            window_size: Size of Gaussian window for SSIM
            sigma: Standard deviation for Gaussian
            data_range: Dynamic range of pixel values
            spatial_weight: Weight for spatial SSIM component
            temporal_weight: Weight for temporal SSIM component
            reduction: Reduction method
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.reduction = reduction

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        self.register_buffer("C1", torch.tensor(C1))
        self.register_buffer("C2", torch.tensor(C2))

        self._window = None
        self._channels = None

    def _get_window(
        self, channels: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create the Gaussian window."""
        if self._window is None or self._channels != channels:
            self._window = _create_gaussian_window_2d(
                self.window_size, self.sigma, channels, device, dtype
            )
            self._channels = channels
        return self._window.to(device=device, dtype=dtype)

    def _compute_ssim_2d(
        self, x: torch.Tensor, y: torch.Tensor, window: torch.Tensor
    ) -> torch.Tensor:
        """Compute 2D SSIM between two frames."""
        C = x.shape[1]
        padding = self.window_size // 2

        mu1 = F.conv2d(x, window, padding=padding, groups=C)
        mu2 = F.conv2d(y, window, padding=padding, groups=C)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(x**2, window, padding=padding, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(y**2, window, padding=padding, groups=C) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=padding, groups=C) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        return ssim_map.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ST-SSIM loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            ST-SSIM loss value
        """
        B, T, C, H, W = input.shape
        window = self._get_window(C, input.device, input.dtype)

        # Spatial SSIM: average SSIM across all frames
        spatial_ssim = 0.0
        for t in range(T):
            spatial_ssim += self._compute_ssim_2d(input[:, t], target[:, t], window)
        spatial_ssim /= T

        # Temporal SSIM: SSIM between frame differences
        temporal_ssim = 0.0
        if T > 1:
            for t in range(T - 1):
                input_diff = input[:, t + 1] - input[:, t]
                target_diff = target[:, t + 1] - target[:, t]
                temporal_ssim += self._compute_ssim_2d(input_diff, target_diff, window)
            temporal_ssim /= T - 1
        else:
            temporal_ssim = spatial_ssim

        # Combined loss (1 - SSIM for minimization)
        loss = self.spatial_weight * (1 - spatial_ssim) + self.temporal_weight * (
            1 - temporal_ssim
        )

        return loss


@register_loss("tssim", ValueRange.UNIT, is_3d_only=True)
class TSSIM(nn.Module):
    """Temporal SSIM focusing only on frame-to-frame consistency.

    Computes SSIM between consecutive frame differences to ensure
    smooth temporal transitions.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        reduction: str = "mean",
    ):
        """Initialize Temporal SSIM.

        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation for Gaussian
            data_range: Dynamic range of pixel values
            reduction: Reduction method
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.reduction = reduction

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        self.register_buffer("C1", torch.tensor(C1))
        self.register_buffer("C2", torch.tensor(C2))

        self._window = None
        self._channels = None

    def _get_window(
        self, channels: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or create the Gaussian window."""
        if self._window is None or self._channels != channels:
            self._window = _create_gaussian_window_2d(
                self.window_size, self.sigma, channels, device, dtype
            )
            self._channels = channels
        return self._window.to(device=device, dtype=dtype)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Temporal SSIM loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            Temporal SSIM loss
        """
        B, T, C, H, W = input.shape

        if T < 2:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        window = self._get_window(C, input.device, input.dtype)
        padding = self.window_size // 2

        # Compute frame differences
        input_diff = input[:, 1:] - input[:, :-1]  # (B, T-1, C, H, W)
        target_diff = target[:, 1:] - target[:, :-1]

        # Reshape for batch processing
        input_diff = input_diff.reshape(-1, C, H, W)
        target_diff = target_diff.reshape(-1, C, H, W)

        # Compute SSIM on differences
        mu1 = F.conv2d(input_diff, window, padding=padding, groups=C)
        mu2 = F.conv2d(target_diff, window, padding=padding, groups=C)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(input_diff**2, window, padding=padding, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target_diff**2, window, padding=padding, groups=C) - mu2_sq
        sigma12 = (
            F.conv2d(input_diff * target_diff, window, padding=padding, groups=C)
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        # Return 1 - SSIM for loss
        loss = 1 - ssim_map

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.view(B, T - 1, C, H, W)


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


@register_loss("temporal_fft", is_3d_only=True)
class TemporalFFTLoss(nn.Module):
    """Temporal FFT Loss.

    Computes FFT along the temporal dimension and applies loss on
    the frequency domain representation. Useful for temporal consistency.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "mse",
        norm_type: NormType = NormType.LOG1P,
        use_amplitude: bool = True,
        use_phase: bool = False,
        phase_weight: float = 0.1,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """Initialize Temporal FFT Loss.

        Args:
            loss_type: Type of loss to compute
            norm_type: Normalization for FFT magnitudes
            use_amplitude: Whether to use amplitude loss
            use_phase: Whether to use phase loss
            phase_weight: Weight for phase loss
            eps: Small constant for numerical stability
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.norm_type = norm_type
        self.use_amplitude = use_amplitude
        self.use_phase = use_phase
        self.phase_weight = phase_weight
        self.eps = eps
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction="none")

    def compute_temporal_fft(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute FFT along temporal dimension.

        Args:
            x: Input tensor (B, T, C, H, W)

        Returns:
            Tuple of (amplitude, phase)
        """
        # FFT along temporal dimension (dim=1)
        fft = torch.fft.fft(x, dim=1, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=1)

        amplitude = torch.abs(fft_shifted)

        # Apply normalization
        if self.norm_type == NormType.L2:
            norm = torch.norm(amplitude, dim=1, keepdim=True) + self.eps
            amplitude = amplitude / norm
        elif self.norm_type == NormType.LOG:
            amplitude = torch.log(amplitude + self.eps)
        elif self.norm_type == NormType.LOG1P:
            amplitude = torch.log1p(amplitude)

        phase = None
        if self.use_phase:
            phase = torch.angle(fft_shifted)

        return amplitude, phase

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute temporal FFT loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            Temporal FFT loss value
        """
        input_amp, input_phase = self.compute_temporal_fft(input)
        target_amp, target_phase = self.compute_temporal_fft(target)

        loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Amplitude loss
        if self.use_amplitude:
            if self.loss_type == "mse":
                amp_loss = F.mse_loss(input_amp, target_amp, reduction="none")
            elif self.loss_type == "l1":
                amp_loss = F.l1_loss(input_amp, target_amp, reduction="none")
            elif self.loss_type == "charbonnier":
                amp_loss = self.charbonnier(input_amp, target_amp)

            if self.reduction == "mean":
                loss = loss + amp_loss.mean()
            elif self.reduction == "sum":
                loss = loss + amp_loss.sum()
            else:
                loss = amp_loss

        # Phase loss
        if self.use_phase and input_phase is not None and target_phase is not None:
            phase_diff = input_phase - target_phase
            phase_loss = 1 - torch.cos(phase_diff)

            if self.reduction == "mean":
                loss = loss + self.phase_weight * phase_loss.mean()
            elif self.reduction == "sum":
                loss = loss + self.phase_weight * phase_loss.sum()
            else:
                loss = loss + self.phase_weight * phase_loss

        return loss


@register_loss("patch_fft_3d", is_3d_only=True)
class PatchFFT3DLoss(nn.Module):
    """3D Patch-based FFT Loss for spatio-temporal frequency analysis.

    Divides video into 3D patches (T, H, W) and computes FFT on each patch.
    Useful for capturing local spatio-temporal frequency patterns.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 8, 8),
        loss_type: Literal["mse", "l1", "charbonnier"] = "mse",
        norm_type: NormType = NormType.LOG1P,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """Initialize 3D Patch FFT Loss.

        Args:
            patch_size: Tuple of (temporal, height, width) patch sizes
            loss_type: Type of loss to compute
            norm_type: Normalization for FFT magnitudes
            eps: Small constant for numerical stability
            reduction: Reduction method
        """
        super().__init__()
        self.patch_size = patch_size
        self.loss_type = loss_type
        self.norm_type = norm_type
        self.eps = eps
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction="none")

    def extract_patches_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping 3D patches from video.

        Args:
            x: Input tensor (B, T, C, H, W)

        Returns:
            Patches tensor (B, C, nt, nh, nw, pt, ph, pw)
        """
        B, T, C, H, W = x.shape
        pt, ph, pw = self.patch_size

        # Pad if needed
        T_pad = (pt - T % pt) % pt
        H_pad = (ph - H % ph) % ph
        W_pad = (pw - W % pw) % pw

        if T_pad > 0 or H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad, 0, T_pad), mode="replicate")

        B, T, C, H, W = x.shape

        # Reshape to extract patches
        # (B, T, C, H, W) -> (B, nt, pt, C, nh, ph, nw, pw)
        x = x.view(B, T // pt, pt, C, H // ph, ph, W // pw, pw)
        # -> (B, C, nt, nh, nw, pt, ph, pw)
        x = x.permute(0, 3, 1, 4, 6, 2, 5, 7).contiguous()

        return x

    def compute_fft_3d(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute 3D FFT on patches.

        Args:
            patches: Tensor (B, C, nt, nh, nw, pt, ph, pw)

        Returns:
            FFT amplitude tensor
        """
        # FFT on last 3 dimensions (pt, ph, pw)
        fft = torch.fft.fftn(patches, dim=(-3, -2, -1), norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-3, -2, -1))

        amplitude = torch.abs(fft_shifted)

        # Apply normalization
        if self.norm_type == NormType.L2:
            norm = torch.norm(amplitude, dim=(-3, -2, -1), keepdim=True) + self.eps
            amplitude = amplitude / norm
        elif self.norm_type == NormType.LOG:
            amplitude = torch.log(amplitude + self.eps)
        elif self.norm_type == NormType.LOG1P:
            amplitude = torch.log1p(amplitude)

        return amplitude

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D patch FFT loss.

        Args:
            input: Predicted video (B, T, C, H, W)
            target: Target video (B, T, C, H, W)

        Returns:
            3D patch FFT loss value
        """
        # Extract patches
        input_patches = self.extract_patches_3d(input)
        target_patches = self.extract_patches_3d(target)

        # Compute FFT
        input_fft = self.compute_fft_3d(input_patches)
        target_fft = self.compute_fft_3d(target_patches)

        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(input_fft, target_fft, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(input_fft, target_fft, reduction="none")
        elif self.loss_type == "charbonnier":
            loss = self.charbonnier(input_fft, target_fft)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
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
