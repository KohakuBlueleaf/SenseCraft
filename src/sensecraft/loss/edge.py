"""
Edge and gradient-based losses using conventional CV methods.

These losses emphasize structural features like edges, gradients, and
high-frequency details using classical computer vision operators.

Includes:
- SobelEdgeLoss: Sobel edge detection loss
- LaplacianEdgeLoss: Laplacian edge detection loss (second-order)
- CannyStyleEdgeLoss: Weighted edge loss emphasizing strong edges
- GradientLoss: Direct first-order gradient loss
- HighFrequencyLoss: High-pass filtered loss for fine details
- MultiScaleGradientLoss: Gradient loss at multiple pyramid scales
- StructureTensorLoss: Texture/orientation-aware loss

Example:
    >>> from sensecraft.loss import SobelEdgeLoss, GradientLoss
    >>>
    >>> # Sobel edge loss
    >>> edge_loss = SobelEdgeLoss(loss_type="l1")
    >>> loss = edge_loss(predicted, target)
    >>>
    >>> # Multi-scale gradient loss
    >>> grad_loss = MultiScaleGradientLoss(num_scales=3)
    >>> loss = grad_loss(predicted, target)
"""

from typing import Literal, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import register_loss, ValueRange
from .general import CharbonnierLoss


@register_loss("sobel", ValueRange.UNIT, is_2d_only=True)
class SobelEdgeLoss(nn.Module):
    """Sobel edge detection loss.

    Computes Sobel edge maps for both input and target, then computes
    the loss between edge maps. This emphasizes structural/edge preservation.

    The Sobel operator detects edges by computing gradients in x and y directions.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize SobelEdgeLoss.

        Args:
            loss_type: Type of loss to compute on edge maps
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        # Sobel kernels for x and y gradients
        self.register_buffer(
            "sobel_kx",
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sobel_ky",
            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32),
            persistent=False,
        )

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel operator.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Edge magnitude tensor (B, C, H, W)
        """
        C = x.shape[1]
        kx = self.sobel_kx.repeat(C, 1, 1, 1).to(x.dtype)
        ky = self.sobel_ky.repeat(C, 1, 1, 1).to(x.dtype)

        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)

        # Edge magnitude
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edge loss.

        Args:
            input: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)

        Returns:
            Edge loss value
        """
        input_edges = self.compute_edges(input)
        target_edges = self.compute_edges(target)

        if self.loss_type == "l1":
            return F.l1_loss(input_edges, target_edges, reduction=self.reduction)
        elif self.loss_type == "mse":
            return F.mse_loss(input_edges, target_edges, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_edges, target_edges)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


@register_loss("laplacian", ValueRange.UNIT, is_2d_only=True)
class LaplacianEdgeLoss(nn.Module):
    """Laplacian edge detection loss.

    Uses the Laplacian operator to detect edges (second-order derivative).
    More sensitive to fine details and noise compared to Sobel.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize LaplacianEdgeLoss.

        Args:
            loss_type: Type of loss to compute on edge maps
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        # Laplacian kernel (8-connected)
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor(
                [[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32
            ),
            persistent=False,
        )

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian response.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Laplacian response tensor (B, C, H, W)
        """
        C = x.shape[1]
        kernel = self.laplacian_kernel.repeat(C, 1, 1, 1).to(x.dtype)
        return F.conv2d(x, kernel, padding=1, groups=C)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian edge loss."""
        input_lap = self.compute_laplacian(input)
        target_lap = self.compute_laplacian(target)

        if self.loss_type == "l1":
            return F.l1_loss(input_lap, target_lap, reduction=self.reduction)
        elif self.loss_type == "mse":
            return F.mse_loss(input_lap, target_lap, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_lap, target_lap)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


@register_loss("canny", ValueRange.UNIT, is_2d_only=True)
class CannyStyleEdgeLoss(nn.Module):
    """Canny-style edge loss using gradient magnitude and direction.

    Combines Sobel gradients with non-maximum suppression-like weighting
    to emphasize strong, well-defined edges.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
        threshold: float = 0.1,
    ):
        """Initialize CannyStyleEdgeLoss.

        Args:
            loss_type: Type of loss to compute
            reduction: Reduction method
            threshold: Edge threshold for soft masking
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.threshold = threshold

        self.register_buffer(
            "sobel_kx",
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sobel_ky",
            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32),
            persistent=False,
        )

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def compute_gradient(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradient magnitude and direction.

        Returns:
            Tuple of (magnitude, gx, gy)
        """
        C = x.shape[1]
        kx = self.sobel_kx.repeat(C, 1, 1, 1).to(x.dtype)
        ky = self.sobel_ky.repeat(C, 1, 1, 1).to(x.dtype)

        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-12)

        return magnitude, gx, gy

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Canny-style edge loss."""
        input_mag, input_gx, input_gy = self.compute_gradient(input)
        target_mag, target_gx, target_gy = self.compute_gradient(target)

        # Soft edge mask based on target edges
        edge_weight = torch.sigmoid((target_mag - self.threshold) * 10)

        # Weighted loss on magnitude
        diff_mag = (input_mag - target_mag).abs() * edge_weight

        if self.reduction == "mean":
            return diff_mag.mean()
        elif self.reduction == "sum":
            return diff_mag.sum()
        else:
            return diff_mag


@register_loss("gradient", ValueRange.UNIT, is_2d_only=True)
class GradientLoss(nn.Module):
    """Direct gradient loss (first-order derivatives).

    Computes loss on x and y gradients separately, useful for preserving
    directional edge information.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize GradientLoss.

        Args:
            loss_type: Type of loss to compute
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute x and y gradients using simple finite differences.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (grad_x, grad_y)
        """
        # Simple finite difference gradients
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad_x, grad_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss."""
        input_gx, input_gy = self.compute_gradients(input)
        target_gx, target_gy = self.compute_gradients(target)

        if self.loss_type == "l1":
            loss_x = F.l1_loss(input_gx, target_gx, reduction=self.reduction)
            loss_y = F.l1_loss(input_gy, target_gy, reduction=self.reduction)
        elif self.loss_type == "mse":
            loss_x = F.mse_loss(input_gx, target_gx, reduction=self.reduction)
            loss_y = F.mse_loss(input_gy, target_gy, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            loss_x = self.charbonnier(input_gx, target_gx)
            loss_y = self.charbonnier(input_gy, target_gy)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss_x + loss_y


@register_loss("high_freq", ValueRange.UNIT, is_2d_only=True)
class HighFrequencyLoss(nn.Module):
    """High-frequency emphasis loss using Gaussian high-pass filter.

    Extracts high-frequency components by subtracting a Gaussian-blurred
    version from the original, then computes loss on high-freq details.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        kernel_size: int = 5,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize HighFrequencyLoss.

        Args:
            sigma: Gaussian blur sigma (larger = more high-freq extracted)
            kernel_size: Gaussian kernel size (must be odd)
            loss_type: Type of loss to compute
            reduction: Reduction method
        """
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.loss_type = loss_type
        self.reduction = reduction

        # Create Gaussian kernel
        self.register_buffer("gaussian_kernel", self._create_gaussian_kernel())

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def _create_gaussian_kernel(self) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        size = self.kernel_size
        sigma = self.sigma

        x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.view(1, 1, size, size)

    def extract_high_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-frequency components.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            High-frequency components
        """
        C = x.shape[1]
        kernel = self.gaussian_kernel.repeat(C, 1, 1, 1).to(x.dtype)
        padding = self.kernel_size // 2

        # Low-pass filter (blur)
        low_freq = F.conv2d(x, kernel, padding=padding, groups=C)

        # High-freq = original - low-freq
        return x - low_freq

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute high-frequency loss."""
        input_hf = self.extract_high_freq(input)
        target_hf = self.extract_high_freq(target)

        if self.loss_type == "l1":
            return F.l1_loss(input_hf, target_hf, reduction=self.reduction)
        elif self.loss_type == "mse":
            return F.mse_loss(input_hf, target_hf, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            return self.charbonnier(input_hf, target_hf)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


@register_loss("multi_scale_gradient", ValueRange.UNIT, is_2d_only=True)
class MultiScaleGradientLoss(nn.Module):
    """Multi-scale gradient loss.

    Computes gradient loss at multiple scales (pyramid) to capture
    both fine and coarse edge structures.
    """

    def __init__(
        self,
        num_scales: int = 3,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
        scale_weights: Optional[list] = None,
    ):
        """Initialize MultiScaleGradientLoss.

        Args:
            num_scales: Number of pyramid scales
            loss_type: Type of loss to compute
            reduction: Reduction method
            scale_weights: Optional weights for each scale (default: equal)
        """
        super().__init__()
        self.num_scales = num_scales
        self.loss_type = loss_type
        self.reduction = reduction

        if scale_weights is None:
            scale_weights = [1.0 / num_scales] * num_scales
        self.scale_weights = scale_weights

        self.gradient_loss = GradientLoss(loss_type=loss_type, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale gradient loss."""
        total_loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)

        current_input = input
        current_target = target

        for i in range(self.num_scales):
            loss = self.gradient_loss(current_input, current_target)
            total_loss = total_loss + self.scale_weights[i] * loss

            # Downsample for next scale
            if i < self.num_scales - 1:
                current_input = F.avg_pool2d(current_input, 2)
                current_target = F.avg_pool2d(current_target, 2)

        return total_loss


@register_loss("structure_tensor", ValueRange.UNIT, is_2d_only=True)
class StructureTensorLoss(nn.Module):
    """Structure tensor loss for texture and edge orientation.

    The structure tensor captures local gradient orientation and coherence,
    useful for preserving texture patterns and edge directions.
    """

    def __init__(
        self,
        window_size: int = 5,
        sigma: float = 1.0,
        loss_type: Literal["l1", "mse", "charbonnier"] = "l1",
        reduction: str = "mean",
    ):
        """Initialize StructureTensorLoss.

        Args:
            window_size: Window size for structure tensor computation
            sigma: Gaussian smoothing sigma
            loss_type: Type of loss to compute
            reduction: Reduction method
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.loss_type = loss_type
        self.reduction = reduction

        # Sobel kernels
        self.register_buffer(
            "sobel_kx",
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sobel_ky",
            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32),
            persistent=False,
        )

        # Gaussian kernel for smoothing
        self.register_buffer("gaussian_kernel", self._create_gaussian_kernel())

        if loss_type == "charbonnier":
            self.charbonnier = CharbonnierLoss(eps=1e-6, reduction=reduction)

    def _create_gaussian_kernel(self) -> torch.Tensor:
        """Create Gaussian smoothing kernel."""
        size = self.window_size
        sigma = self.sigma

        x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.view(1, 1, size, size)

    def compute_structure_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute structure tensor components.

        Returns:
            Tuple of (Jxx, Jxy, Jyy) - structure tensor components
        """
        C = x.shape[1]
        kx = self.sobel_kx.repeat(C, 1, 1, 1).to(x.dtype)
        ky = self.sobel_ky.repeat(C, 1, 1, 1).to(x.dtype)

        # Compute gradients
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)

        # Structure tensor components (outer product of gradient)
        Jxx = gx * gx
        Jxy = gx * gy
        Jyy = gy * gy

        # Smooth with Gaussian
        gauss = self.gaussian_kernel.repeat(C, 1, 1, 1).to(x.dtype)
        padding = self.window_size // 2

        Jxx = F.conv2d(Jxx, gauss, padding=padding, groups=C)
        Jxy = F.conv2d(Jxy, gauss, padding=padding, groups=C)
        Jyy = F.conv2d(Jyy, gauss, padding=padding, groups=C)

        return Jxx, Jxy, Jyy

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute structure tensor loss."""
        input_Jxx, input_Jxy, input_Jyy = self.compute_structure_tensor(input)
        target_Jxx, target_Jxy, target_Jyy = self.compute_structure_tensor(target)

        if self.loss_type == "l1":
            loss = (
                F.l1_loss(input_Jxx, target_Jxx, reduction=self.reduction)
                + F.l1_loss(input_Jxy, target_Jxy, reduction=self.reduction)
                + F.l1_loss(input_Jyy, target_Jyy, reduction=self.reduction)
            )
        elif self.loss_type == "mse":
            loss = (
                F.mse_loss(input_Jxx, target_Jxx, reduction=self.reduction)
                + F.mse_loss(input_Jxy, target_Jxy, reduction=self.reduction)
                + F.mse_loss(input_Jyy, target_Jyy, reduction=self.reduction)
            )
        elif self.loss_type == "charbonnier":
            loss = (
                self.charbonnier(input_Jxx, target_Jxx)
                + self.charbonnier(input_Jxy, target_Jxy)
                + self.charbonnier(input_Jyy, target_Jyy)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss / 3  # Average over three components
