"""Edge and gradient losses on classical operators (Sobel, Laplacian, Gaussian)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..pixel.regression import Distance, check_distance, distance
from ..registry import Range, register

SOBEL_X = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
SOBEL_Y = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))
LAPLACIAN = ((-1, -1, -1), (-1, 8, -1), (-1, -1, -1))

# Keeps the gradient magnitude's sqrt differentiable at zero
MAGNITUDE_EPS = 1e-12


def kernel_buffer(rows: tuple[tuple[int, ...], ...]) -> Tensor:
    """A fixed 2D kernel as a (1, 1, k, k) fp32 tensor."""
    return torch.tensor(rows, dtype=torch.float32)[None, None]


def gaussian_1d(size: int, sigma: float) -> Tensor:
    """Normalized 1D Gaussian of `size` taps, centred, built in fp32."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_2d(size: int, sigma: float) -> Tensor:
    """Normalized 2D Gaussian as a (1, 1, size, size) fp32 tensor."""
    kernel = gaussian_1d(size, sigma)
    return (kernel[:, None] * kernel[None, :]).view(1, 1, size, size)


def depthwise(x: Tensor, kernel: Tensor, padding: int) -> Tensor:
    """Convolve every channel of `x` with the same (1, 1, k, k) kernel."""
    channels = x.shape[1]
    weight = kernel.to(x.dtype).expand(channels, -1, -1, -1)
    return F.conv2d(x, weight, padding=padding, groups=channels)


class SobelGradients(nn.Module):
    """Sobel x and y responses, zero-padded to the input size."""

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel_x", kernel_buffer(SOBEL_X), persistent=False)
        self.register_buffer("kernel_y", kernel_buffer(SOBEL_Y), persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return depthwise(x, self.kernel_x, 1), depthwise(x, self.kernel_y, 1)


def magnitude(grad_x: Tensor, grad_y: Tensor) -> Tensor:
    """Gradient magnitude, sqrt(gx^2 + gy^2 + eps)."""
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + MAGNITUDE_EPS)


@register("sobel", range=Range.UNIT, dims="2d", fp32=True)
class SobelEdgeLoss(nn.Module):
    """Distance between Sobel gradient magnitudes."""

    def __init__(self, loss_type: Distance = "l1"):
        super().__init__()
        self.loss_type = check_distance(loss_type)
        self.sobel = SobelGradients()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_edges = magnitude(*self.sobel(input))
        target_edges = magnitude(*self.sobel(target))
        return distance(self.loss_type, input_edges, target_edges)


@register("laplacian", range=Range.UNIT, dims="2d", fp32=True)
class LaplacianEdgeLoss(nn.Module):
    """Distance between 8-connected Laplacian responses."""

    def __init__(self, loss_type: Distance = "l1"):
        super().__init__()
        self.loss_type = check_distance(loss_type)
        self.register_buffer("kernel", kernel_buffer(LAPLACIAN), persistent=False)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_response = depthwise(input, self.kernel, 1)
        target_response = depthwise(target, self.kernel, 1)
        return distance(self.loss_type, input_response, target_response)


@register("canny", range=Range.UNIT, dims="2d", fp32=True)
class CannyStyleEdgeLoss(nn.Module):
    """L1 between Sobel magnitudes, weighted toward the target's strong edges.

    The weight is sigmoid(10 (|grad target| - threshold)).
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold
        self.sobel = SobelGradients()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_magnitude = magnitude(*self.sobel(input))
        target_magnitude = magnitude(*self.sobel(target))

        edge_weight = torch.sigmoid((target_magnitude - self.threshold) * 10)
        return ((input_magnitude - target_magnitude).abs() * edge_weight).mean()


def finite_differences(x: Tensor) -> tuple[Tensor, Tensor]:
    """Forward differences along width and along height."""
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    return grad_x, grad_y


@register("gradient", range=Range.UNIT, dims="2d", fp32=True)
class GradientLoss(nn.Module):
    """Distance between x finite differences plus distance between y ones."""

    def __init__(self, loss_type: Distance = "l1"):
        super().__init__()
        self.loss_type = check_distance(loss_type)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_x, input_y = finite_differences(input)
        target_x, target_y = finite_differences(target)
        loss_x = distance(self.loss_type, input_x, target_x)
        loss_y = distance(self.loss_type, input_y, target_y)
        return loss_x + loss_y


@register("high_freq", range=Range.UNIT, dims="2d", fp32=True)
class HighFrequencyLoss(nn.Module):
    """Distance between high-pass residuals, x - gaussian_blur(x)."""

    def __init__(
        self, sigma: float = 1.0, kernel_size: int = 5, loss_type: Distance = "l1"
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.loss_type = check_distance(loss_type)
        self.padding = kernel_size // 2
        self.register_buffer(
            "kernel", gaussian_2d(kernel_size, sigma), persistent=False
        )

    def high_pass(self, x: Tensor) -> Tensor:
        return x - depthwise(x, self.kernel, self.padding)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return distance(self.loss_type, self.high_pass(input), self.high_pass(target))


@register("multi_scale_gradient", range=Range.UNIT, dims="2d", fp32=True)
class MultiScaleGradientLoss(nn.Module):
    """Weighted sum of the gradient loss over a 2x average-pool pyramid."""

    def __init__(
        self,
        num_scales: int = 3,
        loss_type: Distance = "l1",
        scale_weights: list[float] | None = None,
    ):
        super().__init__()
        if scale_weights is None:
            scale_weights = [1.0 / num_scales] * num_scales
        if len(scale_weights) != num_scales:
            raise ValueError(
                f"{len(scale_weights)} scale_weights for {num_scales} scales"
            )
        self.scale_weights = scale_weights
        self.gradient_loss = GradientLoss(loss_type)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        terms = []
        for scale, weight in enumerate(self.scale_weights):
            if scale > 0:
                input = F.avg_pool2d(input, 2)
                target = F.avg_pool2d(target, 2)
            terms.append(weight * self.gradient_loss(input, target))
        return sum(terms[1:], start=terms[0])


@register("structure_tensor", range=Range.UNIT, dims="2d", fp32=True)
class StructureTensorLoss(nn.Module):
    """Mean distance over the three Gaussian-smoothed structure tensor components."""

    def __init__(
        self, window_size: int = 5, sigma: float = 1.0, loss_type: Distance = "l1"
    ):
        super().__init__()
        self.loss_type = check_distance(loss_type)
        self.padding = window_size // 2
        self.sobel = SobelGradients()
        self.register_buffer(
            "kernel", gaussian_2d(window_size, sigma), persistent=False
        )

    def components(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Smoothed gx^2, gx gy and gy^2."""
        grad_x, grad_y = self.sobel(x)
        products = (grad_x * grad_x, grad_x * grad_y, grad_y * grad_y)
        return tuple(
            depthwise(product, self.kernel, self.padding) for product in products
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_xx, input_xy, input_yy = self.components(input)
        target_xx, target_xy, target_yy = self.components(target)
        loss = (
            distance(self.loss_type, input_xx, target_xx)
            + distance(self.loss_type, input_xy, target_xy)
            + distance(self.loss_type, input_yy, target_yy)
        )
        return loss / 3
