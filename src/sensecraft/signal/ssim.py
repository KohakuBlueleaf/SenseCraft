"""SSIM and MS-SSIM losses, 1 - similarity, on top of pytorch_msssim.

pytorch_msssim builds its Gaussian window on the host and copies it to the device on
every call unless one is passed in, so these losses keep the window as a buffer.
"""

from pytorch_msssim import ms_ssim, ssim
from torch import Tensor, nn

from ..registry import Range, register
from .edge import gaussian_1d


class GaussianWindow(nn.Module):
    """The 1D Gaussian window pytorch_msssim filters with, kept as a buffer."""

    def __init__(self, size: int = 11, sigma: float = 1.5):
        super().__init__()
        if size % 2 != 1:
            raise ValueError(f"window size must be odd, got {size}")
        self.register_buffer(
            "window", gaussian_1d(size, sigma).view(1, 1, 1, size), persistent=False
        )

    def forward(self, channels: int) -> Tensor:
        """The window in the (channels, 1, 1, size) layout pytorch_msssim takes."""
        return self.window.expand(channels, -1, -1, -1)


def ssim_score(
    input: Tensor,
    target: Tensor,
    window: GaussianWindow,
    data_range: float = 1.0,
    k: tuple[float, float] = (0.01, 0.03),
) -> Tensor:
    """Mean SSIM of (batch, channels, height, width) images."""
    return ssim(input, target, data_range=data_range, win=window(input.shape[1]), K=k)


@register("ssim", range=Range.UNIT, dims="2d", fp32=True)
class SSIMLoss(nn.Module):
    """1 - SSIM, with a Gaussian window of `win_size` taps and std `win_sigma`."""

    def __init__(
        self,
        data_range: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.data_range = data_range
        self.k = (k1, k2)
        self.window = GaussianWindow(win_size, win_sigma)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return 1 - ssim_score(input, target, self.window, self.data_range, self.k)


@register("ms_ssim", range=Range.UNIT, dims="2d", fp32=True)
class MSSSIMLoss(nn.Module):
    """1 - MS-SSIM over five scales; the shorter side must exceed (win_size - 1) * 16.

    `weights` are the per-scale exponents; None takes pytorch_msssim's defaults.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        weights: list[float] | None = None,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.data_range = data_range
        self.weights = weights
        self.k = (k1, k2)
        self.window = GaussianWindow(win_size, win_sigma)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        score = ms_ssim(
            input,
            target,
            data_range=self.data_range,
            win=self.window(input.shape[1]),
            weights=self.weights,
            K=self.k,
        )
        return 1 - score
