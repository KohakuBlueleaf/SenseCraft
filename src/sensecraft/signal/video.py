"""Video losses on (batch, frames, channels, height, width) clips: temporal SSIMs,
frame-difference distances and temporal or spatio-temporal FFT distances."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..pixel.regression import Distance, check_distance, distance, distance_map
from ..registry import Range, register
from .edge import gaussian_2d
from .fft import NormType, normalize_part, spectrum_distance
from .ssim import GaussianWindow, ssim_score


def causal_decay_kernel(size: int, decay: float) -> Tensor:
    """Normalized one-sided temporal kernel over [t - size + 1, ..., t], built in fp32.

    The weight of the frame k steps back is exp(-(k + 1) decay).
    """
    weights = torch.exp(-torch.arange(size, 0, -1).float() * decay)
    return weights / weights.sum()


@register("ssim3d", range=Range.UNIT, dims="3d", fp32=True)
class SSIM3D(nn.Module):
    """1 - SSIM with local statistics from a spatial Gaussian and a causal time kernel.

    The two filters run one after the other; the temporal one only looks back and
    replicates the first frame as padding. Clips are read in [0, 1].
    """

    def __init__(
        self,
        kernel_t: int = 5,
        kernel_s: int = 11,
        sigma_s: float = 1.5,
        temporal_decay: float = 0.5,
        c1: float = 0.01**2,
        c2: float = 0.03**2,
    ):
        super().__init__()
        self.kernel_t = kernel_t
        self.kernel_s = kernel_s
        self.c1 = c1
        self.c2 = c2
        self.register_buffer(
            "spatial_kernel", gaussian_2d(kernel_s, sigma_s), persistent=False
        )
        self.register_buffer(
            "temporal_kernel",
            causal_decay_kernel(kernel_t, temporal_decay)[None, None],
            persistent=False,
        )

    def local_mean(self, x: Tensor) -> Tensor:
        """Spatial Gaussian mean of every frame, then causal temporal mean."""
        batch, frames, channels, height, width = x.shape

        # Spatial: depthwise 2D conv over all frames at once
        spatial_kernel = self.spatial_kernel.expand(channels, 1, -1, -1).to(x.dtype)
        x = F.conv2d(
            x.flatten(0, 1),
            spatial_kernel,
            padding=self.kernel_s // 2,
            groups=channels,
        )
        x = x.view(batch, frames, channels, height, width)

        # Temporal: one 1D conv per pixel over time, left-padded by repeating frame 0
        series = x.permute(0, 2, 3, 4, 1).flatten(0, 3)
        series = F.pad(series, (self.kernel_t - 1, 0), mode="replicate")
        series = F.conv1d(series[:, None, :], self.temporal_kernel.to(x.dtype))[:, 0]
        return series.view(batch, channels, height, width, frames).permute(
            0, 4, 1, 2, 3
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        mu_x = self.local_mean(input)
        mu_y = self.local_mean(target)

        # Variances clamped at zero against cancellation in E[x^2] - E[x]^2
        sigma_x_sq = (self.local_mean(input * input) - mu_x * mu_x).clamp(min=0)
        sigma_y_sq = (self.local_mean(target * target) - mu_y * mu_y).clamp(min=0)
        sigma_xy = self.local_mean(input * target) - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        denominator = (mu_x**2 + mu_y**2 + self.c1) * (
            sigma_x_sq + sigma_y_sq + self.c2
        )
        return 1 - (numerator / denominator).mean()


@register("stssim", range=Range.UNIT, dims="3d", fp32=True)
class STSSIM(nn.Module):
    """1 - weighted SSIM over x-y frames, x-t slices and y-t slices.

    With `sample_slices`, that many x-t and y-t slices are drawn at random per call
    (only when it is below the height, respectively width). Clips are read in [0, 1].
    Slice sides shorter than the 11-tap window are left unsmoothed by pytorch_msssim,
    with a warning.
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
        self.window = GaussianWindow()

    def sample(self, input: Tensor, target: Tensor, side: int) -> tuple[Tensor, Tensor]:
        """A random subset of `sample_slices` slices when that is below `side`."""
        if self.sample_slices is None or self.sample_slices >= side:
            return input, target
        indices = torch.randperm(input.shape[0], device=input.device)
        indices = indices[: self.sample_slices]
        return input[indices], target[indices]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch, frames, channels, height, width = input.shape

        ssim_xy = ssim_score(input.flatten(0, 1), target.flatten(0, 1), self.window)

        # x-t slices: one (frames, width) image per row
        input_xt = input.permute(0, 3, 2, 1, 4).reshape(
            batch * height, channels, frames, width
        )
        target_xt = target.permute(0, 3, 2, 1, 4).reshape(
            batch * height, channels, frames, width
        )
        input_xt, target_xt = self.sample(input_xt, target_xt, height)
        ssim_xt = ssim_score(input_xt, target_xt, self.window)

        # y-t slices: one (frames, height) image per column
        input_yt = input.permute(0, 4, 2, 1, 3).reshape(
            batch * width, channels, frames, height
        )
        target_yt = target.permute(0, 4, 2, 1, 3).reshape(
            batch * width, channels, frames, height
        )
        input_yt, target_yt = self.sample(input_yt, target_yt, width)
        ssim_yt = ssim_score(input_yt, target_yt, self.window)

        score = self.alpha * ssim_xy + self.beta * ssim_xt + self.gamma * ssim_yt
        return 1 - score


def frame_differences(x: Tensor) -> Tensor:
    """x[t + 1] - x[t] along the frame axis."""
    return x[:, 1:] - x[:, :-1]


@register("tssim", range=Range.SYMMETRIC, dims="3d", fp32=True)
class TSSIM(nn.Module):
    """1 - SSIM of frame differences, optionally blended with per-frame SSIM.

    Reads [-1, 1] clips: differences are mapped to [0, 1] by (d + 2) / 4 and frames by
    x / 2 + 1 / 2. Clips with fewer than 2 frames give 0.
    """

    def __init__(self, combine_with_spatial: bool = False, spatial_weight: float = 0.7):
        super().__init__()
        self.combine_with_spatial = combine_with_spatial
        self.spatial_weight = spatial_weight
        self.window = GaussianWindow()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape[1] < 2:
            return input.new_zeros(())

        input_diff = (frame_differences(input) + 2) / 4
        target_diff = (frame_differences(target) + 2) / 4
        score = ssim_score(
            input_diff.flatten(0, 1), target_diff.flatten(0, 1), self.window
        )

        if self.combine_with_spatial:
            input_frames = (input * 0.5 + 0.5).flatten(0, 1)
            target_frames = (target * 0.5 + 0.5).flatten(0, 1)
            spatial = ssim_score(input_frames, target_frames, self.window)
            score = self.spatial_weight * spatial + (1 - self.spatial_weight) * score

        return 1 - score


@register("fdb", dims="3d")
class FDBLoss(nn.Module):
    """Frame difference boundary loss: distance between frame differences.

    Clips with fewer than 2 frames give 0. Same value as `temporal_gradient`.
    """

    def __init__(self, loss_type: Distance = "l1"):
        super().__init__()
        self.loss_type = check_distance(loss_type)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape[1] < 2:
            return input.new_zeros(())
        return distance(
            self.loss_type, frame_differences(input), frame_differences(target)
        )


@register("temporal_accel", dims="3d")
class TemporalAccelerationLoss(nn.Module):
    """Distance between second temporal differences, x[t + 1] - 2 x[t] + x[t - 1].

    Clips with fewer than 3 frames give 0.
    """

    def __init__(self, loss_type: Distance = "l1"):
        super().__init__()
        self.loss_type = check_distance(loss_type)

    @staticmethod
    def acceleration(x: Tensor) -> Tensor:
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape[1] < 3:
            return input.new_zeros(())
        return distance(
            self.loss_type, self.acceleration(input), self.acceleration(target)
        )


@register("temporal_fft", dims="3d", fp32=True)
class TemporalFFTLoss(nn.Module):
    """Distance between real FFTs along the frame axis.

    With `amplitude_only` the magnitudes are compared (L2 norm over frequency, or
    log(|v| + eps), or log1p(|v|)); otherwise the real and imaginary parts are.
    """

    def __init__(
        self,
        loss_type: Distance = "l1",
        norm_type: NormType | str = NormType.NONE,
        amplitude_only: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.loss_type = check_distance(loss_type)
        self.norm_type = NormType.parse(norm_type)
        self.amplitude_only = amplitude_only
        self.eps = eps

    def normalize_amplitude(self, amplitude: Tensor) -> Tensor:
        """Amplitudes are nonnegative, so LOG and LOG1P drop the sign."""
        match self.norm_type:
            case NormType.LOG:
                return torch.log(amplitude + self.eps)
            case NormType.LOG1P:
                return torch.log1p(amplitude)
        return normalize_part(amplitude, self.norm_type, self.eps, (1,))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_fft = torch.fft.rfft(input, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)

        if not self.amplitude_only:
            return spectrum_distance(
                input_fft, target_fft, self.loss_type, self.norm_type, self.eps, (1,)
            )

        input_amplitude = self.normalize_amplitude(input_fft.abs())
        target_amplitude = self.normalize_amplitude(target_fft.abs())
        return distance_map(self.loss_type, input_amplitude, target_amplitude).mean()


@register("patch_fft_3d", dims="3d", fp32=True)
class PatchFFT3DLoss(nn.Module):
    """Distance between the 3D FFTs of matching (frames, height, width) patches.

    With `skip_keyframe` frame 0 is dropped first. The clip sides must be multiples of
    `patch_size`.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (8, 16, 16),
        loss_type: Distance = "mse",
        norm_type: NormType | str = NormType.NONE,
        skip_keyframe: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.loss_type = check_distance(loss_type)
        self.norm_type = NormType.parse(norm_type)
        self.skip_keyframe = skip_keyframe
        self.eps = eps

    def patches(self, x: Tensor) -> Tensor:
        """(batch, frames, channels, height, width) -> (patch count, pt, ph, pw)."""
        batch, frames, channels, height, width = x.shape
        patch_t, patch_h, patch_w = self.patch_size
        for name, side, patch in (
            ("frames", frames, patch_t),
            ("height", height, patch_h),
            ("width", width, patch_w),
        ):
            if side % patch != 0:
                raise ValueError(f"{name} {side} is not a multiple of patch {patch}")

        x = x.view(
            batch,
            frames // patch_t,
            patch_t,
            channels,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )

        # -> (batch, channels, nt, nh, nw, pt, ph, pw)
        x = x.permute(0, 3, 1, 4, 6, 2, 5, 7)
        return x.reshape(-1, patch_t, patch_h, patch_w)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.skip_keyframe:
            input = input[:, 1:]
            target = target[:, 1:]

        dims = (1, 2, 3)
        input_fft = torch.fft.fftn(self.patches(input), dim=dims)
        target_fft = torch.fft.fftn(self.patches(target), dim=dims)
        return spectrum_distance(
            input_fft, target_fft, self.loss_type, self.norm_type, self.eps, dims
        )


@register("temporal_gradient", dims="3d")
class TemporalGradientLoss(FDBLoss):
    """Distance between frame differences; the same loss as `fdb` under another name."""
