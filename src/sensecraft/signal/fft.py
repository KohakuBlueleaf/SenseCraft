"""Frequency-domain losses: distances between the FFTs of input and target."""

from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..pixel.regression import Distance, check_distance, distance_map
from ..registry import register


class NormType(StrEnum):
    """How each real or imaginary FFT part is rescaled before the distance."""

    NONE = "none"
    L2 = "l2"  # divide by the part's L2 norm over the transformed dims
    LOG = "log"  # sign(v) log(|v| + eps)
    LOG1P = "log1p"  # sign(v) log1p(|v|)

    @classmethod
    def parse(cls, value: "str | NormType") -> "NormType":
        """The member named by `value`, matched case-insensitively."""
        return cls(str(value).lower())


def normalize_part(
    part: Tensor, norm_type: NormType, eps: float, norm_dims: tuple[int, ...]
) -> Tensor:
    """One real or imaginary FFT part, rescaled by `norm_type`."""
    match norm_type:
        case NormType.L2:
            norm = torch.linalg.vector_norm(part, dim=norm_dims, keepdim=True)
            return part / (norm + eps)
        case NormType.LOG:
            return torch.sign(part) * torch.log(torch.abs(part) + eps)
        case NormType.LOG1P:
            return torch.sign(part) * torch.log1p(torch.abs(part))
    return part


def spectrum_distance(
    input_fft: Tensor,
    target_fft: Tensor,
    kind: Distance,
    norm_type: NormType,
    eps: float,
    norm_dims: tuple[int, ...],
) -> Tensor:
    """Mean of the real-part distance plus the imaginary-part distance."""
    parts = []
    for input_part, target_part in (
        (input_fft.real, target_fft.real),
        (input_fft.imag, target_fft.imag),
    ):
        input_part = normalize_part(input_part, norm_type, eps, norm_dims)
        target_part = normalize_part(target_part, norm_type, eps, norm_dims)
        parts.append(distance_map(kind, input_part, target_part))

    real_loss, imag_loss = parts
    return (real_loss + imag_loss).mean()


@register("patch_fft", dims="2d", fp32=True)
class PatchFFTLoss(nn.Module):
    """Distance between the 2D FFTs of matching square patches.

    Images whose sides are not a multiple of `patch_size` are reflect-padded first.
    """

    def __init__(
        self,
        patch_size: int = 8,
        loss_type: Distance = "mse",
        norm_type: NormType | str = NormType.NONE,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.loss_type = check_distance(loss_type)
        self.norm_type = NormType.parse(norm_type)
        self.eps = eps

    def patches(self, x: Tensor) -> Tensor:
        """(batch, channels, height, width) -> (patch count, patch, patch)."""
        patch = self.patch_size
        height, width = x.shape[-2:]

        # Reflect-pad the bottom and right edge up to a multiple of the patch size
        pad_height = (patch - height % patch) % patch
        pad_width = (patch - width % patch) % patch
        if pad_height > 0 or pad_width > 0:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="reflect")

        batch, channels, height, width = x.shape
        rows, cols = height // patch, width // patch
        x = x.view(batch, channels, rows, patch, cols, patch)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        return x.reshape(-1, patch, patch)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_fft = torch.fft.fft2(self.patches(input))
        target_fft = torch.fft.fft2(self.patches(target))
        return spectrum_distance(
            input_fft, target_fft, self.loss_type, self.norm_type, self.eps, (-2, -1)
        )


@register("fft", dims="2d", fp32=True)
class FFTLoss(nn.Module):
    """Distance between the 2D FFTs of the whole images."""

    def __init__(
        self,
        loss_type: Distance = "mse",
        norm_type: NormType | str = NormType.NONE,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.loss_type = check_distance(loss_type)
        self.norm_type = NormType.parse(norm_type)
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_fft = torch.fft.fft2(input)
        target_fft = torch.fft.fft2(target)
        return spectrum_distance(
            input_fft, target_fft, self.loss_type, self.norm_type, self.eps, (-2, -1)
        )
