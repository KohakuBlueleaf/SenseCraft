"""Stage S8: what a random shift does to a perceptual term's gradient.

A net with pooling sends its gradient through a fixed grid, so the same pixels of a prediction get
the same push at every step, and a pattern locked to that grid can build up. With `Term(shift=s)`
the pair is cropped at a random offset per call and the grid moves. For one prediction this stage
takes the gradient at `fake` over `DRAWS` calls, without and with each shift, and reports:

- persistence: |mean gradient|^2 / mean |gradient|^2, 1 when every call pushes the same way;
- the mean gradient's peak over its RMS, and its power above a quarter of the sampling rate.

Predictions are the images blurred and lightly noised, a stand-in for a reconstruction.

    python measure/s8_shift.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F
from common import REFS, images

from sensecraft import Loss, Term

DEVICE: str = "cuda:0"
# Real images; None uses the seeded smooth noise of `common.images`
FOLDER: str | None = None
BATCH: int = 8
SIZE: int = 256
DRAWS: int = 16
SHIFTS: tuple = (0, 8, 16, 32)
HIGH: float = 0.25
TERMS: dict[str, tuple[str, dict]] = {
    "lpips_vgg": ("lpips", {"net": "vgg"}),
    "lpips_alex": ("lpips", {"net": "alex"}),
}
OUT: Path = REFS / "s8_shift.json"


def degraded(real: torch.Tensor) -> torch.Tensor:
    """A reconstruction stand-in: a 5-tap blur and noise of 0.02."""
    kernel = torch.tensor(
        [1.0, 4.0, 6.0, 4.0, 1.0], device=real.device, dtype=real.dtype
    )
    kernel = (kernel[:, None] * kernel[None, :] / 256).expand(3, 1, 5, 5)
    blurred = F.conv2d(F.pad(real, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)
    return blurred + 0.02 * torch.randn_like(real)


def high_share(grad: torch.Tensor) -> float:
    power = torch.fft.fft2(grad.double()).abs().square()
    fy = torch.fft.fftfreq(grad.shape[-2], device=grad.device).abs()
    fx = torch.fft.fftfreq(grad.shape[-1], device=grad.device).abs()
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return (power[..., radius > HIGH].sum() / power.sum()).item()


def main() -> None:
    torch.manual_seed(0)
    real = images(BATCH, SIZE, FOLDER).to(DEVICE).float()
    fake = degraded(real)
    rows: dict = {}
    for key, (name, kwargs) in TERMS.items():
        for shift in SHIFTS:
            loss = Loss([Term(name, shift=shift, **kwargs)]).to(DEVICE).eval()
            draws = torch.stack(
                [loss.value_and_seed(fake, real).seed.double() for _ in range(DRAWS)]
            )
            mean = draws.mean(0)
            row = {
                "persistence": (
                    mean.square().sum() / draws.square().sum(dim=(1, 2, 3, 4)).mean()
                ).item(),
                "mean_peak_over_rms": (
                    mean.abs().max() / mean.square().mean().sqrt()
                ).item(),
                "mean_high_share": high_share(mean),
                "draw_high_share": sum(high_share(d) for d in draws) / DRAWS,
            }
            rows.setdefault(key, {})[str(shift)] = row
            print(
                key,
                f"shift {shift:2d}",
                "  ".join(f"{k} {v:.4f}" for k, v in row.items()),
            )
    REFS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
