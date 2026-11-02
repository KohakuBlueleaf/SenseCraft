"""Stage S6: `Loss` routes ranges, weights, monitors and video frames as specified.

Stub terms with known outputs stand in for real ones, so this checks only the composing. Runs
on the CPU in seconds.

    python measure/s6_compose.py
"""

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch
from torch import Tensor, nn

from sensecraft import Loss, Range, Term, register


@register("stub_unit_mean", Range.UNIT, dims="2d")
class UnitMean(nn.Module):
    """Mean of the input in its range, so the range conversion shows in the value."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.mean()


@register("stub_symmetric_mean", Range.SYMMETRIC, dims="any")
class SymmetricMean(UnitMean):
    pass


@register("stub_video_frames", dims="3d")
class Frames(nn.Module):
    """The frame count of a video, to show a 3d term sees the clip whole."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.new_tensor(float(input.shape[1]))


@register("stub_fp32", fp32=True)
class Dtype(nn.Module):
    """1 when the input arrived in fp32."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.new_tensor(float(input.dtype == torch.float32))


def check(name: str, ok: bool) -> None:
    """Print one line, and stop at the first failure."""
    print(("ok  " if ok else "FAIL") + "  " + name)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    """Every composing rule on one pair."""
    torch.manual_seed(0)
    fake = torch.rand(2, 3, 8, 8) * 2 - 1
    real = torch.rand(2, 3, 8, 8) * 2 - 1

    # Range: the input range maps to what each term reads
    out = Loss(
        [Term("stub_unit_mean"), Term("stub_symmetric_mean")], input_range=(-1, 1)
    )(fake, real)
    check(
        "unit term reads [0, 1]",
        torch.allclose(out.terms["stub_unit_mean"], (fake.mean() + 1) / 2),
    )
    check(
        "symmetric term reads [-1, 1]",
        torch.allclose(out.terms["stub_symmetric_mean"], fake.mean()),
    )
    out = Loss([Term("stub_symmetric_mean")], input_range=(0, 1))(
        fake * 0.5 + 0.5, real
    )
    check("from [0, 1] to [-1, 1]", torch.allclose(out.total, fake.mean()))

    # Weights, monitors and repeated names
    terms = [
        Term("stub_unit_mean", 2.0),
        Term("stub_unit_mean", 3.0),
        Term("stub_symmetric_mean", 5.0, monitor=True),
    ]
    out = Loss(terms)(fake, real)
    unit = (fake.mean() + 1) / 2
    check("weighted sum, monitor left out", torch.allclose(out.total, 5 * unit))
    check(
        "repeated names get their own keys",
        set(out.terms) == {"stub_unit_mean", "stub_unit_mean_2", "stub_symmetric_mean"},
    )
    check("monitor has no graph", not out.terms["stub_symmetric_mean"].requires_grad)

    # value_and_seed equals autograd on the same total
    loss = Loss([Term("stub_unit_mean", 2.0), Term("stub_symmetric_mean", 0.5)])
    leaf = fake.clone().requires_grad_()
    (expected,) = torch.autograd.grad(loss(leaf, real).total * 4.0, leaf)
    out = loss.value_and_seed(fake, real, scale=4.0)
    check("seed equals the autograd gradient", torch.allclose(out.seed, expected))
    check("value_and_seed keeps no graph", out.total.grad_fn is None)

    # A loss with nothing to differentiate gives a zero seed
    out = Loss([Term("stub_unit_mean", monitor=True)]).value_and_seed(fake, real)
    check("all-monitor loss gives a zero seed", bool((out.seed == 0).all()))

    # A term survives dataclass helpers
    term = dataclasses.replace(Term("stub_unit_mean", 2.0, extra=1), weight=3.0)
    check(
        "Term keeps its kwargs through replace",
        term.kwargs == {"extra": 1} and term.weight == 3.0,
    )

    # Video: image terms see frames, video terms see the clip
    clip_fake, clip_real = torch.rand(2, 5, 3, 8, 8), torch.rand(2, 5, 3, 8, 8)
    out = Loss([Term("stub_unit_mean"), Term("stub_video_frames")], dims=3)(
        clip_fake, clip_real
    )
    check("video term sees 5 frames", out.terms["stub_video_frames"].item() == 5)
    check(
        "image term on frames",
        torch.allclose(out.terms["stub_unit_mean"], (clip_fake.mean() + 1) / 2),
    )
    try:
        Loss([Term("stub_video_frames")], dims=2)
        refused = False
    except ValueError:
        refused = True
    check("video term refused on images", refused)

    # fp32 terms run with autocast off on fp32 inputs
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = Loss([Term("stub_fp32")])(fake.bfloat16(), real.bfloat16())
    check("fp32 term gets fp32", out.total.item() == 1)

    # Unknown names come back with the closest ones
    try:
        Loss([Term("stub_unit_maen")])
        message = ""
    except KeyError as err:
        message = str(err)
    check("unknown term suggests a name", "stub_unit_mean" in message)


if __name__ == "__main__":
    main()
