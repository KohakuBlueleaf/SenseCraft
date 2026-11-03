"""`Loss`: weighted terms on one image pair, with range conversion and video handling.

loss = Loss([Term("mse"), Term("lpips", 0.5, net="vgg")], dims=2, input_range=(-1, 1))
out = loss.value_and_seed(fake, real)   # out.total, out.terms, out.seed
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from sensecraft.kernels import wide
from sensecraft.precision import strict_fp32
from sensecraft.prepare import prepare
from sensecraft.registry import Range, TermInfo, lookup


@dataclass
class Term:
    """One weighted loss term; extra keywords go to the term's constructor.

    A `monitor` term is computed without gradient and left out of the total. `shift` is a random
    shift range in pixels: on every call the pair is cropped together by `shift` at a random
    offset, so the term's pooling grid lands on a new phase each step. Monitor terms see the
    whole image.
    """

    name: str
    weight: float = 1.0
    monitor: bool = False
    shift: int = 0
    kwargs: dict = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        monitor: bool = False,
        shift: int = 0,
        kwargs: dict | None = None,
        **extra,
    ):
        if shift < 0:
            raise ValueError(f"shift must be 0 or more, got {shift}")
        # `kwargs` is the dataclass field, so `replace` and `asdict` round-trip; extras join it
        self.name, self.weight, self.monitor, self.shift = name, weight, monitor, shift
        self.kwargs = {**(kwargs or {}), **extra}


@dataclass
class Out:
    """The weighted total, every term's own value, and the gradient at `fake` when asked."""

    total: Tensor
    terms: dict[str, Tensor]
    seed: Tensor | None = None


@torch.compiler.disable
def shifted(fake: Tensor, real: Tensor, shift: int) -> tuple[Tensor, Tensor]:
    """`fake` and `real` cropped together by `shift` pixels at one random offset.

    The offset is drawn on the host from torch's CPU generator, so no device value is read, and
    the window is always `(H - shift, W - shift)`, so a compiled term never sees a new shape. Kept
    out of compiled graphs, where each offset would specialize a new one.
    """
    if shift == 0:
        return fake, real
    height, width = fake.shape[-2:]
    if shift >= min(height, width):
        raise ValueError(f"shift {shift} leaves nothing of a {height}x{width} image")
    top, left = torch.randint(shift + 1, (2,)).tolist()
    window = (..., slice(top, top + height - shift), slice(left, left + width - shift))
    return fake[window], real[window]


def label_of(term: Term, taken: set[str]) -> str:
    """A unique key for `term`: its name, numbered when the name repeats."""
    label, count = term.name, 1
    while label in taken:
        count += 1
        label = f"{term.name}_{count}"
    return label


class Loss(nn.Module):
    """Weighted sum of registered terms over `(B, C, H, W)` images or `(B, T, C, H, W)` video."""

    def __init__(
        self,
        terms: list[Term],
        dims: int = 2,
        input_range: tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        if dims not in (2, 3):
            raise ValueError(f"dims must be 2 or 3, got {dims}")
        self.dims = dims
        self.low, self.high = input_range
        self.parts = nn.ModuleDict()
        self.specs: dict[str, tuple[Term, TermInfo]] = {}

        # Build every term, refusing ones that cannot read this layout
        for term in terms:
            info = lookup(term.name)
            if dims == 2 and info.dims == "3d":
                raise ValueError(f"term {term.name!r} takes video, but dims is 2")
            label = label_of(term, set(self.specs))
            self.parts[label] = info.factory(**term.kwargs)
            self.specs[label] = (term, info)

    def _ranged(self, x: Tensor, wanted: Range) -> Tensor:
        """`x` from the input range to the range a term reads."""
        if wanted is Range.ANY:
            return x
        unit = (x - self.low) / (self.high - self.low)
        return unit if wanted is Range.UNIT else unit * 2 - 1

    def _value(self, label: str, fake: Tensor, real: Tensor, shift: int = 0) -> Tensor:
        """One term's value on the pair, in its range, layout and precision, `shift`ed."""
        info = self.specs[label][1]
        fake, real = self._ranged(fake, info.range), self._ranged(real, info.range)

        # Image terms see a video as a batch of frames
        if self.dims == 3 and info.dims == "2d":
            fake, real = fake.flatten(0, 1), real.flatten(0, 1)
        fake, real = shifted(fake, real, shift)

        # Plain terms run in strict fp32; feature-net terms keep their own mixed precision
        if info.fp32:
            with torch.autocast(fake.device.type, enabled=False), strict_fp32():
                return self.parts[label](wide(fake), wide(real))
        return self.parts[label](fake, real)

    def forward(self, fake: Tensor, real: Tensor) -> Out:
        """Every term's value and the weighted total; no host reads."""
        values, total = {}, None
        for label, (term, _) in self.specs.items():
            if term.monitor:
                with torch.no_grad():
                    values[label] = self._value(label, fake, real)
                continue
            values[label] = self._value(label, fake, real, term.shift)
            weighted = term.weight * values[label]
            total = weighted if total is None else total + weighted
        if total is None:
            total = fake.new_zeros((), dtype=torch.float32)
        return Out(total, values)

    def prepare(
        self,
        device,
        dtype=None,
        memory_format=torch.preserve_format,
        compile="off",
        **kwargs,
    ):
        """This loss on `device`; see `sensecraft.prepare.prepare`."""
        return prepare(self, device, dtype, memory_format, compile, **kwargs)

    def value_and_seed(self, fake: Tensor, real: Tensor, scale: float = 1.0) -> Out:
        """Values, and the gradient of `scale * total` at `fake`; no graph is kept."""
        leaf = fake.detach().requires_grad_()
        out = self(leaf, real.detach())

        # A loss with no trainable term, or terms that gave a constant, has a zero gradient
        if out.total.requires_grad:
            (seed,) = torch.autograd.grad(out.total * scale, leaf)
        else:
            seed = torch.zeros_like(fake)
        terms = {label: value.detach() for label, value in out.terms.items()}
        return Out(out.total.detach(), terms, seed)
