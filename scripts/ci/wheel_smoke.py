"""Smoke test of an installed wheel: it imports, registers every term, and composes on the CPU.

Uses only terms that need no download, and the LPIPS heads shipped inside the package.
"""

import torch

import sensecraft
from sensecraft import Loss, Term
from sensecraft.weights import packaged


def main() -> None:
    """Fail loudly on anything a fresh install should do."""
    assert "site-packages" in sensecraft.__file__, sensecraft.__file__
    assert len(sensecraft.TERMS) >= 39, sorted(sensecraft.TERMS)
    for net in ("vgg", "alex", "squeeze"):
        assert packaged(f"lpips_{net}"), net

    torch.manual_seed(0)
    fake, real = torch.rand(2, 3, 64, 64) * 2 - 1, torch.rand(2, 3, 64, 64) * 2 - 1
    loss = Loss(
        [
            Term("mse"),
            Term("charbonnier", 0.5),
            Term("fft", 0.1),
            Term("ssim", monitor=True),
        ]
    )
    out = loss.value_and_seed(fake, real)
    assert torch.isfinite(out.total) and torch.isfinite(out.seed).all()
    print(
        f"sensecraft {sensecraft.__file__}: {len(sensecraft.TERMS)} terms, total {out.total.item():.4f}"
    )


if __name__ == "__main__":
    main()
