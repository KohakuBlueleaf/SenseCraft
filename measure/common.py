"""Shared by the measurement stages: the fixed input batch and the recorded reference files."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

REFS = Path(__file__).parent / "refs"
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def images(batch: int, size: int, folder: str | None = None, seed: int = 0) -> Tensor:
    """`batch` images in [-1, 1], fp64: real ones from `folder`, else smooth seeded noise."""
    if folder is not None:
        from PIL import Image
        from torchvision.transforms.functional import center_crop, pil_to_tensor, resize

        paths = sorted(
            p
            for p in Path(folder).rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        frames = []
        for path in paths[:: max(len(paths) // batch, 1)][:batch]:
            with Image.open(path) as image:
                pixels = pil_to_tensor(image.convert("RGB"))
            frames.append(center_crop(resize(pixels, size), size))
        return torch.stack(frames).double() / 127.5 - 1

    # Coarse structure plus fine noise, so every scale of a net sees signal
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(
        batch, 3, size // 16, size // 16, generator=generator, dtype=torch.float64
    )
    fine = torch.randn(batch, 3, size, size, generator=generator, dtype=torch.float64)
    smooth = F.interpolate(coarse, size=size, mode="bicubic", align_corners=False)
    return (0.6 * smooth + 0.2 * fine).tanh()


def cotangents(features: list[Tensor], seed: int = 1) -> list[Tensor]:
    """A fixed fp64 cotangent per feature map, for input gradients that see every channel."""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(f.shape, generator=generator, dtype=torch.float64) for f in features
    ]


def input_gradient(
    features_of, x: Tensor, cots: list[Tensor]
) -> tuple[list[Tensor], Tensor]:
    """Features of `x` and the gradient at `x` of their dot with `cots`, both detached."""
    leaf = x.detach().requires_grad_()
    features = features_of(leaf)
    dot = sum(
        (f.double() * c.to(f.device)).sum() for f, c in zip(features, cots, strict=True)
    )
    (grad,) = torch.autograd.grad(dot, leaf)
    return [f.detach() for f in features], grad.detach()


def _plain(value: Tensor) -> Tensor:
    """A CPU copy in plain row-major layout, whatever the source layout."""
    return value.detach().cpu().clone(memory_format=torch.contiguous_format)


def digest(tensors: dict[str, Tensor]) -> dict[str, str]:
    """A sha256 per tensor, for comparing two sets of weights exactly."""
    return {
        key: hashlib.sha256(_plain(value).view(torch.uint8).numpy()).hexdigest()
        for key, value in sorted(tensors.items())
    }
