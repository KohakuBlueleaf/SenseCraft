"""Record upstream: the reference for stages S0, S1 and S2.

Runs the installed upstream code (sensecraft 0.3.13 in site-packages, with torchvision, the
official LPIPS heads and transformers), never `src/`. Per term in `terms.py`: weight digests, and
at fp64 and at each test dtype the features at every tap, the input gradient for a fixed
cotangent, and the term value with its input gradient. Set `HF_HOME` to reach DINOv3. Real
compute.

    python measure/s1_record.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if not p.startswith(str(HERE.parent / "src"))]
sys.path.insert(0, str(HERE))

import torch
from common import (
    DTYPES,
    REFS,
    cotangents,
    digest,
    images,
    input_gradient,
)

import sensecraft

assert "site-packages" in sensecraft.__file__, sensecraft.__file__
from terms import TERMS, upstream

DEVICE: str = "cuda:0"
FP16_SCALE: float = 2.0**14  # the loss scale an fp16 run backs the value through
BATCH: int = 2
SIZE: int = 128
IMAGES: str | None = None
KEYS: list[str] = list(TERMS)


def record(key: str, x: torch.Tensor) -> dict:
    """Everything later stages compare term `key` against."""
    term, features_of_any, net = upstream(*TERMS[key])
    out = {"weights": digest(net.state_dict())}

    for dtype_name, dtype in {"float64": torch.float64, **DTYPES}.items():
        term = term.to(DEVICE, dtype)
        typed = x.to(DEVICE, dtype)
        reference = out.get("float64")
        cots = cotangents(
            reference["features"] if reference else features_of_any(typed)
        )
        features, grad = input_gradient(features_of_any, typed, cots)

        leaf = typed.detach().requires_grad_()
        value = term(leaf, typed.flip(0))
        scale = FP16_SCALE if dtype == torch.float16 else 1.0
        (value_grad,) = torch.autograd.grad(value * scale, leaf)
        value_grad = value_grad / scale
        out[dtype_name] = {
            "features": [f.cpu() for f in features],
            "input_grad": grad.cpu(),
            "value": value.detach().cpu(),
            "value_grad": value_grad.cpu(),
        }
        print(key, dtype_name, f"value {value.item():.6f}", flush=True)
    return out


def main() -> None:
    """Write `refs/s1_<key>.pt` per term."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    REFS.mkdir(parents=True, exist_ok=True)
    x = images(BATCH, SIZE, IMAGES)
    for key in KEYS:
        torch.save({"input": x, **record(key, x)}, REFS / f"s1_{key}.pt")


if __name__ == "__main__":
    main()
