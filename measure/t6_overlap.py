"""T6: does running the target forward on a second stream beat running it in line?

Per perceptual term, the features and the distance are compiled once, then the value and
the gradient at the prediction are timed two ways: target features before the input path on one
stream, or on a side stream while the input path runs. Real compute.

    python measure/t6_overlap.py
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import torch
from terms import TERMS

from sensecraft.compose import Loss, Term

DEVICE: str = "cuda:0"
KEYS: list[str] = ["lpips_vgg", "dino_base_mse"]
BATCH: int = 32
SIZE: int = 256
DTYPE: str = "bfloat16"
ROUNDS: int = 30


def best(run) -> float:
    """Best milliseconds of `run` after five warm-ups."""
    seconds = float("inf")
    for index in range(ROUNDS):
        torch.cuda.synchronize()
        began = time.perf_counter()
        run()
        torch.cuda.synchronize()
        if index >= 5:
            seconds = min(seconds, time.perf_counter() - began)
    return seconds * 1e3


def main() -> None:
    """Print in-line and side-stream times per term."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    dtype = getattr(torch, DTYPE)
    side = torch.cuda.Stream()
    for key in KEYS:
        kind, kwargs = TERMS[key]
        name = "lpips" if kind == "lpips" else "dino_vit"
        loss = Loss([Term(name, **kwargs)]).prepare(DEVICE, dtype, torch.channels_last)
        term = next(iter(loss.parts.values()))
        features = torch.compile(term.features)
        distance = torch.compile(term.distance)
        real = (torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1).to(
            memory_format=torch.channels_last
        )
        fake = (real + 0.1 * torch.randn_like(real)).clamp(-1, 1)

        def in_line():
            leaf = fake.detach().requires_grad_()
            with torch.autocast("cuda", dtype=dtype):
                with torch.no_grad():
                    real_features = features(real)
                value = distance(features(leaf), real_features)
            torch.autograd.grad(value, leaf)

        def overlapped():
            leaf = fake.detach().requires_grad_()
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with (
                torch.cuda.stream(side),
                torch.no_grad(),
                torch.autocast("cuda", dtype=dtype),
            ):
                real_features = features(real)
            with torch.autocast("cuda", dtype=dtype):
                fake_features = features(leaf)
            main.wait_stream(side)
            for tensor in real_features:
                tensor.record_stream(main)
            with torch.autocast("cuda", dtype=dtype):
                value = distance(fake_features, real_features)
            torch.autograd.grad(value, leaf)

        print(
            f"{key:15s} in line {best(in_line):6.2f} ms  side stream {best(overlapped):6.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
