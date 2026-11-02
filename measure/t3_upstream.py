"""T5 for sensecraft 0.3.13: the same whole-term timing as `t3_terms.py`, on the installed release.

Needs an environment with 0.3.13 installed; never imports `src/`. Weights in the compute dtype and
layout, each term compiled, forward and the gradient at the prediction. Real compute.

    python measure/t3_upstream.py
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if not p.startswith(str(HERE.parent / "src"))]
sys.path.insert(0, str(HERE))

import torch
from common import REFS
from terms import TERMS, upstream

import sensecraft

assert "site-packages" in sensecraft.__file__, sensecraft.__file__

DEVICE: str = "cuda:0"
KEYS: list[str] = ["lpips_vgg", "dino_base_mse"]
BATCH: int = 32
SIZE: int = 256
ROUNDS: int = 20
DTYPE: str = "bfloat16"
OUT: Path = REFS / "t3_upstream.jsonl"


def best(run) -> float:
    """Best milliseconds of `run` after three warm-ups."""
    seconds = float("inf")
    for index in range(ROUNDS):
        torch.cuda.synchronize()
        began = time.perf_counter()
        run()
        torch.cuda.synchronize()
        if index >= 3:
            seconds = min(seconds, time.perf_counter() - began)
    return seconds * 1e3


def main() -> None:
    """Print and append one row per term."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    dtype = getattr(torch, DTYPE)
    layout = torch.channels_last
    with OUT.open("a", encoding="utf-8") as file:
        for key in KEYS:
            term, _, _ = upstream(*TERMS[key])
            term = term.to(DEVICE, memory_format=layout)
            for param in term.parameters():
                param.data = param.data.to(dtype)
            term.compile()
            real = (torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1).to(
                memory_format=layout
            )
            fake = (real + 0.1 * torch.randn_like(real)).clamp(-1, 1)

            def whole():
                leaf = fake.detach().requires_grad_()
                with torch.autocast("cuda", dtype=dtype):
                    value = term(leaf, real)
                torch.autograd.grad(value, leaf)

            row = {
                "term": key,
                "release": "0.3.13",
                "dtype": DTYPE,
                "term_ms": best(whole),
            }
            file.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
