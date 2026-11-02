"""T3 to T5: each perceptual term as a whole, then its net and its distance alone.

Per term and arm (dtype, layout, compile): T3 the net's target forward without gradient and its
input forward plus backward; T4 the distance alone on recorded features, forward and backward;
T5 the whole term through `Loss.value_and_seed`. The T5 time against T3 plus T4 is what composing
costs. Run on an idle GPU. Real compute.

    python measure/t3_terms.py
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import torch
from common import REFS
from terms import TERMS

from sensecraft.compose import Loss, Term

DEVICE: str = "cuda:0"
KEYS: list[str] = ["lpips_vgg", "dino_base_mse"]
BATCH: int = 32
SIZE: int = 256
ROUNDS: int = 20
ARMS: list[dict] = [
    {"dtype": "bfloat16", "channels_last": False, "compile": "off"},
    {"dtype": "bfloat16", "channels_last": True, "compile": "off"},
    {"dtype": "bfloat16", "channels_last": True, "compile": "term"},
    {"dtype": "float16", "channels_last": True, "compile": "term"},
]
OUT: Path = REFS / "t3_terms.jsonl"


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


def row_of(key: str, arm: dict) -> dict:
    """The T3, T4 and T5 times of one term under one arm."""
    kind, kwargs = TERMS[key]
    name = "lpips" if kind == "lpips" else "dino_vit"
    dtype = getattr(torch, arm["dtype"])
    layout = torch.channels_last if arm["channels_last"] else torch.preserve_format
    loss = Loss([Term(name, **kwargs)]).prepare(DEVICE, dtype, layout, arm["compile"])
    term = next(iter(loss.parts.values()))

    real = (torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1).to(
        memory_format=layout
    )
    fake = (real + 0.1 * torch.randn_like(real)).clamp(-1, 1)
    autocast = torch.autocast("cuda", dtype=dtype)

    # T3: the net alone
    def target_forward():
        with torch.no_grad(), autocast:
            term.features(real)

    def input_both():
        leaf = fake.detach().requires_grad_()
        with autocast:
            features = term.features(leaf)
        torch.autograd.grad(features, leaf, [torch.ones_like(f) for f in features])

    # T4: the distance alone on recorded features
    with torch.no_grad(), autocast:
        fake_features, real_features = term.features(fake), term.features(real)
    distance = term.distance if hasattr(term, "distance") else None

    def distance_both():
        leaves = [f.detach().requires_grad_() for f in fake_features]
        value = distance(leaves, real_features)
        torch.autograd.grad(value, leaves)

    # T5: the whole term
    def whole():
        with autocast:
            loss.value_and_seed(fake, real)

    return {
        "term": key,
        **arm,
        "target_forward_ms": best(target_forward),
        "input_both_ms": best(input_both),
        "distance_ms": best(distance_both) if distance else None,
        "term_ms": best(whole),
    }


def main() -> None:
    """Print and append one row per term and arm."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    with OUT.open("a", encoding="utf-8") as file:
        for key in KEYS:
            for arm in ARMS:
                row = row_of(key, arm)
                file.write(json.dumps(row) + "\n")
                print(
                    {
                        k: round(v, 2) if isinstance(v, float) else v
                        for k, v in row.items()
                    },
                    flush=True,
                )


if __name__ == "__main__":
    main()
