"""T1: what this card does at best, the floor every later speed stage is a ratio to.

Times a bare conv in both layouts, a bf16 and fp16 matmul, a read and a copy far past the cache,
and SDPA at ViT shapes. Writes one JSON per card. Run on an idle GPU. Real compute.

    python measure/t1_floor.py
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F
from common import REFS

DEVICE: str = "cuda:0"
ROUNDS: int = 20


def best(run) -> float:
    """Best seconds of `run` after two warm-ups."""
    seconds = float("inf")
    for index in range(ROUNDS):
        torch.cuda.synchronize()
        began = time.perf_counter()
        run()
        torch.cuda.synchronize()
        if index >= 2:
            seconds = min(seconds, time.perf_counter() - began)
    return seconds


@torch.no_grad()
def floors() -> dict[str, float]:
    """Rates in T MAC/s and GB/s."""
    out = {}
    for dtype in (torch.bfloat16, torch.float16):
        name = str(dtype).removeprefix("torch.")
        a = torch.randn(8192, 8192, device=DEVICE, dtype=dtype)
        out[f"matmul_{name}_tmac"] = 8192**3 / best(lambda: a @ a) / 1e12
        for layout in (torch.contiguous_format, torch.channels_last):
            x = torch.randn(16, 256, 128, 128, device=DEVICE, dtype=dtype).to(
                memory_format=layout
            )
            w = torch.randn(256, 256, 3, 3, device=DEVICE, dtype=dtype).to(
                memory_format=layout
            )
            macs = 16 * 128 * 128 * 256 * 256 * 9
            label = "nhwc" if layout is torch.channels_last else "nchw"
            out[f"conv_{name}_{label}_tmac"] = (
                macs / best(lambda: F.conv2d(x, w, padding=1)) / 1e12
            )
        q = torch.randn(32, 12, 1024, 64, device=DEVICE, dtype=dtype)
        attention_macs = 2 * 32 * 12 * 1024 * 1024 * 64
        out[f"sdpa_{name}_tmac"] = (
            attention_macs
            / best(lambda: F.scaled_dot_product_attention(q, q, q))
            / 1e12
        )

    big = torch.ones(2**30, device=DEVICE)
    other = torch.empty_like(big)
    out["read_gbs"] = big.numel() * 4 / best(lambda: big.sum()) / 1e9
    out["copy_gbs"] = 2 * big.numel() * 4 / best(lambda: other.copy_(big)) / 1e9
    return out


def main() -> None:
    """Print the floors and write `refs/floor_<card>.json`."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    result = {"card": torch.cuda.get_device_name(), **floors()}
    card = result["card"].lower().replace(" ", "_")
    REFS.mkdir(parents=True, exist_ok=True)
    (REFS / f"floor_{card}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
