"""T2: every layer of a perceptual term alone, forward and input backward, as a ratio to T1.

Feeds each layer its real input from one traced forward, times it with the weights frozen, and
reports conv layers against the conv floor and the rest against the copy floor. Reads
`refs/floor_<card>.json`. Run on an idle GPU. Real compute.

    python measure/t2_layers.py
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
from terms import TERMS, ours
from torch import nn

from sensecraft.nets import FeatureNet

DEVICE: str = "cuda:0"
KEYS: list[str] = ["lpips_vgg", "dino_base_mse"]
BATCH: int = 32
SIZE: int = 256
DTYPE: str = "bfloat16"
CHANNELS_LAST: bool = True
ROUNDS: int = 10


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


def leaves_with_inputs(term, x: torch.Tensor) -> list[tuple[str, nn.Module, tuple]]:
    """Every leaf layer of the term's net with the input it saw in one forward."""
    seen = []
    names = {module: name for name, module in term.named_modules()}

    def record(module, inputs):
        seen.append((names[module], module, tuple(t.detach() for t in inputs)))

    leaves = [m for m in term.modules() if not list(m.children()) and m is not term]
    handles = [m.register_forward_pre_hook(record) for m in leaves]
    with torch.no_grad():
        term.features(x)
    for handle in handles:
        handle.remove()
    return seen


def work(module: nn.Module, inputs: tuple, output: torch.Tensor) -> tuple[float, float]:
    """MACs and bytes moved by one call."""
    moved = sum(t.numel() * t.element_size() for t in (*inputs, output))
    if isinstance(module, nn.Conv2d):
        per_output = (
            module.in_channels
            // module.groups
            * module.kernel_size[0]
            * module.kernel_size[1]
        )
        return output.numel() * per_output, moved
    if isinstance(module, nn.Linear):
        return output.numel() * module.in_features, moved
    return 0.0, moved


def main() -> None:
    """Print one row per layer and a summary per term."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    card = torch.cuda.get_device_name().lower().replace(" ", "_")
    floor = json.loads((REFS / f"floor_{card}.json").read_text())
    dtype = getattr(torch, DTYPE)
    layout = torch.channels_last if CHANNELS_LAST else torch.contiguous_format
    conv_floor = (
        floor[f"conv_{DTYPE}_{'nhwc' if CHANNELS_LAST else 'nchw'}_tmac"] * 1e12
    )
    matmul_floor = floor[f"matmul_{DTYPE}_tmac"] * 1e12
    copy_floor = floor["copy_gbs"] * 1e9

    for key in KEYS:
        term, _ = ours(*TERMS[key])
        term.to(DEVICE)
        for module in term.modules():
            if isinstance(module, FeatureNet):
                module.to(dtype=dtype, memory_format=layout)
        x = (
            (torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1)
            .to(dtype)
            .to(memory_format=layout)
        )

        totals = {"forward": 0.0, "backward": 0.0}
        print(
            f"{key}: layer, forward ms, backward ms, rate, ratio to floor", flush=True
        )
        for name, module, inputs in leaves_with_inputs(term, x):
            leaf = inputs[0].clone().requires_grad_()
            with torch.no_grad():
                output = module(leaf.detach().clone(), *inputs[1:])
            forward = best(lambda: module(leaf.detach().clone(), *inputs[1:]))
            cot = torch.randn_like(output)

            def both():
                out = module(leaf.clone(), *inputs[1:])
                torch.autograd.grad(out, leaf, cot)

            backward = max(best(both) - forward, 0.0)
            macs, moved = work(module, inputs, output)
            if macs:
                rate = macs / forward
                floor_rate = (
                    conv_floor if isinstance(module, nn.Conv2d) else matmul_floor
                )
                ratio, unit = rate / floor_rate, f"{rate / 1e12:6.1f}T"
            else:
                rate = moved / forward
                ratio, unit = rate / copy_floor, f"{rate / 1e9:6.0f}G"
            totals["forward"] += forward
            totals["backward"] += backward
            print(
                f"  {name:38s} {forward * 1e3:7.3f} {backward * 1e3:7.3f} {unit} {ratio:5.2f}",
                flush=True,
            )
        print(
            f"{key} total forward {totals['forward'] * 1e3:.2f} ms, backward {totals['backward'] * 1e3:.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
