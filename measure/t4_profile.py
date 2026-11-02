"""T4: the device kernels of one compiled term, by time, grouped into what they do.

Profiles `Loss.value_and_seed` of each term compiled per term, after warm-up, and prints the
kernels that hold the time with their share, plus totals per group: conv, matmul, attention,
elementwise and reductions, layout copies. Run on an idle GPU. Real compute.

    python measure/t4_profile.py
"""

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import torch
from terms import TERMS
from torch.profiler import ProfilerActivity, profile

from sensecraft.compose import Loss, Term

DEVICE: str = "cuda:0"
KEYS: list[str] = ["lpips_vgg", "dino_base_mse"]
BATCH: int = 32
SIZE: int = 256
DTYPE: str = "bfloat16"
STEPS: int = 5
TOP: int = 14
SHOW_GROUPS: list[str] = []  # groups whose kernels are listed in full
GROUPS = {
    "conv": (
        "conv",
        "cudnn",
        "implicit_gemm",
        "xmma",
        "cutlass__5x",
        "wgrad",
        "dgrad",
        "fprop",
    ),
    "matmul": ("nvjet", "gemm", "cublas"),
    "attention": ("flash", "fmha", "attention", "sdpa"),
    "layout": ("nchwToNhwc", "nhwcToNchw", "copy", "transpose"),
    "pool": ("pool", "max_pool"),
}


def group_of(name: str) -> str:
    """The group a kernel name falls in; Triton and the rest count as elementwise."""
    lowered = name.lower()
    for group, needles in GROUPS.items():
        if any(needle.lower() in lowered for needle in needles):
            return group
    return "elementwise"


def main() -> None:
    """Print the top kernels and group totals per term, in ms per call."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    dtype = getattr(torch, DTYPE)
    for key in KEYS:
        kind, kwargs = TERMS[key]
        name = "lpips" if kind == "lpips" else "dino_vit"
        loss = Loss([Term(name, **kwargs)]).prepare(
            DEVICE, dtype, torch.channels_last, "term"
        )
        real = (torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1).to(
            memory_format=torch.channels_last
        )
        fake = (real + 0.1 * torch.randn_like(real)).clamp(-1, 1)

        def call():
            with torch.autocast("cuda", dtype=dtype):
                loss.value_and_seed(fake, real)

        for _ in range(5):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            for _ in range(STEPS):
                call()
            torch.cuda.synchronize()

        kernels = defaultdict(float)
        for event in trace.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernels[event.name] += event.device_time / 1e3 / STEPS
        total = sum(kernels.values())
        groups = defaultdict(float)
        for kernel, ms in kernels.items():
            groups[group_of(kernel)] += ms
        print(f"{key}: {total:.2f} ms device time per call")
        for group, ms in sorted(groups.items(), key=lambda item: -item[1]):
            print(f"  {group:12s} {ms:7.2f} ms {ms / total:6.1%}")
        for kernel, ms in sorted(kernels.items(), key=lambda item: -item[1])[:TOP]:
            print(f"    {ms:6.2f} ms  {kernel[:90]}")
        for group in SHOW_GROUPS:
            print(f"  {group} kernels:")
            members = [(k, ms) for k, ms in kernels.items() if group_of(k) == group]
            for kernel, ms in sorted(members, key=lambda item: -item[1])[:8]:
                print(f"    {ms:6.2f} ms  {kernel[:100]}")


if __name__ == "__main__":
    main()
