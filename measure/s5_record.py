"""S5 record: the installed sensecraft 0.3.13 on fixed inputs, as s5_fixed's reference.

For every fixed (non-network) loss: one seeded fp64 input pair in the loss's range,
and the old loss's value and input gradient at fp64, fp32 and (unless fp32-only) bf16.
Writes REF_PATH. Run with plain `python measure/s5_record.py`.
"""

import sys
from pathlib import Path

import torch

SEED = 0
REF_PATH = Path(__file__).parent / "refs" / "s5_fixed.pt"

IMAGE_SHAPE = (4, 3, 64, 64)
LARGE_IMAGE_SHAPE = (4, 3, 192, 192)  # ms_ssim needs a side above 160
VIDEO_SHAPE = (2, 6, 3, 64, 64)
LONG_VIDEO_SHAPE = (2, 9, 3, 64, 64)  # patch_fft_3d: frames - 1 a multiple of 8
NOISE = 0.1  # std of the input's offset from the target

# (case, registered name, old init kwargs, old forward kwargs, shape, value range)
# Default kwargs everywhere; the three "+" cases run code paths rewritten in the port.
CASES = [
    *[
        (name, name, {}, {}, IMAGE_SHAPE, None)
        for name in [
            "mse",
            "l1",
            "huber",
            "smooth_l1",
            "rmse",
            "mape",
            "smape",
            "log_cosh",
            "quantile",
            "cauchy",
            "geman_mcclure",
            "welsch",
            "tukey",
            "wing",
            "charbonnier",
            "gaussian_noise",
            "fft",
            "patch_fft",
            "sobel",
            "laplacian",
            "canny",
            "gradient",
            "high_freq",
            "multi_scale_gradient",
            "structure_tensor",
            "ssim",
        ]
    ],
    ("ms_ssim", "ms_ssim", {}, {}, LARGE_IMAGE_SHAPE, None),
    *[
        (name, name, {}, {}, VIDEO_SHAPE, None)
        for name in [
            "ssim3d",
            "stssim",
            "tssim",
            "fdb",
            "temporal_accel",
            "temporal_fft",
            "temporal_gradient",
        ]
    ],
    ("patch_fft_3d", "patch_fft_3d", {}, {}, LONG_VIDEO_SHAPE, None),
    (
        "gaussian_noise+noise",
        "gaussian_noise",
        {"sigma_range": (0.05, 0.2)},
        {"add_noise_to_target": True},
        IMAGE_SHAPE,
        None,
    ),
    ("ssim3d+symmetric", "ssim3d", {}, {}, VIDEO_SHAPE, "symmetric"),
    ("stssim+symmetric", "stssim", {}, {}, VIDEO_SHAPE, "symmetric"),
]


def import_old():
    """The installed sensecraft's loss registry; refuses a checkout on sys.path."""
    assert not any(path.startswith("/xg7/SenseCraft/src") for path in sys.path)
    from sensecraft.loss.config import LOSS_REGISTRY

    import sensecraft

    assert "site-packages" in sensecraft.__file__, sensecraft.__file__
    return LOSS_REGISTRY


def input_pair(
    shape: tuple[int, ...], value_range: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded fp64 (input, target): target uniform in range, input = target + noise."""
    generator = torch.Generator().manual_seed(seed)
    target = torch.rand(shape, generator=generator, dtype=torch.float64)
    offset = NOISE * torch.randn(shape, generator=generator, dtype=torch.float64)

    if value_range == "unit":
        return (target + offset).clamp(0, 1), target
    target = target * 2 - 1
    if value_range == "symmetric":
        return (target + offset).clamp(-1, 1), target
    return target + offset, target


def run(
    loss, input, target, dtype, forward_kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Value and input gradient of one forward and backward at `dtype`."""
    input = input.detach().to(dtype).clone().requires_grad_(True)
    torch.manual_seed(SEED)
    value = loss(input, target.to(dtype), **forward_kwargs)
    value.backward()
    return value.detach(), input.grad


def main():
    registry = import_old()

    record = {}
    for index, (case, name, kwargs, forward_kwargs, shape, value_range) in enumerate(
        CASES
    ):
        info = registry[name]
        value_range = value_range or info.required_range.value
        input, target = input_pair(shape, value_range, SEED + index)
        loss = info.loss_class(**kwargs)

        dtypes = [torch.float64, torch.float32]
        if not info.requires_fp32:
            dtypes.append(torch.bfloat16)
        runs = {
            str(dtype).removeprefix("torch."): run(
                loss, input, target, dtype, forward_kwargs
            )
            for dtype in dtypes
        }

        record[case] = {
            "name": name,
            "kwargs": kwargs,
            "input": input,
            "target": target,
            "runs": runs,
            "flags": {
                "range": info.required_range.value,
                "is_2d_only": info.is_2d_only,
                "is_3d_only": info.is_3d_only,
                "requires_fp32": info.requires_fp32,
            },
        }
        print(f"{case:24s} {runs['float64'][0].item():.10g}")

    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, REF_PATH)
    print(f"wrote {len(record)} cases to {REF_PATH}")


if __name__ == "__main__":
    main()
