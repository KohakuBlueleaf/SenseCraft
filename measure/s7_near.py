"""Stage S7: the gradient at the prediction when the prediction is close to the target.

A perceptual term's gradient at `fake` depends on the difference between the features of `fake` and
`real`. When the two images differ by less than the spacing of the compute dtype, that difference
is lost and the gradient keeps little of its direction. Each perceptual term is run on `real` and
`real + noise` for shrinking noise, in fp16, bf16, fp32 with TF32 and strict fp32, and its gradient
at `fake` is compared with an fp64 run by cosine. Rows are written to `refs/s7_near_<LABEL>.json`,
and every recorded label is drawn into `figures/s7_near.svg`.

    python measure/s7_near.py
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import torch
from common import REFS, images
from torch.nn.functional import cosine_similarity

from sensecraft import Loss, Term

DEVICE: str = "cuda:0"
LABEL: str = "after"  # the name this run is recorded under, e.g. "before" a change
FOLDER: str | None = (
    None  # real images; None uses the seeded smooth noise of `common.images`
)
BATCH: int = 8
SIZE: int = 256
NOISE: tuple = (3e-2, 1e-2, 1e-3, 1e-4, 1e-5)
FP16_SCALE: float = 2.0**15
# DINOv3 terms need their gated checkpoints in the Hugging Face cache; drop them otherwise
TERMS: dict[str, tuple[str, dict]] = {
    "lpips_vgg": ("lpips", {"net": "vgg"}),
    "lpips_alex": ("lpips", {"net": "alex"}),
    "lpips_squeeze": ("lpips", {"net": "squeeze"}),
    "dino_vit_base": ("dino_vit", {"model": "base", "use_gram": False}),
    "convnext_small": ("convnext", {"model": "small"}),
    "dino_convnext_tiny": ("dino_convnext", {"model": "tiny"}),
}
MODES: tuple = ("fp16", "bf16", "tf32", "fp32")
FIGURE: Path = HERE / "figures" / "s7_near.svg"


def precision(conv: str, matmul: str) -> tuple[str, str]:
    """Set the fp32 precision of cuDNN convs and of matmuls; returns the previous pair."""
    old = (
        torch.backends.cudnn.conv.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
    )
    torch.backends.cudnn.conv.fp32_precision = conv
    torch.backends.cuda.matmul.fp32_precision = matmul
    return old


def seed(name: str, kwargs: dict, mode: str, fake, real) -> torch.Tensor:
    """The gradient of the term at `fake`, in `mode`."""
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mode, torch.float32)
    loss = Loss([Term(name, **kwargs)]).prepare(DEVICE, dtype, torch.channels_last)
    fake, real = fake.float(), real.float()
    if mode in ("fp16", "bf16"):
        scale = FP16_SCALE if mode == "fp16" else 1.0
        with torch.autocast("cuda", dtype=dtype):
            return loss.value_and_seed(fake, real, scale).seed / scale
    strict = "ieee" if mode == "fp32" else "tf32"
    old = precision(strict, strict)
    try:
        return loss.value_and_seed(fake, real).seed
    finally:
        precision(*old)


def reference(name: str, kwargs: dict, fake, real) -> torch.Tensor:
    """The gradient in fp64."""
    loss = Loss([Term(name, **kwargs)]).to(DEVICE).double()
    return loss.value_and_seed(fake.double(), real.double()).seed


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def record() -> dict:
    """Cosine to fp64 per term, mode and noise."""
    torch.manual_seed(0)
    real = images(BATCH, SIZE, FOLDER).to(DEVICE)
    rows: dict = {}
    for key, (name, kwargs) in TERMS.items():
        for noise in NOISE:
            fake = real + noise * torch.randn(
                real.shape, dtype=real.dtype, device=DEVICE
            )
            truth = reference(name, kwargs, fake, real)
            for mode in MODES:
                value = cos(seed(name, kwargs, mode, fake, real), truth)
                rows.setdefault(key, {}).setdefault(mode, {})[f"{noise:g}"] = value
            print(
                key,
                f"noise {noise:g}",
                "  ".join(f"{m} {rows[key][m][f'{noise:g}']:.5f}" for m in MODES),
                flush=True,
            )
    return rows


def draw(labels: dict[str, dict]) -> str:
    """An SVG grid: one panel per term, cosine to fp64 against noise, one line per mode and label."""
    terms = sorted({key for rows in labels.values() for key in rows})
    columns, panel_w, panel_h, pad = 3, 300, 220, 46
    rows_n = math.ceil(len(terms) / columns)
    width, height = columns * panel_w, rows_n * panel_h + 40
    colors = {
        "fp16": "#d62728",
        "bf16": "#9467bd",
        "tf32": "#ff7f0e",
        "fp32": "#2ca02c",
    }
    dashes = {
        label: "" if i == len(labels) - 1 else "5,4" for i, label in enumerate(labels)
    }
    noises = [f"{n:g}" for n in NOISE]
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'font-family="sans-serif" font-size="11">'
        '<rect width="100%" height="100%" fill="white"/>'
    )
    out = [header]
    for index, term in enumerate(terms):
        left = (index % columns) * panel_w + pad
        top = (index // columns) * panel_h + 24
        plot_w, plot_h = panel_w - pad - 12, panel_h - 70

        def x_of(i):
            return left + plot_w * i / (len(noises) - 1)

        def y_of(value):
            return top + plot_h * (1 - max(value, 0.0))

        out.append(f'<text x="{left}" y="{top - 8}" font-weight="bold">{term}</text>')
        out.append(
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
            f'fill="none" stroke="#999"/>'
        )
        for tick in (0.0, 0.5, 1.0):
            out.append(
                f'<text x="{left - 6}" y="{y_of(tick) + 4}" text-anchor="end">{tick:g}</text>'
            )
        for i, noise in enumerate(noises):
            out.append(
                f'<text x="{x_of(i)}" y="{top + plot_h + 14}" text-anchor="middle">{noise}</text>'
            )
        for label, rows in labels.items():
            for mode, color in colors.items():
                values = rows.get(term, {}).get(mode)
                if not values:
                    continue
                points = " ".join(
                    f"{x_of(i):.1f},{y_of(values[n]):.1f}" for i, n in enumerate(noises)
                )
                out.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    f'stroke-width="1.8" stroke-dasharray="{dashes[label]}"/>'
                )
        out.append(
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 28}" text-anchor="middle">'
            "noise added to the target</text>"
        )

    # Legend: colors are modes, dashed lines the earlier labels
    legend = ", ".join(f'<tspan fill="{c}">{m}</tspan>' for m, c in colors.items())
    styles = "; ".join(
        f"{label}: {'solid' if not dash else 'dashed'}"
        for label, dash in dashes.items()
    )
    out.append(
        f'<text x="10" y="{height - 10}">cosine to fp64 of the gradient at the prediction. '
        f"{legend}. {styles}</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    rows = record()
    REFS.mkdir(exist_ok=True)
    (REFS / f"s7_near_{LABEL}.json").write_text(json.dumps(rows, indent=1))
    labels = {}
    for path in sorted(REFS.glob("s7_near_*.json"), key=lambda p: p.stat().st_mtime):
        labels[path.stem.removeprefix("s7_near_")] = json.loads(path.read_text())
    FIGURE.parent.mkdir(exist_ok=True)
    FIGURE.write_text(draw(labels))


if __name__ == "__main__":
    main()
