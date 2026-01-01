"""
Test script for comparing FFT losses vs pixel-based losses (L1/L2).

This script:
1. Creates synthetic test data (2D images and 3D video)
2. Compares gradient norms and loss values across FFT losses and pixel losses
3. Tests different FFT normalization methods (none, l2, log, log1p)
4. Saves plots to results/fft_comparison/*.png

Usage:
    python examples/test_fft_vs_pixel_loss.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensecraft.loss.general import (
    PatchFFTLoss,
    FFTLoss,
    NormType,
    MSELoss,
    L1Loss,
    CharbonnierLoss,
)
from sensecraft.loss.video import PatchFFT3DLoss, TemporalFFTLoss


# Output directory for plots
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "fft_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simple_table(data: list[dict], headers: list[str] = None) -> str:
    """Simple table formatter without external dependencies."""
    if not data:
        return ""
    if headers is None:
        headers = list(data[0].keys())

    widths = {h: len(str(h)) for h in headers}
    for row in data:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    header_line = " | ".join(str(h).ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)

    rows = []
    for row in data:
        row_line = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        rows.append(row_line)

    return "\n".join([header_line, separator] + rows)


def compute_grad_norm(
    loss_fn: nn.Module, input: torch.Tensor, target: torch.Tensor
) -> dict:
    """Compute loss value and gradient norm for a loss function."""
    input = input.clone().requires_grad_(True)

    loss = loss_fn(input, target)
    loss.backward()

    grad = input.grad
    grad_norm = grad.norm().item()
    grad_max = grad.abs().max().item()
    grad_min = grad.abs().min().item()

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm,
        "grad_max": grad_max,
        "grad_min": grad_min,
    }


def test_2d_fft_vs_pixel():
    """Compare 2D FFT losses vs pixel losses."""
    print("=" * 80)
    print("2D FFT Loss vs Pixel Loss Comparison")
    print("=" * 80)

    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    input = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    # Pixel losses
    pixel_losses = [
        ("MSE (L2)", MSELoss()),
        ("L1", L1Loss()),
        ("Charbonnier", CharbonnierLoss()),
    ]

    # FFT losses with different norm types
    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]

    print("\n### Pixel Losses ###\n")
    results = []
    for name, loss_fn in pixel_losses:
        stats = compute_grad_norm(loss_fn, input.clone(), target)
        results.append(
            {
                "loss_name": name,
                "loss": f"{stats['loss']:.4e}",
                "grad_norm": f"{stats['grad_norm']:.4e}",
                "grad_max": f"{stats['grad_max']:.4e}",
            }
        )
    print(simple_table(results))

    print("\n### PatchFFTLoss (patch_size=8, loss_type=mse) ###\n")
    results = []
    for norm_type in norm_types:
        loss_fn = PatchFFTLoss(patch_size=8, loss_type="mse", norm_type=norm_type)
        stats = compute_grad_norm(loss_fn, input.clone(), target)
        results.append(
            {
                "norm_type": norm_type.value,
                "loss": f"{stats['loss']:.4e}",
                "grad_norm": f"{stats['grad_norm']:.4e}",
                "grad_max": f"{stats['grad_max']:.4e}",
            }
        )
    print(simple_table(results))

    print("\n### FFTLoss (global, loss_type=mse) ###\n")
    results = []
    for norm_type in norm_types:
        loss_fn = FFTLoss(loss_type="mse", norm_type=norm_type)
        stats = compute_grad_norm(loss_fn, input.clone(), target)
        results.append(
            {
                "norm_type": norm_type.value,
                "loss": f"{stats['loss']:.4e}",
                "grad_norm": f"{stats['grad_norm']:.4e}",
                "grad_max": f"{stats['grad_max']:.4e}",
            }
        )
    print(simple_table(results))


def test_3d_fft_vs_pixel():
    """Compare 3D FFT losses vs pixel losses."""
    print("\n" + "=" * 80)
    print("3D FFT Loss vs Pixel Loss Comparison")
    print("=" * 80)

    torch.manual_seed(42)
    B, T, C, H, W = 1, 9, 3, 64, 64
    input = torch.randn(B, T, C, H, W)
    target = torch.randn(B, T, C, H, W)

    # Flatten for pixel losses
    input_flat = input.view(-1, C, H, W)
    target_flat = target.view(-1, C, H, W)

    # Pixel losses (applied frame-wise)
    pixel_losses = [
        ("MSE (L2)", MSELoss()),
        ("L1", L1Loss()),
        ("Charbonnier", CharbonnierLoss()),
    ]

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]

    print("\n### Pixel Losses (applied to 3D data as B*T frames) ###\n")
    results = []
    for name, loss_fn in pixel_losses:
        stats = compute_grad_norm(loss_fn, input_flat.clone(), target_flat)
        results.append(
            {
                "loss_name": name,
                "loss": f"{stats['loss']:.4e}",
                "grad_norm": f"{stats['grad_norm']:.4e}",
                "grad_max": f"{stats['grad_max']:.4e}",
            }
        )
    print(simple_table(results))

    print("\n### PatchFFT3DLoss (patch_size=(8,16,16), skip_keyframe=True) ###\n")
    results = []
    for norm_type in norm_types:
        loss_fn = PatchFFT3DLoss(
            patch_size=(8, 16, 16),
            loss_type="mse",
            norm_type=norm_type,
            skip_keyframe=True,
        )
        stats = compute_grad_norm(loss_fn, input.clone(), target)
        results.append(
            {
                "norm_type": norm_type.value,
                "loss": f"{stats['loss']:.4e}",
                "grad_norm": f"{stats['grad_norm']:.4e}",
                "grad_max": f"{stats['grad_max']:.4e}",
            }
        )
    print(simple_table(results))

    print("\n### TemporalFFTLoss (amplitude_only=True) ###\n")
    results = []
    for norm_type in norm_types:
        loss_fn = TemporalFFTLoss(
            loss_type="l1", norm_type=norm_type, amplitude_only=True
        )
        stats = compute_grad_norm(loss_fn, input.clone(), target)
        results.append(
            {
                "norm_type": norm_type.value,
                "loss": f"{stats['loss']:.4e}",
                "grad_norm": f"{stats['grad_norm']:.4e}",
                "grad_max": f"{stats['grad_max']:.4e}",
            }
        )
    print(simple_table(results))


def plot_2d_comparison():
    """Plot comparison of 2D FFT losses vs pixel losses."""
    print("\nGenerating 2D comparison plots...")

    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    input = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    # All losses to compare
    loss_configs = [
        # Pixel losses
        ("MSE (L2)", lambda: MSELoss(), "pixel"),
        ("L1", lambda: L1Loss(), "pixel"),
        ("Charbonnier", lambda: CharbonnierLoss(), "pixel"),
        # FFT losses with different norms
        (
            "PatchFFT\n(none)",
            lambda: PatchFFTLoss(patch_size=8, norm_type="none"),
            "fft",
        ),
        ("PatchFFT\n(l2)", lambda: PatchFFTLoss(patch_size=8, norm_type="l2"), "fft"),
        ("PatchFFT\n(log)", lambda: PatchFFTLoss(patch_size=8, norm_type="log"), "fft"),
        (
            "PatchFFT\n(log1p)",
            lambda: PatchFFTLoss(patch_size=8, norm_type="log1p"),
            "fft",
        ),
        ("FFT\n(none)", lambda: FFTLoss(norm_type="none"), "fft"),
        ("FFT\n(log1p)", lambda: FFTLoss(norm_type="log1p"), "fft"),
    ]

    names = []
    losses = []
    grad_norms = []
    categories = []

    for name, factory, cat in loss_configs:
        loss_fn = factory()
        stats = compute_grad_norm(loss_fn, input.clone(), target)
        names.append(name)
        losses.append(stats["loss"])
        grad_norms.append(stats["grad_norm"])
        categories.append(cat)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("2D Losses: FFT vs Pixel-Based (L1/L2)", fontsize=14)

    x = np.arange(len(names))
    colors = ["steelblue" if c == "pixel" else "coral" for c in categories]

    # Loss values
    ax1 = axes[0]
    bars1 = ax1.bar(x, losses, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Loss Value (log scale)")
    ax1.set_title("Loss Values")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="Pixel Loss"),
        Patch(facecolor="coral", edgecolor="black", label="FFT Loss"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Gradient norms
    ax2 = axes[1]
    bars2 = ax2.bar(x, grad_norms, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Gradient Norm (log scale)")
    ax2.set_title("Gradient Norms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2d_fft_vs_pixel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '2d_fft_vs_pixel.png'}")


def plot_3d_comparison():
    """Plot comparison of 3D FFT losses vs pixel losses."""
    print("\nGenerating 3D comparison plots...")

    torch.manual_seed(42)
    B, T, C, H, W = 1, 9, 3, 64, 64
    input_3d = torch.randn(B, T, C, H, W)
    target_3d = torch.randn(B, T, C, H, W)
    input_flat = input_3d.view(-1, C, H, W)
    target_flat = target_3d.view(-1, C, H, W)

    # All losses to compare
    loss_configs = [
        # Pixel losses (applied to flattened data)
        ("MSE (L2)", lambda: MSELoss(), input_flat, target_flat, "pixel"),
        ("L1", lambda: L1Loss(), input_flat, target_flat, "pixel"),
        ("Charbonnier", lambda: CharbonnierLoss(), input_flat, target_flat, "pixel"),
        # 3D FFT losses
        (
            "PatchFFT3D\n(none)",
            lambda: PatchFFT3DLoss(patch_size=(8, 16, 16), norm_type="none"),
            input_3d,
            target_3d,
            "fft",
        ),
        (
            "PatchFFT3D\n(l2)",
            lambda: PatchFFT3DLoss(patch_size=(8, 16, 16), norm_type="l2"),
            input_3d,
            target_3d,
            "fft",
        ),
        (
            "PatchFFT3D\n(log1p)",
            lambda: PatchFFT3DLoss(patch_size=(8, 16, 16), norm_type="log1p"),
            input_3d,
            target_3d,
            "fft",
        ),
        (
            "TemporalFFT\n(none)",
            lambda: TemporalFFTLoss(norm_type="none"),
            input_3d,
            target_3d,
            "fft",
        ),
        (
            "TemporalFFT\n(log1p)",
            lambda: TemporalFFTLoss(norm_type="log1p"),
            input_3d,
            target_3d,
            "fft",
        ),
    ]

    names = []
    losses = []
    grad_norms = []
    categories = []

    for name, factory, inp, tgt, cat in loss_configs:
        loss_fn = factory()
        stats = compute_grad_norm(loss_fn, inp.clone(), tgt)
        names.append(name)
        losses.append(stats["loss"])
        grad_norms.append(stats["grad_norm"])
        categories.append(cat)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("3D Losses: FFT vs Pixel-Based (L1/L2)", fontsize=14)

    x = np.arange(len(names))
    colors = ["steelblue" if c == "pixel" else "coral" for c in categories]

    # Loss values
    ax1 = axes[0]
    ax1.bar(x, losses, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Loss Value (log scale)")
    ax1.set_title("Loss Values")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="Pixel Loss"),
        Patch(facecolor="coral", edgecolor="black", label="FFT Loss"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Gradient norms
    ax2 = axes[1]
    ax2.bar(x, grad_norms, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Gradient Norm (log scale)")
    ax2.set_title("Gradient Norms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3d_fft_vs_pixel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '3d_fft_vs_pixel.png'}")


def plot_gradient_scaling():
    """Plot how gradients scale with input magnitude for FFT vs pixel losses."""
    print("\nGenerating gradient scaling plots...")

    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64

    scales = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    loss_configs = [
        ("MSE", lambda: MSELoss(), "steelblue"),
        ("L1", lambda: L1Loss(), "navy"),
        (
            "PatchFFT (none)",
            lambda: PatchFFTLoss(patch_size=8, norm_type="none"),
            "coral",
        ),
        ("PatchFFT (l2)", lambda: PatchFFTLoss(patch_size=8, norm_type="l2"), "red"),
        (
            "PatchFFT (log1p)",
            lambda: PatchFFTLoss(patch_size=8, norm_type="log1p"),
            "orange",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gradient Scaling: FFT vs Pixel Losses", fontsize=14)

    ax1, ax2 = axes

    for name, factory, color in loss_configs:
        grad_norms = []
        loss_values = []
        for scale in scales:
            inp = torch.randn(B, C, H, W) * scale
            tgt = torch.randn(B, C, H, W) * scale
            loss_fn = factory()
            stats = compute_grad_norm(loss_fn, inp, tgt)
            grad_norms.append(stats["grad_norm"])
            loss_values.append(stats["loss"])

        ax1.plot(
            scales, grad_norms, "o-", label=name, color=color, linewidth=2, markersize=6
        )
        ax2.plot(
            scales,
            loss_values,
            "o-",
            label=name,
            color=color,
            linewidth=2,
            markersize=6,
        )

    ax1.set_xlabel("Input Scale")
    ax1.set_ylabel("Gradient Norm")
    ax1.set_title("Gradient Norm vs Input Scale")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Input Scale")
    ax2.set_ylabel("Loss Value")
    ax2.set_title("Loss Value vs Input Scale")
    ax2.legend()
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradient_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'gradient_scaling.png'}")


def plot_heatmap_comparison():
    """Create a comprehensive heatmap comparing all losses."""
    print("\nGenerating heatmap comparison...")

    torch.manual_seed(42)

    # 2D data
    B, C, H, W = 2, 3, 64, 64
    input_2d = torch.randn(B, C, H, W)
    target_2d = torch.randn(B, C, H, W)

    # 3D data
    B, T, C, H, W = 1, 9, 3, 64, 64
    input_3d = torch.randn(B, T, C, H, W)
    target_3d = torch.randn(B, T, C, H, W)
    input_flat = input_3d.view(-1, C, H, W)
    target_flat = target_3d.view(-1, C, H, W)

    loss_configs = [
        # Pixel losses (2D)
        ("MSE (2D)", input_2d, target_2d, lambda: MSELoss()),
        ("L1 (2D)", input_2d, target_2d, lambda: L1Loss()),
        ("Charbonnier (2D)", input_2d, target_2d, lambda: CharbonnierLoss()),
        # FFT losses (2D)
        (
            "PatchFFT (none)",
            input_2d,
            target_2d,
            lambda: PatchFFTLoss(patch_size=8, norm_type="none"),
        ),
        (
            "PatchFFT (l2)",
            input_2d,
            target_2d,
            lambda: PatchFFTLoss(patch_size=8, norm_type="l2"),
        ),
        (
            "PatchFFT (log1p)",
            input_2d,
            target_2d,
            lambda: PatchFFTLoss(patch_size=8, norm_type="log1p"),
        ),
        ("FFT (none)", input_2d, target_2d, lambda: FFTLoss(norm_type="none")),
        ("FFT (log1p)", input_2d, target_2d, lambda: FFTLoss(norm_type="log1p")),
        # Pixel losses (3D as flat)
        ("MSE (3D flat)", input_flat, target_flat, lambda: MSELoss()),
        ("L1 (3D flat)", input_flat, target_flat, lambda: L1Loss()),
        # FFT losses (3D)
        (
            "PatchFFT3D (none)",
            input_3d,
            target_3d,
            lambda: PatchFFT3DLoss(patch_size=(8, 16, 16), norm_type="none"),
        ),
        (
            "PatchFFT3D (log1p)",
            input_3d,
            target_3d,
            lambda: PatchFFT3DLoss(patch_size=(8, 16, 16), norm_type="log1p"),
        ),
        (
            "TemporalFFT (none)",
            input_3d,
            target_3d,
            lambda: TemporalFFTLoss(norm_type="none"),
        ),
        (
            "TemporalFFT (log1p)",
            input_3d,
            target_3d,
            lambda: TemporalFFTLoss(norm_type="log1p"),
        ),
    ]

    loss_names = []
    loss_values = []
    grad_norms = []

    for name, inp, tgt, factory in loss_configs:
        loss_fn = factory()
        stats = compute_grad_norm(loss_fn, inp.clone(), tgt)
        loss_names.append(name)
        loss_values.append(stats["loss"])
        grad_norms.append(stats["grad_norm"])

    # Create figure with 2 bar plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Comprehensive Loss Comparison: FFT vs Pixel-Based", fontsize=14)

    x = np.arange(len(loss_names))

    # Categorize losses
    colors = []
    for name in loss_names:
        if "FFT" in name or "fft" in name.lower():
            colors.append("coral")
        else:
            colors.append("steelblue")

    # Loss values
    ax1 = axes[0]
    ax1.bar(x, loss_values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Loss Value (log scale)")
    ax1.set_title("Loss Values Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(loss_names, rotation=45, ha="right", fontsize=9)
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="Pixel Loss"),
        Patch(facecolor="coral", edgecolor="black", label="FFT Loss"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Gradient norms
    ax2 = axes[1]
    ax2.bar(x, grad_norms, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Gradient Norm (log scale)")
    ax2.set_title("Gradient Norms Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(loss_names, rotation=45, ha="right", fontsize=9)
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "comprehensive_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'comprehensive_comparison.png'}")


def plot_norm_type_comparison():
    """Plot comparison of different FFT normalization types."""
    print("\nGenerating norm type comparison...")

    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    input = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    norm_types = ["none", "l2", "log", "log1p"]
    loss_types = ["mse", "l1"]
    colors = {"mse": "steelblue", "l1": "coral"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("FFT Normalization Types: Impact on Loss and Gradients", fontsize=14)

    # PatchFFTLoss - Loss values
    ax1 = axes[0, 0]
    x = np.arange(len(norm_types))
    width = 0.35
    for i, lt in enumerate(loss_types):
        values = []
        for nt in norm_types:
            loss_fn = PatchFFTLoss(patch_size=8, loss_type=lt, norm_type=nt)
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            values.append(stats["loss"])
        ax1.bar(
            x + i * width - width / 2, values, width, label=lt.upper(), color=colors[lt]
        )
    ax1.set_ylabel("Loss Value (log scale)")
    ax1.set_title("PatchFFTLoss - Loss Values")
    ax1.set_xticks(x)
    ax1.set_xticklabels(norm_types)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # PatchFFTLoss - Gradient norms
    ax2 = axes[0, 1]
    for i, lt in enumerate(loss_types):
        values = []
        for nt in norm_types:
            loss_fn = PatchFFTLoss(patch_size=8, loss_type=lt, norm_type=nt)
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            values.append(stats["grad_norm"])
        ax2.bar(
            x + i * width - width / 2, values, width, label=lt.upper(), color=colors[lt]
        )
    ax2.set_ylabel("Gradient Norm (log scale)")
    ax2.set_title("PatchFFTLoss - Gradient Norms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(norm_types)
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # FFTLoss - Loss values
    ax3 = axes[1, 0]
    for i, lt in enumerate(loss_types):
        values = []
        for nt in norm_types:
            loss_fn = FFTLoss(loss_type=lt, norm_type=nt)
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            values.append(stats["loss"])
        ax3.bar(
            x + i * width - width / 2, values, width, label=lt.upper(), color=colors[lt]
        )
    ax3.set_ylabel("Loss Value (log scale)")
    ax3.set_title("FFTLoss (Global) - Loss Values")
    ax3.set_xticks(x)
    ax3.set_xticklabels(norm_types)
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # FFTLoss - Gradient norms
    ax4 = axes[1, 1]
    for i, lt in enumerate(loss_types):
        values = []
        for nt in norm_types:
            loss_fn = FFTLoss(loss_type=lt, norm_type=nt)
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            values.append(stats["grad_norm"])
        ax4.bar(
            x + i * width - width / 2, values, width, label=lt.upper(), color=colors[lt]
        )
    ax4.set_ylabel("Gradient Norm (log scale)")
    ax4.set_title("FFTLoss (Global) - Gradient Norms")
    ax4.set_xticks(x)
    ax4.set_xticklabels(norm_types)
    ax4.set_yscale("log")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "norm_type_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'norm_type_comparison.png'}")


if __name__ == "__main__":
    print("=" * 60)
    print("FFT vs Pixel Loss Comparison Test")
    print("=" * 60)

    # Run text-based tests
    test_2d_fft_vs_pixel()
    test_3d_fft_vs_pixel()

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_2d_comparison()
    plot_3d_comparison()
    plot_gradient_scaling()
    plot_heatmap_comparison()
    plot_norm_type_comparison()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
