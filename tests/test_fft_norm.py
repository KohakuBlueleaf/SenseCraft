"""
Test script for FFT loss normalization methods and gradient norm comparison.

This script tests different normalization methods (NONE, L2, LOG, LOG1P) for FFT losses
and compares their gradient norms to understand which methods produce more stable gradients.

Usage:
    python tests/test_fft_norm.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sensecraft.loss.general import PatchFFTLoss, FFTLoss, NormType
from sensecraft.loss.video import PatchFFT3DLoss, TemporalFFTLoss


# Output directory for plots
OUTPUT_DIR = Path(__file__).parent / "fft_norm_plots"
OUTPUT_DIR.mkdir(exist_ok=True)


def simple_table(data: list[dict], headers: list[str] = None) -> str:
    """Simple table formatter without external dependencies."""
    if not data:
        return ""
    if headers is None:
        headers = list(data[0].keys())

    # Calculate column widths
    widths = {h: len(str(h)) for h in headers}
    for row in data:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    # Format header
    header_line = " | ".join(str(h).ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)

    # Format rows
    rows = []
    for row in data:
        row_line = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        rows.append(row_line)

    return "\n".join([header_line, separator] + rows)


def compute_grad_norm(
    loss_fn: nn.Module, input: torch.Tensor, target: torch.Tensor
) -> dict:
    """Compute loss value and gradient norm for a loss function.

    Returns:
        dict with 'loss', 'grad_norm', 'grad_max', 'grad_min'
    """
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


def test_2d_fft_losses():
    """Test 2D FFT losses with different norm types."""
    print("=" * 80)
    print("2D FFT Loss - Gradient Norm Comparison")
    print("=" * 80)

    # Create test data
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    input = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]
    loss_types = ["mse", "l1"]

    # Test PatchFFTLoss
    print("\n### PatchFFTLoss (patch_size=8) ###\n")
    results = []
    for norm_type in norm_types:
        for loss_type in loss_types:
            loss_fn = PatchFFTLoss(
                patch_size=8, loss_type=loss_type, norm_type=norm_type
            )
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            results.append(
                {
                    "norm_type": norm_type.value,
                    "loss_type": loss_type,
                    "loss": f"{stats['loss']:.4e}",
                    "grad_norm": f"{stats['grad_norm']:.4e}",
                    "grad_max": f"{stats['grad_max']:.4e}",
                    "grad_min": f"{stats['grad_min']:.4e}",
                }
            )

    print(simple_table(results))

    # Test FFTLoss
    print("\n### FFTLoss (global) ###\n")
    results = []
    for norm_type in norm_types:
        for loss_type in loss_types:
            loss_fn = FFTLoss(loss_type=loss_type, norm_type=norm_type)
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            results.append(
                {
                    "norm_type": norm_type.value,
                    "loss_type": loss_type,
                    "loss": f"{stats['loss']:.4e}",
                    "grad_norm": f"{stats['grad_norm']:.4e}",
                    "grad_max": f"{stats['grad_max']:.4e}",
                    "grad_min": f"{stats['grad_min']:.4e}",
                }
            )

    print(simple_table(results))


def test_3d_fft_losses():
    """Test 3D FFT losses with different norm types."""
    print("\n" + "=" * 80)
    print("3D FFT Loss - Gradient Norm Comparison")
    print("=" * 80)

    # Create test data (video)
    torch.manual_seed(42)
    B, T, C, H, W = (
        1,
        9,
        3,
        64,
        64,
    )  # T=9 so after skip_keyframe T=8, divisible by patch_t=8
    input = torch.randn(B, T, C, H, W)
    target = torch.randn(B, T, C, H, W)

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]
    loss_types = ["mse", "l1"]

    # Test PatchFFT3DLoss
    print("\n### PatchFFT3DLoss (patch_size=(8,16,16), skip_keyframe=True) ###\n")
    results = []
    for norm_type in norm_types:
        for loss_type in loss_types:
            loss_fn = PatchFFT3DLoss(
                patch_size=(8, 16, 16),
                loss_type=loss_type,
                norm_type=norm_type,
                skip_keyframe=True,
            )
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            results.append(
                {
                    "norm_type": norm_type.value,
                    "loss_type": loss_type,
                    "loss": f"{stats['loss']:.4e}",
                    "grad_norm": f"{stats['grad_norm']:.4e}",
                    "grad_max": f"{stats['grad_max']:.4e}",
                    "grad_min": f"{stats['grad_min']:.4e}",
                }
            )

    print(simple_table(results))

    # Test TemporalFFTLoss
    print("\n### TemporalFFTLoss (amplitude_only=True) ###\n")
    results = []
    for norm_type in norm_types:
        for loss_type in loss_types:
            loss_fn = TemporalFFTLoss(
                loss_type=loss_type, norm_type=norm_type, amplitude_only=True
            )
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            results.append(
                {
                    "norm_type": norm_type.value,
                    "loss_type": loss_type,
                    "loss": f"{stats['loss']:.4e}",
                    "grad_norm": f"{stats['grad_norm']:.4e}",
                    "grad_max": f"{stats['grad_max']:.4e}",
                    "grad_min": f"{stats['grad_min']:.4e}",
                }
            )

    print(simple_table(results))

    print("\n### TemporalFFTLoss (amplitude_only=False) ###\n")
    results = []
    for norm_type in norm_types:
        for loss_type in loss_types:
            loss_fn = TemporalFFTLoss(
                loss_type=loss_type, norm_type=norm_type, amplitude_only=False
            )
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            results.append(
                {
                    "norm_type": norm_type.value,
                    "loss_type": loss_type,
                    "loss": f"{stats['loss']:.4e}",
                    "grad_norm": f"{stats['grad_norm']:.4e}",
                    "grad_max": f"{stats['grad_max']:.4e}",
                    "grad_min": f"{stats['grad_min']:.4e}",
                }
            )

    print(simple_table(results))


def test_gradient_scaling():
    """Test how gradients scale with input magnitude."""
    print("\n" + "=" * 80)
    print("Gradient Scaling with Input Magnitude")
    print("=" * 80)

    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64

    # Test with different input scales
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    norm_types = [NormType.NONE, NormType.LOG1P]

    print("\n### PatchFFTLoss - Gradient norm vs input scale ###\n")
    results = []
    for scale in scales:
        input = torch.randn(B, C, H, W) * scale
        target = torch.randn(B, C, H, W) * scale

        for norm_type in norm_types:
            loss_fn = PatchFFTLoss(patch_size=8, norm_type=norm_type)
            stats = compute_grad_norm(loss_fn, input.clone(), target)
            results.append(
                {
                    "scale": scale,
                    "norm_type": norm_type.value,
                    "loss": f"{stats['loss']:.4e}",
                    "grad_norm": f"{stats['grad_norm']:.4e}",
                }
            )

    print(simple_table(results))


def test_nan_inf_check():
    """Test for NaN/Inf issues with different norm types."""
    print("\n" + "=" * 80)
    print("NaN/Inf Check")
    print("=" * 80)

    torch.manual_seed(42)

    # Test 2D
    B, C, H, W = 2, 3, 64, 64
    input_2d = torch.randn(B, C, H, W)
    target_2d = torch.randn(B, C, H, W)

    # Test 3D
    B, T, C, H, W = 1, 9, 3, 64, 64
    input_3d = torch.randn(B, T, C, H, W)
    target_3d = torch.randn(B, T, C, H, W)

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]

    print("\nChecking for NaN/Inf in loss values and gradients...\n")

    issues = []

    # Check 2D losses
    for norm_type in norm_types:
        for LossClass, name, inp, tgt in [
            (PatchFFTLoss, "PatchFFTLoss", input_2d, target_2d),
            (FFTLoss, "FFTLoss", input_2d, target_2d),
        ]:
            if LossClass == PatchFFTLoss:
                loss_fn = LossClass(patch_size=8, norm_type=norm_type)
            else:
                loss_fn = LossClass(norm_type=norm_type)

            inp_clone = inp.clone().requires_grad_(True)
            loss = loss_fn(inp_clone, tgt)
            loss.backward()

            has_nan_loss = torch.isnan(loss).any().item()
            has_inf_loss = torch.isinf(loss).any().item()
            has_nan_grad = torch.isnan(inp_clone.grad).any().item()
            has_inf_grad = torch.isinf(inp_clone.grad).any().item()

            if has_nan_loss or has_inf_loss or has_nan_grad or has_inf_grad:
                issues.append(
                    {
                        "loss_class": name,
                        "norm_type": norm_type.value,
                        "nan_loss": has_nan_loss,
                        "inf_loss": has_inf_loss,
                        "nan_grad": has_nan_grad,
                        "inf_grad": has_inf_grad,
                    }
                )

    # Check 3D losses
    for norm_type in norm_types:
        for LossClass, name, kwargs in [
            (
                PatchFFT3DLoss,
                "PatchFFT3DLoss",
                {"patch_size": (8, 16, 16), "skip_keyframe": True},
            ),
            (TemporalFFTLoss, "TemporalFFTLoss", {"amplitude_only": True}),
        ]:
            loss_fn = LossClass(norm_type=norm_type, **kwargs)

            inp_clone = input_3d.clone().requires_grad_(True)
            loss = loss_fn(inp_clone, target_3d)
            loss.backward()

            has_nan_loss = torch.isnan(loss).any().item()
            has_inf_loss = torch.isinf(loss).any().item()
            has_nan_grad = torch.isnan(inp_clone.grad).any().item()
            has_inf_grad = torch.isinf(inp_clone.grad).any().item()

            if has_nan_loss or has_inf_loss or has_nan_grad or has_inf_grad:
                issues.append(
                    {
                        "loss_class": name,
                        "norm_type": norm_type.value,
                        "nan_loss": has_nan_loss,
                        "inf_loss": has_inf_loss,
                        "nan_grad": has_nan_grad,
                        "inf_grad": has_inf_grad,
                    }
                )

    if issues:
        print("Issues found:")
        print(simple_table(issues))
    else:
        print("No NaN/Inf issues found in any configuration!")


def plot_2d_fft_comparison():
    """Create bar plots comparing 2D FFT losses across norm types."""
    print("\nGenerating 2D FFT comparison plots...")

    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64
    input = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]
    norm_names = ["none", "l2", "log", "log1p"]
    loss_types = ["mse", "l1"]

    # Collect data for PatchFFTLoss and FFTLoss
    loss_classes = [
        (
            "PatchFFTLoss",
            lambda nt, lt: PatchFFTLoss(patch_size=8, loss_type=lt, norm_type=nt),
        ),
        ("FFTLoss", lambda nt, lt: FFTLoss(loss_type=lt, norm_type=nt)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2D FFT Loss: Loss Value and Gradient Norm Comparison", fontsize=14)

    for idx, (loss_name, loss_factory) in enumerate(loss_classes):
        # Collect data
        data = {lt: {"loss": [], "grad_norm": []} for lt in loss_types}

        for norm_type in norm_types:
            for loss_type in loss_types:
                loss_fn = loss_factory(norm_type, loss_type)
                stats = compute_grad_norm(loss_fn, input.clone(), target)
                data[loss_type]["loss"].append(stats["loss"])
                data[loss_type]["grad_norm"].append(stats["grad_norm"])

        x = np.arange(len(norm_names))
        width = 0.35

        # Plot loss values
        ax1 = axes[idx, 0]
        bars1 = ax1.bar(
            x - width / 2, data["mse"]["loss"], width, label="MSE", color="steelblue"
        )
        bars2 = ax1.bar(
            x + width / 2, data["l1"]["loss"], width, label="L1", color="coral"
        )
        ax1.set_xlabel("Normalization Type")
        ax1.set_ylabel("Loss Value")
        ax1.set_title(f"{loss_name} - Loss Values")
        ax1.set_xticks(x)
        ax1.set_xticklabels(norm_names)
        ax1.legend()
        ax1.set_yscale("log")
        ax1.grid(axis="y", alpha=0.3)

        # Plot gradient norms
        ax2 = axes[idx, 1]
        bars3 = ax2.bar(
            x - width / 2,
            data["mse"]["grad_norm"],
            width,
            label="MSE",
            color="steelblue",
        )
        bars4 = ax2.bar(
            x + width / 2, data["l1"]["grad_norm"], width, label="L1", color="coral"
        )
        ax2.set_xlabel("Normalization Type")
        ax2.set_ylabel("Gradient Norm")
        ax2.set_title(f"{loss_name} - Gradient Norms")
        ax2.set_xticks(x)
        ax2.set_xticklabels(norm_names)
        ax2.legend()
        ax2.set_yscale("log")
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2d_fft_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '2d_fft_comparison.png'}")


def plot_3d_fft_comparison():
    """Create bar plots comparing 3D FFT losses across norm types."""
    print("\nGenerating 3D FFT comparison plots...")

    torch.manual_seed(42)
    B, T, C, H, W = 1, 9, 3, 64, 64
    input = torch.randn(B, T, C, H, W)
    target = torch.randn(B, T, C, H, W)

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]
    norm_names = ["none", "l2", "log", "log1p"]
    loss_types = ["mse", "l1"]

    # Collect data for 3D losses
    loss_classes = [
        (
            "PatchFFT3DLoss",
            lambda nt, lt: PatchFFT3DLoss(
                patch_size=(8, 16, 16), loss_type=lt, norm_type=nt, skip_keyframe=True
            ),
        ),
        (
            "TemporalFFTLoss (amp)",
            lambda nt, lt: TemporalFFTLoss(
                loss_type=lt, norm_type=nt, amplitude_only=True
            ),
        ),
        (
            "TemporalFFTLoss (full)",
            lambda nt, lt: TemporalFFTLoss(
                loss_type=lt, norm_type=nt, amplitude_only=False
            ),
        ),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("3D FFT Loss: Loss Value and Gradient Norm Comparison", fontsize=14)

    for idx, (loss_name, loss_factory) in enumerate(loss_classes):
        # Collect data
        data = {lt: {"loss": [], "grad_norm": []} for lt in loss_types}

        for norm_type in norm_types:
            for loss_type in loss_types:
                loss_fn = loss_factory(norm_type, loss_type)
                stats = compute_grad_norm(loss_fn, input.clone(), target)
                data[loss_type]["loss"].append(stats["loss"])
                data[loss_type]["grad_norm"].append(stats["grad_norm"])

        x = np.arange(len(norm_names))
        width = 0.35

        # Plot loss values
        ax1 = axes[idx, 0]
        ax1.bar(
            x - width / 2, data["mse"]["loss"], width, label="MSE", color="steelblue"
        )
        ax1.bar(x + width / 2, data["l1"]["loss"], width, label="L1", color="coral")
        ax1.set_xlabel("Normalization Type")
        ax1.set_ylabel("Loss Value")
        ax1.set_title(f"{loss_name} - Loss Values")
        ax1.set_xticks(x)
        ax1.set_xticklabels(norm_names)
        ax1.legend()
        ax1.set_yscale("log")
        ax1.grid(axis="y", alpha=0.3)

        # Plot gradient norms
        ax2 = axes[idx, 1]
        ax2.bar(
            x - width / 2,
            data["mse"]["grad_norm"],
            width,
            label="MSE",
            color="steelblue",
        )
        ax2.bar(
            x + width / 2, data["l1"]["grad_norm"], width, label="L1", color="coral"
        )
        ax2.set_xlabel("Normalization Type")
        ax2.set_ylabel("Gradient Norm")
        ax2.set_title(f"{loss_name} - Gradient Norms")
        ax2.set_xticks(x)
        ax2.set_xticklabels(norm_names)
        ax2.legend()
        ax2.set_yscale("log")
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3d_fft_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / '3d_fft_comparison.png'}")


def plot_gradient_scaling():
    """Plot how gradients scale with input magnitude for different norm types."""
    print("\nGenerating gradient scaling plots...")

    torch.manual_seed(42)
    B, C, H, W = 2, 3, 64, 64

    scales = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]
    norm_names = ["none", "l2", "log", "log1p"]
    colors = ["steelblue", "coral", "seagreen", "orchid"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gradient Norm vs Input Scale", fontsize=14)

    # PatchFFTLoss
    ax1 = axes[0, 0]
    for norm_type, name, color in zip(norm_types, norm_names, colors):
        grad_norms = []
        for scale in scales:
            inp = torch.randn(B, C, H, W) * scale
            tgt = torch.randn(B, C, H, W) * scale
            loss_fn = PatchFFTLoss(patch_size=8, norm_type=norm_type)
            stats = compute_grad_norm(loss_fn, inp, tgt)
            grad_norms.append(stats["grad_norm"])
        ax1.plot(
            scales, grad_norms, "o-", label=name, color=color, linewidth=2, markersize=6
        )

    ax1.set_xlabel("Input Scale")
    ax1.set_ylabel("Gradient Norm")
    ax1.set_title("PatchFFTLoss - Gradient Norm vs Scale")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # FFTLoss
    ax2 = axes[0, 1]
    for norm_type, name, color in zip(norm_types, norm_names, colors):
        grad_norms = []
        for scale in scales:
            inp = torch.randn(B, C, H, W) * scale
            tgt = torch.randn(B, C, H, W) * scale
            loss_fn = FFTLoss(norm_type=norm_type)
            stats = compute_grad_norm(loss_fn, inp, tgt)
            grad_norms.append(stats["grad_norm"])
        ax2.plot(
            scales, grad_norms, "o-", label=name, color=color, linewidth=2, markersize=6
        )

    ax2.set_xlabel("Input Scale")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("FFTLoss - Gradient Norm vs Scale")
    ax2.legend()
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Loss values vs scale for PatchFFTLoss
    ax3 = axes[1, 0]
    for norm_type, name, color in zip(norm_types, norm_names, colors):
        losses = []
        for scale in scales:
            inp = torch.randn(B, C, H, W) * scale
            tgt = torch.randn(B, C, H, W) * scale
            loss_fn = PatchFFTLoss(patch_size=8, norm_type=norm_type)
            stats = compute_grad_norm(loss_fn, inp, tgt)
            losses.append(stats["loss"])
        ax3.plot(
            scales, losses, "o-", label=name, color=color, linewidth=2, markersize=6
        )

    ax3.set_xlabel("Input Scale")
    ax3.set_ylabel("Loss Value")
    ax3.set_title("PatchFFTLoss - Loss Value vs Scale")
    ax3.legend()
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Loss values vs scale for FFTLoss
    ax4 = axes[1, 1]
    for norm_type, name, color in zip(norm_types, norm_names, colors):
        losses = []
        for scale in scales:
            inp = torch.randn(B, C, H, W) * scale
            tgt = torch.randn(B, C, H, W) * scale
            loss_fn = FFTLoss(norm_type=norm_type)
            stats = compute_grad_norm(loss_fn, inp, tgt)
            losses.append(stats["loss"])
        ax4.plot(
            scales, losses, "o-", label=name, color=color, linewidth=2, markersize=6
        )

    ax4.set_xlabel("Input Scale")
    ax4.set_ylabel("Loss Value")
    ax4.set_title("FFTLoss - Loss Value vs Scale")
    ax4.legend()
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradient_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'gradient_scaling.png'}")


def plot_all_losses_summary():
    """Create a comprehensive summary plot of all FFT losses."""
    print("\nGenerating summary plot...")

    torch.manual_seed(42)

    # 2D data
    B, C, H, W = 2, 3, 64, 64
    input_2d = torch.randn(B, C, H, W)
    target_2d = torch.randn(B, C, H, W)

    # 3D data
    B, T, C, H, W = 1, 9, 3, 64, 64
    input_3d = torch.randn(B, T, C, H, W)
    target_3d = torch.randn(B, T, C, H, W)

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]
    norm_names = ["none", "l2", "log", "log1p"]

    # All loss configurations
    loss_configs = [
        (
            "PatchFFT (2D)",
            input_2d,
            target_2d,
            lambda nt: PatchFFTLoss(patch_size=8, norm_type=nt),
        ),
        ("FFT (2D)", input_2d, target_2d, lambda nt: FFTLoss(norm_type=nt)),
        (
            "PatchFFT3D",
            input_3d,
            target_3d,
            lambda nt: PatchFFT3DLoss(
                patch_size=(8, 16, 16), norm_type=nt, skip_keyframe=True
            ),
        ),
        (
            "TemporalFFT (amp)",
            input_3d,
            target_3d,
            lambda nt: TemporalFFTLoss(norm_type=nt, amplitude_only=True),
        ),
        (
            "TemporalFFT (full)",
            input_3d,
            target_3d,
            lambda nt: TemporalFFTLoss(norm_type=nt, amplitude_only=False),
        ),
    ]

    # Collect all data
    all_data = {}
    for loss_name, inp, tgt, factory in loss_configs:
        all_data[loss_name] = {"loss": [], "grad_norm": []}
        for norm_type in norm_types:
            loss_fn = factory(norm_type)
            stats = compute_grad_norm(loss_fn, inp.clone(), tgt)
            all_data[loss_name]["loss"].append(stats["loss"])
            all_data[loss_name]["grad_norm"].append(stats["grad_norm"])

    # Create grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("FFT Loss Comparison Across All Losses and Norm Types", fontsize=14)

    x = np.arange(len(norm_names))
    n_losses = len(loss_configs)
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, n_losses))

    # Loss values
    ax1 = axes[0]
    for i, (loss_name, _, _, _) in enumerate(loss_configs):
        offset = (i - n_losses / 2 + 0.5) * width
        ax1.bar(
            x + offset,
            all_data[loss_name]["loss"],
            width,
            label=loss_name,
            color=colors[i],
        )

    ax1.set_xlabel("Normalization Type")
    ax1.set_ylabel("Loss Value (log scale)")
    ax1.set_title("Loss Values")
    ax1.set_xticks(x)
    ax1.set_xticklabels(norm_names)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)

    # Gradient norms
    ax2 = axes[1]
    for i, (loss_name, _, _, _) in enumerate(loss_configs):
        offset = (i - n_losses / 2 + 0.5) * width
        ax2.bar(
            x + offset,
            all_data[loss_name]["grad_norm"],
            width,
            label=loss_name,
            color=colors[i],
        )

    ax2.set_xlabel("Normalization Type")
    ax2.set_ylabel("Gradient Norm (log scale)")
    ax2.set_title("Gradient Norms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(norm_names)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_losses_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'all_losses_summary.png'}")


def plot_grad_norm_heatmap():
    """Create a heatmap of gradient norms across all losses and norm types."""
    print("\nGenerating gradient norm heatmap...")

    torch.manual_seed(42)

    # 2D data
    B, C, H, W = 2, 3, 64, 64
    input_2d = torch.randn(B, C, H, W)
    target_2d = torch.randn(B, C, H, W)

    # 3D data
    B, T, C, H, W = 1, 9, 3, 64, 64
    input_3d = torch.randn(B, T, C, H, W)
    target_3d = torch.randn(B, T, C, H, W)

    norm_types = [NormType.NONE, NormType.L2, NormType.LOG, NormType.LOG1P]
    norm_names = ["none", "l2", "log", "log1p"]

    loss_configs = [
        (
            "PatchFFT\n(2D, MSE)",
            input_2d,
            target_2d,
            lambda nt: PatchFFTLoss(patch_size=8, norm_type=nt, loss_type="mse"),
        ),
        (
            "PatchFFT\n(2D, L1)",
            input_2d,
            target_2d,
            lambda nt: PatchFFTLoss(patch_size=8, norm_type=nt, loss_type="l1"),
        ),
        (
            "FFT\n(2D, MSE)",
            input_2d,
            target_2d,
            lambda nt: FFTLoss(norm_type=nt, loss_type="mse"),
        ),
        (
            "FFT\n(2D, L1)",
            input_2d,
            target_2d,
            lambda nt: FFTLoss(norm_type=nt, loss_type="l1"),
        ),
        (
            "PatchFFT3D\n(MSE)",
            input_3d,
            target_3d,
            lambda nt: PatchFFT3DLoss(
                patch_size=(8, 16, 16),
                norm_type=nt,
                loss_type="mse",
                skip_keyframe=True,
            ),
        ),
        (
            "PatchFFT3D\n(L1)",
            input_3d,
            target_3d,
            lambda nt: PatchFFT3DLoss(
                patch_size=(8, 16, 16), norm_type=nt, loss_type="l1", skip_keyframe=True
            ),
        ),
        (
            "TemporalFFT\n(amp, L1)",
            input_3d,
            target_3d,
            lambda nt: TemporalFFTLoss(
                norm_type=nt, amplitude_only=True, loss_type="l1"
            ),
        ),
        (
            "TemporalFFT\n(full, L1)",
            input_3d,
            target_3d,
            lambda nt: TemporalFFTLoss(
                norm_type=nt, amplitude_only=False, loss_type="l1"
            ),
        ),
    ]

    # Build matrix
    grad_matrix = np.zeros((len(loss_configs), len(norm_types)))
    loss_names = []

    for i, (loss_name, inp, tgt, factory) in enumerate(loss_configs):
        loss_names.append(loss_name)
        for j, norm_type in enumerate(norm_types):
            loss_fn = factory(norm_type)
            stats = compute_grad_norm(loss_fn, inp.clone(), tgt)
            grad_matrix[i, j] = stats["grad_norm"]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use log scale for colors
    log_matrix = np.log10(grad_matrix + 1e-10)

    im = ax.imshow(log_matrix, cmap="RdYlGn_r", aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("log10(Gradient Norm)", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(len(norm_names)))
    ax.set_yticks(np.arange(len(loss_names)))
    ax.set_xticklabels(norm_names)
    ax.set_yticklabels(loss_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(loss_names)):
        for j in range(len(norm_names)):
            val = grad_matrix[i, j]
            text = f"{val:.2e}"
            color = "white" if log_matrix[i, j] > np.median(log_matrix) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Gradient Norm Heatmap (lower = more stable)")
    ax.set_xlabel("Normalization Type")
    ax.set_ylabel("Loss Function")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "grad_norm_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'grad_norm_heatmap.png'}")


if __name__ == "__main__":
    print("=" * 60)
    print("FFT Loss Normalization Test")
    print("=" * 60)

    # Run text-based tests
    test_2d_fft_losses()
    test_3d_fft_losses()
    test_gradient_scaling()
    test_nan_inf_check()

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_2d_fft_comparison()
    plot_3d_fft_comparison()
    plot_gradient_scaling()
    plot_all_losses_summary()
    plot_grad_norm_heatmap()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
