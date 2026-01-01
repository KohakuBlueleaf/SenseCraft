"""
Test script for comparing FFT losses vs pixel-based losses (L1/L2) under various distortions.

This script:
1. Loads test images from images/test*.{jpg,jpeg,png,webp}
2. Applies distortions: JPEG/WebP compression, Gaussian noise, Gaussian blur
3. Compares FFT losses (with different norm types and loss modes) vs pixel losses
4. Saves plots to results/fft_comparison/*.png

Usage:
    python examples/test_fft_vs_pixel_loss.py [--device cuda/cpu] [--image PATH]
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensecraft.loss.general import (
    CharbonnierLoss,
    FFTLoss,
    L1Loss,
    MSELoss,
    NormType,
    PatchFFTLoss,
)
from sensecraft.loss.video import PatchFFT3DLoss, TemporalFFTLoss


# Output directory for plots
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "fft_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Distortion Functions
# ============================================================================


def apply_jpeg_compression(img_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply JPEG compression to image tensor."""
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    img_compressed = Image.open(buffer)
    img_np = np.array(img_compressed).astype(np.float32) / 255.0

    return torch.from_numpy(img_np).permute(2, 0, 1).to(img_tensor.device)


def apply_gaussian_noise(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian noise to image tensor."""
    noise = torch.randn_like(img_tensor) * sigma
    return torch.clamp(img_tensor + noise, 0, 1)


def apply_gaussian_blur(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to image tensor."""
    if sigma <= 0:
        return img_tensor

    kernel_size = int(np.ceil(sigma * 3) * 2 + 1)
    kernel_size = max(kernel_size, 3)

    x = torch.arange(kernel_size, dtype=torch.float32, device=img_tensor.device)
    x = x - kernel_size // 2
    kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(3, 1, 1, 1)

    padding = kernel_size // 2
    img_4d = img_tensor.unsqueeze(0)
    blurred = F.conv2d(img_4d, kernel_2d, padding=padding, groups=3)

    return blurred.squeeze(0)


# ============================================================================
# Loss Function Definitions
# ============================================================================


def get_pixel_losses(device: torch.device) -> Dict[str, nn.Module]:
    """Get pixel-based loss functions."""
    return {
        "MSE": MSELoss().to(device),
        "L1": L1Loss().to(device),
        "Charbonnier": CharbonnierLoss(eps=1e-6).to(device),
    }


def get_fft_losses(device: torch.device) -> Dict[str, nn.Module]:
    """Get FFT loss functions with different norm types and loss modes."""
    losses = {}

    # PatchFFT with different norm types and loss modes
    for norm_type in ["none", "l2", "log1p"]:
        for loss_type in ["mse", "l1"]:
            name = f"PatchFFT_{loss_type}_{norm_type}"
            losses[name] = PatchFFTLoss(
                patch_size=8, loss_type=loss_type, norm_type=norm_type
            ).to(device)

    # Global FFT with different norm types and loss modes
    for norm_type in ["none", "log1p"]:
        for loss_type in ["mse", "l1"]:
            name = f"FFT_{loss_type}_{norm_type}"
            losses[name] = FFTLoss(loss_type=loss_type, norm_type=norm_type).to(device)

    return losses


def get_all_losses(device: torch.device) -> Dict[str, nn.Module]:
    """Get all loss functions (pixel + FFT)."""
    losses = get_pixel_losses(device)
    losses.update(get_fft_losses(device))
    return losses


# ============================================================================
# Evaluation Functions
# ============================================================================


def compute_loss_and_grad(
    loss_fn: nn.Module,
    gt: torch.Tensor,
    distorted: torch.Tensor,
) -> Tuple[float, float]:
    """Compute loss value and gradient norm."""
    distorted = distorted.clone().requires_grad_(True)

    loss = loss_fn(distorted, gt)
    loss.backward()

    grad_norm = distorted.grad.norm().item()
    loss_value = loss.item()

    return loss_value, grad_norm


def evaluate_distortion(
    loss_fns: Dict[str, nn.Module],
    gt: torch.Tensor,
    distort_fn: Callable,
    distort_levels: List,
    distort_name: str,
) -> Dict[str, Dict[str, List[float]]]:
    """Evaluate all losses across distortion levels."""
    results = {name: {"loss": [], "grad": []} for name in loss_fns.keys()}

    gt_batch = gt.unsqueeze(0)

    for level in distort_levels:
        distorted = distort_fn(gt, level)
        distorted_batch = distorted.unsqueeze(0)

        for name, loss_fn in loss_fns.items():
            try:
                loss_val, grad_norm = compute_loss_and_grad(
                    loss_fn, gt_batch, distorted_batch
                )
                results[name]["loss"].append(loss_val)
                results[name]["grad"].append(grad_norm)
            except Exception as e:
                print(f"  Warning: {name} failed at level {level}: {e}")
                results[name]["loss"].append(float("nan"))
                results[name]["grad"].append(float("nan"))

    return results


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_results(
    results: Dict[str, Dict[str, List[float]]],
    x_values: List,
    x_label: str,
    title: str,
    output_path: Path,
    log_scale_y: bool = False,
):
    """Plot loss and gradient norm results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Separate pixel and FFT losses for coloring
    pixel_losses = ["MSE", "L1", "Charbonnier"]
    pixel_colors = {"MSE": "steelblue", "L1": "navy", "Charbonnier": "royalblue"}

    # FFT loss colors based on norm type
    fft_colors = {
        "none": "coral",
        "l2": "red",
        "log": "orange",
        "log1p": "darkorange",
    }

    for name, data in results.items():
        if name in pixel_losses:
            color = pixel_colors.get(name, "blue")
            linestyle = "-"
            marker = "o"
        else:
            # Determine norm type from name
            for norm in ["none", "l2", "log1p", "log"]:
                if norm in name.lower():
                    color = fft_colors.get(norm, "red")
                    break
            else:
                color = "red"
            linestyle = "--" if "_l1_" in name.lower() else "-"
            marker = "s" if "_l1_" in name.lower() else "^"

        ax1.plot(
            x_values, data["loss"], marker=marker, label=name, markersize=4,
            color=color, linestyle=linestyle
        )
        ax2.plot(
            x_values, data["grad"], marker=marker, label=name, markersize=4,
            color=color, linestyle=linestyle
        )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Loss Value")
    ax1.set_title(f"{title} - Loss Values")
    ax1.legend(loc="upper left", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    if log_scale_y:
        ax1.set_yscale("log")

    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title(f"{title} - Gradient Norms")
    ax2.legend(loc="upper left", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    if log_scale_y:
        ax2.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pixel_vs_fft(
    results: Dict[str, Dict[str, List[float]]],
    x_values: List,
    x_label: str,
    title: str,
    output_path: Path,
):
    """Plot pixel losses vs FFT losses separately for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pixel_losses = {k: v for k, v in results.items() if k in ["MSE", "L1", "Charbonnier"]}
    fft_losses = {k: v for k, v in results.items() if k not in ["MSE", "L1", "Charbonnier"]}

    # Pixel losses - Loss values
    ax = axes[0, 0]
    for name, data in pixel_losses.items():
        ax.plot(x_values, data["loss"], marker="o", label=name, markersize=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss Value")
    ax.set_title(f"{title} - Pixel Losses")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pixel losses - Gradient norms
    ax = axes[0, 1]
    for name, data in pixel_losses.items():
        ax.plot(x_values, data["grad"], marker="o", label=name, markersize=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title} - Pixel Losses (Grad)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # FFT losses - Loss values
    ax = axes[1, 0]
    for name, data in fft_losses.items():
        linestyle = "--" if "_l1_" in name.lower() else "-"
        ax.plot(x_values, data["loss"], marker="s", label=name, markersize=3, linestyle=linestyle)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss Value")
    ax.set_title(f"{title} - FFT Losses")
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # FFT losses - Gradient norms
    ax = axes[1, 1]
    for name, data in fft_losses.items():
        linestyle = "--" if "_l1_" in name.lower() else "-"
        ax.plot(x_values, data["grad"], marker="s", label=name, markersize=3, linestyle=linestyle)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title} - FFT Losses (Grad)")
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison_grid(
    all_results: Dict[str, Dict[str, Dict[str, List[float]]]],
    distortion_configs: Dict[str, Tuple[List, str]],
    output_path: Path,
):
    """Plot a grid comparing all distortions and losses."""
    n_distortions = len(all_results)
    fig, axes = plt.subplots(n_distortions, 2, figsize=(14, 5 * n_distortions))

    if n_distortions == 1:
        axes = axes.reshape(1, -1)

    for i, (dist_name, results) in enumerate(all_results.items()):
        x_values, x_label = distortion_configs[dist_name]

        # Loss values
        for name, data in results.items():
            axes[i, 0].plot(
                x_values, data["loss"], marker="o", label=name, markersize=3
            )
        axes[i, 0].set_xlabel(x_label)
        axes[i, 0].set_ylabel("Loss Value")
        axes[i, 0].set_title(f"{dist_name} - Loss Values")
        axes[i, 0].legend(loc="upper right", fontsize=5, ncol=2)
        axes[i, 0].grid(True, alpha=0.3)

        # Gradient norms
        for name, data in results.items():
            axes[i, 1].plot(
                x_values, data["grad"], marker="o", label=name, markersize=3
            )
        axes[i, 1].set_xlabel(x_label)
        axes[i, 1].set_ylabel("Gradient Norm")
        axes[i, 1].set_title(f"{dist_name} - Gradient Norms")
        axes[i, 1].legend(loc="upper right", fontsize=5, ncol=2)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_norm_comparison(
    results: Dict[str, Dict[str, List[float]]],
    x_values: List,
    x_label: str,
    title: str,
    output_path: Path,
):
    """Plot comparison of different FFT norm types for same loss mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    norm_types = ["none", "l2", "log1p"]
    colors = {"none": "coral", "l2": "red", "log1p": "darkorange"}

    # PatchFFT MSE mode
    ax = axes[0, 0]
    for norm in norm_types:
        name = f"PatchFFT_mse_{norm}"
        if name in results:
            ax.plot(x_values, results[name]["loss"], marker="o", label=f"norm={norm}",
                    markersize=4, color=colors[norm])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss Value")
    ax.set_title(f"{title} - PatchFFT (MSE mode)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # PatchFFT L1 mode
    ax = axes[0, 1]
    for norm in norm_types:
        name = f"PatchFFT_l1_{norm}"
        if name in results:
            ax.plot(x_values, results[name]["loss"], marker="s", label=f"norm={norm}",
                    markersize=4, color=colors[norm], linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss Value")
    ax.set_title(f"{title} - PatchFFT (L1 mode)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Gradient norms - MSE mode
    ax = axes[1, 0]
    for norm in norm_types:
        name = f"PatchFFT_mse_{norm}"
        if name in results:
            ax.plot(x_values, results[name]["grad"], marker="o", label=f"norm={norm}",
                    markersize=4, color=colors[norm])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title} - PatchFFT Grad Norm (MSE mode)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Gradient norms - L1 mode
    ax = axes[1, 1]
    for norm in norm_types:
        name = f"PatchFFT_l1_{norm}"
        if name in results:
            ax.plot(x_values, results[name]["grad"], marker="s", label=f"norm={norm}",
                    markersize=4, color=colors[norm], linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title} - PatchFFT Grad Norm (L1 mode)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_mse_vs_l1_mode(
    results: Dict[str, Dict[str, List[float]]],
    x_values: List,
    x_label: str,
    title: str,
    output_path: Path,
):
    """Plot comparison of MSE vs L1 mode for same norm type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    norm_types = ["none", "log1p"]
    colors_mse = {"none": "steelblue", "log1p": "navy"}
    colors_l1 = {"none": "coral", "log1p": "darkorange"}

    # PatchFFT - Loss values
    ax = axes[0, 0]
    for norm in norm_types:
        mse_name = f"PatchFFT_mse_{norm}"
        l1_name = f"PatchFFT_l1_{norm}"
        if mse_name in results:
            ax.plot(x_values, results[mse_name]["loss"], marker="o",
                    label=f"MSE (norm={norm})", markersize=4, color=colors_mse[norm])
        if l1_name in results:
            ax.plot(x_values, results[l1_name]["loss"], marker="s",
                    label=f"L1 (norm={norm})", markersize=4, color=colors_l1[norm], linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss Value")
    ax.set_title(f"{title} - PatchFFT: MSE vs L1 mode")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # PatchFFT - Gradient norms
    ax = axes[0, 1]
    for norm in norm_types:
        mse_name = f"PatchFFT_mse_{norm}"
        l1_name = f"PatchFFT_l1_{norm}"
        if mse_name in results:
            ax.plot(x_values, results[mse_name]["grad"], marker="o",
                    label=f"MSE (norm={norm})", markersize=4, color=colors_mse[norm])
        if l1_name in results:
            ax.plot(x_values, results[l1_name]["grad"], marker="s",
                    label=f"L1 (norm={norm})", markersize=4, color=colors_l1[norm], linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title} - PatchFFT Grad: MSE vs L1 mode")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # FFT - Loss values
    ax = axes[1, 0]
    for norm in norm_types:
        mse_name = f"FFT_mse_{norm}"
        l1_name = f"FFT_l1_{norm}"
        if mse_name in results:
            ax.plot(x_values, results[mse_name]["loss"], marker="o",
                    label=f"MSE (norm={norm})", markersize=4, color=colors_mse[norm])
        if l1_name in results:
            ax.plot(x_values, results[l1_name]["loss"], marker="s",
                    label=f"L1 (norm={norm})", markersize=4, color=colors_l1[norm], linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss Value")
    ax.set_title(f"{title} - FFT (Global): MSE vs L1 mode")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # FFT - Gradient norms
    ax = axes[1, 1]
    for norm in norm_types:
        mse_name = f"FFT_mse_{norm}"
        l1_name = f"FFT_l1_{norm}"
        if mse_name in results:
            ax.plot(x_values, results[mse_name]["grad"], marker="o",
                    label=f"MSE (norm={norm})", markersize=4, color=colors_mse[norm])
        if l1_name in results:
            ax.plot(x_values, results[l1_name]["grad"], marker="s",
                    label=f"L1 (norm={norm})", markersize=4, color=colors_l1[norm], linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"{title} - FFT Grad: MSE vs L1 mode")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# Image Loading
# ============================================================================


def load_image(path: Path, device: torch.device, max_size: int = 512) -> torch.Tensor:
    """Load image and convert to tensor."""
    img = Image.open(path).convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

    return img_tensor


def find_test_images(images_dir: Path) -> List[Path]:
    """Find all test images in the images directory."""
    patterns = ["test*.jpg", "test*.jpeg", "test*.png", "test*.webp"]
    images = []
    for pattern in patterns:
        images.extend(images_dir.glob(pattern))
    return sorted(images)


def get_best_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare FFT losses vs pixel losses under distortions"
    )
    default_device = get_best_device()
    parser.add_argument(
        "--device",
        type=str,
        default=str(default_device),
        help=f"Device to use (default: {default_device})",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to specific image to test"
    )
    parser.add_argument("--max-size", type=int, default=512, help="Max image size")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Setup paths
    project_root = Path(__file__).parent.parent
    images_dir = project_root / "images"

    # Find images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = find_test_images(images_dir)

    if not image_paths:
        print(f"No test images found in {images_dir}")
        print("Using synthetic test data instead...")
        use_synthetic = True
    else:
        use_synthetic = False
        print(f"Found {len(image_paths)} test image(s)")

    # Load loss functions
    print("\nLoading loss functions...")
    loss_fns = get_all_losses(device)
    print(f"  Loaded {len(loss_fns)} losses:")
    for name in sorted(loss_fns.keys()):
        print(f"    - {name}")

    # Define distortion configurations
    distortions = {
        "JPEG_Compression": {
            "fn": apply_jpeg_compression,
            "levels": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5],
            "x_label": "JPEG Quality",
        },
        "Gaussian_Noise": {
            "fn": apply_gaussian_noise,
            "levels": [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
            "x_label": "Noise Sigma",
        },
        "Gaussian_Blur": {
            "fn": apply_gaussian_blur,
            "levels": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0],
            "x_label": "Blur Sigma",
        },
    }

    if use_synthetic:
        # Use synthetic data
        torch.manual_seed(42)
        C, H, W = 3, 256, 256
        gt = torch.rand(C, H, W, device=device)
        img_name = "synthetic"
        image_paths = [None]  # Dummy for loop

    for img_idx, img_path in enumerate(image_paths):
        if not use_synthetic:
            img_name = img_path.stem
            print(f"\nProcessing: {img_path.name}")
            try:
                gt = load_image(img_path, device, max_size=args.max_size)
                print(f"  Image size: {gt.shape[1]}x{gt.shape[2]}")
            except Exception as e:
                print(f"  Error loading image: {e}")
                continue
        else:
            print(f"\nProcessing: synthetic image")

        # Store all results
        all_results = {}
        distortion_configs = {}

        # Run each distortion
        for dist_name, dist_config in distortions.items():
            print(f"  Testing: {dist_name}")

            results = evaluate_distortion(
                loss_fns=loss_fns,
                gt=gt,
                distort_fn=dist_config["fn"],
                distort_levels=dist_config["levels"],
                distort_name=dist_name,
            )

            all_results[dist_name] = results
            distortion_configs[dist_name] = (
                dist_config["levels"],
                dist_config["x_label"],
            )

            # Save individual plot - all losses combined
            output_path = OUTPUT_DIR / img_name / f"{dist_name}_all.png"
            plot_results(
                results=results,
                x_values=dist_config["levels"],
                x_label=dist_config["x_label"],
                title=f"{img_name} - {dist_name}",
                output_path=output_path,
            )

            # Save pixel vs FFT comparison
            output_path = OUTPUT_DIR / img_name / f"{dist_name}_pixel_vs_fft.png"
            plot_pixel_vs_fft(
                results=results,
                x_values=dist_config["levels"],
                x_label=dist_config["x_label"],
                title=f"{img_name} - {dist_name}",
                output_path=output_path,
            )

            # Save norm type comparison
            output_path = OUTPUT_DIR / img_name / f"{dist_name}_norm_comparison.png"
            plot_norm_comparison(
                results=results,
                x_values=dist_config["levels"],
                x_label=dist_config["x_label"],
                title=f"{img_name} - {dist_name}",
                output_path=output_path,
            )

            # Save MSE vs L1 mode comparison
            output_path = OUTPUT_DIR / img_name / f"{dist_name}_mse_vs_l1.png"
            plot_mse_vs_l1_mode(
                results=results,
                x_values=dist_config["levels"],
                x_label=dist_config["x_label"],
                title=f"{img_name} - {dist_name}",
                output_path=output_path,
            )

        # Save grid comparison for all distortions
        output_path = OUTPUT_DIR / img_name / "all_distortions_grid.png"
        plot_comparison_grid(
            all_results=all_results,
            distortion_configs=distortion_configs,
            output_path=output_path,
        )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
