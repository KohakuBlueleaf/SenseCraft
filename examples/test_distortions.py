"""
Test script for evaluating different loss functions and metrics under various image distortions.

This script:
1. Loads test images from images/test*.{jpg,jpeg,png,webp}
2. Applies distortions: JPEG/WebP compression, Gaussian noise, Gaussian blur
3. Computes loss values and gradient norms for each distortion level
4. Computes metric values (PSNR, SSIM, MS-SSIM, etc.) for each distortion level
5. Saves plots to results/**/*.png

Usage:
    python examples/test_distortions.py [--device cuda/cpu] [--image PATH]
"""

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensecraft.loss import (
    CharbonnierLoss,
    PatchFFTLoss,
    FFTLoss,
    NormType,
)
from sensecraft.metrics import psnr, ssim, ms_ssim, rmse, mae, mape, lpips


# ============================================================================
# Distortion Functions
# ============================================================================


def apply_jpeg_compression(img_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply JPEG compression to image tensor.

    Args:
        img_tensor: Image tensor of shape (C, H, W) in range [0, 1]
        quality: JPEG quality (1-100)

    Returns:
        Compressed image tensor
    """
    # Convert to PIL
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    # Compress to JPEG in memory
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Load back
    img_compressed = Image.open(buffer)
    img_np = np.array(img_compressed).astype(np.float32) / 255.0

    return torch.from_numpy(img_np).permute(2, 0, 1).to(img_tensor.device)


def apply_webp_compression(img_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply WebP compression to image tensor.

    Args:
        img_tensor: Image tensor of shape (C, H, W) in range [0, 1]
        quality: WebP quality (1-100)

    Returns:
        Compressed image tensor
    """
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)

    img_compressed = Image.open(buffer).convert("RGB")
    img_np = np.array(img_compressed).astype(np.float32) / 255.0

    return torch.from_numpy(img_np).permute(2, 0, 1).to(img_tensor.device)


def apply_gaussian_noise(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian noise to image tensor.

    Args:
        img_tensor: Image tensor of shape (C, H, W) in range [0, 1]
        sigma: Standard deviation of noise

    Returns:
        Noisy image tensor (clamped to [0, 1])
    """
    noise = torch.randn_like(img_tensor) * sigma
    return torch.clamp(img_tensor + noise, 0, 1)


def apply_gaussian_blur(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to image tensor.

    Args:
        img_tensor: Image tensor of shape (C, H, W) in range [0, 1]
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Blurred image tensor
    """
    if sigma <= 0:
        return img_tensor

    # Compute kernel size (should be odd, at least 3*sigma)
    kernel_size = int(np.ceil(sigma * 3) * 2 + 1)
    kernel_size = max(kernel_size, 3)

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=img_tensor.device)
    x = x - kernel_size // 2
    kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(3, 1, 1, 1)

    # Apply convolution
    padding = kernel_size // 2
    img_4d = img_tensor.unsqueeze(0)
    blurred = F.conv2d(img_4d, kernel_2d, padding=padding, groups=3)

    return blurred.squeeze(0)


# ============================================================================
# Loss Function Definitions
# ============================================================================


def get_basic_losses(device: torch.device) -> Dict[str, torch.nn.Module]:
    """Get basic loss functions (MSE, L1, Charbonnier, FFT variants)."""
    return {
        "MSE": torch.nn.MSELoss().to(device),
        "L1": torch.nn.L1Loss().to(device),
        "Charbonnier": CharbonnierLoss(eps=1e-6).to(device),
        "FFT_MSE": FFTLoss(loss_type="mse", norm_type=NormType.LOG1P).to(device),
        "FFT_L1": FFTLoss(loss_type="l1", norm_type=NormType.LOG1P).to(device),
        "PatchFFT_MSE_8": PatchFFTLoss(
            patch_size=8, loss_type="mse", norm_type=NormType.LOG1P
        ).to(device),
        "PatchFFT_MSE_16": PatchFFTLoss(
            patch_size=16, loss_type="mse", norm_type=NormType.LOG1P
        ).to(device),
        "PatchFFT_L1_8": PatchFFTLoss(
            patch_size=8, loss_type="l1", norm_type=NormType.LOG1P
        ).to(device),
    }


def get_dinov3_losses(device: torch.device) -> Dict[str, torch.nn.Module]:
    """Get DINOv3 ViT perceptual losses (requires transformers)."""
    try:
        from sensecraft.loss import ViTDinoV3PerceptualLoss
        from sensecraft.loss.gram_dinov3 import ModelType

        return {
            "DINOv3_ViT": ViTDinoV3PerceptualLoss(
                model_type=ModelType.SMALL,
                use_norm=True,
                use_gram=False,
                input_range=(0, 1),
            ).to(device),
            "DINOv3_ViT_Gram": ViTDinoV3PerceptualLoss(
                model_type=ModelType.SMALL,
                use_norm=True,
                use_gram=True,
                input_range=(0, 1),
            ).to(device),
        }
    except ImportError as e:
        print(f"Warning: Could not load DINOv3 losses: {e}")
        return {}


def get_perceptual_losses(device: torch.device) -> Dict[str, torch.nn.Module]:
    """Get LPIPS and SSIM/MS-SSIM losses."""
    losses = {}

    # LPIPS
    try:
        from sensecraft.loss import LPIPS

        losses["LPIPS_VGG"] = LPIPS(net="vgg").to(device)
        losses["LPIPS_Alex"] = LPIPS(net="alex").to(device)
    except ImportError as e:
        print(f"Warning: Could not load LPIPS: {e}")

    # SSIM and MS-SSIM (using pytorch_msssim)
    try:
        from pytorch_msssim import SSIM, MS_SSIM

        # SSIM as loss (1 - SSIM)
        class SSIMLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

            def forward(self, x, y):
                return 1 - self.ssim(x, y)

        # MS-SSIM as loss (1 - MS_SSIM)
        class MSSSIMLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)

            def forward(self, x, y):
                return 1 - self.ms_ssim(x, y)

        losses["SSIM"] = SSIMLoss().to(device)
        losses["MS_SSIM"] = MSSSIMLoss().to(device)
    except ImportError as e:
        print(f"Warning: Could not load SSIM/MS-SSIM (pip install pytorch-msssim): {e}")

    return losses


def get_metrics(device: torch.device) -> Dict[str, Callable]:
    """Get evaluation metrics as callables (functional API).

    Returns dict of metric functions that take (input, target) tensors.
    """
    metrics = {
        "PSNR": lambda x, y: psnr(x, y, data_range=1.0),
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
    }

    # SSIM (regular and dB)
    try:
        metrics["SSIM"] = lambda x, y: ssim(x, y, data_range=1.0)
        metrics["SSIM_dB"] = lambda x, y: ssim(x, y, data_range=1.0, as_db=True)
        metrics["MS-SSIM"] = lambda x, y: ms_ssim(x, y, data_range=1.0)
        metrics["MS-SSIM_dB"] = lambda x, y: ms_ssim(x, y, data_range=1.0, as_db=True)
    except Exception as e:
        print(f"Warning: Could not load SSIM/MS-SSIM metrics: {e}")

    # LPIPS metric (auto-caches model)
    try:
        # Wrap to convert from [0,1] to [-1,1]
        metrics["LPIPS"] = lambda x, y: lpips(x * 2 - 1, y * 2 - 1, net="alex")
    except Exception as e:
        print(f"Warning: Could not load LPIPS metric: {e}")

    return metrics


# ============================================================================
# Evaluation Functions
# ============================================================================


def compute_loss_and_grad(
    loss_fn: torch.nn.Module,
    gt: torch.Tensor,
    distorted: torch.Tensor,
) -> Tuple[float, float]:
    """Compute loss value and gradient norm.

    Args:
        loss_fn: Loss function module
        gt: Ground truth image tensor (B, C, H, W)
        distorted: Distorted image tensor (B, C, H, W), requires_grad=True

    Returns:
        Tuple of (loss_value, grad_norm)
    """
    distorted = distorted.clone().requires_grad_(True)

    loss = loss_fn(distorted, gt)
    loss.backward()

    grad_norm = distorted.grad.norm().item()
    loss_value = loss.item()

    return loss_value, grad_norm


@torch.no_grad()
def compute_metric(
    metric_fn: Callable,
    gt: torch.Tensor,
    distorted: torch.Tensor,
) -> float:
    """Compute metric value (no gradient).

    Args:
        metric_fn: Metric function (callable)
        gt: Ground truth image tensor (B, C, H, W)
        distorted: Distorted image tensor (B, C, H, W)

    Returns:
        Metric value
    """
    value = metric_fn(distorted, gt)
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def evaluate_distortion(
    loss_fns: Dict[str, torch.nn.Module],
    gt: torch.Tensor,
    distort_fn: Callable,
    distort_levels: List,
    distort_name: str,
) -> Dict[str, Dict[str, List[float]]]:
    """Evaluate all losses across distortion levels.

    Args:
        loss_fns: Dictionary of loss functions
        gt: Ground truth image (C, H, W) in range [0, 1]
        distort_fn: Distortion function (img, level) -> distorted_img
        distort_levels: List of distortion levels to test
        distort_name: Name of the distortion for logging

    Returns:
        Dictionary mapping loss_name -> {"loss": [...], "grad": [...]}
    """
    results = {name: {"loss": [], "grad": []} for name in loss_fns.keys()}

    gt_batch = gt.unsqueeze(0)  # Add batch dimension

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


def evaluate_distortion_metrics(
    metric_fns: Dict[str, torch.nn.Module],
    gt: torch.Tensor,
    distort_fn: Callable,
    distort_levels: List,
    distort_name: str,
) -> Dict[str, List[float]]:
    """Evaluate all metrics across distortion levels (no gradient).

    Args:
        metric_fns: Dictionary of metric functions
        gt: Ground truth image (C, H, W) in range [0, 1]
        distort_fn: Distortion function (img, level) -> distorted_img
        distort_levels: List of distortion levels to test
        distort_name: Name of the distortion for logging

    Returns:
        Dictionary mapping metric_name -> [values...]
    """
    results = {name: [] for name in metric_fns.keys()}

    gt_batch = gt.unsqueeze(0)  # Add batch dimension

    for level in distort_levels:
        distorted = distort_fn(gt, level)
        distorted_batch = distorted.unsqueeze(0)

        for name, metric_fn in metric_fns.items():
            try:
                metric_val = compute_metric(metric_fn, gt_batch, distorted_batch)
                results[name].append(metric_val)
            except Exception as e:
                print(f"  Warning: Metric {name} failed at level {level}: {e}")
                results[name].append(float("nan"))

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
    """Plot loss and gradient norm results.

    Args:
        results: Dictionary mapping loss_name -> {"loss": [...], "grad": [...]}
        x_values: X-axis values (distortion levels)
        x_label: Label for x-axis
        title: Plot title
        output_path: Path to save the plot
        log_scale_y: Whether to use log scale for y-axis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss values
    for name, data in results.items():
        ax1.plot(x_values, data["loss"], marker="o", label=name, markersize=4)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Loss Value")
    ax1.set_title(f"{title} - Loss Values")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    if log_scale_y:
        ax1.set_yscale("log")

    # Plot gradient norms
    for name, data in results.items():
        ax2.plot(x_values, data["grad"], marker="o", label=name, markersize=4)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title(f"{title} - Gradient Norms")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    if log_scale_y:
        ax2.set_yscale("log")

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
    """Plot a grid comparing all distortions and losses.

    Args:
        all_results: {distortion_name: {loss_name: {"loss": [...], "grad": [...]}}}
        distortion_configs: {distortion_name: (x_values, x_label)}
        output_path: Path to save the plot
    """
    n_distortions = len(all_results)
    fig, axes = plt.subplots(n_distortions, 2, figsize=(14, 4 * n_distortions))

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
        axes[i, 0].legend(loc="upper right", fontsize=6)
        axes[i, 0].grid(True, alpha=0.3)

        # Gradient norms
        for name, data in results.items():
            axes[i, 1].plot(
                x_values, data["grad"], marker="o", label=name, markersize=3
            )
        axes[i, 1].set_xlabel(x_label)
        axes[i, 1].set_ylabel("Gradient Norm")
        axes[i, 1].set_title(f"{dist_name} - Gradient Norms")
        axes[i, 1].legend(loc="upper right", fontsize=6)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pairwise_comparison(
    results: Dict[str, Dict[str, List[float]]],
    x_values: List,
    x_label: str,
    dist_name: str,
    loss_a: str,
    loss_b: str,
    output_path: Path,
):
    """Plot pairwise comparison between two losses for a single distortion.

    Creates two subplots:
    1. Loss A vs Loss B values
    2. Distortion level vs grad norm ratio (A/B)

    Args:
        results: {loss_name: {"loss": [...], "grad": [...]}}
        x_values: Distortion levels
        x_label: Label for x-axis
        dist_name: Name of the distortion
        loss_a: Name of first loss
        loss_b: Name of second loss
        output_path: Path to save the plot
    """
    # Check if both losses exist in results
    if loss_a not in results or loss_b not in results:
        return  # Skip if either loss is missing

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    a_losses = results[loss_a]["loss"]
    b_losses = results[loss_b]["loss"]
    a_grads = np.array(results[loss_a]["grad"])
    b_grads = np.array(results[loss_b]["grad"])

    # Plot 1: Loss A vs Loss B values
    ax1.plot(a_losses, b_losses, marker="o", markersize=4, color="blue")
    ax1.set_xlabel(f"{loss_a} Loss")
    ax1.set_ylabel(f"{loss_b} Loss")
    ax1.set_title(f"{dist_name}: {loss_a} vs {loss_b} Loss Values")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distortion level vs grad norm ratio (A/B)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a_grads / b_grads
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    ax2.plot(x_values, ratio, marker="o", markersize=4, color="green")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(f"Grad Norm Ratio ({loss_a} / {loss_b})")
    ax2.set_title(f"{dist_name}: Gradient Norm Ratio")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics(
    results: Dict[str, List[float]],
    x_values: List,
    x_label: str,
    title: str,
    output_path: Path,
):
    """Plot metric results.

    Args:
        results: Dictionary mapping metric_name -> [values...]
        x_values: X-axis values (distortion levels)
        x_label: Label for x-axis
        title: Plot title
        output_path: Path to save the plot
    """
    # Separate metrics into groups for better visualization
    # Group 1: dB metrics (PSNR, SSIM_dB, MS-SSIM_dB)
    # Group 2: 0-1 metrics (SSIM, MS-SSIM)
    # Group 3: Error metrics (RMSE, MAE, MAPE, LPIPS) - lower is better

    db_metrics = {k: v for k, v in results.items() if "dB" in k or k == "PSNR"}
    similarity_metrics = {k: v for k, v in results.items() if k in ["SSIM", "MS-SSIM"]}
    error_metrics = {
        k: v for k, v in results.items() if k in ["RMSE", "MAE", "MAPE", "LPIPS"]
    }

    # Determine number of subplots needed
    n_plots = sum(
        [
            1 if db_metrics else 0,
            1 if similarity_metrics else 0,
            1 if error_metrics else 0,
        ]
    )
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot dB metrics
    if db_metrics:
        ax = axes[plot_idx]
        for name, data in db_metrics.items():
            ax.plot(x_values, data, marker="o", label=name, markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value (dB)")
        ax.set_title(f"{title} - Quality Metrics (dB)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot similarity metrics (0-1)
    if similarity_metrics:
        ax = axes[plot_idx]
        for name, data in similarity_metrics.items():
            ax.plot(x_values, data, marker="o", label=name, markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Similarity (0-1)")
        ax.set_title(f"{title} - Similarity Metrics")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plot_idx += 1

    # Plot error metrics
    if error_metrics:
        ax = axes[plot_idx]
        for name, data in error_metrics.items():
            ax.plot(x_values, data, marker="o", label=name, markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Error (lower is better)")
        ax.set_title(f"{title} - Error Metrics")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_metrics_grid(
    all_results: Dict[str, Dict[str, List[float]]],
    distortion_configs: Dict[str, Tuple[List, str]],
    output_path: Path,
):
    """Plot a grid comparing all distortions for metrics.

    Args:
        all_results: {distortion_name: {metric_name: [values...]}}
        distortion_configs: {distortion_name: (x_values, x_label)}
        output_path: Path to save the plot
    """
    n_distortions = len(all_results)
    fig, axes = plt.subplots(n_distortions, 2, figsize=(14, 4 * n_distortions))

    if n_distortions == 1:
        axes = axes.reshape(1, -1)

    for i, (dist_name, results) in enumerate(all_results.items()):
        x_values, x_label = distortion_configs[dist_name]

        # Left plot: dB metrics (PSNR, SSIM_dB, MS-SSIM_dB)
        db_metrics = {k: v for k, v in results.items() if "dB" in k or k == "PSNR"}
        for name, data in db_metrics.items():
            axes[i, 0].plot(x_values, data, marker="o", label=name, markersize=3)
        axes[i, 0].set_xlabel(x_label)
        axes[i, 0].set_ylabel("Value (dB)")
        axes[i, 0].set_title(f"{dist_name} - Quality Metrics (dB)")
        axes[i, 0].legend(loc="best", fontsize=6)
        axes[i, 0].grid(True, alpha=0.3)

        # Right plot: similarity/error metrics
        other_metrics = {
            k: v for k, v in results.items() if "dB" not in k and k != "PSNR"
        }
        for name, data in other_metrics.items():
            axes[i, 1].plot(x_values, data, marker="o", label=name, markersize=3)
        axes[i, 1].set_xlabel(x_label)
        axes[i, 1].set_ylabel("Value")
        axes[i, 1].set_title(f"{dist_name} - Other Metrics")
        axes[i, 1].legend(loc="best", fontsize=6)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================


def load_image(path: Path, device: torch.device, max_size: int = 512) -> torch.Tensor:
    """Load image and convert to tensor.

    Args:
        path: Path to image file
        device: Device to load tensor to
        max_size: Maximum size for any dimension (for memory efficiency)

    Returns:
        Image tensor of shape (C, H, W) in range [0, 1]
    """
    img = Image.open(path).convert("RGB")

    # Resize if too large
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
    """Detect and return the best available device.

    Checks for availability in order: CUDA, MPS, XPU, then falls back to CPU.

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Test loss functions under distortions"
    )
    default_device = get_best_device()
    parser.add_argument(
        "--device",
        type=str,
        default=str(default_device),
        help=f"Device to use (default: {default_device}). Options: cuda, mps, xpu, cpu",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to specific image to test"
    )
    parser.add_argument("--no-dinov3", action="store_true", help="Skip DINOv3 losses")
    parser.add_argument("--max-size", type=int, default=256, help="Max image size")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Setup paths
    project_root = Path(__file__).parent.parent
    images_dir = project_root / "images"
    results_dir = project_root / "results"

    # Find images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = find_test_images(images_dir)

    if not image_paths:
        print(f"No test images found in {images_dir}")
        print(
            "Please add images named test*.{{jpg,jpeg,png,webp}} to the images/ directory"
        )
        return

    print(f"Found {len(image_paths)} test image(s)")

    # Load loss functions
    print("\nLoading loss functions...")
    loss_fns = get_basic_losses(device)
    print(f"  Loaded {len(loss_fns)} basic losses")

    perceptual_losses = get_perceptual_losses(device)
    loss_fns.update(perceptual_losses)
    print(f"  Loaded {len(perceptual_losses)} perceptual losses (LPIPS, SSIM)")

    if not args.no_dinov3:
        dinov3_losses = get_dinov3_losses(device)
        loss_fns.update(dinov3_losses)
        print(f"  Loaded {len(dinov3_losses)} DINOv3 losses")

    # Load metrics
    print("\nLoading metrics...")
    metric_fns = get_metrics(device)
    print(f"  Loaded {len(metric_fns)} metrics")

    # Define distortion configurations
    distortions = {
        "JPEG_Compression": {
            "fn": apply_jpeg_compression,
            "levels": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5],
            "x_label": "JPEG Quality",
        },
        "WebP_Compression": {
            "fn": apply_webp_compression,
            "levels": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5],
            "x_label": "WebP Quality",
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

    # Process each image
    for img_path in image_paths:
        img_name = img_path.stem
        print(f"\nProcessing: {img_path.name}")

        try:
            gt = load_image(img_path, device, max_size=args.max_size)
            print(f"  Image size: {gt.shape[1]}x{gt.shape[2]}")
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue

        # Store all results for this image
        all_results = {}
        all_metric_results = {}
        distortion_configs = {}

        # Run each distortion
        for dist_name, dist_config in distortions.items():
            print(f"  Testing: {dist_name}")

            # Evaluate losses (with gradients)
            results = evaluate_distortion(
                loss_fns=loss_fns,
                gt=gt,
                distort_fn=dist_config["fn"],
                distort_levels=dist_config["levels"],
                distort_name=dist_name,
            )

            # Evaluate metrics (no gradients)
            metric_results = evaluate_distortion_metrics(
                metric_fns=metric_fns,
                gt=gt,
                distort_fn=dist_config["fn"],
                distort_levels=dist_config["levels"],
                distort_name=dist_name,
            )

            all_results[dist_name] = results
            all_metric_results[dist_name] = metric_results
            distortion_configs[dist_name] = (
                dist_config["levels"],
                dist_config["x_label"],
            )

            # Save individual loss plot
            output_path = results_dir / img_name / "losses" / f"{dist_name}.png"
            plot_results(
                results=results,
                x_values=dist_config["levels"],
                x_label=dist_config["x_label"],
                title=f"{img_name} - {dist_name}",
                output_path=output_path,
            )

            # Save individual metric plot
            metric_output_path = results_dir / img_name / "metrics" / f"{dist_name}.png"
            plot_metrics(
                results=metric_results,
                x_values=dist_config["levels"],
                x_label=dist_config["x_label"],
                title=f"{img_name} - {dist_name}",
                output_path=metric_output_path,
            )

        # Save combined grid plots
        grid_path = results_dir / img_name / "all_distortions_losses.png"
        plot_comparison_grid(all_results, distortion_configs, grid_path)

        metrics_grid_path = results_dir / img_name / "all_distortions_metrics.png"
        plot_metrics_grid(all_metric_results, distortion_configs, metrics_grid_path)

        # Generate pairwise comparison plots (perceptual vs Charbonnier)
        # Focus on comparing perceptual losses against Charbonnier
        loss_pairs = [
            ("LPIPS_VGG", "Charbonnier"),
            ("LPIPS_Alex", "Charbonnier"),
            ("SSIM", "Charbonnier"),
            ("MS_SSIM", "Charbonnier"),
            ("DINOv3_ViT", "Charbonnier"),
            ("DINOv3_ViT_Gram", "Charbonnier"),
            ("FFT_L1", "Charbonnier"),
            ("PatchFFT_L1_8", "Charbonnier"),
        ]

        print(f"  Generating pairwise comparisons...")
        comparison_dir = results_dir / img_name / "comparisons"
        for dist_name, results in all_results.items():
            x_values, x_label = distortion_configs[dist_name]
            for loss_a, loss_b in loss_pairs:
                output_path = comparison_dir / dist_name / f"{loss_a}-{loss_b}.png"
                plot_pairwise_comparison(
                    results, x_values, x_label, dist_name, loss_a, loss_b, output_path
                )

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
