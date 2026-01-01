"""
Demo training script using SenseCraftLoss with torchvision datasets.

This example shows how to:
1. Use SenseCraftLoss with multiple loss components
2. Train a simple autoencoder/denoiser network
3. Compare different loss configurations

Usage:
    python demo_training.py --dataset flowers102 --epochs 10
    python demo_training.py --dataset food101 --epochs 5 --batch-size 16
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from sensecraft.loss import SenseCraftLoss


def get_best_device() -> torch.device:
    """Get best available device (cuda > mps > xpu > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")


class SimpleAutoencoder(nn.Module):
    """Simple convolutional autoencoder for demo purposes."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class DenoisingAutoencoder(nn.Module):
    """Autoencoder trained to denoise images."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Simple U-Net style architecture
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))

        # Decoder with skip connections
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return d1


def get_dataset(
    name: str, root: str = "./data", image_size: int = 128
) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """Get train and test datasets.

    Args:
        name: Dataset name ('flowers102', 'food101', 'cifar10')
        root: Data root directory
        image_size: Target image size

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # -> [-1, 1]
        ]
    )

    if name == "flowers102":
        train_dataset = datasets.Flowers102(
            root=root, split="train", download=True, transform=transform
        )
        test_dataset = datasets.Flowers102(
            root=root, split="test", download=True, transform=transform
        )
    elif name == "food101":
        train_dataset = datasets.Food101(
            root=root, split="train", download=True, transform=transform
        )
        test_dataset = datasets.Food101(
            root=root, split="test", download=True, transform=transform
        )
    elif name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_dataset, test_dataset


def add_noise(x: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to tensor."""
    noise = torch.randn_like(x) * noise_level
    return torch.clamp(x + noise, -1, 1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: SenseCraftLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_level: float = 0.1,
) -> dict[str, float]:
    """Train for one epoch.

    Returns:
        Dict of average loss values
    """
    model.train()
    total_losses = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)

        # Add noise for denoising task
        noisy_images = add_noise(images, noise_level)

        # Forward pass
        outputs = model(noisy_images)

        # Compute loss
        losses = loss_fn(outputs, images)

        # Backward pass
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: SenseCraftLoss,
    device: torch.device,
    noise_level: float = 0.1,
) -> dict[str, float]:
    """Evaluate model.

    Returns:
        Dict of average loss values
    """
    model.eval()
    total_losses = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            noisy_images = add_noise(images, noise_level)

            outputs = model(noisy_images)
            losses = loss_fn(outputs, images)

            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1

    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Demo training with SenseCraftLoss")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["flowers102", "food101", "cifar10"],
        help="Dataset to use",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=64, help="Image size")
    parser.add_argument("--noise-level", type=float, default=0.15, help="Noise level")
    parser.add_argument("--data-root", type=str, default="./data", help="Data root")
    parser.add_argument(
        "--loss-config",
        type=str,
        default="perceptual",
        choices=["simple", "perceptual", "full"],
        help="Loss configuration preset",
    )
    args = parser.parse_args()

    # Device
    device = get_best_device()
    print(f"Using device: {device}")

    # Dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset = get_dataset(
        args.dataset, args.data_root, args.image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Model
    model = DenoisingAutoencoder(in_channels=3, base_channels=32).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss configuration presets using new {name: weight} format
    if args.loss_config == "simple":
        # Simple reconstruction loss
        loss_config = [
            {"charbonnier": 1.0},
        ]
    elif args.loss_config == "perceptual":
        # Perceptual loss (if LPIPS available)
        loss_config = [
            {"charbonnier": 1.0},
            {"sobel": 0.1},
        ]
        # Try to add LPIPS if available
        try:
            from lpips import LPIPS as _

            loss_config.append({"lpips": 0.05})
        except ImportError:
            print("LPIPS not available, skipping...")
    elif args.loss_config == "full":
        # Full loss with multiple components
        # For losses with extra kwargs, use GeneralConfig
        from sensecraft.loss import GeneralConfig, PatchFFTConfig

        loss_config = [
            {"charbonnier": 1.0},
            {"sobel": 0.1},
            PatchFFTConfig(weight=0.05, patch_size=8),
        ]
        # Try to add SSIM if available
        try:
            from pytorch_msssim import SSIM as _

            loss_config.append({"ssim": 0.1})
        except ImportError:
            print("pytorch_msssim not available, skipping SSIM...")

    print(f"\nLoss configuration ({args.loss_config}):")
    for cfg in loss_config:
        if isinstance(cfg, dict):
            name, weight = next(iter(cfg.items()))
            print(f"  - {name}: weight={weight}")
        else:
            print(f"  - {cfg.get_name()}: weight={cfg.weight}")

    # Create SenseCraftLoss
    loss_fn = SenseCraftLoss(
        loss_config=loss_config,
        input_range=(-1, 1),  # Our data is normalized to [-1, 1]
        mode="2d",
        return_dict=True,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    print("\nStarting training...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, device, args.noise_level
        )

        # Evaluate
        eval_losses = evaluate(model, test_loader, loss_fn, device, args.noise_level)

        # Update scheduler
        scheduler.step()

        # Print losses
        print(f"Train losses:")
        for k, v in train_losses.items():
            print(f"  {k}: {v:.6f}")

        print(f"Eval losses:")
        for k, v in eval_losses.items():
            print(f"  {k}: {v:.6f}")

        # Save best model
        if eval_losses["loss"] < best_loss:
            best_loss = eval_losses["loss"]
            print(f"New best loss: {best_loss:.6f}")

    print("\nTraining complete!")
    print(f"Best eval loss: {best_loss:.6f}")

    # Print available losses for reference
    print("\n" + "=" * 50)
    print("Available losses in SenseCraftLoss:")
    print(SenseCraftLoss.loss_info_str())


if __name__ == "__main__":
    main()
