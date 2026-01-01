# SenseCraft: Unified Perceptual Feature Loss Framework

A PyTorch framework providing various perceptual loss functions for image processing tasks including super-resolution, image restoration, style transfer, and more.

## Features

- **Compound Loss System**: `SenseCraftLoss` for easy multi-loss configuration with automatic value range handling
- **Multiple Perceptual Loss Types**: ConvNext, DINOv3 (ConvNext & ViT), LPIPS
- **Frequency Domain Losses**: FFT and Patch-FFT losses with configurable normalization
- **Edge & Structure Losses**: Sobel, Laplacian, Gradient, Structure Tensor losses
- **Video/3D Losses**: Temporal SSIM, 3D SSIM, Frame Difference losses
- **General Losses**: Charbonnier, SSIM, MS-SSIM, Gaussian noise-aware losses
- **Self-Supervised Features**: DINOv3 models provide better generalization than supervised features
- **Flexible Configuration**: Layer selection, normalization options, Gram matrix support
- **Gradient Flow**: Proper gradient handling for training neural networks

## Installation

```bash
# Basic installation
pip install sensecraft

# With DINOv3 support (requires transformers >= 4.56.0)
pip install sensecraft[dinov3]

# Full installation with all optional dependencies
pip install sensecraft[full]
```

For development:
```bash
git clone https://github.com/KohakuBlueleaf/SenseCraft.git
cd SenseCraft
pip install -e ".[full]"
```

## Quick Start

### Using SenseCraftLoss (Recommended)

The easiest way to use multiple losses is through `SenseCraftLoss`:

```python
import torch
from sensecraft.loss import SenseCraftLoss

# Simple configuration with {name: weight} format
loss_fn = SenseCraftLoss(
    loss_config=[
        {"charbonnier": 1.0},    # Main reconstruction loss
        {"sobel": 0.1},          # Edge preservation
        {"ssim": 0.05},          # Structural similarity
        {"lpips": 0.1},          # Perceptual quality
    ],
    input_range=(-1, 1),  # Your data's value range
    mode="2d",            # "2d" for images, "3d" for video
)

# Create sample images
predicted = torch.randn(1, 3, 256, 256)
target = torch.randn(1, 3, 256, 256)

# Compute all losses at once
losses = loss_fn(predicted, target)
print(losses["loss"])        # Total weighted loss (for backprop)
print(losses["charbonnier"]) # Individual loss values
print(losses["sobel"])
```

### Typed Configs for Complex Losses

For losses with many parameters, use typed config classes:

```python
from sensecraft.loss import (
    SenseCraftLoss,
    DinoV3LossConfig,
    LPIPSConfig,
    PatchFFTConfig,
)

loss_fn = SenseCraftLoss(
    loss_config=[
        {"charbonnier": 1.0},
        DinoV3LossConfig(
            weight=0.1,
            model_type="small_plus",
            loss_layer=-4,
            use_gram=False,
            use_norm=True,
        ),
        LPIPSConfig(weight=0.05, net="alex"),
        PatchFFTConfig(weight=0.05, patch_size=16),
    ],
    input_range=(-1, 1),
)
```

### Monitoring Losses (weight=0)

Losses with `weight=0` are computed under `torch.no_grad()` for efficiency but still returned:

```python
loss_fn = SenseCraftLoss(
    loss_config=[
        {"charbonnier": 1.0},
        {"ssim": 0.0},      # Computed for logging, not in gradient
        {"ms_ssim": 0.0},   # Same here
    ],
    input_range=(0, 1),
)

losses = loss_fn(pred, target)
# losses["loss"] only includes charbonnier
# losses["ssim"] and losses["ms_ssim"] available for logging
```

### 3D/Video Mode

For video data (B, T, C, H, W), use `mode="3d"`. 2D-only losses are applied frame-by-frame:

```python
loss_fn = SenseCraftLoss(
    loss_config=[
        {"charbonnier": 1.0},
        {"sobel": 0.1},           # Applied per-frame
        {"temporal_gradient": 0.1},  # 3D-specific loss
        {"stssim": 0.05},         # Spatio-temporal SSIM
    ],
    input_range=(-1, 1),
    mode="3d",
)

video_pred = torch.randn(2, 8, 3, 64, 64)  # (B, T, C, H, W)
video_target = torch.randn(2, 8, 3, 64, 64)
losses = loss_fn(video_pred, video_target)
```

### Automatic Value Range Handling

`SenseCraftLoss` automatically converts inputs to the required range for each loss:
- **UNIT [0, 1]**: SSIM, MS-SSIM, edge losses (Sobel, Laplacian, etc.)
- **SYMMETRIC [-1, 1]**: Perceptual losses (LPIPS, DINOv3, ConvNext)
- **ANY**: Charbonnier, MSE, L1, FFT losses

You just specify your data's range via `input_range`, and the conversion is handled automatically.

### Using Individual Losses

You can also use losses directly:

```python
from sensecraft.loss import (
    ConvNextDinoV3PerceptualLoss,
    CharbonnierLoss,
    PatchFFTLoss,
    SSIMLoss,
)

# Perceptual loss with DINOv3 ConvNext
loss_fn = ConvNextDinoV3PerceptualLoss(
    loss_layer=-1,      # Use last layer
    use_norm=True,      # L2 normalize features
    use_gram=False,     # Direct MSE loss
    input_range=(0, 1), # Input value range
)
loss = loss_fn(predicted, target)

# Charbonnier loss (smooth L1)
charbonnier = CharbonnierLoss(eps=1e-6)
loss = charbonnier(predicted, target)

# Patch FFT loss
fft_loss = PatchFFTLoss(patch_size=8, loss_type="l1")
loss = fft_loss(predicted, target)

# SSIM loss (returns 1 - SSIM for minimization)
ssim_loss = SSIMLoss(data_range=1.0)
loss = ssim_loss(predicted, target)
```

## Available Losses

### Registered Loss Names

All losses can be used with `SenseCraftLoss` via their registered names:

| Category | Names | Value Range |
|----------|-------|-------------|
| **Basic** | `mse`, `l1`, `charbonnier`, `smooth_l1` | ANY |
| **FFT** | `fft`, `patch_fft`, `gaussian_noise` | ANY |
| **Edge** | `sobel`, `laplacian`, `canny`, `gradient`, `high_freq`, `multi_scale_gradient`, `structure_tensor` | UNIT [0,1] |
| **SSIM** | `ssim`, `ms_ssim` | UNIT [0,1] |
| **Perceptual** | `lpips`, `convnext`, `dino_convnext`, `dino_vit` | SYMMETRIC [-1,1] |
| **Video/3D** | `ssim3d`, `stssim`, `tssim`, `fdb`, `temporal_accel`, `temporal_fft`, `patch_fft_3d`, `temporal_gradient` | varies |

### Config Classes

For complex losses, use typed config classes:

| Config Class | Loss | Key Parameters |
|--------------|------|----------------|
| `GeneralConfig` | Any loss | `name`, `weight`, `**kwargs` |
| `DinoV3LossConfig` | `dino_vit` | `model_type`, `loss_layer`, `use_gram`, `use_norm` |
| `ConvNextDinoV3LossConfig` | `dino_convnext` | `model_type`, `loss_layer`, `use_gram`, `use_norm` |
| `LPIPSConfig` | `lpips` | `net` ("vgg", "alex", "squeeze") |
| `SSIMConfig` | `ssim` | `win_size`, `win_sigma` |
| `MSSSIMConfig` | `ms_ssim` | `win_size`, `win_sigma`, `weights` |
| `PatchFFTConfig` | `patch_fft` | `patch_size`, `loss_type`, `norm_type`, `use_phase` |

## Loss Functions

### Perceptual Losses

#### ConvNextPerceptualLoss

Uses ImageNet-pretrained ConvNext models from torchvision.

```python
from sensecraft.loss import ConvNextPerceptualLoss
from sensecraft.loss.convnext import ConvNextType

loss_fn = ConvNextPerceptualLoss(
    model_type=ConvNextType.SMALL,      # TINY, SMALL, BASE, LARGE
    feature_layers=[2, 4, 8, 14],       # Layer indices to extract
    use_gram=False,                      # True for style/texture loss
    input_range=(-1, 1),                # Expected input range
    layer_weight_decay=1.0,             # Weight decay for layers
)
```

#### ConvNextDinoV3PerceptualLoss

Uses DINOv3 self-supervised ConvNext models (requires `transformers >= 4.56.0`).

```python
from sensecraft.loss import ConvNextDinoV3PerceptualLoss
from sensecraft.loss.convnext_dinov3 import ConvNextType

# Single-layer mode (recommended)
loss_fn = ConvNextDinoV3PerceptualLoss(
    model_type=ConvNextType.SMALL,
    loss_layer=-1,                      # -1 for last layer
    use_norm=True,                      # L2 normalize features
    use_gram=False,                     # MSE on normalized features
    input_range=(0, 1),
)

# Multi-layer mode
loss_fn = ConvNextDinoV3PerceptualLoss(
    model_type=ConvNextType.SMALL,
    feature_layers=[2, 4, 8, 14, 20],   # Multiple layers
    feature_weights=[1.0] * 5,          # Optional explicit weights
    use_gram=True,                      # Gram matrix loss
)
```

#### ViTDinoV3PerceptualLoss

Uses DINOv3 Vision Transformer models for sequence-based perceptual loss.

> **Note**: When using `use_norm=True` and `use_gram=False`, this is equivalent to
> the DINO perceptual loss described in [NA-VAE](https://na-vae.github.io/dino_perceptual/).

```python
from sensecraft.loss import ViTDinoV3PerceptualLoss
from sensecraft.loss.gram_dinov3 import ModelType

loss_fn = ViTDinoV3PerceptualLoss(
    model_type=ModelType.SMALL_PLUS,    # SMALL, SMALL_PLUS, BASE, LARGE
    use_norm=True,                       # L2 normalize features
    use_gram=True,                       # Gram matrix for texture
    loss_layer=-4,                       # Layer index (supports negative, default -4)
    input_range=(0, 1),
)
```

#### LPIPS

Learned Perceptual Image Patch Similarity from Zhang et al.

```python
from sensecraft.loss import LPIPS

loss_fn = LPIPS(
    net_type="vgg",     # "vgg", "alex", "squeeze"
    version="0.1",      # "0.0" or "0.1"
)
```

### Frequency Domain Losses

#### FFTLoss

Global FFT loss operating on the entire image.

```python
from sensecraft.loss import FFTLoss, NormType

loss_fn = FFTLoss(
    loss_type="mse",                # "mse", "l1", "charbonnier"
    norm_type=NormType.LOG1P,       # NONE, L2, LOG, LOG1P
    use_amplitude=True,             # Loss on magnitude
    use_phase=False,                # Loss on phase
    phase_weight=0.1,               # Weight for phase loss
)
```

#### PatchFFTLoss

Patch-based FFT loss for local frequency analysis.

```python
from sensecraft.loss import PatchFFTLoss, NormType

loss_fn = PatchFFTLoss(
    patch_size=8,                   # 8x8 or 16x16 patches
    loss_type="l1",                 # "mse", "l1", "charbonnier"
    norm_type=NormType.LOG1P,       # Normalization for FFT magnitudes
    use_amplitude=True,
    use_phase=False,
)
```

**Normalization Types:**
- `NormType.NONE`: No normalization (may produce very large values)
- `NormType.L2`: L2 normalization per patch
- `NormType.LOG`: `log(x + eps)`
- `NormType.LOG1P`: `log(1 + x)` (recommended)

### General Losses

#### CharbonnierLoss

Smooth approximation to L1 loss, differentiable everywhere.

```python
from sensecraft.loss import CharbonnierLoss

loss_fn = CharbonnierLoss(
    eps=1e-6,           # Smoothness parameter
    reduction="mean",   # "none", "mean", "sum"
)
```

The Charbonnier loss is defined as: `L(x, y) = sqrt((x - y)^2 + eps^2)`

#### GaussianNoiseLoss

Noise-aware loss for denoising tasks.

```python
from sensecraft.loss import GaussianNoiseLoss

loss_fn = GaussianNoiseLoss(
    sigma=0.1,                      # Fixed noise sigma
    sigma_range=(0.01, 0.2),        # Or random range
    loss_type="l1",                 # "mse", "l1", "charbonnier"
)

# Can add noise to target during training
loss = loss_fn(predicted, target, add_noise_to_target=True)
```

### Edge and Structure Losses

Losses for preserving edges and structural details:

```python
from sensecraft.loss import (
    SobelEdgeLoss,
    LaplacianEdgeLoss,
    GradientLoss,
    MultiScaleGradientLoss,
    StructureTensorLoss,
)

# Sobel edge loss
sobel = SobelEdgeLoss(loss_type="l1")  # "l1", "mse", "charbonnier"
loss = sobel(predicted, target)

# Multi-scale gradient for coarse-to-fine edge matching
msg = MultiScaleGradientLoss(num_scales=3)
loss = msg(predicted, target)

# Structure tensor for texture/orientation
st = StructureTensorLoss(window_size=5, sigma=1.0)
loss = st(predicted, target)
```

### Video/3D Losses

Losses for temporal consistency in video:

```python
from sensecraft.loss import (
    STSSIM,
    TSSIM,
    TemporalGradientLoss,
    TemporalFFTLoss,
    FDBLoss,
)

# Spatio-temporal SSIM
stssim = STSSIM(spatial_weight=0.5, temporal_weight=0.5)
loss = stssim(video_pred, video_target)  # (B, T, C, H, W)

# Temporal gradient loss (frame differences)
tg = TemporalGradientLoss(loss_type="l1")
loss = tg(video_pred, video_target)

# Temporal FFT for frequency consistency over time
tfft = TemporalFFTLoss()
loss = tfft(video_pred, video_target)
```

## Comparison: When to Use Which Loss

| Loss Type | Best For | Characteristics |
|-----------|----------|-----------------|
| **MSE** | Pixel-accurate reconstruction | Simple, can be blurry |
| **L1** | General reconstruction | Less blurry than MSE |
| **Charbonnier** | Restoration tasks | Smooth L1, robust to outliers |
| **SSIM/MS-SSIM** | Structural quality | Window-based, perceptually motivated |
| **LPIPS** | Perceptual similarity | Learned, correlates with human perception |
| **ConvNext** | Content matching | Multi-scale features |
| **DINOv3 ConvNext** | Semantic matching | Self-supervised, better generalization |
| **DINOv3 ViT** | Global structure | Transformer-based, sequence features |
| **FFT** | Frequency content | Captures textures, patterns |
| **PatchFFT** | Local frequency | Better for high-frequency details |
| **Sobel/Gradient** | Edge preservation | First-order derivatives |
| **Laplacian** | Fine details | Second-order derivatives |
| **Structure Tensor** | Texture orientation | Captures local anisotropy |
| **Temporal losses** | Video consistency | Frame-to-frame coherence |

## Example: Testing Distortions

The package includes an example script to compare loss behavior under various distortions:

```bash
# Run the distortion test
python examples/test_distortions.py --device cuda

# Test specific image
python examples/test_distortions.py --image path/to/image.png

# Skip DINOv3 losses (faster, no transformers needed)
python examples/test_distortions.py --no-dinov3
```

This generates plots in `results/` showing:
- Loss values vs distortion level
- Gradient norms vs distortion level

For distortion types:
- JPEG compression (quality 5-100)
- WebP compression (quality 5-100)
- Gaussian noise (sigma 0-0.3)
- Gaussian blur (sigma 0-7)

## API Reference

### Common Parameters

All perceptual losses share these parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_range` | `Tuple[float, float]` | Expected (min, max) of input values |
| `use_gram` | `bool` | Use Gram matrix (L1) vs direct features (MSE) |
| `use_norm` | `bool` | L2 normalize features before loss |

### DINOv3 Models

Available model types for DINOv3 losses:

**ConvNext:**
- `ConvNextType.TINY`: ~28M params
- `ConvNextType.SMALL`: ~50M params (recommended)
- `ConvNextType.BASE`: ~89M params
- `ConvNextType.LARGE`: ~198M params

**ViT:**
- `ModelType.SMALL`: ~22M params
- `ModelType.SMALL_PLUS`: Larger hidden dim
- `ModelType.BASE`: ~86M params
- `ModelType.LARGE`: ~307M params

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchvision
- numpy

**Optional:**
- transformers >= 4.56.0 (for DINOv3 losses)
- scikit-image (for color space conversions)
- matplotlib (for example scripts)
- Pillow (for example scripts)

## License

Apache License 2.0

## Citation

If you use SenseCraft in your research, please cite:

```bibtex
@software{sensecraft,
  author = {Shih-Ying Yeh (KohakuBlueleaf)},
  title = {SenseCraft: Unified Perceptual Feature Loss Framework},
  url = {https://github.com/KohakuBlueleaf/SenseCraft},
  year = {2024}
}
```

## Acknowledgments

- [ConvNext Perceptual Loss](https://github.com/sypsyp97/convnext_perceptual_loss) by sypsyp97
- [LPIPS](https://richzhang.github.io/PerceptualSimilarity/) by Richard Zhang et al.
- [DINOv3](https://github.com/facebookresearch/dinov3) by Facebook Research
