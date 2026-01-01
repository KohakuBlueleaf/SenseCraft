# SenseCraft: Unified Perceptual Feature Loss Framework

A PyTorch framework providing various perceptual loss functions and evaluation metrics for image processing tasks including super-resolution, image restoration, style transfer, and more.

## Features

- **Compound Loss System**: `SenseCraftLoss` for easy multi-loss configuration with automatic value range handling
- **Evaluation Metrics**: PSNR, SSIM, MS-SSIM, LPIPS with functional API and dB scale support
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

## Evaluation Metrics

SenseCraft provides evaluation metrics separate from loss functions. Metrics return actual quality values (not loss formulations).

### Functional API (Recommended)

The functional API is simple and auto-manages resources like LPIPS models:

```python
from sensecraft.metrics import psnr, ssim, ms_ssim, rmse, mae, mape, lpips

# Compute metrics (tensors should be in [0, 1] range for most metrics)
pred = torch.rand(1, 3, 256, 256)
target = torch.rand(1, 3, 256, 256)

# PSNR (always in dB, higher is better)
print(f"PSNR: {psnr(pred, target):.2f} dB")

# SSIM (0-1, higher is better)
print(f"SSIM: {ssim(pred, target):.4f}")

# SSIM in dB scale (higher is better)
print(f"SSIM: {ssim(pred, target, as_db=True):.2f} dB")

# MS-SSIM (requires ~160x160 minimum image size)
print(f"MS-SSIM: {ms_ssim(pred, target):.4f}")

# Error metrics (lower is better)
print(f"RMSE: {rmse(pred, target):.4f}")
print(f"MAE: {mae(pred, target):.4f}")
print(f"MAPE: {mape(pred, target):.4f}")  # Mean Absolute Percentage Error

# LPIPS (lower is better, expects [-1, 1] range)
pred_sym = pred * 2 - 1
target_sym = target * 2 - 1
print(f"LPIPS: {lpips(pred_sym, target_sym):.4f}")
```

### Data Range

For PSNR and SSIM metrics, `data_range` specifies the dynamic range:
- `data_range=1.0` for images in `[0, 1]`
- `data_range=2.0` for images in `[-1, 1]`
- `data_range=255.0` for images in `[0, 255]`

```python
# For [-1, 1] normalized images
psnr_val = psnr(pred, target, data_range=2.0)
ssim_val = ssim(pred, target, data_range=2.0)
```

### dB Scale for SSIM/MS-SSIM

SSIM and MS-SSIM can be returned in dB scale using `as_db=True`:

```python
# dB scale: -10 * log10(1 - ssim_value)
# Gives values like 15-25 dB for typical quality ranges
ssim_db = ssim(pred, target, as_db=True)  # e.g., 18.5 dB
ms_ssim_db = ms_ssim(pred, target, as_db=True)  # e.g., 22.1 dB
```

### LPIPS Auto-Caching

The `lpips()` function automatically caches models and moves them to the correct device:

```python
# First call loads the model
val1 = lpips(pred1, target1, net="alex")  # Loads AlexNet model

# Subsequent calls reuse the cached model
val2 = lpips(pred2, target2, net="alex")  # Uses cached model

# Different network types are cached separately
val3 = lpips(pred3, target3, net="vgg")  # Loads and caches VGG model
```

### Class-based API

For repeated use with the same settings:

```python
from sensecraft.metrics import PSNR, SSIM, MSSSIM, RMSE, MAE, MAPE

psnr_metric = PSNR(data_range=1.0)
ssim_metric = SSIM(data_range=1.0, as_db=True)

# Use like nn.Module
psnr_val = psnr_metric(pred, target)
ssim_val = ssim_metric(pred, target)
```

### Available Metrics

| Function | Class | Description | Range | Better |
|----------|-------|-------------|-------|--------|
| `psnr()` | `PSNR` | Peak Signal-to-Noise Ratio | dB | Higher |
| `ssim()` | `SSIM` | Structural Similarity | 0-1 or dB | Higher |
| `ms_ssim()` | `MSSSIM` | Multi-Scale SSIM | 0-1 or dB | Higher |
| `rmse()` | `RMSE` | Root Mean Squared Error | 0+ | Lower |
| `mae()` | `MAE` | Mean Absolute Error | 0+ | Lower |
| `mape()` | `MAPE` | Mean Absolute Percentage Error | 0+ | Lower |
| `lpips()` | `LPIPSMetric` | Learned Perceptual Similarity | 0+ | Lower |

## Available Losses

### Registered Loss Names

All losses can be used with `SenseCraftLoss` via their registered names:

| Category | Names | Value Range |
|----------|-------|-------------|
| **Basic** | `mse`, `l1`, `huber`, `charbonnier` | ANY |
| **Regression** | `rmse`, `mape`, `smape`, `log_cosh`, `quantile` | ANY |
| **Robust** | `cauchy`, `geman_mcclure`, `welsch`, `tukey`, `wing` | ANY |
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

Global FFT loss operating on the entire image. Computes loss on real and imaginary parts separately.

```python
from sensecraft.loss import FFTLoss
from sensecraft.loss.general import NormType

loss_fn = FFTLoss(
    loss_type="mse",                # "mse", "l1", "charbonnier"
    norm_type=NormType.LOG1P,       # NONE, L2, LOG, LOG1P
)
```

#### PatchFFTLoss

Patch-based FFT loss for local frequency analysis.

```python
from sensecraft.loss import PatchFFTLoss
from sensecraft.loss.general import NormType

loss_fn = PatchFFTLoss(
    patch_size=8,                   # 8x8 or 16x16 patches
    loss_type="l1",                 # "mse", "l1", "charbonnier"
    norm_type=NormType.LOG1P,       # Normalization for FFT real/imag parts
)
```

**Normalization Types:**
- `NormType.NONE`: No normalization (may produce very large values)
- `NormType.L2`: L2 normalization per patch
- `NormType.LOG`: Sign-preserving `sign(x) * log(|x| + eps)`
- `NormType.LOG1P`: Sign-preserving `sign(x) * log(1 + |x|)` (recommended)

> **Note:** FFT losses automatically cast inputs to fp32 to avoid ComplexHalf warnings with mixed precision training.

### General Losses

#### Standard Regression Losses

```python
from sensecraft.loss import (
    MSELoss,           # Mean Squared Error (L2)
    L1Loss,            # Mean Absolute Error (L1)
    HuberLoss,         # Smooth L1 / Huber loss
    CharbonnierLoss,   # Differentiable L1 variant
    RMSELoss,          # Root Mean Squared Error
    MAPELoss,          # Mean Absolute Percentage Error
    SMAPELoss,         # Symmetric MAPE (more stable near zero)
    LogCoshLoss,       # Log-Cosh (smooth L1/L2 hybrid)
    QuantileLoss,      # Pinball loss for quantile regression
)

# Huber loss (smooth L1)
huber = HuberLoss(delta=1.0)  # Quadratic when |x-y| < delta, linear otherwise

# Charbonnier loss (differentiable L1)
charbonnier = CharbonnierLoss(eps=1e-3)  # L(x,y) = sqrt((x-y)^2 + eps^2)

# Log-Cosh (behaves like L2 for small errors, L1 for large)
log_cosh = LogCoshLoss()

# Quantile loss (q=0.5 is equivalent to L1/median regression)
quantile = QuantileLoss(quantile=0.5)
```

#### Robust Regression Losses (Outlier-resistant)

These losses are designed to be robust against outliers:

```python
from sensecraft.loss import (
    CauchyLoss,        # Cauchy/Lorentzian loss
    GemanMcClureLoss,  # Bounded loss, very robust
    WelschLoss,        # Bounded with exponential decay
    TukeyBiweightLoss, # Completely ignores large outliers
    WingLoss,          # High precision for small errors (facial landmarks)
)

# Cauchy loss: log(1 + ((x-y)/scale)^2)
cauchy = CauchyLoss(scale=1.0)

# Geman-McClure: bounded loss that saturates for large errors
geman = GemanMcClureLoss(scale=1.0)

# Welsch: 1 - exp(-0.5 * ((x-y)/scale)^2)
welsch = WelschLoss(scale=1.0)

# Tukey: completely ignores outliers beyond threshold c
tukey = TukeyBiweightLoss(c=4.685)

# Wing loss: high precision for small errors, linear for large
wing = WingLoss(width=5.0, curvature=0.5)
```

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
| **Huber** | Balanced reconstruction | Quadratic near zero, linear for large errors |
| **Charbonnier** | Restoration tasks | Smooth L1, differentiable everywhere |
| **Log-Cosh** | Smooth regression | Like L2 for small, L1 for large errors |
| **Cauchy/Welsch/Tukey** | Noisy data | Robust to outliers, bounded influence |
| **MAPE/SMAPE** | Relative error | Scale-invariant, good for varied magnitudes |
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

The package includes an example script to compare loss and metric behavior under various distortions:

```bash
# Run the distortion test
python examples/test_distortions.py --device cuda

# Test specific image
python examples/test_distortions.py --image path/to/image.png

# Skip DINOv3 losses (faster, no transformers needed)
python examples/test_distortions.py --no-dinov3
```

This generates plots in `results/{image_name}/`:

**Loss plots** (`losses/`):
- Loss values vs distortion level
- Gradient norms vs distortion level

**Metric plots** (`metrics/`):
- PSNR, SSIM, MS-SSIM (dB scale) vs distortion level
- SSIM, MS-SSIM (0-1 scale) vs distortion level
- RMSE, MAE, MAPE, LPIPS vs distortion level

**Combined plots**:
- `all_distortions_losses.png` - Grid of all loss plots
- `all_distortions_metrics.png` - Grid of all metric plots

Distortion types tested:
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
- pytorch-msssim (for SSIM/MS-SSIM metrics and losses)
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
