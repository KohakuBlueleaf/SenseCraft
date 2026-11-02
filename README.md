# SenseCraft

Perceptual and signal losses for image and video models, with our own feature nets, a compose layer
that returns the gradient at the prediction directly, and a measured precision and speed record for
every term.

Work in progress.

## Install

```bash
pip install sensecraft            # torch, safetensors, pytorch-msssim
pip install "sensecraft[dinov3]"  # adds huggingface_hub for the DINOv3 terms
```

Python 3.13. Pretrained weights are converted once to safetensors under `~/.cache/sensecraft`
(`SENSECRAFT_CACHE` moves it). VGG, AlexNet and SqueezeNet come from the torchvision URLs; the LPIPS
heads ship in the package. DINOv3 checkpoints are gated on Hugging Face, so their first use needs a
token.

## Use

```python
import torch
from sensecraft import Loss, Term

loss = Loss(
    [
        Term("mse"),
        Term("lpips", 0.5, net="vgg"),
        Term("dino_vit", 250, model="base", use_gram=False),
        Term("ssim", monitor=True),          # logged, not in the total
    ],
    dims=2,                                  # 3 for (B, T, C, H, W) video
    input_range=(-1, 1),
).prepare("cuda", torch.bfloat16, torch.channels_last, compile="term")

out = loss(fake, real)                       # out.total, out.terms
out = loss.value_and_seed(fake, real)        # also out.seed = d total / d fake, no graph kept
```

- Each term reads its images in the range it was trained for; `Loss` converts from `input_range`.
- Image terms run per frame on video; video terms see the clip.
- `prepare` puts the feature nets in the compute dtype and layout and compiles by scope: `off`,
  `term` or `loss`. Distances stay in fp32.
- Terms that need fp32 (FFT, SSIM) run with autocast off.
- Nothing reads the host: totals and terms stay on the device.

Metrics on the same units, without gradient:

```python
from sensecraft.metrics import lpips, ms_ssim, psnr, ssim

psnr(fake, real, data_range=2.0)
```

## Terms

| Group | Names |
| --- | --- |
| Perceptual | `lpips` (vgg, alex, squeeze), `dino_vit`, `dino_convnext`, `convnext` |
| Pixel | `mse`, `l1`, `huber`, `smooth_l1`, `rmse`, `mape`, `smape`, `log_cosh`, `quantile`, `cauchy`, `geman_mcclure`, `welsch`, `tukey`, `wing`, `charbonnier`, `gaussian_noise` |
| Frequency | `fft`, `patch_fft` |
| Edge | `sobel`, `laplacian`, `canny`, `gradient`, `high_freq`, `multi_scale_gradient`, `structure_tensor` |
| Structure | `ssim`, `ms_ssim` |
| Video | `ssim3d`, `stssim`, `tssim`, `fdb`, `temporal_accel`, `temporal_fft`, `patch_fft_3d`, `temporal_gradient` |

A new term is an `nn.Module` with `forward(input, target) -> scalar`, registered with
`@register(name, Range.UNIT, dims="2d", fp32=False)`.

## Layout

```text
src/sensecraft/
  nets/        feature nets, features only: VGG16, AlexNet, SqueezeNet, ConvNeXt, DINOv3 ViT and ConvNeXt
  weights/     upstream sources, conversion to cached safetensors, the LPIPS heads
  distance/    LPIPS and feature distances, in fp32
  perceptual/  terms: a net and a distance
  pixel/       pixel regression terms
  signal/      FFT, edge, SSIM and video terms
  kernels/     fused ops with plain PyTorch fallbacks
  compose.py   Term, Loss, value_and_seed
  prepare.py   dtype, layout and compile scope
  metrics.py   metrics on device tensors
measure/       the precision and speed chain, see measure/README.md
checks/        behavior checks run by hand, see checks/README.md
```

## Precision

Every perceptual term matches upstream at fp64 (S1, error at the level of fp64 rounding), and every
fixed term matches 0.3.13 exactly at fp64 (S5). Against fp64, at the same dtype as upstream:

| Term | dtype | Value gradient cosine, ours | upstream |
| --- | --- | --- | --- |
| LPIPS SqueezeNet | fp16 | 0.992 | 0.607 |
| LPIPS VGG | bf16 | 0.988 | 0.985 |
| LPIPS VGG | fp16 | 0.995 | 0.995 |

The LPIPS distance runs in fp32: in fp16 the channel sum of squares overflows, and in bf16 the
difference of two close unit vectors cancels. Value error drops 3 to 5x in bf16 and from 28% to
0.02% for SqueezeNet in fp16. DINOv3 rows are measured on random stand-in weights until the gated
checkpoints are available.

Changed on purpose against 0.3.13: `tssim` reads [-1, 1] clips, which its formulas assume; `ssim3d`
and `stssim` read [0, 1] clips instead of guessing the range from the input's minimum.

## Speed

Batch 32, 256 px, one RTX 5090, bf16, channels-last, compiled per term, forward and the gradient at
the prediction:

| Term | 0.3.13 | now |
| --- | --- | --- |
| LPIPS VGG | 29.5 ms | 25.8 ms |
| DINOv3 ViT-B | 21.5 ms | 21.5 ms |
| ConvNeXt small | | 8.6 ms |
| DINOv3 ConvNeXt tiny | | 8.1 ms |

LPIPS runs every conv, bias and ReLU as one cuDNN op with an input-only backward, since feature net
weights never take gradients. The full numbers and per-layer floor ratios come from `measure/`.

## License

Apache-2.0. The LPIPS heads and the pretrained nets keep their own licenses; see `LICENSE-NOTICES`.
