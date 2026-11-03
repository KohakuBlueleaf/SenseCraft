# Change Log

## 2026-11-03 update to 0.4.1

### Highlights

* A perceptual term's gradient no longer loses its direction when the prediction is near the
  target. The layer that reads the image keeps the image's low bits, and the DINOv3 and ConvNeXt
  residual streams, norms, activations and layer scales run in fp32. LPIPS VGG in fp16 at noise
  1e-4 goes from 0.55 to 0.93 cosine to fp64. DINOv3 ViT-B in fp16 at noise 1e-3 goes from 0.06
  to 0.89. Costs: 2.2 ms on LPIPS VGG and under 1 ms on the others, at batch 32 and 256 px.
* Plain terms run in strict fp32: autocast off, fp32 inputs, no TF32.

### Full change log

#### New Features

* `measure/s7_near.py`: stage S7, the gradient near the target in fp16, bf16, TF32 and fp32,
  drawn to `measure/figures/s7_near.svg`.
* `sensecraft.precision.strict_fp32`, and `kernels.split_layer` for a first layer that keeps its
  input's low bits.

#### Changes

* Every pixel term and `fdb`, `temporal_accel` and `temporal_gradient` are registered `fp32=True`.
  `fp32=True` now also turns off TF32 for convs and matmuls.
* Feature nets return their features in at least fp32.

#### Fixes

* LPIPS unit vectors take the eps inside the root. An all-zero feature vector gave a NaN gradient
  in any dtype.
* `charbonnier` and the inner Charbonnier distance could give `sqrt(0)` under fp16 inputs, where
  `eps ** 2` underflows. They now always run in fp32.

## 2026-11-02 update to 0.4.0

A rewrite: own feature nets, a compose layer that returns the gradient at the prediction, fp32
distances, fused kernels, and a measured precision and speed record for every term. The API
breaks against 0.3.

### Highlights

* `Loss([Term(...)])` with `prepare(device, dtype, memory_format, compile)` and
  `value_and_seed(fake, real)`, which returns the gradient at the prediction with no graph kept and
  no host reads.
* Own VGG16, AlexNet, SqueezeNet, ConvNeXt and DINOv3 ViT and ConvNeXt feature nets; weights are
  converted once to safetensors, keyed by upstream revision. `torchvision` and `transformers` are no
  longer needed at runtime.
* LPIPS VGG, batch 32 at 256 px, bf16, compiled: 29.5 to 23.3 ms and 2.8 to 1.8 GB peak, in either
  layout. DINOv3 ViT-B: 21.5 to 20.6 ms.
* LPIPS distances run in fp32: SqueezeNet LPIPS in fp16 goes from 28% value error and 0.61 gradient
  cosine to 0.02% and 0.99.

### Full change log

#### New Features

* `sensecraft.Loss`, `Term`, `Out`, `prepare` with compile scopes `off`, `term` and `loss`.
* `register(name, Range, dims, fp32)` for new terms; unknown names suggest the closest ones.
* A fused frozen conv, bias and ReLU with an input-only backward; index-free 2x2 pooling.
* `measure/`: the precision chain (S0 to S6) and the speed chain (T1 to T6). `checks/`: behavior
  checks run by hand.

#### Changes

* Python 3.13 and torch 2.9 or newer. Dependencies: torch, safetensors, pytorch-msssim;
  `sensecraft[dinov3]` adds huggingface_hub.
* `SenseCraftLoss` and the `*Config` classes are replaced by `Loss` and `Term`; `weight=0` terms by
  `monitor=True`.
* `reduction` and `size_average` are gone; terms return the mean.
* Metrics return device tensors and take `per_sample` and `db`.

#### Fixes

* `tssim` reads [-1, 1] clips, which its formulas assume.
* `ssim3d` and `stssim` read [0, 1] clips instead of guessing the range from the input's minimum.
* The DINOv3 ConvNeXt config path works; it imported a missing name and passed an unknown keyword.
* Five host syncs per call are gone (`gaussian_noise`, `ssim3d`, `stssim`, `multi_scale_gradient`,
  the compound total).
