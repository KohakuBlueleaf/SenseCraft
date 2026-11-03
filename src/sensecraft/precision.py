"""Strict fp32 for the terms registered with `fp32=True`.

TF32 multiplies with a 10-bit mantissa, the same as fp16, so an fp32 term on TF32 convs or matmuls
loses what fp32 is asked for. Inside `strict_fp32` cuDNN convs and matmuls run in IEEE fp32; the
previous settings come back on exit. It uses the per-backend `fp32_precision` flags only, so it
never mixes with the legacy `allow_tf32` flags.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def strict_fp32() -> Iterator[None]:
    """IEEE fp32 convs and matmuls inside, the previous settings after."""
    conv = torch.backends.cudnn.conv
    matmul = torch.backends.cuda.matmul
    old = conv.fp32_precision, matmul.fp32_precision
    conv.fp32_precision = matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        conv.fp32_precision, matmul.fp32_precision = old
