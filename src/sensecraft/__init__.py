"""
SenseCraft: Unified Perceptual Feature Loss Framework.

A PyTorch framework providing various perceptual loss functions and
evaluation metrics for image/video processing tasks.

Submodules:
- sensecraft.loss: Loss functions for training
- sensecraft.metrics: Evaluation metrics
"""

from . import loss
from . import metrics

__all__ = ["loss", "metrics"]
