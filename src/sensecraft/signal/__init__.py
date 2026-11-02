"""Signal-processing losses: frequency, edge, SSIM and video terms."""

from .edge import (
    CannyStyleEdgeLoss,
    GradientLoss,
    HighFrequencyLoss,
    LaplacianEdgeLoss,
    MultiScaleGradientLoss,
    SobelEdgeLoss,
    StructureTensorLoss,
)
from .fft import FFTLoss, NormType, PatchFFTLoss
from .ssim import MSSSIMLoss, SSIMLoss
from .video import (
    SSIM3D,
    STSSIM,
    TSSIM,
    FDBLoss,
    PatchFFT3DLoss,
    TemporalAccelerationLoss,
    TemporalFFTLoss,
    TemporalGradientLoss,
)

__all__ = [
    "SSIM3D",
    "STSSIM",
    "TSSIM",
    "CannyStyleEdgeLoss",
    "FDBLoss",
    "FFTLoss",
    "GradientLoss",
    "HighFrequencyLoss",
    "LaplacianEdgeLoss",
    "MSSSIMLoss",
    "MultiScaleGradientLoss",
    "NormType",
    "PatchFFT3DLoss",
    "PatchFFTLoss",
    "SSIMLoss",
    "SobelEdgeLoss",
    "StructureTensorLoss",
    "TemporalAccelerationLoss",
    "TemporalFFTLoss",
    "TemporalGradientLoss",
]
