# Config and registry
from .config import (
    LOSS_REGISTRY,
    ValueRange,
    LossConfig,
    GeneralConfig,
    DinoV3LossConfig,
    ConvNextDinoV3LossConfig,
    LPIPSConfig,
    SSIMConfig,
    MSSSIMConfig,
    PatchFFTConfig,
    register_loss,
)

# Perceptual losses
from .convnext import ConvNextPerceptualLoss
from .convnext_dinov3 import ConvNextDinoV3PerceptualLoss
from .gram_dinov3 import ViTDinoV3PerceptualLoss
from .lpips import LPIPS

# General losses
from .general import (
    CharbonnierLoss,
    PatchFFTLoss,
    FFTLoss,
    NormType,
    GaussianNoiseLoss,
)

# Edge losses
from .edge import (
    SobelEdgeLoss,
    LaplacianEdgeLoss,
    CannyStyleEdgeLoss,
    GradientLoss,
    HighFrequencyLoss,
    MultiScaleGradientLoss,
    StructureTensorLoss,
)

# SSIM losses
from .ssim import SSIMLoss, MSSSIMLoss

# Video/3D losses
from .video import (
    SSIM3D,
    STSSIM,
    TSSIM,
    FDBLoss,
    TemporalAccelerationLoss,
    TemporalFFTLoss,
    PatchFFT3DLoss,
    TemporalGradientLoss,
)

# Compound loss
from .sensecraft import SenseCraftLoss
