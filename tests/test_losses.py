"""
Comprehensive test suite for SenseCraft loss functions.

Tests all loss functions for:
- 2D input (B, C, H, W)
- 3D input (B, T, C, H, W)
- Various input sizes including non-divisible dimensions
- Output shape (should be scalar)
- Gradient flow
- SenseCraftLoss compound loss in both modes
"""

import unittest
import torch
import torch.nn as nn

from sensecraft.loss import (
    # General losses
    CharbonnierLoss,
    FFTLoss,
    PatchFFTLoss,
    GaussianNoiseLoss,
    NormType,
    # Edge losses
    SobelEdgeLoss,
    LaplacianEdgeLoss,
    CannyStyleEdgeLoss,
    GradientLoss,
    HighFrequencyLoss,
    MultiScaleGradientLoss,
    StructureTensorLoss,
    # SSIM losses
    SSIMLoss,
    MSSSIMLoss,
    # Perceptual losses
    LPIPS,
    # Video losses
    SSIM3D,
    STSSIM,
    TSSIM,
    FDBLoss,
    TemporalAccelerationLoss,
    TemporalFFTLoss,
    PatchFFT3DLoss,
    TemporalGradientLoss,
    # Compound loss
    SenseCraftLoss,
)
from sensecraft.loss.config import LOSS_REGISTRY, LPIPSConfig, DinoV3LossConfig


class TestLossOutputShape(unittest.TestCase):
    """Test that all losses return scalar output."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        # 2D inputs: (B, C, H, W)
        self.input_2d = torch.randn(2, 3, 64, 64, device=self.device)
        self.target_2d = torch.randn(2, 3, 64, 64, device=self.device)
        # 3D inputs: (B, T, C, H, W)
        self.input_3d = torch.randn(2, 8, 3, 64, 64, device=self.device)
        self.target_3d = torch.randn(2, 8, 3, 64, 64, device=self.device)

    def _test_loss_scalar_output(self, loss_fn, input_tensor, target_tensor, name):
        """Helper to test that loss returns scalar."""
        loss = loss_fn(input_tensor, target_tensor)
        self.assertEqual(
            loss.shape,
            torch.Size([]),
            f"{name} should return scalar, got shape {loss.shape}",
        )
        self.assertFalse(torch.isnan(loss), f"{name} returned NaN")
        self.assertFalse(torch.isinf(loss), f"{name} returned Inf")

    # === General Losses (2D) ===

    def test_charbonnier_2d(self):
        loss_fn = CharbonnierLoss()
        self._test_loss_scalar_output(
            loss_fn, self.input_2d, self.target_2d, "CharbonnierLoss"
        )

    def test_fft_2d(self):
        loss_fn = FFTLoss()
        self._test_loss_scalar_output(loss_fn, self.input_2d, self.target_2d, "FFTLoss")

    def test_patch_fft_2d(self):
        loss_fn = PatchFFTLoss(patch_size=8)
        self._test_loss_scalar_output(
            loss_fn, self.input_2d, self.target_2d, "PatchFFTLoss"
        )

    def test_gaussian_noise_2d(self):
        loss_fn = GaussianNoiseLoss()
        self._test_loss_scalar_output(
            loss_fn, self.input_2d, self.target_2d, "GaussianNoiseLoss"
        )

    # === Edge Losses (2D only) ===

    def test_sobel_2d(self):
        # Sobel needs [0, 1] range
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = SobelEdgeLoss()
        self._test_loss_scalar_output(loss_fn, input_unit, target_unit, "SobelEdgeLoss")

    def test_laplacian_2d(self):
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = LaplacianEdgeLoss()
        self._test_loss_scalar_output(
            loss_fn, input_unit, target_unit, "LaplacianEdgeLoss"
        )

    def test_canny_2d(self):
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = CannyStyleEdgeLoss()
        self._test_loss_scalar_output(
            loss_fn, input_unit, target_unit, "CannyStyleEdgeLoss"
        )

    def test_gradient_2d(self):
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = GradientLoss()
        self._test_loss_scalar_output(loss_fn, input_unit, target_unit, "GradientLoss")

    def test_high_freq_2d(self):
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = HighFrequencyLoss()
        self._test_loss_scalar_output(
            loss_fn, input_unit, target_unit, "HighFrequencyLoss"
        )

    def test_multi_scale_gradient_2d(self):
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = MultiScaleGradientLoss()
        self._test_loss_scalar_output(
            loss_fn, input_unit, target_unit, "MultiScaleGradientLoss"
        )

    def test_structure_tensor_2d(self):
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = StructureTensorLoss()
        self._test_loss_scalar_output(
            loss_fn, input_unit, target_unit, "StructureTensorLoss"
        )

    # === SSIM Losses (2D only) ===

    def test_ssim_2d(self):
        input_unit = (self.input_2d + 1) / 2
        target_unit = (self.target_2d + 1) / 2
        loss_fn = SSIMLoss()
        self._test_loss_scalar_output(loss_fn, input_unit, target_unit, "SSIMLoss")

    def test_msssim_2d(self):
        # MS-SSIM needs larger images (at least 161x161 for 5 scales with win_size=11)
        input_large = torch.randn(2, 3, 176, 176, device=self.device)
        target_large = torch.randn(2, 3, 176, 176, device=self.device)
        input_unit = (input_large + 1) / 2
        target_unit = (target_large + 1) / 2
        loss_fn = MSSSIMLoss()
        self._test_loss_scalar_output(loss_fn, input_unit, target_unit, "MSSSIMLoss")

    # === Perceptual Losses (2D only) ===

    def test_lpips_2d(self):
        loss_fn = LPIPS(net="alex")
        self._test_loss_scalar_output(loss_fn, self.input_2d, self.target_2d, "LPIPS")

    # === Video/3D Losses ===

    def test_ssim3d(self):
        input_unit = (self.input_3d + 1) / 2
        target_unit = (self.target_3d + 1) / 2
        loss_fn = SSIM3D()
        self._test_loss_scalar_output(loss_fn, input_unit, target_unit, "SSIM3D")

    def test_stssim(self):
        input_unit = (self.input_3d + 1) / 2
        target_unit = (self.target_3d + 1) / 2
        loss_fn = STSSIM()
        self._test_loss_scalar_output(loss_fn, input_unit, target_unit, "STSSIM")

    def test_tssim(self):
        input_unit = (self.input_3d + 1) / 2
        target_unit = (self.target_3d + 1) / 2
        loss_fn = TSSIM()
        self._test_loss_scalar_output(loss_fn, input_unit, target_unit, "TSSIM")

    def test_fdb(self):
        loss_fn = FDBLoss()
        self._test_loss_scalar_output(loss_fn, self.input_3d, self.target_3d, "FDBLoss")

    def test_temporal_accel(self):
        loss_fn = TemporalAccelerationLoss()
        self._test_loss_scalar_output(
            loss_fn, self.input_3d, self.target_3d, "TemporalAccelerationLoss"
        )

    def test_temporal_fft(self):
        loss_fn = TemporalFFTLoss()
        self._test_loss_scalar_output(
            loss_fn, self.input_3d, self.target_3d, "TemporalFFTLoss"
        )

    def test_patch_fft_3d(self):
        loss_fn = PatchFFT3DLoss(patch_size=(4, 8, 8))
        self._test_loss_scalar_output(
            loss_fn, self.input_3d, self.target_3d, "PatchFFT3DLoss"
        )

    def test_temporal_gradient(self):
        loss_fn = TemporalGradientLoss()
        self._test_loss_scalar_output(
            loss_fn, self.input_3d, self.target_3d, "TemporalGradientLoss"
        )


class TestLossNonDivisibleDimensions(unittest.TestCase):
    """Test losses with non-divisible input dimensions."""

    def setUp(self):
        self.device = torch.device("cpu")

    def test_patch_fft_2d_odd_dims(self):
        """Test PatchFFTLoss with non-divisible dimensions."""
        loss_fn = PatchFFTLoss(patch_size=8)
        # 67 and 71 are not divisible by 8
        x = torch.randn(2, 3, 67, 71, device=self.device)
        y = torch.randn(2, 3, 67, 71, device=self.device)
        loss = loss_fn(x, y)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_patch_fft_3d_odd_dims(self):
        """Test PatchFFT3DLoss with non-divisible dimensions."""
        loss_fn = PatchFFT3DLoss(patch_size=(4, 8, 8))
        # T=25, H=257, W=255 - all non-divisible
        x = torch.randn(2, 25, 3, 257, 255, device=self.device)
        y = torch.randn(2, 25, 3, 257, 255, device=self.device)
        loss = loss_fn(x, y)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_patch_fft_3d_small_temporal(self):
        """Test PatchFFT3DLoss with T < patch_t."""
        loss_fn = PatchFFT3DLoss(patch_size=(4, 8, 8))
        # T=3 < patch_t=4
        x = torch.randn(2, 3, 3, 64, 64, device=self.device)
        y = torch.randn(2, 3, 3, 64, 64, device=self.device)
        loss = loss_fn(x, y)
        self.assertEqual(loss.shape, torch.Size([]))


class TestLossGradientFlow(unittest.TestCase):
    """Test that gradients flow correctly through losses."""

    def setUp(self):
        self.device = torch.device("cpu")

    def _test_gradient_flow(self, loss_fn, input_tensor, target_tensor, name):
        """Helper to test gradient flow."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        loss = loss_fn(input_tensor, target_tensor)
        loss.backward()
        self.assertIsNotNone(
            input_tensor.grad, f"{name} should have gradients on input"
        )
        self.assertFalse(
            torch.all(input_tensor.grad == 0),
            f"{name} gradients should not be all zero",
        )

    def test_charbonnier_gradient(self):
        x = torch.randn(2, 3, 64, 64, device=self.device)
        y = torch.randn(2, 3, 64, 64, device=self.device)
        self._test_gradient_flow(CharbonnierLoss(), x, y, "CharbonnierLoss")

    def test_fft_gradient(self):
        x = torch.randn(2, 3, 64, 64, device=self.device)
        y = torch.randn(2, 3, 64, 64, device=self.device)
        self._test_gradient_flow(FFTLoss(), x, y, "FFTLoss")

    def test_lpips_gradient(self):
        x = torch.randn(2, 3, 64, 64, device=self.device)
        y = torch.randn(2, 3, 64, 64, device=self.device)
        self._test_gradient_flow(LPIPS(net="alex"), x, y, "LPIPS")

    def test_patch_fft_3d_gradient(self):
        x = torch.randn(2, 8, 3, 64, 64, device=self.device)
        y = torch.randn(2, 8, 3, 64, 64, device=self.device)
        self._test_gradient_flow(
            PatchFFT3DLoss(patch_size=(4, 8, 8)), x, y, "PatchFFT3DLoss"
        )


class TestSenseCraftLoss2D(unittest.TestCase):
    """Test SenseCraftLoss in 2D mode."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.input = torch.randn(2, 3, 64, 64, device=self.device)
        self.target = torch.randn(2, 3, 64, 64, device=self.device)

    def test_basic_losses(self):
        """Test with basic losses (no perceptual)."""
        loss_fn = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                {"sobel": 0.1},
            ],
            input_range=(-1, 1),
            mode="2d",
        )
        result = loss_fn(self.input, self.target, return_dict=True)
        self.assertIn("loss", result)
        self.assertEqual(result["loss"].shape, torch.Size([]))

    def test_return_dict_false(self):
        """Test return_dict=False returns only total loss."""
        loss_fn = SenseCraftLoss(
            [{"charbonnier": 1.0}],
            input_range=(-1, 1),
            mode="2d",
        )
        result = loss_fn(self.input, self.target, return_dict=False)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([]))

    def test_lpips_integration(self):
        """Test with LPIPS loss."""
        loss_fn = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                LPIPSConfig(net="alex", weight=0.1),
            ],
            input_range=(-1, 1),
            mode="2d",
        )
        result = loss_fn(self.input, self.target, return_dict=True)
        self.assertIn("lpips", result)
        self.assertEqual(result["lpips"].shape, torch.Size([]))

    def test_gradient_flow(self):
        """Test gradients flow through compound loss."""
        loss_fn = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                {"fft": 0.1},
            ],
            input_range=(-1, 1),
            mode="2d",
        )
        input_grad = self.input.clone().requires_grad_(True)
        result = loss_fn(input_grad, self.target, return_dict=True)
        result["loss"].backward()
        self.assertIsNotNone(input_grad.grad)

    def test_param_count_multiple_perceptual(self):
        """Test that multiple perceptual losses have correct param count."""
        # LPIPS VGG alone
        lpips_only = SenseCraftLoss(
            [LPIPSConfig(net="vgg", weight=1.0)],
            input_range=(-1, 1),
        )
        lpips_params = sum(p.numel() for p in lpips_only.parameters())

        # Combined with basic loss (shouldn't add params)
        combined = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                LPIPSConfig(net="vgg", weight=0.1),
            ],
            input_range=(-1, 1),
        )
        combined_params = sum(p.numel() for p in combined.parameters())

        # Should be the same since charbonnier has no params
        self.assertEqual(lpips_params, combined_params)


class TestSenseCraftLoss3D(unittest.TestCase):
    """Test SenseCraftLoss in 3D mode."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.input = torch.randn(2, 8, 3, 64, 64, device=self.device)
        self.target = torch.randn(2, 8, 3, 64, 64, device=self.device)

    def test_3d_only_losses(self):
        """Test with 3D-only losses."""
        loss_fn = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                {"temporal_gradient": 0.1},
                {"fdb": 0.1},
            ],
            input_range=(-1, 1),
            mode="3d",
        )
        result = loss_fn(self.input, self.target, return_dict=True)
        self.assertIn("loss", result)
        self.assertEqual(result["loss"].shape, torch.Size([]))

    def test_2d_loss_applied_framewise(self):
        """Test that 2D losses are applied frame-by-frame in 3D mode."""
        loss_fn = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                {"sobel": 0.1},  # 2D-only loss
            ],
            input_range=(-1, 1),
            mode="3d",
        )
        result = loss_fn(self.input, self.target, return_dict=True)
        self.assertIn("sobel", result)
        self.assertEqual(result["sobel"].shape, torch.Size([]))

    def test_patch_fft_3d_integration(self):
        """Test PatchFFT3DLoss in compound loss."""
        loss_fn = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                {"patch_fft_3d": 0.1},
            ],
            input_range=(-1, 1),
            mode="3d",
        )
        result = loss_fn(self.input, self.target, return_dict=True)
        self.assertIn("patch_fft_3d", result)

    def test_non_divisible_3d(self):
        """Test 3D mode with non-divisible dimensions."""
        # T=25, H=67, W=71
        input_odd = torch.randn(2, 25, 3, 67, 71, device=self.device)
        target_odd = torch.randn(2, 25, 3, 67, 71, device=self.device)

        loss_fn = SenseCraftLoss(
            [
                {"charbonnier": 1.0},
                {"patch_fft_3d": 0.1},
            ],
            input_range=(-1, 1),
            mode="3d",
        )
        result = loss_fn(input_odd, target_odd, return_dict=True)
        self.assertEqual(result["loss"].shape, torch.Size([]))


class TestAllRegisteredLosses(unittest.TestCase):
    """Test all losses in the registry work with various input sizes."""

    def setUp(self):
        self.device = torch.device("cpu")
        # Test sizes: standard + edge cases (non-divisible dimensions)
        # 2D: (H, W) - need at least 161 for MS-SSIM with 5 scales
        self.sizes_2d = [
            (256, 256),  # Standard power of 2
            (256 + 17, 256 + 23),  # Non-divisible by common patch sizes
            (192 + 5, 192 + 11),  # Another edge case
        ]
        # 3D: (T, H, W)
        self.sizes_3d = [
            (24, 64, 64),  # Standard
            (24 + 5, 64 + 7, 64 + 11),  # Non-divisible T, H, W
            (16 + 3, 96 + 13, 96 + 17),  # Another edge case
        ]

    def test_all_2d_losses(self):
        """Test all 2D-compatible losses with various sizes."""
        for H, W in self.sizes_2d:
            input_2d = torch.randn(2, 3, H, W, device=self.device)
            target_2d = torch.randn(2, 3, H, W, device=self.device)
            input_unit = torch.rand(2, 3, H, W, device=self.device)
            target_unit = torch.rand(2, 3, H, W, device=self.device)

            for name, info in LOSS_REGISTRY.items():
                if info.is_3d_only:
                    continue

                with self.subTest(loss=name, size=(H, W)):
                    try:
                        loss_fn = info.loss_class()

                        if info.required_range.value == "unit":
                            loss = loss_fn(input_unit, target_unit)
                        else:
                            loss = loss_fn(input_2d, target_2d)

                        self.assertEqual(
                            loss.shape,
                            torch.Size([]),
                            f"{name} should return scalar for size {(H, W)}",
                        )
                    except Exception as e:
                        self.fail(f"{name} failed with size {(H, W)}: {e}")

    def test_all_3d_losses(self):
        """Test all 3D-compatible losses with various sizes."""
        for T, H, W in self.sizes_3d:
            input_3d = torch.randn(2, T, 3, H, W, device=self.device)
            target_3d = torch.randn(2, T, 3, H, W, device=self.device)
            input_unit = torch.rand(2, T, 3, H, W, device=self.device)
            target_unit = torch.rand(2, T, 3, H, W, device=self.device)

            for name, info in LOSS_REGISTRY.items():
                if info.is_2d_only:
                    continue

                with self.subTest(loss=name, size=(T, H, W)):
                    try:
                        loss_fn = info.loss_class()

                        if info.required_range.value == "unit":
                            loss = loss_fn(input_unit, target_unit)
                        else:
                            loss = loss_fn(input_3d, target_3d)

                        self.assertEqual(
                            loss.shape,
                            torch.Size([]),
                            f"{name} should return scalar for size {(T, H, W)}",
                        )
                    except Exception as e:
                        self.fail(f"{name} failed with size {(T, H, W)}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
