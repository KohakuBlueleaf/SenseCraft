"""
Test suite to verify SenseCraft LPIPS implementation matches the official lpips package.

This test ensures:
1. Forward pass produces identical loss values
2. Backward pass produces identical gradients
3. Both implementations agree across different network types (vgg, alex, squeeze)
"""

import unittest

import torch
import numpy as np

# Check if official lpips is available
try:
    import lpips as official_lpips

    HAS_OFFICIAL_LPIPS = True
except ImportError:
    HAS_OFFICIAL_LPIPS = False

from sensecraft.loss import LPIPS


@unittest.skipUnless(HAS_OFFICIAL_LPIPS, "Official lpips package not installed")
class TestLPIPSCompatibility(unittest.TestCase):
    """Test that SenseCraft LPIPS matches official lpips package."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use deterministic inputs for reproducibility
        torch.manual_seed(42)
        self.input1 = torch.randn(2, 3, 64, 64, device=self.device)
        self.input2 = torch.randn(2, 3, 64, 64, device=self.device)

        # Clamp to [-1, 1] as LPIPS expects
        self.input1 = self.input1.clamp(-1, 1)
        self.input2 = self.input2.clamp(-1, 1)

    def _test_network_type(self, net_type: str, atol: float = 1e-5, rtol: float = 1e-4):
        """Test a specific network type for loss and gradient compatibility."""
        # Create both implementations
        official = official_lpips.LPIPS(net=net_type, verbose=False).to(self.device)
        ours = LPIPS(net=net_type).to(self.device)

        # Both should be in eval mode
        official.eval()
        ours.eval()

        # Test forward pass
        with torch.no_grad():
            official_loss = official(self.input1, self.input2)
            our_loss = ours(self.input1, self.input2)

        # Official returns [B, 1, 1, 1], ours returns scalar
        official_loss_scalar = official_loss.mean()

        self.assertTrue(
            torch.allclose(official_loss_scalar, our_loss, atol=atol, rtol=rtol),
            f"[{net_type}] Loss mismatch: official={official_loss_scalar.item():.6f}, "
            f"ours={our_loss.item():.6f}, diff={abs(official_loss_scalar.item() - our_loss.item()):.6f}",
        )

        # Test gradient flow
        input1_official = self.input1.clone().requires_grad_(True)
        input1_ours = self.input1.clone().requires_grad_(True)

        official_loss = official(input1_official, self.input2).mean()
        our_loss = ours(input1_ours, self.input2)

        official_loss.backward()
        our_loss.backward()

        # Compare gradients
        grad_diff = (input1_official.grad - input1_ours.grad).abs()
        max_grad_diff = grad_diff.max().item()
        mean_grad_diff = grad_diff.mean().item()

        # Gradients should be very close
        self.assertTrue(
            torch.allclose(
                input1_official.grad, input1_ours.grad, atol=atol, rtol=rtol
            ),
            f"[{net_type}] Gradient mismatch: max_diff={max_grad_diff:.6f}, "
            f"mean_diff={mean_grad_diff:.6f}",
        )

        return {
            "loss_diff": abs(official_loss.item() - our_loss.item()),
            "max_grad_diff": max_grad_diff,
            "mean_grad_diff": mean_grad_diff,
        }

    def test_vgg_compatibility(self):
        """Test VGG network compatibility."""
        results = self._test_network_type("vgg")
        print(
            f"\nVGG: loss_diff={results['loss_diff']:.2e}, "
            f"max_grad_diff={results['max_grad_diff']:.2e}"
        )

    def test_alex_compatibility(self):
        """Test AlexNet network compatibility."""
        results = self._test_network_type("alex")
        print(
            f"\nAlexNet: loss_diff={results['loss_diff']:.2e}, "
            f"max_grad_diff={results['max_grad_diff']:.2e}"
        )

    def test_squeeze_compatibility(self):
        """Test SqueezeNet network compatibility."""
        results = self._test_network_type("squeeze")
        print(
            f"\nSqueezeNet: loss_diff={results['loss_diff']:.2e}, "
            f"max_grad_diff={results['max_grad_diff']:.2e}"
        )

    def test_batch_consistency(self):
        """Test that batch processing gives consistent results."""
        ours = LPIPS(net="alex").to(self.device).eval()

        # Single sample
        with torch.no_grad():
            loss_single_0 = ours(self.input1[0:1], self.input2[0:1])
            loss_single_1 = ours(self.input1[1:2], self.input2[1:2])
            loss_batch = ours(self.input1, self.input2)

        # Batch mean should equal mean of singles
        expected_mean = (loss_single_0 + loss_single_1) / 2

        self.assertTrue(
            torch.allclose(loss_batch, expected_mean, atol=1e-5),
            f"Batch consistency failed: batch={loss_batch.item():.6f}, "
            f"expected={expected_mean.item():.6f}",
        )

    def test_identical_images(self):
        """Test that identical images produce zero loss."""
        ours = LPIPS(net="alex").to(self.device).eval()

        with torch.no_grad():
            loss = ours(self.input1, self.input1)

        self.assertTrue(
            loss.item() < 1e-6,
            f"Identical images should have ~0 loss, got {loss.item():.6f}",
        )

    def test_symmetry(self):
        """Test that LPIPS(a, b) == LPIPS(b, a)."""
        ours = LPIPS(net="alex").to(self.device).eval()

        with torch.no_grad():
            loss_ab = ours(self.input1, self.input2)
            loss_ba = ours(self.input2, self.input1)

        self.assertTrue(
            torch.allclose(loss_ab, loss_ba, atol=1e-6),
            f"Symmetry failed: LPIPS(a,b)={loss_ab.item():.6f}, "
            f"LPIPS(b,a)={loss_ba.item():.6f}",
        )


class TestLPIPSBasic(unittest.TestCase):
    """Basic LPIPS tests that don't require official package."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

    def test_output_shape(self):
        """Test that LPIPS returns scalar output."""
        lpips_fn = LPIPS(net="alex").to(self.device).eval()

        input1 = torch.randn(2, 3, 64, 64, device=self.device)
        input2 = torch.randn(2, 3, 64, 64, device=self.device)

        with torch.no_grad():
            loss = lpips_fn(input1, input2)

        self.assertEqual(
            loss.shape, torch.Size([]), f"Expected scalar, got {loss.shape}"
        )

    def test_gradient_flow(self):
        """Test that gradients flow through LPIPS."""
        lpips_fn = LPIPS(net="alex").to(self.device)

        input1 = torch.randn(1, 3, 64, 64, device=self.device, requires_grad=True)
        input2 = torch.randn(1, 3, 64, 64, device=self.device)

        loss = lpips_fn(input1, input2)
        loss.backward()

        self.assertIsNotNone(input1.grad)
        self.assertFalse(torch.isnan(input1.grad).any())
        self.assertFalse(torch.isinf(input1.grad).any())

    def test_different_sizes(self):
        """Test LPIPS with various input sizes."""
        lpips_fn = LPIPS(net="alex").to(self.device).eval()

        sizes = [(64, 64), (128, 128), (256, 256), (64, 128), (100, 100)]

        for h, w in sizes:
            input1 = torch.randn(1, 3, h, w, device=self.device)
            input2 = torch.randn(1, 3, h, w, device=self.device)

            with torch.no_grad():
                loss = lpips_fn(input1, input2)

            self.assertEqual(loss.shape, torch.Size([]), f"Size {h}x{w} failed")
            self.assertFalse(torch.isnan(loss), f"NaN for size {h}x{w}")

    def test_all_network_types(self):
        """Test all supported network types."""
        for net_type in ["vgg", "alex", "squeeze"]:
            lpips_fn = LPIPS(net=net_type).to(self.device).eval()

            input1 = torch.randn(1, 3, 64, 64, device=self.device)
            input2 = torch.randn(1, 3, 64, 64, device=self.device)

            with torch.no_grad():
                loss = lpips_fn(input1, input2)

            self.assertFalse(torch.isnan(loss), f"NaN for {net_type}")
            self.assertTrue(loss.item() >= 0, f"Negative loss for {net_type}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
