from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectrum_qwen21.chebyshev import (  # noqa: E402
    HistoryWeightChebyshevForecaster,
    chebyshev_basis,
    normalize_step_position,
)


class ChebyshevBasisTest(unittest.TestCase):
    def test_basis_shape_and_values(self) -> None:
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        basis = chebyshev_basis(x, 4)
        self.assertEqual(tuple(basis.shape), (4, 5))
        # T_0 = 1, T_1 = x, T_2 = 2x^2 - 1
        self.assertTrue(torch.allclose(basis[:, 0], torch.ones(4)))
        self.assertTrue(torch.allclose(basis[:, 1], x))
        self.assertTrue(torch.allclose(basis[:, 2], 2 * x * x - 1))

    def test_normalize_step_position(self) -> None:
        self.assertAlmostEqual(normalize_step_position(0, 20), -1.0)
        self.assertAlmostEqual(normalize_step_position(19, 20), 1.0)
        self.assertAlmostEqual(normalize_step_position(10, 20), 2 * 10 / 19 - 1, places=5)
        self.assertEqual(normalize_step_position(0, 1), 0.0)
        # clamped for out-of-range indices
        self.assertEqual(normalize_step_position(25, 20), 1.0)


class ForecasterMathTest(unittest.TestCase):
    def _make(self, degree=4, lam=0.1, history=8, blend=1.0):
        return HistoryWeightChebyshevForecaster(
            degree=degree, ridge_lambda=lam, max_history=history, blend_weight=blend
        )

    def test_matches_direct_ridge_solve(self) -> None:
        """History-weight prediction must equal the paper's Eq. (12)-(14)."""
        torch.manual_seed(3)
        degree, lam, k = 4, 0.1, 7
        coords = [-1.0, -0.7, -0.4, -0.1, 0.3, 0.6, 0.9]
        feature = torch.randn(2, 11, 5, dtype=torch.float32)

        forecaster = self._make(degree=degree, lam=lam, history=k, blend=1.0)
        for c in coords:
            forecaster.update(c, feature + c)  # distinct values per anchor

        design = chebyshev_basis(torch.tensor(coords), degree)  # (K, P)
        target_matrix = torch.stack([feature + c for c in coords]).reshape(k, -1)
        gram = design.t() @ design + lam * torch.eye(degree + 1)
        coeffs = torch.linalg.solve(gram, design.t() @ target_matrix)

        for probe in (-0.85, 0.0, 0.45, 0.99):
            phi = chebyshev_basis(torch.tensor([probe]), degree)
            expected = (phi @ coeffs).reshape(feature.shape)
            predicted = forecaster.predict(probe)
            self.assertTrue(
                torch.allclose(predicted, expected, atol=2e-3, rtol=1e-3),
                f"mismatch at probe={probe}",
            )

    def test_exact_recovery_of_low_degree_polynomial(self) -> None:
        degree = 4
        forecaster = self._make(degree=degree, lam=0.0, history=8)
        # h(coord) = 3 + 2*coord - coord^2 (degree 2 < 4 -> exact fit)
        for c in [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]:
            value = 3.0 + 2.0 * c - c * c
            forecaster.update(c, torch.full((1, 4, 3), value))
        for probe in (-0.9, -0.35, 0.11, 0.77, 0.98):
            expected = 3.0 + 2.0 * probe - probe * probe
            predicted = forecaster.predict(probe)
            self.assertTrue(
                torch.allclose(predicted, torch.full((1, 4, 3), expected), atol=1e-4)
            )

    def test_blend_zero_is_pure_linear_extrapolation(self) -> None:
        forecaster = self._make(degree=4, lam=0.1, history=6, blend=0.0)
        # five collinear anchors; v(coord) = 1 + 2*(coord + 1)
        for c in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            forecaster.update(c, torch.full((2, 3), 1.0 + 2.0 * (c + 1.0)))
        for probe in (0.25, 0.9):
            expected = 1.0 + 2.0 * (probe + 1.0)
            predicted = forecaster.predict(probe)
            self.assertTrue(
                torch.allclose(predicted, torch.full((2, 3), expected), atol=1e-4)
            )

    def test_blend_mixes_spectral_and_linear(self) -> None:
        torch.manual_seed(7)
        blend = 0.5
        forecaster_s = self._make(blend=1.0)
        forecaster_l = self._make(blend=0.0)
        forecaster_m = self._make(blend=blend)
        for c in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            f = torch.randn(1, 6)
            forecaster_s.update(c, f)
            forecaster_l.update(c, f.clone())
            forecaster_m.update(c, f.clone())
        probe = 0.75
        expected = blend * forecaster_s.predict(probe) + (1 - blend) * forecaster_l.predict(probe)
        self.assertTrue(torch.allclose(forecaster_m.predict(probe), expected, atol=1e-5))

    def test_history_trim_is_fifo(self) -> None:
        forecaster = self._make(degree=2, history=3)
        for i in range(6):
            forecaster.update(float(i) / 5.0, torch.full((1, 2), float(i)))
        self.assertEqual(forecaster.history_size, 3)
        self.assertTrue(torch.allclose(forecaster._anchors[0].feature, torch.tensor([[3.0, 3.0]])))

    def test_not_ready_raises(self) -> None:
        forecaster = self._make(degree=4, history=8)
        forecaster.update(0.0, torch.zeros(1, 2))
        with self.assertRaises(RuntimeError):
            forecaster.predict(0.5)

    def test_shape_change_raises_and_reset_recovers(self) -> None:
        forecaster = self._make(degree=2, history=5)
        forecaster.update(0.0, torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            forecaster.update(0.2, torch.zeros(1, 3))
        forecaster.reset()
        forecaster.update(0.0, torch.zeros(1, 3))
        self.assertEqual(tuple(forecaster.feature_shape), (1, 3))

    def test_dtype_preserved_and_sane(self) -> None:
        forecaster = self._make(degree=1, history=5, blend=0.0)
        # values that extrapolate beyond fp16 range must be clamped
        forecaster.update(-1.0, torch.full((1, 4), 60000.0, dtype=torch.float16))
        forecaster.update(-0.8, torch.full((1, 4), -60000.0, dtype=torch.float16))
        out = forecaster.predict(0.5)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.isfinite(out.float()).all())
        self.assertLessEqual(out.float().abs().max().item(), 65504.0 + 1e-3)

    def test_release_features_keeps_signature(self) -> None:
        forecaster = self._make(degree=2, history=5)
        forecaster.update(0.0, torch.zeros(2, 4))
        forecaster.release_features()
        self.assertEqual(forecaster.history_size, 0)
        self.assertEqual(tuple(forecaster.feature_shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
