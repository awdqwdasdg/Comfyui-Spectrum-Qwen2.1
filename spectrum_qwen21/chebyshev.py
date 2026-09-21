from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Element count per chunk when computing the weighted prediction sum.
# 2**22 = ~4.2M elements -> ~16 MB fp32 temporaries per anchor slice.
_PREDICT_CHUNK = 1 << 22


def normalize_step_position(step_index: int, total_steps: int) -> float:
    """Map a solver step index onto the Chebyshev support [-1, 1].

    This generalizes the paper's timestep projection g(t) = 2t - 1
    (arXiv 2603.01623, Eq. 8) to an arbitrary number of solver steps,
    matching the official implementation which indexes diffusion time by
    step position.
    """
    if total_steps <= 1:
        return 0.0
    coord = 2.0 * float(step_index) / float(total_steps - 1) - 1.0
    return max(-1.0, min(1.0, coord))


def chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate Chebyshev T-polynomials T_0..T_degree at points x.

    Args:
        x: tensor of shape (K,) with coordinates in [-1, 1].
        degree: polynomial degree M >= 0.

    Returns:
        Float32 tensor of shape (K, degree + 1); column m holds T_m(x).
    """
    x = x.to(dtype=torch.float32)
    columns = [torch.ones_like(x)]
    if degree >= 1:
        columns.append(x)
    for _ in range(2, degree + 1):
        columns.append(2.0 * x * columns[-1] - columns[-2])
    return torch.stack(columns[: degree + 1], dim=-1)


@dataclass
class _Anchor:
    """One real (non-forecast) observation of the target feature."""

    coord: float
    feature: torch.Tensor


class HistoryWeightChebyshevForecaster:
    """Ridge-regularized Chebyshev forecaster in history-weight form.

    Implements the paper's estimator (Eq. 12-14):

        C = (Phi^T Phi + lambda I)^{-1} Phi^T H        (coefficients)
        h(t*) = phi(g(t*)) C                            (forecast)

    but never materializes the (M+1) x F coefficient matrix. Instead the
    prediction is expanded as a weighted sum over the K stored anchors,

        h(t*) = sum_k w_k h_k,   w = phi(tau*) (Phi^T Phi + lambda I)^{-1} Phi^T,

    which is algebraically identical while scaling memory with the anchor
    count instead of the feature size. This matters for Qwen-Image-2.1
    where F = batch * H * W * 4096 can exceed 2.5e8 channels.

    Optionally blends the spectral weights with a two-point (discrete
    Taylor order-1) extrapolation on the most recent anchors, mirroring
    the official repository's post-publication robustness extension:

        w = blend * w_spectral + (1 - blend) * w_linear

    All fitting math runs in float32 on CPU regardless of where the
    features are stored.
    """

    def __init__(
        self,
        degree: int,
        ridge_lambda: float,
        max_history: int,
        blend_weight: float = 1.0,
    ) -> None:
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = max(2, int(max_history))
        self.blend_weight = min(1.0, max(0.0, float(blend_weight)))
        self._anchors: list[_Anchor] = []
        self._feature_shape: Optional[torch.Size] = None
        self._feature_dtype: Optional[torch.dtype] = None
        self._storage_device: Optional[torch.device] = None
        self._chol: Optional[torch.Tensor] = None

    # -- basic state ---------------------------------------------------

    def reset(self) -> None:
        self._anchors.clear()
        self._feature_shape = None
        self._feature_dtype = None
        self._storage_device = None
        self._chol = None

    def release_features(self) -> None:
        """Drop stored features (keeps nothing else)."""
        self._anchors.clear()
        self._chol = None

    def ready(self) -> bool:
        return (
            self._feature_shape is not None
            and len(self._anchors) >= self.degree + 1
        )

    @property
    def history_size(self) -> int:
        return len(self._anchors)

    @property
    def feature_shape(self) -> Optional[torch.Size]:
        return self._feature_shape

    @property
    def feature_dtype(self) -> Optional[torch.dtype]:
        return self._feature_dtype

    @property
    def storage_device(self) -> Optional[torch.device]:
        return self._storage_device

    # -- recording -----------------------------------------------------

    def update(self, coord: float, feature: torch.Tensor) -> None:
        """Record one real observation. The tensor is stored as given.

        Raises ValueError when shape or dtype differ from previous
        anchors; callers are expected to reset() on geometry changes.
        """
        if self._feature_shape is None:
            self._feature_shape = torch.Size(feature.shape)
            self._feature_dtype = feature.dtype
            self._storage_device = feature.device
        elif tuple(feature.shape) != tuple(self._feature_shape):
            raise ValueError(
                f"Spectrum feature shape changed from "
                f"{tuple(self._feature_shape)} to {tuple(feature.shape)}."
            )
        elif feature.dtype != self._feature_dtype:
            raise ValueError(
                f"Spectrum feature dtype changed from {self._feature_dtype} "
                f"to {feature.dtype}."
            )

        self._anchors.append(_Anchor(float(coord), feature))
        if len(self._anchors) > self.max_history:
            self._anchors.pop(0)
        # Cached factorization belongs to a specific history window.
        self._chol = None

    # -- fitting -------------------------------------------------------

    def _design(self) -> torch.Tensor:
        coords = torch.tensor(
            [a.coord for a in self._anchors], dtype=torch.float32
        )
        return chebyshev_basis(coords, self.degree)  # (K, P)

    def _factorize(self) -> torch.Tensor:
        """Cholesky factor of (Phi^T Phi + lambda I), shape (P, P)."""
        if self._chol is not None:
            return self._chol
        design = self._design()
        gram = design.t().matmul(design)
        p = gram.shape[0]
        eye = torch.eye(p, dtype=torch.float32)
        lhs = gram + self.ridge_lambda * eye
        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            diag_scale = max(float(gram.diagonal().abs().mean().item()), 1.0)
            jittered = False
            for multiplier in (1e-8, 1e-7, 1e-6, 1e-5):
                try:
                    chol = torch.linalg.cholesky(
                        lhs + (diag_scale * multiplier) * eye
                    )
                    jittered = True
                    break
                except RuntimeError:
                    continue
            if not jittered:
                raise
        self._chol = chol
        return chol

    def _spectral_weights(self, coord: float) -> torch.Tensor:
        """w = phi(tau*) (Phi^T Phi + lambda I)^{-1} Phi^T, shape (K,)."""
        design = self._design()  # (K, P)
        chol = self._factorize()  # (P, P)
        phi = chebyshev_basis(torch.tensor([float(coord)]), self.degree)  # (1, P)
        solved = torch.cholesky_solve(design.t(), chol)  # (P, K)
        return (phi.matmul(solved)).reshape(-1)  # (K,)

    def _linear_weights(self, coord: float) -> torch.Tensor:
        """Discrete Taylor order-1 (Newton forward difference) weights."""
        k = len(self._anchors)
        weights = torch.zeros(k, dtype=torch.float32)
        if k == 1:
            weights[0] = 1.0
            return weights
        prev = self._anchors[-2].coord
        last = self._anchors[-1].coord
        spacing = last - prev
        if abs(spacing) <= 1e-12:
            weights[-1] = 1.0
            return weights
        ratio = (float(coord) - last) / spacing
        weights[-2] = -ratio
        weights[-1] = 1.0 + ratio
        return weights

    def combined_weights(self, coord: float) -> torch.Tensor:
        """Blend of spectral and linear extrapolation weights, shape (K,)."""
        blend = self.blend_weight
        if blend <= 1e-9:
            return self._linear_weights(coord)
        spectral = self._spectral_weights(coord)
        if blend >= 1.0 - 1e-9:
            return spectral
        linear = self._linear_weights(coord)
        return blend * spectral + (1.0 - blend) * linear

    # -- prediction ----------------------------------------------------

    def predict(self, coord: float) -> torch.Tensor:
        """Forecast the feature at `coord`.

        Returns a tensor with the recorded shape/dtype on the recorded
        storage device. Non-finite values are sanitized to the target
        dtype's finite range before casting.
        """
        if not self.ready():
            raise RuntimeError("Spectrum forecaster is not ready yet.")
        assert self._feature_shape is not None
        assert self._feature_dtype is not None
        assert self._storage_device is not None

        weights = self.combined_weights(coord).tolist()
        out = torch.empty(
            self._feature_shape,
            dtype=self._feature_dtype,
            device=self._storage_device,
        )
        flats = [a.feature.contiguous().reshape(-1) for a in self._anchors]
        out_flat = out.view(-1)
        numel = out_flat.numel()
        if self._feature_dtype.is_floating_point:
            finfo = torch.finfo(self._feature_dtype)
            low, high = float(finfo.min), float(finfo.max)
        else:
            low, high = None, None

        for start in range(0, numel, _PREDICT_CHUNK):
            stop = min(start + _PREDICT_CHUNK, numel)
            acc = torch.zeros(
                stop - start, dtype=torch.float32, device=out.device
            )
            for weight, flat in zip(weights, flats):
                if weight == 0.0:
                    continue
                acc += weight * flat[start:stop].to(torch.float32)
            if low is not None:
                torch.nan_to_num_(acc, nan=0.0, posinf=high, neginf=low)
                acc.clamp_(min=low, max=high)
            out_flat[start:stop] = acc.to(self._feature_dtype)
        return out
