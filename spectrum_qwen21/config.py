from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrumConfig:
    """All tunables of the Spectrum forecaster for Qwen-Image-2.1.

    Defaults follow the paper's "moderate" configuration
    (arXiv 2603.01623: W=5, N=2, alpha=0.75, M=4, lambda=0.1) with the
    authors' post-publication robustness recommendation of blending the
    Chebyshev prediction with a first-order (linear) extrapolation at
    weight 0.5, plus a small amount of ComfyUI-specific safety padding
    (protected tail steps, consecutive-forecast cap).
    """

    warmup_steps: int = 5
    tail_actual_steps: int = 2
    window_size: float = 2.0
    flex_window: float = 0.75
    max_consecutive_forecasts: int = 8
    history_points: int = 8
    chebyshev_degree: int = 4
    ridge_lambda: float = 0.1
    blend_weight: float = 0.5
    cache_device: str = "main_device"
    force_actual_on_control: bool = True
    debug: bool = False

    def validate(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.tail_actual_steps < 0:
            raise ValueError("tail_actual_steps must be >= 0")
        if self.history_points < 2:
            raise ValueError("history_points must be >= 2")
        if self.chebyshev_degree < 1:
            raise ValueError("chebyshev_degree must be >= 1")
        if self.chebyshev_degree + 1 > self.history_points:
            raise ValueError(
                "history_points must be at least chebyshev_degree + 1 "
                f"(got degree={self.chebyshev_degree}, "
                f"history_points={self.history_points})"
            )
        if self.max_consecutive_forecasts < 0:
            raise ValueError("max_consecutive_forecasts must be >= 0")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if not (0.0 <= self.blend_weight <= 1.0):
            raise ValueError("blend_weight must be within [0.0, 1.0]")
        if self.window_size < 1.0:
            raise ValueError("window_size must be >= 1.0")
        if self.flex_window < 0.0:
            raise ValueError("flex_window must be >= 0.0")
        if self.cache_device not in {"main_device", "offload_device", "cpu"}:
            raise ValueError(
                "cache_device must be one of: main_device, offload_device, cpu"
            )

    @property
    def window_growth_cap(self) -> float:
        """Upper bound for the growing recomputation interval.

        The paper's schedule lets the interval grow without bound; for a
        fixed history window we cap it at max(window_size, history_points),
        which for the defaults (2, 8) reproduces the interval sizes the
        paper reaches at 50 steps with alpha=0.75.
        """
        return max(self.window_size, float(self.history_points), 1.0)
