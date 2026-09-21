from __future__ import annotations

from .spectrum_qwen21.config import SpectrumConfig
from .spectrum_qwen21.patcher import apply_spectrum

_NODE_CATEGORY = "model/optimization"


class SpectrumQwenImage21:
    """Apply Spectrum (arXiv 2603.01623) to a Qwen-Image-2.1 MODEL.

    Training-free sampling acceleration: on selected steps the 32-block
    Qwen-Image-2.1 transformer is skipped entirely and its final hidden
    state is forecast with an online ridge-regularized Chebyshev fit over
    the real steps, after which only the cheap output head runs.
    """

    CATEGORY = _NODE_CATEGORY
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Initial solver steps always run as real forwards "
                            "(paper: W = 5). These steps also seed the forecaster."
                        ),
                    },
                ),
                "tail_actual_steps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Final solver steps always kept on the real path; the "
                            "last steps are where details are resolved."
                        ),
                    },
                ),
                "window_size": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 16.0,
                        "step": 0.25,
                        "tooltip": (
                            "Initial recomputation interval (paper: N = 2): one "
                            "real step every N-th step before growth."
                        ),
                    },
                ),
                "flex_window": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": (
                            "Interval growth per real step (paper: alpha; "
                            "0.75 = moderate, 3.0 = aggressive)."
                        ),
                    },
                ),
                "max_consecutive_forecasts": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Safety cap: at most this many skipped forwards in a "
                            "row before a real step is forced."
                        ),
                    },
                ),
                "history_points": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Number of real hidden-state snapshots (anchors) kept "
                            "for the Chebyshev fit (sliding window)."
                        ),
                    },
                ),
                "chebyshev_degree": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 12,
                        "step": 1,
                        "tooltip": "Polynomial degree M of the spectral fit (paper: 4).",
                    },
                ),
                "ridge_lambda": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Ridge regularization strength (paper: 0.1).",
                    },
                ),
                "blend_weight": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "1.0 = pure Chebyshev (paper-exact); 0.0 = pure linear "
                            "extrapolation; the authors recommend 0.5 for "
                            "robustness across acceleration levels."
                        ),
                    },
                ),
                "cache_device": (
                    ["main_device", "offload_device", "cpu"],
                    {
                        "default": "main_device",
                        "tooltip": (
                            "Where anchor hidden states are stored. At 2048x2048 "
                            "each anchor is ~512 MB (bf16); use cpu/offload_device "
                            "if VRAM is tight."
                        ),
                    },
                ),
                "force_actual_on_control": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Run real forwards whenever control residuals are present.",
                    },
                ),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Print per-step actual/forecast decisions and a run summary.",
                    },
                ),
            }
        }

    def patch(
        self,
        model,
        warmup_steps: int,
        tail_actual_steps: int,
        window_size: float,
        flex_window: float,
        max_consecutive_forecasts: int,
        history_points: int,
        chebyshev_degree: int,
        ridge_lambda: float,
        blend_weight: float,
        cache_device: str,
        force_actual_on_control: bool,
        debug: bool,
    ):
        config = SpectrumConfig(
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            window_size=window_size,
            flex_window=flex_window,
            max_consecutive_forecasts=max_consecutive_forecasts,
            history_points=history_points,
            chebyshev_degree=chebyshev_degree,
            ridge_lambda=ridge_lambda,
            blend_weight=blend_weight,
            cache_device=cache_device,
            force_actual_on_control=force_actual_on_control,
            debug=debug,
        )
        patched = apply_spectrum(model, config)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "SpectrumQwenImage21": SpectrumQwenImage21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumQwenImage21": "Spectrum (Qwen-Image-2.1)",
}
