from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .chebyshev import HistoryWeightChebyshevForecaster
from .config import SpectrumConfig


@dataclass
class SpectrumBranchState:
    """Per-branch (per cond/uncond combination) forecaster state.

    ComfyUI batches conditioning chunks into single model calls and
    exposes the combination through transformer_options["cond_or_uncond"]
    (e.g. ``(0,)`` cond only, ``(1,)`` uncond only, ``(0, 1)`` both). Each
    combination gets its own forecaster so that CFG branches never share
    history, exactly like the official code keeps separate forecasters
    for cond/uncond passes.
    """

    config: SpectrumConfig
    forecaster: HistoryWeightChebyshevForecaster = field(init=False)
    consecutive_forecasts: int = 0
    current_window: float = 2.0
    actual_count: int = 0
    forecast_count: int = 0
    model_feature_dtype: Optional[torch.dtype] = None
    last_coord: Optional[float] = None
    last_sigma: Optional[float] = None

    def __post_init__(self) -> None:
        self.current_window = float(self.config.window_size)
        self.forecaster = HistoryWeightChebyshevForecaster(
            degree=self.config.chebyshev_degree,
            ridge_lambda=self.config.ridge_lambda,
            max_history=self.config.history_points,
            blend_weight=self.config.blend_weight,
        )

    def reset(self) -> None:
        self.forecaster.reset()
        self.consecutive_forecasts = 0
        self.current_window = float(self.config.window_size)
        self.actual_count = 0
        self.forecast_count = 0
        self.model_feature_dtype = None
        self.last_coord = None
        self.last_sigma = None

    def signature(self) -> Optional[tuple]:
        shape = self.forecaster.feature_shape
        if shape is None:
            return None
        return tuple(shape)

    def record_actual(
        self,
        coord: float,
        sigma: Optional[float],
        feature: torch.Tensor,
        model_feature_dtype: Optional[torch.dtype],
    ) -> None:
        """Record one real observation (pure data; counters are owned by
        controller.note_decision, which runs exactly once per call)."""
        self.forecaster.update(coord, feature)
        if model_feature_dtype is not None:
            self.model_feature_dtype = model_feature_dtype
        self.last_coord = float(coord)
        self.last_sigma = None if sigma is None else float(sigma)

    def record_forecast(self, coord: float) -> None:
        self.last_coord = float(coord)


@dataclass
class SpectrumRootState:
    """State shared by every branch of one sampling run.

    A "run" is one execution of a sampler over a sigma schedule. New runs
    are detected through the identity of the sample_sigmas tensor, a
    change in the total step count, or a step index moving backwards.
    """

    config: SpectrumConfig
    branch_states: dict[tuple[int, ...], SpectrumBranchState] = field(
        default_factory=dict
    )
    last_sigmas_id: Optional[int] = None
    last_total_steps: Optional[int] = None
    last_step_index: Optional[int] = None
    completed_branches: set[tuple[int, ...]] = field(default_factory=set)
    summary_logged: bool = False

    def reset_run(self) -> None:
        self.branch_states.clear()
        self.last_sigmas_id = None
        self.last_total_steps = None
        self.last_step_index = None
        self.completed_branches.clear()
        self.summary_logged = False

    def get_branch_state(
        self, branch_key: tuple[int, ...]
    ) -> SpectrumBranchState:
        state = self.branch_states.get(branch_key)
        if state is None:
            state = SpectrumBranchState(config=self.config)
            self.branch_states[branch_key] = state
        return state
