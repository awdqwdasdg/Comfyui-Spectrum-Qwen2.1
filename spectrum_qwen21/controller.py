from __future__ import annotations

import math

import torch

from .config import SpectrumConfig
from .state import SpectrumBranchState


def find_step_index(sigmas: torch.Tensor, timestep: torch.Tensor) -> int:
    """Locate the current solver step index inside the sigma schedule.

    ComfyUI sets ``transformer_options["sample_sigmas"]`` to the full sigma
    schedule of the sampling run, and (for the flow-matching Qwen-Image-2.1)
    passes the per-call timestep unchanged (ModelSamplingFlux.timestep is
    the identity), so the timestep value matches one schedule entry.

    Falls back to bracketing and finally nearest-neighbour matching.

    Returns -1 when the inputs are too deggenerate to reason about.
    """
    if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 1:
        return -1
    if not isinstance(timestep, torch.Tensor) or timestep.numel() < 1:
        return -1

    target = timestep.reshape(-1)[0]
    if target.device != sigmas.device:
        sigmas = sigmas.to(target.device)

    diffs = sigmas.float() - target.float()
    exact = (diffs == 0).nonzero()
    if exact.numel() > 0:
        return int(exact[0, 0].item())

    sign = diffs.sign()
    if sign.numel() > 1:
        crossings = (sign[:-1] * sign[1:] <= 0).nonzero()
        if crossings.numel() > 0:
            return int(crossings[0, 0].item())

    return int(diffs.abs().argmin().item())


def decide_actual_or_forecast(
    state: SpectrumBranchState,
    step_index: int,
    total_steps: int,
    control_present: bool,
    config: SpectrumConfig,
) -> tuple[bool, str]:
    """Decide whether this model call runs the real transformer or a forecast.

    Priority order (fail-closed):
      1. unknown step -> real
      2. warm-up region -> real
      3. protected tail -> real
      4. control residuals present (and guarded) -> real
      5. not enough anchors for the fit -> real
      6. consecutive-forecast cap hit -> real
      7. paper's adaptive growing-window rule decides
    """
    if step_index < 0:
        return True, "unknown_step"
    if step_index < config.warmup_steps:
        return True, "warmup"
    if config.tail_actual_steps > 0 and step_index >= max(
        0, total_steps - config.tail_actual_steps
    ):
        return True, "tail"
    if control_present and config.force_actual_on_control:
        return True, "control"
    if not state.forecaster.ready():
        return True, "insufficient_history"
    if state.consecutive_forecasts >= config.max_consecutive_forecasts:
        return True, "refresh"

    interval = max(1, int(math.floor(state.current_window)))
    if (state.consecutive_forecasts + 1) % interval == 0:
        return True, "window"
    return False, "forecast"


def note_decision(
    state: SpectrumBranchState,
    actual: bool,
    reason: str,
    config: SpectrumConfig,
) -> None:
    """Update branch counters after the decision has been executed.

    The recomputation interval only grows when an actual step was selected
    by the adaptive window rule itself (mirroring the official
    implementation, where curr_ws += flex_window after a window-triggered
    real forward).
    """
    if actual:
        state.consecutive_forecasts = 0
        state.actual_count += 1
        if reason == "window":
            state.current_window = min(
                state.current_window + config.flex_window,
                config.window_growth_cap,
            )
    else:
        state.consecutive_forecasts += 1
        state.forecast_count += 1
