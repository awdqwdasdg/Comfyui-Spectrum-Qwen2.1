from __future__ import annotations

from typing import Any, Callable

import torch

from .config import SpectrumConfig
from .state import SpectrumBranchState
from .utils import log_debug, log_warning, resolve_cache_target


def _compute_temb(model: Any, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Replicate QwenImage21Transformer2DModel._forward timestep embedding.

    Source (comfy/ldm/qwen_image21/model.py):

        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)

    The second row embeds t = 0 and is used for the step-invariant prefix
    rows (text / references). The output head only consumes temb[:-1].
    """
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
    return model.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)


def run_forecast_head(
    model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    feature: torch.Tensor,
) -> torch.Tensor:
    """Run only the cheap output tail on a forecasted hidden state.

    Replicates, verbatim, the tail of
    QwenImage21Transformer2DModel._forward:

        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, out_channels, H, W)

    The captured/forecast feature already is the target-token slice that
    norm_out consumes, so no prefix handling is needed here.
    """
    dtype = x.dtype
    batch = x.shape[0]
    height, width = int(x.shape[-2]), int(x.shape[-1])
    out_channels = int(getattr(model, "out_channels", x.shape[1]))

    predicted = feature.to(device=x.device, dtype=dtype)
    temb = _compute_temb(model, timesteps, dtype)
    hidden = model.norm_out(predicted, temb[:-1])
    hidden = model.proj_out(hidden)
    return hidden.transpose(1, 2).reshape(batch, out_channels, height, width)


def _make_capture_hook(
    config: SpectrumConfig,
    captured: dict,
) -> Callable[[Any, tuple], None]:
    """Build the forward-pre-hook installed on model.norm_out."""

    def capture_norm_out_input(module: Any, args: tuple) -> None:
        if not args:
            return
        hidden = args[0]
        if not torch.is_tensor(hidden):
            return
        target_device, target_dtype = resolve_cache_target(
            hidden, config.cache_device
        )
        try:
            captured["feature"] = (
                hidden.detach()
                .to(device=target_device, dtype=target_dtype)
                .contiguous()
            )
            captured["model_feature_dtype"] = hidden.dtype
        except RuntimeError:
            # Storage target rejected the allocation (e.g. VRAM pressure);
            # fall back to host memory rather than failing the step.
            try:
                captured["feature"] = (
                    hidden.detach().to(device="cpu").contiguous()
                )
                captured["model_feature_dtype"] = hidden.dtype
            except RuntimeError:
                return

    return capture_norm_out_input


def run_actual_step(
    executor: Callable[..., Any],
    model: Any,
    config: SpectrumConfig,
    branch: SpectrumBranchState,
    reason: str,
    step_index: int,
    total_steps: int,
    sigma: float,
    coord: float,
    branch_key: tuple[int, ...],
    args_tuple: tuple,
    kwargs: dict,
) -> Any:
    """Execute a real forward pass, capturing the pre-norm_out feature.

    The capture hook is registered only for the duration of this call and
    removed afterwards, so nothing is permanently attached to the shared
    model instance. The returned output is exactly what the untouched
    model would have produced.
    """
    captured: dict = {}
    hook = _make_capture_hook(config, captured)
    handle = model.norm_out.register_forward_pre_hook(hook)
    try:
        out = executor(*args_tuple, **kwargs)
    finally:
        handle.remove()

    feature = captured.get("feature")
    if feature is not None:
        try:
            branch.record_actual(
                coord=coord,
                sigma=sigma,
                feature=feature,
                model_feature_dtype=captured.get("model_feature_dtype"),
            )
        except ValueError:
            # Geometry changed mid-run (area conds, batch re-splitting):
            # rebuild this branch's history instead of failing the run.
            branch.reset()
            try:
                branch.record_actual(
                    coord=coord,
                    sigma=sigma,
                    feature=feature,
                    model_feature_dtype=captured.get("model_feature_dtype"),
                )
            except ValueError:
                if not getattr(branch, "warned_record_once", False):
                    branch.warned_record_once = True  # type: ignore[attr-defined]
                    log_warning(
                        "could not record captured feature after branch reset; "
                        "continuing without this anchor"
                    )
    elif not getattr(branch, "warned_capture_once", False):
        branch.warned_capture_once = True  # type: ignore[attr-defined]
        log_warning("failed to capture the pre-norm_out hidden state")

    log_debug(
        config.debug,
        f"step {step_index + 1}/{total_steps} branch={list(branch_key)} "
        f"mode=actual reason={reason} history={branch.forecaster.history_size}",
    )
    return out


def run_forecast_step(
    model: Any,
    config: SpectrumConfig,
    branch: SpectrumBranchState,
    step_index: int,
    total_steps: int,
    coord: float,
    branch_key: tuple[int, ...],
    x: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Predict the final hidden state and run the output head only."""
    predicted = branch.forecaster.predict(coord)
    out = run_forecast_head(model, x, timesteps, predicted)
    branch.record_forecast(coord)
    log_debug(
        config.debug,
        f"step {step_index + 1}/{total_steps} branch={list(branch_key)} "
        f"mode=forecast history={branch.forecaster.history_size}",
    )
    return out
