from __future__ import annotations

import threading
from typing import Any, Callable

import torch

from .chebyshev import normalize_step_position
from .config import SpectrumConfig
from .constants import WRAPPER_KEY
from .controller import decide_actual_or_forecast, find_step_index, note_decision
from .forward import run_actual_step, run_forecast_step
from .state import SpectrumRootState
from .utils import is_qwen_image21_core, log_debug


def create_spectrum_wrapper(
    config: SpectrumConfig,
) -> Callable[..., Any]:
    """Build the DIFFUSION_MODEL wrapper closure.

    The wrapper is registered (see apply_spectrum) under
    model_options["transformer_options"]["wrappers"]["diffusion_model"] so
    that ComfyUI's own WrapperExecutor invokes it around
    QwenImage21Transformer2DModel._forward on every model call. All state
    lives inside this closure: the underlying model instance is never
    mutated, which keeps the patch local to the cloned MODEL and lets
    ComfyUI's prefix-KV-cache optimization stay active on real steps.

    The returned function carries a ``root_state`` attribute for
    introspection (used by the test-suite).
    """
    # MultiGPU CFG Split runs cond/uncond on different GPUs in parallel
    # threads, all sharing this one closure. Keep one independent state per
    # device so the threads never reset or overwrite each other's history.
    roots: dict[str, SpectrumRootState] = {}
    roots_lock = threading.Lock()
    root = SpectrumRootState(config=config)
    roots["__default__"] = root

    def _get_root(device: Any) -> SpectrumRootState:
        key = str(device) if device is not None else "__default__"
        with roots_lock:
            r = roots.get(key)
            if r is None:
                if key != "__default__" and not roots["__default__"].branch_states and len(roots) == 1:
                    r = roots["__default__"]
                else:
                    r = SpectrumRootState(config=config)
                roots[key] = r
            return r

    def _sigmas_signature(sigmas: torch.Tensor) -> tuple:
        # Value-based run fingerprint. MultiGPU CFG Split hands each
        # non-primary GPU a fresh sample_sigmas.to(device) copy on every
        # call, so id(sigmas) changes every step and must not be used.
        return tuple(round(v, 6) for v in sigmas.detach().float().cpu().tolist())

    def spectrum_diffusion_model_wrapper(
        executor: Any,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Any = None,
        ref_latents: Any = None,
        image_slots: Any = None,
        transformer_options: Any = None,
        **kwargs: Any,
    ) -> Any:
        model = getattr(executor, "class_obj", None)
        if not isinstance(transformer_options, dict):
            transformer_options = {}
        args_tuple = (
            x,
            timesteps,
            context,
            ref_latents,
            image_slots,
            transformer_options,
        )

        # Fail-closed: anything we cannot reason about runs the real path.
        if not is_qwen_image21_core(model):
            return executor(*args_tuple, **kwargs)
        sigmas = transformer_options.get("sample_sigmas")
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 2:
            return executor(*args_tuple, **kwargs)
        if getattr(model, "gradient_checkpointing", False):
            return executor(*args_tuple, **kwargs)

        root = _get_root(
            transformer_options.get("multigpu_thread_device", x.device)
        )
        step_index = find_step_index(sigmas, timesteps)
        total_steps = max(1, sigmas.numel() - 1)
        sigmas_sig = _sigmas_signature(sigmas)

        # New-run detection.
        if (
            root.last_sigmas_id is not None
            and root.last_sigmas_id != sigmas_sig
        ) or (
            root.last_total_steps is not None
            and root.last_total_steps != total_steps
        ) or (
            root.last_step_index is not None
            and step_index < root.last_step_index
        ):
            root.reset_run()

        branch_key = tuple(transformer_options.get("cond_or_uncond") or ())
        branch = root.get_branch_state(branch_key)

        # Geometry guard: the forecast target must keep a fixed shape for
        # the anchors to be combinable. A mismatch resets this branch's
        # history (falls back to real steps until re-learned).
        expected_shape = (
            int(x.shape[0]),
            int(x.shape[-2]) * int(x.shape[-1]),
            int(getattr(model, "inner_dim", -1)),
        )
        known_shape = branch.forecaster.feature_shape
        if known_shape is not None and tuple(known_shape) != expected_shape:
            branch.reset()

        control_present = (
            kwargs.get("control") is not None
            or transformer_options.get("control") is not None
        )

        actual, reason = decide_actual_or_forecast(
            state=branch,
            step_index=step_index,
            total_steps=total_steps,
            control_present=control_present,
            config=config,
        )

        coord = normalize_step_position(step_index, total_steps)
        sigma: Any = None
        if step_index >= 0:
            clamped = min(max(step_index, 0), sigmas.numel() - 1)
            sigma = float(sigmas[clamped].item())

        try:
            if actual:
                out = run_actual_step(
                    executor=executor,
                    model=model,
                    config=config,
                    branch=branch,
                    reason=reason,
                    step_index=step_index,
                    total_steps=total_steps,
                    sigma=sigma,
                    coord=coord,
                    branch_key=branch_key,
                    args_tuple=args_tuple,
                    kwargs=kwargs,
                )
            else:
                try:
                    out = run_forecast_step(
                        model=model,
                        config=config,
                        branch=branch,
                        step_index=step_index,
                        total_steps=total_steps,
                        coord=coord,
                        branch_key=branch_key,
                        x=x,
                        timesteps=timesteps,
                    )
                except Exception:
                    # Any forecast failure degrades to a real step.
                    actual = True
                    reason = "forecast_failed"
                    out = run_actual_step(
                        executor=executor,
                        model=model,
                        config=config,
                        branch=branch,
                        reason=reason,
                        step_index=step_index,
                        total_steps=total_steps,
                        sigma=sigma,
                        coord=coord,
                        branch_key=branch_key,
                        args_tuple=args_tuple,
                        kwargs=kwargs,
                    )
        finally:
            note_decision(branch, actual, reason, config)
            root.last_step_index = step_index
            root.last_sigmas_id = sigmas_sig
            root.last_total_steps = total_steps

        # Run completion: log a summary and release anchor memory.
        if step_index >= 0 and step_index >= total_steps - 1:
            root.completed_branches.add(branch_key)
            if len(root.completed_branches) >= len(root.branch_states):
                if config.debug and not root.summary_logged:
                    total_actual = sum(
                        b.actual_count for b in root.branch_states.values()
                    )
                    total_forecast = sum(
                        b.forecast_count for b in root.branch_states.values()
                    )
                    log_debug(
                        config.debug,
                        f"run complete: branches={len(root.branch_states)} "
                        f"actual_calls={total_actual} "
                        f"forecast_calls={total_forecast} "
                        f"total_calls={total_actual + total_forecast}",
                    )
                    root.summary_logged = True
                for b in root.branch_states.values():
                    b.forecaster.release_features()

        return out

    spectrum_diffusion_model_wrapper.root_state = root  # type: ignore[attr-defined]
    spectrum_diffusion_model_wrapper.roots = roots  # type: ignore[attr-defined]
    return spectrum_diffusion_model_wrapper


def apply_spectrum(model: Any, config: SpectrumConfig) -> Any:
    """Patch a ComfyUI MODEL (Qwen-Image-2.1) with the Spectrum wrapper.

    Returns a cloned ModelPatcher; the input model is untouched.
    """
    config.validate()

    model_clone = model.clone()
    core = model_clone.get_model_object("diffusion_model")
    if not is_qwen_image21_core(core):
        raise ValueError(
            "Spectrum (Qwen-Image-2.1) requires ComfyUI's native "
            "Qwen-Image-2.1 transformer core. Load the model with the "
            "native Qwen-Image-2.1 checkpoint loader, or connect a MODEL "
            "that still exposes the same transformer internals. "
            "Qwen-Image 1.x models use a different core layout and are "
            "not supported by this node."
        )

    import comfy.patcher_extension

    wrapper = create_spectrum_wrapper(config)
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        wrapper,
        model_clone.model_options,
        is_model_options=True,
    )
    return model_clone
