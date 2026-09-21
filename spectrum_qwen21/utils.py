from __future__ import annotations

from typing import Any

import torch

from .constants import LOG_TAG


def log_debug(enabled: bool, message: str) -> None:
    """Debug logging routed to stdout (visible in the ComfyUI console)."""
    if enabled:
        print(f"[{LOG_TAG}] {message}", flush=True)


def log_info(message: str) -> None:
    print(f"[{LOG_TAG}] {message}", flush=True)


def log_warning(message: str) -> None:
    print(f"[{LOG_TAG}] WARNING: {message}", flush=True)


def resolve_cache_target(
    hidden: torch.Tensor, cache_device: str
) -> tuple[torch.device, torch.dtype]:
    """Pick the device/dtype used to store captured anchor features.

    Features are always stored in their native dtype; the fitting math is
    float32 regardless, so no precision is lost by keeping bf16 storage.
    """
    if cache_device == "cpu":
        return torch.device("cpu"), hidden.dtype
    if cache_device == "offload_device":
        try:
            import comfy.model_management as model_management

            return torch.device(model_management.unet_offload_device()), hidden.dtype
        except Exception:
            return torch.device("cpu"), hidden.dtype
    return hidden.device, hidden.dtype


def is_qwen_image21_core(obj: Any) -> bool:
    """Structural check for ComfyUI's native Qwen-Image-2.1 transformer.

    Matches the class name first (comfy.ldm.qwen_image21.model.
    QwenImage21Transformer2DModel); falls back to a structural signature
    that positively identifies the 2.1 single-stream layout and
    deliberately excludes Qwen-Image 1.x cores (which expose txt_norm and
    a different time_text_embed calling convention).
    """
    if obj is None:
        return False
    type_name = type(obj).__name__
    if "QwenImage21" in type_name:
        return True
    if "QwenImage" in type_name and "21" not in type_name:
        return False
    from .constants import EXCLUDE_CORE_FIELDS, REQUIRED_CORE_FIELDS

    for field_name in EXCLUDE_CORE_FIELDS:
        if hasattr(obj, field_name):
            return False
    for field_name in REQUIRED_CORE_FIELDS:
        if not hasattr(obj, field_name):
            return False
    blocks = getattr(obj, "transformer_blocks", None)
    return hasattr(blocks, "__len__") and len(blocks) > 0
