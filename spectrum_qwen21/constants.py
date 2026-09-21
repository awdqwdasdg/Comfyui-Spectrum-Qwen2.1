from __future__ import annotations

# Key under which the DIFFUSION_MODEL wrapper is registered in
# model_options["transformer_options"]["wrappers"].
WRAPPER_KEY = "spectrum_qwen21"

# Logger tag used for debug output.
LOG_TAG = "Spectrum-QwenImage21"

# Attributes that the native ComfyUI Qwen-Image-2.1 transformer core is
# expected to expose. The 2.1 core has a single-stream layout:
#   transformer_blocks (ModuleList), img_in, txt_in, modulation,
#   time_text_embed (TimestepProjEmbeddings), norm_out (LastLayer),
#   proj_out (Linear), inner_dim, out_channels.
# Qwen-Image 1.x cores additionally expose txt_norm and a different
# time_text_embed signature, and are intentionally not matched here.
REQUIRED_CORE_FIELDS = (
    "transformer_blocks",
    "img_in",
    "txt_in",
    "norm_out",
    "proj_out",
    "time_text_embed",
    "inner_dim",
    "out_channels",
)

# Present on Qwen-Image 1.x cores but not on the 2.1 core; used to reject
# the previous generation instead of silently mis-calling its head.
EXCLUDE_CORE_FIELDS = ("txt_norm",)
