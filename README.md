# ComfyUI-Spectrum-QwenImage21

# **Disclaimer: This was fully vibe coded in one shot by GLM-5.3**

# Quick T2I Notes
* Generations with spectrum often look very similar to the base step generation, but softer and more airbrushed. Good for quick prompt iteration.

Random tests done by me with the default spectrum node values (3060 12GB, 928x1664, 25 steps, **diffusion time only**, fixed seed, CFG 1, int8convrot, **SAMPLE SIZE 1**, Euler + Simple, Sage attention):
| Model / Test | Steps | CFG | Diffusion Time | Speed |
| :--- | :---: | :---: | :---: | :---: |
| Base | 25 | 1 | ~49s | 1.97s/it |
| Easy Cache | 25 | 1 | ~30s | 1.23s/it |
| Spectrum | 25 | 1 | ~22s | 1.10it/s |
| Spectrum | **45** | 1 | ~28s | 1.56it/s |
| Spectrum | 25 | **4** | ~45s | 2.81s/it |

* Easy cache was ran with `reuse_threshold = 0.20`, `start_percent = 0.20`, and `end_percent = 0.70`
* Tests ran with sage attention, though fully compatible with comfy kitchen attention (add '--use-ck-attention' to your startup flags or use the `Model Attention Backend` node with `comfy kitchen attention` selected)
* 45 steps was chosen as I found this is generally where quality becomes more consistent.

For equivalent steps, it is arguably equal quality compared to easy cache (in some cases easy cache ends up creating artifacts while spectrum always looks airbrushed), but it seems to be far faster, allowing me to fit up to around 45 steps while still being faster than Easy Cache.

# Quick I2I notes
* Transparency still works fine, though I recommend following the prompting tip from the official [HuggingFace space](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1)
> For transparent image generation, use the following prompt format and replace xxxxx with your image description: `This is an RGBA image with transparency. xxxxx The image has alpha channel and the background is transparent.`
* Reminder: You can't use the Flux 2/Mage Flow VAE Encode/Decode trick to reduce the lattice grid for transparent images as these VAEs do not support an alpha channel. It will turn the alpha channel pink/magenta.
* Tested with up to 3 reference images. Worked perfectly fine.

Random tests done by me with default spectrum node values (3060 12GB, 1 megapixel, 45 steps, **diffusion time only**, fixed seed, CFG 1, int8convrot, **SAMPLE SIZE 1**, Euler + Simple, Sage attention)
| Ref. Imgs | Steps | Diffusion Time | Speed |
| :--- | :---: | :---: | :---: |
| 1 | 45 | ~21s | 2.08 it/s |
| 2\* | 45 | ~26s | 1.73 it/s |
| 3\* | 45 | ~30s | 1.49 it/s |
| 1 | **25** | ~16s | 1.49 it/s |
| 2\* | **25** | ~20s | 1.22 it/s |
| 3\* | **25** | ~23s| 1.07 it/s |

*_You will feel a larger time gap as reference images increase due to the extra conditioning required._
* All images were passed at 1 megapixel, and all tests were ran with spectrum.
* Tests ran with sage attention, though fully compatible with comfy kitchen attention (add '--use-ck-attention' to your startup flags or use the `Model Attention Backend` node with `comfy kitchen attention` selected)
* 45 steps was chosen as I found this is generally where quality becomes more consistent.
* I did not do in-depth testing against the base/easy cache as I did some light testing and found similar results to T2I

That's all from me, everything after this is AI slop.

# **Spectrum sampling acceleration for Qwen-Image-2.1 in ComfyUI.**

This custom node applies **Spectrum** — the training-free diffusion sampling
accelerator from the paper *"Adaptive Spectral Feature Forecasting for
Diffusion Sampling Acceleration"* (arXiv:2603.01623, CVPR 2026) — to
ComfyUI's native **Qwen-Image-2.1** model (text-to-image and editing,
including RGBA and multi-reference workflows).

On selected denoising steps the node skips the entire 32-block Qwen
transformer and instead *forecasts* its final hidden state with an online,
ridge-regularized Chebyshev fit over the real steps, running only the cheap
output head (`time_text_embed → norm_out → proj_out`). This typically skips
~65–70% of the transformer passes at the paper's "moderate" setting while
keeping outputs visually close to full sampling.

- Paper: <https://arxiv.org/abs/2603.01623>
- Official code: <https://github.com/hanjq17/Spectrum>
- Qwen-Image-2.1: <https://huggingface.co/Qwen/Qwen-Image-2.1>

## Why this works on Qwen-Image-2.1

Qwen-Image-2.1's diffusion transformer is an ideal Spectrum target:

1. **Single-stream DiT + cheap tail.** 32 `QwenImage21TransformerBlock`
   layers process the packed sequence, and only the target-image tokens
   enter the output head. The tail (`norm_out`/`proj_out`) costs a small
   fraction of one block, so skipping the stack yields near-full step
   savings.
2. **Step-invariant prefix.** Text and reference tokens are modulated at
   t = 0 and attended to through a causal prefix, so the pre-`norm_out`
   target hidden state — the feature Spectrum forecasts — is a smooth,
   fixed-shape function of the diffusion timestep. That is exactly the
   setting the Chebyshev forecaster is designed and theoretically bounded
   for.
3. **Flow-matching schedule.** Like FLUX and SD3.5 (both validated in the
   paper), Qwen-Image-2.1 is a rectified-flow model whose per-step features
   vary smoothly along the trajectory.

The node integrates through ComfyUI's official wrapper mechanism
(`WrappersMP.DIFFUSION_MODEL` via `transformer_options`), so nothing is
monkey-patched, the patched model is a true clone, and ComfyUI's own
prefix-KV-cache optimization keeps working on the real steps.

## Installation

Clone this folder into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/awdqwdasdg/Comfyui-Spectrum-Qwen2.1.git
```

Restart ComfyUI. No extra Python dependencies (only PyTorch, which ComfyUI
already provides).

## Usage

```text
Load Qwen-Image-2.1 checkpoint
        │
        ▼
(any LoRA / model modifying nodes)
        │
        ▼
Spectrum (Qwen-Image-2.1)   ◄── place after all model mutations,
        │                       before the sampler
        ▼
KSampler / SamplerCustom   (cfg = 1.0 recommended, 20–40 steps)
```

- Place the node **after** everything that modifies the model and **before**
  the sampler.
- Defaults follow the paper's *moderate* configuration; leave them as-is
  for a first run.
- Enable `debug` on the first run to see the actual/forecast pattern in
  the console (e.g. `step 13/40 ... mode=forecast history=8`).

### Example: expected schedule at defaults (40 steps)

5 warm-up real steps → growing forecast gaps (1, 1, 2, 2, 3, 4, 4, 5, 5, 6)
→ 2 protected tail steps = **14 real / 26 forecast steps (~2.9× fewer
transformer passes)**.

## Parameters

| Input | Default | Meaning |
|---|---|---|
| `model` | — | The Qwen-Image-2.1 `MODEL` to patch. |
| `warmup_steps` | `5` | Initial steps always run real (paper `W`). Also seeds the fit. |
| `tail_actual_steps` | `2` | Final steps always run real; the last steps resolve fine detail. |
| `window_size` | `2.0` | Initial forecast gap (paper `N`): one real step every N-th step. |
| `flex_window` | `0.75` | Gap growth per real step (paper `alpha`). `0.75` = moderate, `3.0` = aggressive (~4–5×). |
| `max_consecutive_forecasts` | `8` | Safety cap on consecutive skipped steps. |
| `history_points` | `8` | Real snapshots (anchors) kept for the fit, sliding window. |
| `chebyshev_degree` | `4` | Polynomial degree `M` (paper default; ablation: 2→4 helps, 6 marginal). |
| `ridge_lambda` | `0.1` | Ridge regularization `lambda` (paper default; 1e-3 and 10 both hurt). |
| `blend_weight` | `0.5` | `1.0` = pure Chebyshev (paper-exact), `0.0` = pure linear extrapolation. The authors recommend `0.5` for robustness across acceleration levels. |
| `cache_device` | `main_device` | Where anchors are stored. Each anchor is ~512 MB (bf16) at 2048×2048 / ~128 MB at 1024×1024. Use `cpu` or `offload_device` if VRAM is tight. |
| `force_actual_on_control` | `True` | Force real forwards while control residuals are present. |
| `debug` | `False` | Print per-step decisions and a run summary. |

### Tuning

- **More speed**: raise `flex_window` toward `3.0` (the paper's aggressive
  setting) or lower `tail_actual_steps` to `1`. Expect a visible quality
  drift at very high acceleration.
- **More quality**: lower `flex_window` (e.g. `0.4`), raise
  `tail_actual_steps` to `3–4`, or set `blend_weight` to `1.0` for the
  paper-exact predictor.
- **Less VRAM**: `cache_device = cpu` (anchors then live in system RAM;
  forecasts add a host→device copy per step).

## How it works

1. **Register.** The node clones the model and attaches one
   `diffusion_model` wrapper (ComfyUI `WrapperExecutor`) under
   `model_options["transformer_options"]["wrappers"]`.
2. **Schedule.** Each model call maps its timestep onto the sigma schedule
   (`sample_sigmas`) to get the step index. A decision function chooses
   *real* or *forecast* per cond/uncond branch, following the paper's
   adaptive rule: real iff `(consecutive_forecasts + 1) % floor(window) == 0`,
   with the window growing by `flex_window` after each window-triggered real
   step, plus warm-up/tail/insufficient-history/control guards.
3. **Real steps.** The wrapper calls the original transformer untouched and
   captures — via a temporary pre-forward hook on `norm_out` — the final
   hidden state of the target image tokens, which becomes a fit anchor.
4. **Forecast steps.** The anchor history is fitted with Chebyshev
   polynomials (degree `M`) under ridge regression (λ). The prediction is
   computed as a weighted sum over anchors,
   `w = φ(τ*)(ΦᵀΦ + λI)⁻¹Φᵀ` — algebraically identical to the paper's
   Eq. (12)–(14) but never materializing the `(M+1)×F` coefficient matrix,
   which matters here because `F = B·H·W·4096` can exceed 2.5×10⁸ at
   2048×2048. The prediction is blended with a two-point (Taylor order-1)
   extrapolation at `blend_weight`.
5. **Head only.** The forecasted hidden state runs through the exact tail
   of `QwenImage21Transformer2DModel._forward`
   (`temb = time_text_embed(cat([t, 0]))`, `norm_out(h, temb[:-1])`,
   `proj_out`, transpose/reshape) and returns a normal velocity output, so
   every ComfyUI sampler integrates it as usual.

Memory: anchors are stored in the model's dtype (bf16 → half the footprint
of fp32) on the configured device, with an automatic CPU fallback if the
capture copy hits VRAM pressure. Anchor memory is released at the end of
every sampling run.

## Safety / fallback behavior

The node fails **closed**: it runs the real transformer whenever it cannot
prove a forecast is safe.

- Not a native Qwen-Image-2.1 core, or `sample_sigmas` unavailable → real.
- Warm-up, protected tail, or fewer anchors than the fit needs → real.
- Control residuals present (with the guard enabled) → real.
- Feature geometry changed mid-run (area conditions, batch re-splitting) →
  branch history resets and real steps rebuild it.
- Any exception during a forecast → that step degrades to a real step.

## Limitations

- **Qwen-Image-2.1 only.** Qwen-Image 1.x / -Edit cores have a different
  layout (patch-2 MMDiT, different `time_text_embed` convention) and are
  rejected with an error message.
- Quality at high acceleration is a tradeoff. The paper reports, for
  FLUX.1-dev at `alpha=0.75` (the default here), PSNR ≈ 24.3 dB vs. full
  50-step sampling with a 3.47× speedup; treat Spectrum outputs as
  *very close* rather than bit-identical.
- The forecaster adds a per-anchor memory cost (see `cache_device`).
- Quantized/wrapped cores that hide the native transformer internals are
  not supported.

## Development

The repository ships a CPU test-suite (no ComfyUI or GPU needed) that
validates the math against a direct ridge solve, the paper's schedule
pattern, and the full wrapper flow against a replica of ComfyUI's
Qwen-Image-2.1 transformer:

```bash
python -m unittest discover -s tests -v
```

## Credits

- Spectrum method: Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan
  Guo, Stefano Ermon — *"Adaptive Spectral Feature Forecasting for
  Diffusion Sampling Acceleration"* (arXiv:2603.01623, CVPR 2026).
- Qwen-Image-2.1 model & ComfyUI integration: Qwen team and the ComfyUI
  contributors (`comfy/ldm/qwen_image21`).
- Community reference ports that informed the ComfyUI integration
  patterns: `xmarre/ComfyUI-Spectrum-*`.

License: MIT (this implementation). The underlying method and model remain
under their respective licenses.
