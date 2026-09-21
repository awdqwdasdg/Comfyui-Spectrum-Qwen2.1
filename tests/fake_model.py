"""A CPU-friendly replica of ComfyUI's Qwen-Image-2.1 transformer.

Mirrors the structure and the exact call-flow of
comfy/ldm/qwen_image21/model.py at a tiny scale, including:

* the WrapperExecutor call pattern around ``_forward``,
* the timestep embedding math (t = ((t*1000)/1000), temb rows
  [batch..., t=0]),
* the output tail (norm_out -> proj_out -> transpose/reshape),
* the structural attributes consumed by the Spectrum node
  (transformer_blocks, img_in, txt_in, inner_dim, out_channels, ...).

The block stack is deliberately simple, but every component is a smooth
(analytic) function of the timestep, so the final hidden state is a
smooth function of diffusion time -- exactly the regime the Spectrum
forecaster is designed for.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal embedding (same as comfy.ldm.flux.layers)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class FakeTimestepProjEmbeddings(nn.Module):
    """Replica of qwen_image21.model.TimestepProjEmbeddings."""

    def __init__(self, embedding_dim: int, freq_dim: int = 64):
        super().__init__()
        self.linear_1 = nn.Linear(freq_dim, embedding_dim)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)
        self.freq_dim = freq_dim

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.linear_2(
            F.silu(self.linear_1(timestep_embedding(timestep.float(), self.freq_dim).to(dtype)))
        )


class FakeTextProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(x), approximate="tanh"))


class FakeLastLayer(nn.Module):
    """Replica of qwen_image21.model.LastLayer (scale-only AdaLN)."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        return self.norm(x) * (1 + scale)


class FakeTransformerBlock(nn.Module):
    """Smooth-in-t block (stand-in for QwenImage21TransformerBlock)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.net1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.net2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor, mod: tuple) -> torch.Tensor:
        scale1, gate1, scale2, gate2 = mod
        x = x + gate1 * self.net1(self.norm1(x) * (1 + scale1))
        x = x + gate2 * self.net2(self.norm2(x) * (1 + scale2))
        return x


class FakeExecutor:
    """Replica of comfy.patcher_extension.WrapperExecutor."""

    def __init__(self, original: Callable, class_obj: Any, wrappers: list, idx: int = 0):
        self.original = original
        self.class_obj = class_obj
        self.wrappers = list(wrappers)
        self.idx = idx
        self.is_last = idx == len(self.wrappers)

    def __call__(self, *args, **kwargs):
        new = FakeExecutor(self.original, self.class_obj, self.wrappers, self.idx + 1)
        return new.execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.is_last:
            return self.original(*args, **kwargs)
        return self.wrappers[self.idx](self, *args, **kwargs)


class FakeQwenImage21Model(nn.Module):
    """Small-scale replica of QwenImage21Transformer2DModel."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        inner_dim: int = 32,
        num_layers: int = 4,
        context_dim: int = 16,
        txt_len: int = 7,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.inner_dim = inner_dim
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.txt_len = txt_len

        self.time_text_embed = FakeTimestepProjEmbeddings(inner_dim)
        self.txt_in = FakeTextProjection(context_dim, inner_dim)
        self.img_in = nn.Linear(in_channels, inner_dim, bias=False)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(inner_dim, 4 * inner_dim, bias=False)
        )
        self.transformer_blocks = nn.ModuleList(
            [FakeTransformerBlock(inner_dim) for _ in range(num_layers)]
        )
        self.norm_out = FakeLastLayer(inner_dim)
        self.proj_out = nn.Linear(inner_dim, out_channels, bias=False)

    def _forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Any = None,
        ref_latents: Any = None,
        image_slots: Any = None,
        transformer_options: dict = {},
        **kwargs: Any,
    ) -> torch.Tensor:
        batch, _, height, width = x.shape
        dtype = x.dtype

        txt = self.txt_in(context)
        img = self.img_in(x.flatten(2).transpose(1, 2))
        hidden = torch.cat([txt, img], dim=1)
        prefix_len = hidden.shape[1] - height * width

        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb[:-1]).chunk(4, dim=-1)
        mod = (
            scale1.unsqueeze(1),
            gate1.tanh().unsqueeze(1),
            scale2.unsqueeze(1),
            gate2.tanh().unsqueeze(1),
        )

        for block in self.transformer_blocks:
            hidden = block(hidden, mod)

        hidden = self.norm_out(hidden[:, prefix_len:], temb[:-1])
        hidden = self.proj_out(hidden)
        return hidden.transpose(1, 2).reshape(batch, self.out_channels, height, width)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Any = None,
        ref_latents: Any = None,
        image_slots: Any = None,
        transformer_options: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not isinstance(transformer_options, dict):
            transformer_options = {}
        wrappers: list = []
        for wrapper_list in (
            transformer_options.get("wrappers", {}).get("diffusion_model", {}).values()
        ):
            wrappers.extend(wrapper_list)
        return FakeExecutor(self._forward, self, wrappers).execute(
            x, timesteps, context, ref_latents, image_slots, transformer_options, **kwargs
        )


class NotAQwenModel(nn.Module):
    """Model failing the structural check (no txt_in/transformer_blocks)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.linear(x)
