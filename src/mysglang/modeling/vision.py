"""Qwen3.5 vision tower: patches -> ViT -> 2x2 merged features in the LLM hidden size.

Each image (or video frame) attends only to itself (non-causal, split by ``cu_seqlens``).
Positions come from two sources: a learned 48x48 table bilinearly resampled to the image
grid, and a 2-D rotary embedding over (row, column). Patches arrive in 2x2-merge-block
order, so the merger simply concatenates every 4 consecutive patch features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mysglang.config import VisionConfig


def _axis_taps(index: torch.Tensor, size: torch.Tensor, side: int):
    """Bilinear (align_corners=True) taps into a ``side``-long table for one axis."""
    source = index.float() * (side - 1) / torch.clamp(size - 1, min=1)
    floor = torch.floor(source)
    offsets = torch.arange(2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    weights = (1 - (source[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
    return taps, weights


def merge_block_coordinates(
    grid_thw: torch.Tensor, merge: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(row, col, height, width) of every patch, in 2x2-merge-block order."""
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(int(counts.sum()), device=grid_thw.device) - starts) % (
        heights * widths
    )
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    return block_row * merge + in_row, block_col * merge + in_col, heights, widths


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segments: list[int],
    ) -> torch.Tensor:
        tokens = x.size(0)
        query, key, value = (
            self.qkv(x).view(tokens, 3, self.num_heads, self.head_dim).unbind(1)
        )
        # 视觉 RoPE 在 float32 中施加，再 cast 回激活精度（与 HF 相同）。
        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            tensor32 = tensor.float()
            half = tensor32.size(-1) // 2
            rotated = torch.cat((-tensor32[..., half:], tensor32[..., :half]), dim=-1)
            return (tensor32 * cos[:, None] + rotated * sin[:, None]).to(tensor.dtype)

        query, key = rotate(query), rotate(key)
        outputs = []
        offset = 0
        for length in segments:
            window = slice(offset, offset + length)
            offset += length
            outputs.append(
                F.scaled_dot_product_attention(
                    query[window].transpose(0, 1),
                    key[window].transpose(0, 1),
                    value[window].transpose(0, 1),
                ).transpose(0, 1)
            )
        return self.proj(torch.cat(outputs).reshape(tokens, -1))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.mlp.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x, cos, sin, segments):
        x = x + self.attn(self.norm1(x), cos, sin, segments)
        hidden = F.gelu(self.mlp.linear_fc1(self.norm2(x)), approximate="tanh")
        return x + self.mlp.linear_fc2(hidden)


class VisionPatchMerger(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.merged_size = config.hidden_size * config.spatial_merge_size**2
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size, kernel, stride=kernel)


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = _PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        rotary_dim = head_dim // 2
        self.register_buffer(
            "inv_freq",
            1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)),
            persistent=False,
        )
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.depth))
        self.merger = VisionPatchMerger(config)

    def reset_buffers(self, device: torch.device | str) -> None:
        rotary_dim = self.inv_freq.numel() * 2
        indices = torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device)
        self.inv_freq = 1.0 / (10000.0 ** (indices / rotary_dim))

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """``[num_patches, patch_dim]`` pixels -> ``[num_patches / 4, out_hidden_size]``."""
        config = self.config
        weight = self.patch_embed.proj.weight
        # Conv3d 的 kernel 等于 stride，所以它就是对展平 patch 的一次线性变换。
        hidden = F.linear(
            pixel_values.to(weight.dtype),
            weight.reshape(weight.size(0), -1),
            self.patch_embed.proj.bias,
        )
        row, col, heights, widths = merge_block_coordinates(grid_thw, config.spatial_merge_size)
        side = self.num_grid_per_side
        row_taps, row_weights = _axis_taps(row, heights, side)
        col_taps, col_weights = _axis_taps(col, widths, side)
        indices = (row_taps[:, :, None] * side + col_taps[:, None, :]).reshape(-1, 4)
        weights = (row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, 4)
        positions = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden = hidden + positions.to(hidden.dtype)

        frequencies = torch.cat(
            (row[:, None] * self.inv_freq, col[:, None] * self.inv_freq), dim=-1
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = embedding.cos(), embedding.sin()
        # 每一帧单独做注意力：图片 t=1，一张图就是一个段。
        segments = [
            int(h * w) for t, h, w in grid_thw.tolist() for _ in range(int(t))
        ]
        for block in self.blocks:
            hidden = block(hidden, cos, sin, segments)
        return self.merger(hidden)
