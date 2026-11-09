"""小模型 FP32 数学基准；不模拟激活量化，也不调用 CUDA 扩展。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import DeepSeekV41Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def rotary(
    x: torch.Tensor,
    positions: torch.Tensor,
    config: DeepSeekV41Config,
    compressed: bool,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """只旋转末尾 RoPE 维度，使用相邻元素配对；输出投影前施加逆旋转。"""
    dim = config.rope_head_dim
    base = config.compress_rope_theta if compressed else config.rope_theta
    freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    if compressed and config.original_seq_len:

        def correction(rotations: float) -> float:
            return (
                dim
                * math.log(config.original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), dim - 1)
        ramp = (torch.arange(dim // 2, device=x.device) - low) / max(high - low, 1e-3)
        ramp = ramp.clamp(0, 1)
        freq = freq * (1 - ramp) + freq / config.rope_factor * ramp
    angles = positions.float().unsqueeze(-1) * freq
    shape = (1, len(positions)) + (1,) * (x.ndim - 3) + (dim // 2,)
    cos, sin = angles.cos().view(shape), angles.sin().view(shape)
    if inverse:
        sin = -sin
    pairs = x[..., -dim:].unflatten(-1, (-1, 2))
    a, b = pairs.unbind(-1)
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)
    return torch.cat((x[..., :-dim], rotated), -1)


def hc_split(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, config: DeepSeekV41Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, eps = config.hc_mult, config.hc_eps
    pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + eps
    post = 2 * torch.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).unflatten(-1, (n, n))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(config.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attention(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    picked = kv[batch, indices.clamp_min(0).long()]
    logits = torch.einsum("bshd,bskd->bshk", q, picked) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(indices.unsqueeze(-2) < 0, -torch.inf)
    # sink 只增加 softmax 分母，没有对应的 value；全无效行应输出零。
    sink_logits = sink.view(1, 1, -1, 1).expand(*logits.shape[:-1], 1)
    weights = torch.cat((logits, sink_logits), -1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", weights, picked)


def candidate_mask(
    scores: torch.Tensor, lengths: torch.Tensor, block_size: int, topk_blocks: int
) -> torch.Tensor:
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    blocks = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    blocks = blocks.unflatten(-1, (-1, block_size)).amax(-1)
    newest = (lengths - 1) // block_size
    blocks = blocks.masked_fill(
        torch.arange(blocks.shape[-1], device=scores.device) == newest, torch.inf
    )
    # 同分按更早的位置优先，避免被 masked future 列数影响（torch.topk 不保证稳定）。
    ids = blocks.argsort(dim=-1, descending=True, stable=True)[..., :topk_blocks]
    values = blocks.gather(-1, ids)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, ids, values > -torch.inf)
    return keep.repeat_interleave(block_size, -1)[..., :width]
