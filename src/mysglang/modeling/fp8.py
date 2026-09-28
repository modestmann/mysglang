"""Block-wise FP8 weights used by official Qwen FP8 checkpoints.

A ``[N, K]`` weight is split into ``[block_n, block_k]`` tiles. Each tile stores e4m3
values plus one float32 ``weight_scale_inv``; the real weight is ``fp8 * scale``.
The reference path dequantizes to the activation dtype and calls ``F.linear``. On CUDA
a Triton kernel reads the 1-byte weights directly and applies each tile's scale to its
partial dot product, so Decode moves half the weight bytes of BF16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _expand_scale(
    scale: torch.Tensor,
    shape: torch.Size,
    block_size: tuple[int, int],
) -> torch.Tensor:
    block_n, block_k = block_size
    rows, columns = shape[-2], shape[-1]
    expanded = scale.repeat_interleave(block_n, dim=-2).repeat_interleave(block_k, dim=-1)
    return expanded[..., :rows, :columns]


@torch.no_grad()
def quantize_block_fp8(
    weight: torch.Tensor,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., N, K]`` weights exactly like the official block FP8 export."""
    block_n, block_k = block_size
    rows, columns = weight.shape[-2], weight.shape[-1]
    padded_rows = ceil_div(rows, block_n) * block_n
    padded_columns = ceil_div(columns, block_k) * block_k
    source = weight.float()
    if (padded_rows, padded_columns) != (rows, columns):
        source = F.pad(source, (0, padded_columns - columns, 0, padded_rows - rows))
    tiles = source.reshape(
        *source.shape[:-2],
        padded_rows // block_n,
        block_n,
        padded_columns // block_k,
        block_k,
    )
    amax = tiles.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    quantized = (tiles / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    quantized = quantized.reshape(source.shape)[..., :rows, :columns].contiguous()
    return quantized, scale.squeeze(-1).squeeze(-2).contiguous()


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize the real weight; used by CPU tests and ``--dequantize-fp8`` loads."""
    return (weight.float() * _expand_scale(scale.float(), weight.shape, block_size)).to(dtype)


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """``x @ dequant(weight).T`` without materializing the dequantized weight on CUDA."""
    if x.is_cuda and x.dtype in {torch.float16, torch.bfloat16}:
        from .triton_fp8 import triton_fp8_block_linear

        return triton_fp8_block_linear(x, weight, scale, block_size)
    return F.linear(x, dequantize_block_fp8(weight, scale, block_size, x.dtype))
