from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - exercised only without the optional extra
    raise ImportError("FP8 CUDA kernels require the 'flash-attn' project extra") from error


# W8A16：权重保持 1 字节 e4m3，在 kernel 内转成激活精度参与 tl.dot。每个 K tile 落在同一个
# scale block 内，因此部分和乘以该列所属 block 的 scale 即等价于先反量化再做 GEMM。
@triton.jit
def _fp8_block_linear_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    SCALE_COLUMNS,
    SCALE_BLOCK_N: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale_rows = columns // SCALE_BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        inputs = tl.load(
            x_ptr + rows[:, None] * stride_xm + k_offsets[None, :],
            mask=(rows[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        weights = tl.load(
            weight_ptr + columns[None, :] * stride_wn + k_offsets[:, None],
            mask=(columns[None, :] < N) & (k_offsets[:, None] < K),
            other=0.0,
        )
        scales = tl.load(
            scale_ptr + scale_rows * SCALE_COLUMNS + k_start // SCALE_BLOCK_K,
            mask=columns < N,
            other=0.0,
        )
        partial = tl.dot(inputs, weights.to(inputs.dtype))
        accumulator += partial * scales[None, :]
    tl.store(
        output_ptr + rows[:, None] * stride_om + columns[None, :],
        accumulator.to(output_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )


def _tile_sizes(rows: int, block_size: tuple[int, int]) -> tuple[int, int, int]:
    block_n, block_k = block_size
    block_m = 16 if rows <= 16 else 32 if rows <= 64 else 64
    tile_n = 64
    tile_k = min(128, block_k)
    while block_k % tile_k:
        tile_k //= 2
    if tile_k < 16:
        raise ValueError("FP8 CUDA kernel requires weight_block_size[1] >= 16")
    return block_m, tile_n, tile_k


def triton_fp8_block_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 linear expects float8_e4m3fn weights")
    if x.size(-1) != weight.size(1):
        raise ValueError("FP8 linear input width does not match the weight")
    original_shape = x.shape
    inputs = x.reshape(-1, x.size(-1)).contiguous()
    weight = weight.contiguous()
    scale = scale.contiguous()
    rows, columns, depth = inputs.size(0), weight.size(0), weight.size(1)
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    if rows == 0:
        return output.reshape(*original_shape[:-1], columns)
    block_m, block_n, block_k = _tile_sizes(rows, block_size)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(columns, block_n))
    _fp8_block_linear_kernel[grid](
        inputs,
        weight,
        scale,
        output,
        rows,
        columns,
        depth,
        inputs.stride(0),
        weight.stride(0),
        output.stride(0),
        scale.size(1),
        SCALE_BLOCK_N=block_size[0],
        SCALE_BLOCK_K=block_size[1],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output.reshape(*original_shape[:-1], columns)
