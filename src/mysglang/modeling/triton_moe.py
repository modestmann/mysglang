from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "the triton_grouped MoE backend requires the 'flash-attn' project extra"
    ) from error


# Each expert owns a variable number of consecutive rows after dispatch sorting.  These
# kernels use row_offsets to find those real rows; unlike the portable grouped baseline,
# they never materialize [num_active_experts, max_tokens_per_expert, ...] padding.
@triton.jit
def _grouped_swiglu_kernel(
    inputs_ptr,
    gate_up_ptr,
    gate_up_scale_ptr,
    row_offsets_ptr,
    tile_offsets_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    output_tile_idx = tl.program_id(1)
    total_tiles = tl.load(tile_offsets_ptr + NUM_EXPERTS)
    if tile_idx < total_tiles:
        # tile_offsets[e] is the first M tile belonging to expert e.  Counting how
        # many expert ends this tile has crossed identifies its expert without a CPU
        # list or one launch per expert.
        expert_idx = 0
        for expert in tl.static_range(NUM_EXPERTS):
            expert_idx += tile_idx >= tl.load(tile_offsets_ptr + expert + 1)

        expert_tile_start = tl.load(tile_offsets_ptr + expert_idx)
        expert_row_start = tl.load(row_offsets_ptr + expert_idx)
        expert_row_end = tl.load(row_offsets_ptr + expert_idx + 1)
        rows = expert_row_start + (tile_idx - expert_tile_start) * BLOCK_M
        rows += tl.arange(0, BLOCK_M)
        columns = output_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        expert_weight_start = expert_idx * 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE
        # FP8 专家：每个 [SCALE_BLOCK_N, SCALE_BLOCK_K] 权重 tile 共享一个 float32 scale。
        scale_columns = tl.cdiv(HIDDEN_SIZE, SCALE_BLOCK_K)
        expert_scale_start = expert_idx * tl.cdiv(2 * INTERMEDIATE_SIZE, SCALE_BLOCK_N)
        expert_scale_start = expert_scale_start * scale_columns
        for hidden_start in range(0, HIDDEN_SIZE, BLOCK_K):
            hidden = hidden_start + tl.arange(0, BLOCK_K)
            inputs = tl.load(
                inputs_ptr + rows[:, None] * HIDDEN_SIZE + hidden[None, :],
                mask=(rows[:, None] < expert_row_end) & (hidden[None, :] < HIDDEN_SIZE),
                other=0.0,
            )
            gate_weights = tl.load(
                gate_up_ptr
                + expert_weight_start
                + columns[None, :] * HIDDEN_SIZE
                + hidden[:, None],
                mask=(hidden[:, None] < HIDDEN_SIZE)
                & (columns[None, :] < INTERMEDIATE_SIZE),
                other=0.0,
            )
            up_weights = tl.load(
                gate_up_ptr
                + expert_weight_start
                + (INTERMEDIATE_SIZE + columns[None, :]) * HIDDEN_SIZE
                + hidden[:, None],
                mask=(hidden[:, None] < HIDDEN_SIZE)
                & (columns[None, :] < INTERMEDIATE_SIZE),
                other=0.0,
            )
            if HAS_SCALE:
                scale_k = hidden_start // SCALE_BLOCK_K
                gate_scales = tl.load(
                    gate_up_scale_ptr
                    + expert_scale_start
                    + (columns // SCALE_BLOCK_N) * scale_columns
                    + scale_k,
                    mask=columns < INTERMEDIATE_SIZE,
                    other=0.0,
                )
                up_scales = tl.load(
                    gate_up_scale_ptr
                    + expert_scale_start
                    + ((INTERMEDIATE_SIZE + columns) // SCALE_BLOCK_N) * scale_columns
                    + scale_k,
                    mask=columns < INTERMEDIATE_SIZE,
                    other=0.0,
                )
                gate += tl.dot(inputs, gate_weights.to(inputs.dtype)) * gate_scales[None, :]
                up += tl.dot(inputs, up_weights.to(inputs.dtype)) * up_scales[None, :]
            else:
                gate += tl.dot(inputs, gate_weights)
                up += tl.dot(inputs, up_weights)

        # Match the two-F.linear reference's BF16/FP16 rounding point before SiLU,
        # while still avoiding a materialized [M, 2 * intermediate] tensor.
        gate = gate.to(output_ptr.dtype.element_ty)
        up = up.to(output_ptr.dtype.element_ty)
        silu = gate * tl.sigmoid(gate.to(tl.float32))
        activated = silu.to(output_ptr.dtype.element_ty) * up
        tl.store(
            output_ptr + rows[:, None] * INTERMEDIATE_SIZE + columns[None, :],
            activated,
            mask=(rows[:, None] < expert_row_end)
            & (columns[None, :] < INTERMEDIATE_SIZE),
        )


@triton.jit
def _grouped_down_kernel(
    inputs_ptr,
    down_ptr,
    down_scale_ptr,
    row_offsets_ptr,
    tile_offsets_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    output_tile_idx = tl.program_id(1)
    total_tiles = tl.load(tile_offsets_ptr + NUM_EXPERTS)
    if tile_idx < total_tiles:
        expert_idx = 0
        for expert in tl.static_range(NUM_EXPERTS):
            expert_idx += tile_idx >= tl.load(tile_offsets_ptr + expert + 1)

        expert_tile_start = tl.load(tile_offsets_ptr + expert_idx)
        expert_row_start = tl.load(row_offsets_ptr + expert_idx)
        expert_row_end = tl.load(row_offsets_ptr + expert_idx + 1)
        rows = expert_row_start + (tile_idx - expert_tile_start) * BLOCK_M
        rows += tl.arange(0, BLOCK_M)
        columns = output_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        expert_weight_start = expert_idx * HIDDEN_SIZE * INTERMEDIATE_SIZE
        scale_columns = tl.cdiv(INTERMEDIATE_SIZE, SCALE_BLOCK_K)
        expert_scale_start = expert_idx * tl.cdiv(HIDDEN_SIZE, SCALE_BLOCK_N) * scale_columns
        for intermediate_start in range(0, INTERMEDIATE_SIZE, BLOCK_K):
            intermediate = intermediate_start + tl.arange(0, BLOCK_K)
            inputs = tl.load(
                inputs_ptr
                + rows[:, None] * INTERMEDIATE_SIZE
                + intermediate[None, :],
                mask=(rows[:, None] < expert_row_end)
                & (intermediate[None, :] < INTERMEDIATE_SIZE),
                other=0.0,
            )
            weights = tl.load(
                down_ptr
                + expert_weight_start
                + columns[None, :] * INTERMEDIATE_SIZE
                + intermediate[:, None],
                mask=(intermediate[:, None] < INTERMEDIATE_SIZE)
                & (columns[None, :] < HIDDEN_SIZE),
                other=0.0,
            )
            if HAS_SCALE:
                scales = tl.load(
                    down_scale_ptr
                    + expert_scale_start
                    + (columns // SCALE_BLOCK_N) * scale_columns
                    + intermediate_start // SCALE_BLOCK_K,
                    mask=columns < HIDDEN_SIZE,
                    other=0.0,
                )
                output += tl.dot(inputs, weights.to(inputs.dtype)) * scales[None, :]
            else:
                output += tl.dot(inputs, weights)

        tl.store(
            output_ptr + rows[:, None] * HIDDEN_SIZE + columns[None, :],
            output,
            mask=(rows[:, None] < expert_row_end) & (columns[None, :] < HIDDEN_SIZE),
        )


def triton_grouped_mlp(
    inputs: torch.Tensor,
    local_experts: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    gate_up_scale: torch.Tensor | None = None,
    down_scale: torch.Tensor | None = None,
    block_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Evaluate real expert assignments with two no-padding Triton launches.

    With ``block_size`` the expert weights are block FP8 and the kernels scale each
    partial dot product instead of materializing dequantized experts.
    """
    if not inputs.is_cuda:
        raise ValueError("Triton grouped MoE requires CUDA tensors")
    if inputs.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Triton grouped MoE supports float16 and bfloat16")
    if inputs.ndim != 2 or local_experts.ndim != 1:
        raise ValueError("expert inputs must be rank 2 and expert IDs rank 1")
    if inputs.size(0) != local_experts.numel():
        raise ValueError("expert inputs and IDs must contain the same assignments")
    tensors_are_contiguous = (
        inputs.is_contiguous()
        and gate_up_proj.is_contiguous()
        and down_proj.is_contiguous()
    )
    if not tensors_are_contiguous:
        raise ValueError("Triton grouped MoE expects contiguous inputs and weights")
    if inputs.size(0) == 0:
        return torch.empty_like(inputs)

    num_experts = gate_up_proj.size(0)
    hidden_size = inputs.size(1)
    intermediate_size = down_proj.size(2)
    if gate_up_proj.shape != (num_experts, 2 * intermediate_size, hidden_size):
        raise ValueError("gate/up expert weights do not match the input shape")
    if down_proj.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("down expert weights do not match the input shape")

    # TP ranks see the same fixed M = batch * top_k assignments, but only own a
    # data-dependent subset of experts.  A sentinel sorts non-local assignments to
    # the end; their zero outputs keep every rank's tensor shapes graph-stable.
    owned = (local_experts >= 0) & (local_experts < num_experts)
    dispatch_experts = torch.where(owned, local_experts, num_experts)
    order = torch.argsort(dispatch_experts, stable=True)
    sorted_inputs = inputs[order].contiguous()
    counts = torch.zeros(num_experts, dtype=torch.int32, device=inputs.device)
    counts.scatter_add_(
        0,
        local_experts.clamp(0, num_experts - 1),
        owned.to(torch.int32),
    )
    zero = torch.zeros(1, dtype=torch.int32, device=inputs.device)
    row_offsets = torch.cat((zero, counts.cumsum(0, dtype=torch.int32)))

    has_scale = block_size is not None
    if has_scale and (gate_up_scale is None or down_scale is None):
        raise ValueError("FP8 grouped MoE requires gate/up and down scales")
    scale_block_n, scale_block_k = block_size if has_scale else (1, 1)
    block_m = 16
    block_n = 64
    block_k = 32
    if has_scale and scale_block_k % block_k:
        raise ValueError("FP8 expert block_k must be a multiple of 32")
    # 无量化时 scale 指针不会被读取，传入任意有效张量即可。
    gate_up_scale = gate_up_scale if has_scale else gate_up_proj
    down_scale = down_scale if has_scale else down_proj
    tile_counts = torch.div(counts + block_m - 1, block_m, rounding_mode="floor")
    tile_offsets = torch.cat((zero, tile_counts.cumsum(0, dtype=torch.int32)))
    # sum(ceil(M_e / BLOCK_M)) <= ceil(sum(M_e) / BLOCK_M) + E - 1.
    # Launching that small upper bound keeps the grid independent of a GPU -> CPU .item().
    max_tiles = triton.cdiv(inputs.size(0), block_m) + num_experts

    activated = torch.empty(
        (inputs.size(0), intermediate_size),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    _grouped_swiglu_kernel[(max_tiles, triton.cdiv(intermediate_size, block_n))](
        sorted_inputs,
        gate_up_proj,
        gate_up_scale,
        row_offsets,
        tile_offsets,
        activated,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        NUM_EXPERTS=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_SCALE=has_scale,
        SCALE_BLOCK_N=scale_block_n,
        SCALE_BLOCK_K=scale_block_k,
        num_warps=4,
    )

    # Invalid/sentinel rows are never visited by the kernels and must contribute zero
    # when every TP rank later all-reduces its owned expert outputs.
    sorted_output = torch.zeros_like(sorted_inputs)
    _grouped_down_kernel[(max_tiles, triton.cdiv(hidden_size, block_n))](
        activated,
        down_proj,
        down_scale,
        row_offsets,
        tile_offsets,
        sorted_output,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        NUM_EXPERTS=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_SCALE=has_scale,
        SCALE_BLOCK_N=scale_block_n,
        SCALE_BLOCK_K=scale_block_k,
        num_warps=4,
    )

    output = torch.empty_like(sorted_output)
    output[order] = sorted_output
    return output
