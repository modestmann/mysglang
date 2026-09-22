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
    row_offsets_ptr,
    tile_offsets_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    row_offsets_ptr,
    tile_offsets_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
) -> torch.Tensor:
    """Evaluate real expert assignments with two no-padding Triton launches."""
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

    order = torch.argsort(local_experts, stable=True)
    sorted_inputs = inputs[order].contiguous()
    sorted_experts = local_experts[order]
    counts = torch.bincount(sorted_experts, minlength=num_experts).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=inputs.device)
    row_offsets = torch.cat((zero, counts.cumsum(0, dtype=torch.int32)))

    block_m = 16
    block_n = 64
    block_k = 32
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
        row_offsets,
        tile_offsets,
        activated,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        NUM_EXPERTS=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    sorted_output = torch.empty_like(sorted_inputs)
    _grouped_down_kernel[(max_tiles, triton.cdiv(hidden_size, block_n))](
        activated,
        down_proj,
        row_offsets,
        tile_offsets,
        sorted_output,
        HIDDEN_SIZE=hidden_size,
        INTERMEDIATE_SIZE=intermediate_size,
        NUM_EXPERTS=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    output = torch.empty_like(sorted_output)
    output[order] = sorted_output
    return output
