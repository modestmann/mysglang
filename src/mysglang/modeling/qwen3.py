from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mysglang.cache import (
    PagedKVBatch,
    PagedKVCache,
    RecurrentStateBatch,
    RecurrentStateSpec,
)
from mysglang.config import LINEAR_ATTENTION, Fp8QuantConfig, ModelConfig

from .attention import AttentionBackend, TorchAttentionBackend
from .deltanet import Qwen3_5GatedDeltaNet
from .fp8 import FP8_DTYPE, ceil_div, dequantize_block_fp8, quantize_block_fp8
from .parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelContext
from .vision import Qwen3_5VisionModel


class RMSNorm(nn.Module):
    """Qwen3 scales the normalized value by ``w``; Qwen3.5 stores ``w - 1`` instead.

    Qwen3.5 checkpoints use a zero-centered weight (initialized to zero) and multiply by
    ``1 + w`` in float32 before casting back, exactly as the Hugging Face reference.
    """

    def __init__(self, hidden_size: int, eps: float, *, zero_centered: bool = False) -> None:
        super().__init__()
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(
            torch.zeros(hidden_size) if zero_centered else torch.ones(hidden_size)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.zero_centered:
            return (normalized * (1.0 + self.weight.float())).to(x.dtype)
        return self.weight * normalized.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Default RoPE on the first ``rotary_dim`` channels of each head.

    Qwen3 rotates the whole head. Qwen3.5 rotates only 25% of it (64 of 256 channels)
    and leaves the rest position-free. Its interleaved M-RoPE assigns the rotary
    frequencies to (t, h, w) position axes as ``[t, h, w, t, h, w, ...]``; text tokens use
    the same position on all axes, which reduces exactly to this 1-D RoPE, so only image
    prompts pass separate ``mrope_positions``.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float,
        rotary_dim: int | None = None,
        mrope_section: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        self.theta = theta
        self.mrope_section = mrope_section
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self, device: torch.device | str | None = None) -> torch.Tensor:
        indices = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
        return 1.0 / self.theta ** (indices / self.rotary_dim)

    def reset_buffer(self, device: torch.device | str) -> None:
        # ``to_empty`` intentionally leaves buffers uninitialized; weights come from the
        # checkpoint, while this deterministic non-persistent RoPE buffer is rebuilt here.
        self.inv_freq = self._make_inv_freq(device)

    def _frequencies(
        self,
        positions: torch.Tensor,
        mrope_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        if mrope_positions is None:
            return positions.float().unsqueeze(-1) * self.inv_freq.float()
        if not self.mrope_section:
            raise ValueError("this model has no M-RoPE sections")
        # [3, ..., rotary/2]：t 轴为底，h/w 轴按交错下标覆盖对应频率。
        axes = mrope_positions.float().unsqueeze(-1) * self.inv_freq.float()
        frequencies = axes[0].clone()
        for axis in (1, 2):
            index = slice(axis, 3 * self.mrope_section[axis], 3)
            frequencies[..., index] = axes[axis][..., index]
        return frequencies

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
        mrope_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = self._frequencies(positions, mrope_positions)
        positions = frequencies[..., 0]
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        cos_values = embeddings.cos().to(dtype=query.dtype)
        sin_values = embeddings.sin().to(dtype=query.dtype)
        if query.ndim == 3 and positions.ndim == 1:
            cos = cos_values[:, None, :]
            sin = sin_values[:, None, :]
        elif query.ndim == 4 and positions.ndim == 1:
            cos = cos_values[None, None, :, :]
            sin = sin_values[None, None, :, :]
        elif query.ndim == 4 and positions.ndim == 2:
            cos = cos_values[:, None, :, :]
            sin = sin_values[:, None, :, :]
        else:
            raise ValueError("positions shape does not match dense or packed Q/K")
        if self.rotary_dim == self.head_dim:
            return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)
        dim = self.rotary_dim
        return (
            torch.cat((_apply_rope(query[..., :dim], cos, sin), query[..., dim:]), dim=-1),
            torch.cat((_apply_rope(key[..., :dim], cos, sin), key[..., dim:]), dim=-1),
        )


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


QuantizeFn = Callable[..., "Fp8QuantConfig | None"]
# (flat token rows, replacement embeddings): image placeholders -> vision features.
EmbeddingOverrides = tuple[torch.Tensor, torch.Tensor]


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        backend: AttentionBackend,
        tensor_parallel: TensorParallelContext,
        *,
        kv_layer_idx: int | None = None,
        module_prefix: str | None = None,
        quantize: QuantizeFn | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # Paged K/V 只为 full-attention 层分配，因此 cache 层号不一定等于模型层号。
        self.kv_layer_idx = layer_idx if kv_layer_idx is None else kv_layer_idx
        self.backend = backend
        self.tensor_parallel = tensor_parallel
        self.num_heads = tensor_parallel.local_size(
            config.num_attention_heads,
            "num_attention_heads",
        )
        # KV 头少于 TP rank 时按 GQA 分组复制：例如 2 个 KV 头、TP=4，rank 0/1 共用头 0。
        self.kv_shards = kv_shard_count(config, tensor_parallel)
        self.num_kv_heads = config.num_key_value_heads // self.kv_shards
        assert config.head_dim is not None
        self.head_dim = config.head_dim
        # Qwen3.5 的 q_proj 每个头输出 [query, gate] 两段，gate 过 sigmoid 后乘到注意力输出上。
        self.output_gate = config.attn_output_gate
        q_width = self.head_dim * (2 if self.output_gate else 1)
        self.global_q_size = config.num_attention_heads * q_width
        self.global_kv_size = config.num_key_value_heads * self.head_dim
        self.q_size = self.num_heads * q_width
        self.kv_size = self.num_kv_heads * self.head_dim
        prefix = module_prefix or f"layers.{layer_idx}.self_attn"
        # Hugging Face stores Q/K/V separately. The loader independently takes this
        # rank's heads from each tensor, then packs them as [Q_rank, K_rank, V_rank].
        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            (self.global_q_size, self.global_kv_size, self.global_kv_size),
            tensor_parallel,
            shard_counts=(tensor_parallel.world_size, self.kv_shards, self.kv_shards),
            quant=(
                quantize(f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj")
                if quantize
                else None
            ),
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            tensor_parallel,
            quant=quantize(f"{prefix}.o_proj") if quantize else None,
        )
        zero_centered = config.is_qwen3_5
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, zero_centered=zero_centered)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, zero_centered=zero_centered)
        self.rope = RotaryEmbedding(
            self.head_dim, config.rope_theta, config.rotary_dim, config.mrope_section
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
        mrope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim not in {2, 3}:
            raise ValueError("hidden states must be dense rank 3 or packed rank 2")
        leading = x.shape[:-1]
        query, key, value = self.qkv_proj(x).split(
            (self.q_size, self.kv_size, self.kv_size),
            dim=-1,
        )
        gate = None
        if self.output_gate:
            query, gate = query.view(*leading, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
            gate = gate.reshape(*leading, self.num_heads * self.head_dim)
        query = self.q_norm(query.reshape(*leading, self.num_heads, self.head_dim))
        key = self.k_norm(key.reshape(*leading, self.num_kv_heads, self.head_dim))
        value = value.reshape(*leading, self.num_kv_heads, self.head_dim)
        if x.ndim == 3:
            # Dense attention uses [batch, heads, sequence, head_dim].
            query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        query, key = self.rope(query, key, positions, mrope_positions)
        output = self.backend.forward(
            query,
            key,
            value,
            layer_idx=self.kv_layer_idx,
            kv_cache=kv_cache,
            cache_batch=cache_batch,
        )
        if output.ndim == 4:
            output = output.transpose(1, 2).contiguous().view(*leading, -1)
        else:
            output = output.reshape(output.size(0), -1)
        if gate is not None:
            output = output * torch.sigmoid(gate)
        return self.o_proj(output)


def kv_shard_count(config: ModelConfig, tensor_parallel: TensorParallelContext) -> int:
    """Distinct K/V head shards; below TP size each shard is replicated on several ranks."""
    world_size = tensor_parallel.world_size
    if config.num_key_value_heads >= world_size:
        tensor_parallel.local_size(config.num_key_value_heads, "num_key_value_heads")
        return world_size
    if world_size % config.num_key_value_heads:
        raise ValueError(
            f"TP={world_size} must be a multiple of num_key_value_heads="
            f"{config.num_key_value_heads} to replicate K/V heads"
        )
    return config.num_key_value_heads


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tensor_parallel: TensorParallelContext,
        *,
        module_prefix: str = "",
        quantize: QuantizeFn | None = None,
    ) -> None:
        super().__init__()
        self.global_intermediate_size = intermediate_size
        self.intermediate_size = tensor_parallel.local_size(
            intermediate_size,
            "intermediate_size",
        )
        # As with QKV, gate and up are independently sharded before local fusion.
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            (intermediate_size, intermediate_size),
            tensor_parallel,
            quant=(
                quantize(f"{module_prefix}.gate_proj", f"{module_prefix}.up_proj")
                if quantize
                else None
            ),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tensor_parallel,
            quant=quantize(f"{module_prefix}.down_proj") if quantize else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen3Experts(nn.Module):
    """Expert-sharded top-k dispatch with selectable communication and GEMM paths.

    ``naive``, ``sorted``, ``grouped`` and ``triton_grouped`` keep token hidden states
    replicated and combine rank-local expert contributions with all-reduce. ``all_to_all``
    first gives each rank a contiguous token shard, dispatches its assignments to expert
    owners, returns expert outputs to token owners, then all-gathers token outputs because
    the following Attention TP layer still requires replicated tokens.
    """

    keep_dtype_parameters: tuple[str, ...] = ()

    def __init__(
        self,
        config: ModelConfig,
        expert_parallel: TensorParallelContext,
        dispatch_backend: str,
        *,
        quant: Fp8QuantConfig | None = None,
    ) -> None:
        super().__init__()
        valid_backends = {"naive", "sorted", "grouped", "triton_grouped", "all_to_all"}
        if dispatch_backend not in valid_backends:
            raise ValueError(
                "MoE dispatch backend must be 'naive', 'sorted', 'grouped', "
                "'triton_grouped' or 'all_to_all'"
            )
        self.num_experts = config.num_experts
        self.expert_parallel = expert_parallel
        self.dispatch_backend = dispatch_backend
        self.cuda_graph_static_dispatch = False
        self.num_local_experts = expert_parallel.local_size(
            config.num_experts,
            "num_experts",
        )
        self.local_expert_start = expert_parallel.rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.intermediate_size = config.moe_intermediate_size
        self.quant = quant
        gate_up_shape = (
            self.num_local_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        down_shape = (self.num_local_experts, config.hidden_size, config.moe_intermediate_size)
        if quant is None:
            self.gate_up_proj = nn.Parameter(torch.empty(gate_up_shape))
            self.down_proj = nn.Parameter(torch.empty(down_shape))
        else:
            block_n, block_k = quant.weight_block_size
            if config.moe_intermediate_size % block_n:
                # gate/up 在输出维拼接，每段必须从 scale tile 边界开始。
                raise ValueError("FP8 expert intermediate size must be a multiple of block_n")
            self.keep_dtype_parameters = (
                "gate_up_proj",
                "down_proj",
                "gate_up_proj_scale_inv",
                "down_proj_scale_inv",
            )
            self.gate_up_proj = nn.Parameter(
                torch.empty(gate_up_shape, dtype=FP8_DTYPE), requires_grad=False
            )
            self.down_proj = nn.Parameter(
                torch.empty(down_shape, dtype=FP8_DTYPE), requires_grad=False
            )
            self.gate_up_proj_scale_inv = nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    ceil_div(gate_up_shape[1], block_n),
                    ceil_div(gate_up_shape[2], block_k),
                ),
                requires_grad=False,
            )
            self.down_proj_scale_inv = nn.Parameter(
                torch.empty(
                    self.num_local_experts,
                    ceil_div(down_shape[1], block_n),
                    ceil_div(down_shape[2], block_k),
                ),
                requires_grad=False,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.quant is None:
            for expert_idx in range(self.num_local_experts):
                nn.init.kaiming_uniform_(self.gate_up_proj[expert_idx], a=5**0.5)
                nn.init.kaiming_uniform_(self.down_proj[expert_idx], a=5**0.5)
            return
        for weight, scale in (
            (self.gate_up_proj, self.gate_up_proj_scale_inv),
            (self.down_proj, self.down_proj_scale_inv),
        ):
            full = torch.empty(weight.shape, device=weight.device)
            for expert_idx in range(self.num_local_experts):
                nn.init.kaiming_uniform_(full[expert_idx], a=5**0.5)
            quantized, scales = quantize_block_fp8(full, self.quant.weight_block_size)
            with torch.no_grad():
                weight.copy_(quantized)
                scale.copy_(scales)

    def _gate_up(self, expert_index, dtype: torch.dtype) -> torch.Tensor:
        """One expert (int) or a batch of experts (tensor) in the activation dtype."""
        if self.quant is None:
            return self.gate_up_proj[expert_index]
        return dequantize_block_fp8(
            self.gate_up_proj[expert_index],
            self.gate_up_proj_scale_inv[expert_index],
            self.quant.weight_block_size,
            dtype,
        )

    def _down(self, expert_index, dtype: torch.dtype) -> torch.Tensor:
        if self.quant is None:
            return self.down_proj[expert_index]
        return dequantize_block_fp8(
            self.down_proj[expert_index],
            self.down_proj_scale_inv[expert_index],
            self.quant.weight_block_size,
            dtype,
        )

    @property
    def source_experts(self) -> slice:
        return slice(self.local_expert_start, self.local_expert_end)

    def local_index(self, global_expert_index: int) -> int | None:
        if not self.local_expert_start <= global_expert_index < self.local_expert_end:
            return None
        return global_expert_index - self.local_expert_start

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.dispatch_backend == "all_to_all":
            return self._forward_all_to_all(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        if self.dispatch_backend in {"grouped", "triton_grouped"}:
            result = self._forward_grouped(
                hidden_states,
                selected_experts,
                routing_weights,
                use_triton=self.dispatch_backend == "triton_grouped",
            )
        elif self.dispatch_backend == "sorted":
            result = self._forward_sorted(hidden_states, selected_experts, routing_weights)
        else:
            result = self._forward_naive(hidden_states, selected_experts, routing_weights)
        return self.expert_parallel.all_reduce(result)

    def _forward_naive(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(hidden_states)
        for local_expert_idx in range(self.num_local_experts):
            global_expert_idx = self.local_expert_start + local_expert_idx
            token_idx, top_k_slot = torch.where(selected_experts == global_expert_idx)
            if token_idx.numel() == 0:
                continue
            current = hidden_states[token_idx]
            gate, up = F.linear(
                current, self._gate_up(local_expert_idx, current.dtype)
            ).chunk(2, dim=-1)
            current = F.linear(F.silu(gate) * up, self._down(local_expert_idx, current.dtype))
            current = current * routing_weights[token_idx, top_k_slot, None]
            result.index_add_(0, token_idx, current.to(result.dtype))
        return result

    def _grouped_expert_gemm(
        self,
        inputs: torch.Tensor,
        local_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Run active experts as two padded batched GEMMs and preserve assignment order.

        Native grouped-GEMM availability differs across supported PyTorch/CUDA builds. This
        portable baseline groups variable-size expert batches into padded active-expert
        matrices, replacing the Python per-expert GEMM loop with two ``bmm`` launches. The
        cloud benchmark decides whether padding or launch reduction wins for a workload.
        """
        if inputs.size(0) != local_experts.numel():
            raise ValueError("expert inputs and IDs must contain the same assignments")
        if inputs.size(0) == 0:
            return torch.empty_like(inputs)

        order = torch.argsort(local_experts, stable=True)
        sorted_experts = local_experts[order]
        sorted_inputs = inputs[order]
        active_experts, counts = torch.unique_consecutive(
            sorted_experts,
            return_counts=True,
        )
        max_count = int(counts.max().item())
        # A badly imbalanced router could otherwise allocate
        # [num_active_experts, max_count, ...] close to num_experts times too large.
        # Preserve correctness and memory safety by falling back only for that skewed case.
        padded_assignments = active_experts.numel() * max_count
        if padded_assignments > inputs.size(0) * 2:
            sorted_output = torch.empty_like(sorted_inputs)
            start = 0
            for expert_idx, count in zip(active_experts.tolist(), counts.tolist()):
                end = start + count
                gate, up = F.linear(
                    sorted_inputs[start:end],
                    self._gate_up(expert_idx, inputs.dtype),
                ).chunk(2, dim=-1)
                sorted_output[start:end] = F.linear(
                    F.silu(gate) * up,
                    self._down(expert_idx, inputs.dtype),
                )
                start = end
            output = torch.empty_like(sorted_output)
            output[order] = sorted_output
            return output
        active_batch = torch.repeat_interleave(
            torch.arange(active_experts.numel(), device=inputs.device),
            counts,
        )
        starts = torch.cumsum(counts, dim=0) - counts
        slots = torch.arange(inputs.size(0), device=inputs.device) - torch.repeat_interleave(
            starts,
            counts,
        )

        padded = inputs.new_zeros((active_experts.numel(), max_count, inputs.size(-1)))
        padded[active_batch, slots] = sorted_inputs
        gate_up = torch.bmm(
            padded,
            self._gate_up(active_experts, inputs.dtype).transpose(1, 2),
        )
        gate, up = gate_up.chunk(2, dim=-1)
        padded_output = torch.bmm(
            F.silu(gate) * up,
            self._down(active_experts, inputs.dtype).transpose(1, 2),
        )
        sorted_output = padded_output[active_batch, slots]
        output = torch.empty_like(sorted_output)
        output[order] = sorted_output
        return output

    def _loop_expert_gemm(
        self,
        inputs: torch.Tensor,
        local_experts: torch.Tensor,
    ) -> torch.Tensor:
        """No-padding reference used when the Triton backend is selected off CUDA."""
        output = torch.zeros_like(inputs)
        for expert_idx in range(self.num_local_experts):
            indices = torch.where(local_experts == expert_idx)[0]
            if indices.numel() == 0:
                continue
            gate, up = F.linear(
                inputs[indices], self._gate_up(expert_idx, inputs.dtype)
            ).chunk(2, dim=-1)
            output[indices] = F.linear(F.silu(gate) * up, self._down(expert_idx, inputs.dtype))
        return output

    def _triton_grouped_expert_gemm(
        self,
        inputs: torch.Tensor,
        local_experts: torch.Tensor,
    ) -> torch.Tensor:
        # CPU/Gloo tests validate routing and fallback semantics. The actual kernel is
        # intentionally imported only on CUDA so Triton remains an optional dependency.
        if not inputs.is_cuda or inputs.dtype not in {torch.float16, torch.bfloat16}:
            return self._loop_expert_gemm(inputs, local_experts)
        from .triton_moe import triton_grouped_mlp

        if self.quant is None:
            return triton_grouped_mlp(
                inputs.contiguous(),
                local_experts,
                self.gate_up_proj,
                self.down_proj,
            )
        return triton_grouped_mlp(
            inputs.contiguous(),
            local_experts,
            self.gate_up_proj,
            self.down_proj,
            gate_up_scale=self.gate_up_proj_scale_inv,
            down_scale=self.down_proj_scale_inv,
            block_size=self.quant.weight_block_size,
        )

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        use_triton: bool = False,
    ) -> torch.Tensor:
        """Compute this rank's replicated-token assignments with grouped expert GEMMs."""
        top_k = selected_experts.size(1)
        flat_experts = selected_experts.reshape(-1)
        if use_triton and self.cuda_graph_static_dispatch:
            # Keep exactly batch * top_k rows on every TP rank. Non-local experts use
            # sentinel local IDs and produce zero; this removes data-dependent shapes
            # so NCCL all-reduce and the whole Decode forward can enter CUDA Graph.
            assignment_indices = torch.arange(flat_experts.numel(), device=flat_experts.device)
            token_indices = torch.div(assignment_indices, top_k, rounding_mode="floor")
            top_k_slots = assignment_indices.remainder(top_k)
            local_experts = flat_experts - self.local_expert_start
            current = self._triton_grouped_expert_gemm(
                hidden_states[token_indices],
                local_experts,
            )
            current = current * routing_weights[token_indices, top_k_slots, None]
            # Rows already follow token-major, top-k-slot order.  A fixed-order
            # reduction avoids conflicting BF16 atomic adds for one token.
            return current.view(hidden_states.size(0), top_k, -1).sum(dim=1)

        owned = (flat_experts >= self.local_expert_start) & (
            flat_experts < self.local_expert_end
        )
        assignment_indices = torch.where(owned)[0]
        token_indices = torch.div(assignment_indices, top_k, rounding_mode="floor")
        top_k_slots = assignment_indices.remainder(top_k)
        local_experts = flat_experts[assignment_indices] - self.local_expert_start
        expert_inputs = hidden_states[token_indices]
        if use_triton:
            current = self._triton_grouped_expert_gemm(expert_inputs, local_experts)
        else:
            current = self._grouped_expert_gemm(expert_inputs, local_experts)
        current = current * routing_weights[token_indices, top_k_slots, None]
        # Each assignment owns one unique (token, top-k slot).  Scatter into those
        # unique rows first, then reduce slots in a deterministic order.  Direct
        # index_add_ would issue conflicting atomics for every token and can change
        # near-tied greedy logits between otherwise identical BF16 runs.
        assignment_output = hidden_states.new_zeros(
            (flat_experts.numel(), hidden_states.size(-1))
        )
        assignment_output.index_copy_(0, assignment_indices, current)
        return assignment_output.view(hidden_states.size(0), top_k, -1).sum(dim=1)

    @staticmethod
    def _token_shard_sizes(total_tokens: int, world_size: int) -> tuple[int, ...]:
        base, remainder = divmod(total_tokens, world_size)
        return tuple(base + (rank < remainder) for rank in range(world_size))

    def _forward_all_to_all(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch token-owned assignments to expert owners and combine them back."""
        parallel = self.expert_parallel
        if not parallel.enabled:
            return self._forward_grouped(hidden_states, selected_experts, routing_weights)

        token_sizes = self._token_shard_sizes(hidden_states.size(0), parallel.world_size)
        token_start = sum(token_sizes[: parallel.rank])
        token_end = token_start + token_sizes[parallel.rank]
        local_hidden = hidden_states[token_start:token_end]
        local_selected = selected_experts[token_start:token_end]
        local_routing = routing_weights[token_start:token_end]
        top_k = selected_experts.size(1)

        flat_experts = local_selected.reshape(-1)
        local_token_indices = torch.arange(
            local_hidden.size(0),
            device=hidden_states.device,
        ).repeat_interleave(top_k)
        destination_ranks = torch.div(
            flat_experts,
            self.num_local_experts,
            rounding_mode="floor",
        )
        local_experts = flat_experts.remainder(self.num_local_experts)
        # Sorting by (destination rank, local expert) lets the receiver reconstruct expert
        # IDs from a small count matrix instead of sending one int64 ID per assignment.
        dispatch_keys = destination_ranks * self.num_local_experts + local_experts
        order = torch.argsort(dispatch_keys, stable=True)
        send_hidden = local_hidden[local_token_indices[order]]
        send_weights = local_routing.reshape(-1)[order]

        send_expert_counts = torch.bincount(
            dispatch_keys,
            minlength=parallel.world_size * self.num_local_experts,
        ).to(dtype=torch.int64)
        metadata_splits = [self.num_local_experts] * parallel.world_size
        recv_expert_counts = parallel.all_to_all_variable(
            send_expert_counts,
            output_split_sizes=metadata_splits,
            input_split_sizes=metadata_splits,
        ).view(parallel.world_size, self.num_local_experts)
        send_counts = send_expert_counts.view(
            parallel.world_size,
            self.num_local_experts,
        ).sum(dim=1).tolist()
        recv_counts = recv_expert_counts.sum(dim=1).tolist()

        received_hidden = parallel.all_to_all_variable(
            send_hidden,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        received_experts = torch.repeat_interleave(
            torch.arange(self.num_local_experts, device=hidden_states.device).repeat(
                parallel.world_size
            ),
            recv_expert_counts.reshape(-1),
        )
        received_outputs = self._grouped_expert_gemm(received_hidden, received_experts)
        returned_outputs = parallel.all_to_all_variable(
            received_outputs,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
        )

        # The transport order is grouped by destination/expert. Restore the original
        # token-major top-k slot order before reducing expert contributions; otherwise
        # BF16 addition order differs from the reference path and can perturb close logits.
        weighted_outputs = torch.empty_like(returned_outputs)
        weighted_outputs[order] = returned_outputs * send_weights[:, None]
        local_result = weighted_outputs.view(
            local_hidden.size(0),
            top_k,
            hidden_states.size(-1),
        ).sum(dim=1)
        return parallel.all_gather_variable_first_dim(local_result, token_sizes)

    def _forward_sorted(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Group this rank's assignments once instead of scanning tokens per expert."""
        top_k = selected_experts.size(1)
        flat_experts = selected_experts.reshape(-1)
        owned = (flat_experts >= self.local_expert_start) & (
            flat_experts < self.local_expert_end
        )
        assignment_indices = torch.where(owned)[0]
        local_experts = flat_experts[assignment_indices] - self.local_expert_start
        order = torch.argsort(local_experts)
        assignment_indices = assignment_indices[order]
        local_experts = local_experts[order]
        token_indices = torch.div(assignment_indices, top_k, rounding_mode="floor")
        top_k_slots = assignment_indices.remainder(top_k)
        counts = torch.bincount(local_experts, minlength=self.num_local_experts).tolist()

        result = torch.zeros_like(hidden_states)
        start = 0
        for local_expert_idx, count in enumerate(counts):
            end = start + count
            if count:
                current_token_indices = token_indices[start:end]
                current = hidden_states[current_token_indices]
                gate, up = F.linear(
                    current,
                    self._gate_up(local_expert_idx, current.dtype),
                ).chunk(2, dim=-1)
                current = F.linear(
                    F.silu(gate) * up,
                    self._down(local_expert_idx, current.dtype),
                )
                current = current * routing_weights[
                    current_token_indices,
                    top_k_slots[start:end],
                    None,
                ]
                result.index_add_(0, current_token_indices, current.to(result.dtype))
            start = end
        return result


class Qwen3SparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        expert_parallel: TensorParallelContext,
        dispatch_backend: str,
        *,
        module_prefix: str = "",
        quantize: QuantizeFn | None = None,
    ) -> None:
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3Experts(
            config,
            expert_parallel,
            dispatch_backend,
            quant=(
                quantize(
                    f"{module_prefix}.experts.0.gate_proj",
                    f"{module_prefix}.experts.0.up_proj",
                    f"{module_prefix}.experts.0.down_proj",
                )
                if quantize
                else None
            ),
        )
        # Qwen3.5 MoE：每个 token 另经过一个共享专家，由 sigmoid 标量门控后与路由专家相加。
        self.shared_expert: Qwen3MLP | None = None
        self.shared_expert_gate: nn.Linear | None = None
        if config.shared_expert_intermediate_size:
            self.shared_expert = Qwen3MLP(
                config.hidden_size,
                config.shared_expert_intermediate_size,
                expert_parallel,
                module_prefix=f"{module_prefix}.shared_expert",
                quantize=quantize,
            )
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        hidden_states = x.reshape(-1, x.size(-1))
        router_logits = self.gate(hidden_states)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probs,
            self.num_experts_per_tok,
            dim=-1,
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        output = self.experts(
            hidden_states,
            selected_experts,
            routing_weights.to(router_logits.dtype),
        )
        if self.shared_expert is not None:
            assert self.shared_expert_gate is not None
            shared = self.shared_expert(hidden_states)
            output = output + torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared
        return output.view(original_shape)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        backend: AttentionBackend,
        tensor_parallel: TensorParallelContext,
        moe_dispatch_backend: str,
        *,
        kv_layer_idx: int | None = None,
        state_layer_idx: int | None = None,
        linear_attention_backend: str = "auto",
        layer_type: str | None = None,
        module_prefix: str | None = None,
        quantize: QuantizeFn | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if layer_type is None else layer_type
        prefix = module_prefix or f"layers.{layer_idx}"
        self.self_attn: Qwen3Attention | None = None
        self.linear_attn: Qwen3_5GatedDeltaNet | None = None
        if self.layer_type == LINEAR_ATTENTION:
            assert state_layer_idx is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                layer_idx,
                state_layer_idx,
                tensor_parallel,
                backend=linear_attention_backend,
                quantize=quantize,
            )
        else:
            self.self_attn = Qwen3Attention(
                config,
                layer_idx,
                backend,
                tensor_parallel,
                kv_layer_idx=kv_layer_idx,
                module_prefix=f"{prefix}.self_attn",
                quantize=quantize,
            )
        sparse = config.is_sparse_layer(layer_idx) if layer_type is None else config.is_moe
        self.mlp = (
            Qwen3SparseMoeBlock(
                config,
                tensor_parallel,
                moe_dispatch_backend,
                module_prefix=f"{prefix}.mlp",
                quantize=quantize,
            )
            if sparse
            else Qwen3MLP(
                config.hidden_size,
                config.intermediate_size,
                tensor_parallel,
                module_prefix=f"{prefix}.mlp",
                quantize=quantize,
            )
        )
        zero_centered = config.is_qwen3_5
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, zero_centered=zero_centered
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, zero_centered=zero_centered
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
        state_batch: RecurrentStateBatch | None = None,
        mrope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(x)
        if self.linear_attn is not None:
            x = x + self.linear_attn(normalized, state_batch)
        else:
            assert self.self_attn is not None
            x = x + self.self_attn(
                normalized, positions, kv_cache, cache_batch, mrope_positions
            )
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3_5MultiTokenPredictor(nn.Module):
    """Qwen3.5 MTP head: predicts token ``t+2`` from ``h_t`` and ``embed(t+1)``.

    The input is ``fc([norm(embed(next_token)), norm(target_hidden)])``, followed by one
    full-attention decoder layer with its own K/V, a final norm and the shared LM head.
    Its K/V lives in a companion paged cache at exactly the target's token positions.
    """

    def __init__(
        self,
        config: ModelConfig,
        backend: AttentionBackend,
        tensor_parallel: TensorParallelContext,
        moe_dispatch_backend: str,
        *,
        quantize: QuantizeFn | None = None,
    ) -> None:
        super().__init__()
        self.pre_fc_norm_embedding = RMSNorm(
            config.hidden_size, config.rms_norm_eps, zero_centered=True
        )
        self.pre_fc_norm_hidden = RMSNorm(
            config.hidden_size, config.rms_norm_eps, zero_centered=True
        )
        # 官方 FP8 checkpoint 把 mtp.fc 列在 modules_to_not_convert 中，这里始终保持 BF16。
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(
                config,
                config.full_attention_layers[0],
                backend,
                tensor_parallel,
                moe_dispatch_backend,
                kv_layer_idx=index,
                layer_type="full_attention",
                module_prefix=f"mtp.layers.{index}",
                quantize=quantize,
            )
            for index in range(config.mtp_num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)


class Qwen3ForCausalLM(nn.Module):
    """Dense/MoE Qwen3 and the Qwen3.5 hybrid family sharing one paged-attention path."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        tensor_parallel: TensorParallelContext | None = None,
        moe_dispatch_backend: str = "sorted",
        linear_attention_backend: str = "auto",
        enable_mtp: bool = False,
        fp8_weights: bool = True,
        enable_vision: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend or TorchAttentionBackend()
        self.tensor_parallel = tensor_parallel or TensorParallelContext()
        self.moe_dispatch_backend = moe_dispatch_backend
        self.linear_attention_backend = linear_attention_backend
        if enable_mtp and not config.mtp_num_hidden_layers:
            raise ValueError("this checkpoint does not contain MTP layers")
        # fp8_weights=False 时，FP8 checkpoint 在加载时反量化为普通权重（用作数值对照）。
        self.fp8 = config.quantization if fp8_weights else None
        quantize = self._quantize_for if self.fp8 is not None else None
        self.kv_cache_num_heads = config.num_key_value_heads // kv_shard_count(
            config,
            self.tensor_parallel,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        kv_layers = {layer: index for index, layer in enumerate(config.full_attention_layers)}
        state_layers = {
            layer: index for index, layer in enumerate(config.linear_attention_layers)
        }
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(
                config,
                layer_idx,
                self.attention_backend,
                self.tensor_parallel,
                moe_dispatch_backend,
                kv_layer_idx=kv_layers.get(layer_idx),
                state_layer_idx=state_layers.get(layer_idx),
                linear_attention_backend=linear_attention_backend,
                quantize=quantize,
            )
            for layer_idx in range(config.num_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, zero_centered=config.is_qwen3_5
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = (
            Qwen3_5MultiTokenPredictor(
                config,
                self.attention_backend,
                self.tensor_parallel,
                moe_dispatch_backend,
                quantize=quantize,
            )
            if enable_mtp
            else None
        )
        if enable_vision and config.vision is None:
            raise ValueError("this checkpoint does not contain a vision tower")
        # 视觉塔始终以激活精度（BF16）保存；官方 FP8 checkpoint 也不量化它。
        self.visual = Qwen3_5VisionModel(config.vision) if enable_vision else None
        self.tie_weights()

    def _quantize_for(self, *hf_module_names: str) -> Fp8QuantConfig | None:
        """FP8 config when every fused checkpoint Linear is stored as block FP8."""
        assert self.fp8 is not None
        decisions = {self.fp8.is_quantized(name) for name in hf_module_names}
        if len(decisions) != 1:
            raise ValueError(f"cannot fuse FP8 and unquantized Linears: {hf_module_names}")
        return self.fp8 if decisions.pop() else None

    @property
    def num_kv_cache_layers(self) -> int:
        return len(self.config.full_attention_layers)

    @property
    def num_draft_kv_layers(self) -> int:
        return 0 if self.mtp is None else len(self.mtp.layers)

    @property
    def linear_attention_modules(self) -> tuple[Qwen3_5GatedDeltaNet, ...]:
        return tuple(layer.linear_attn for layer in self.layers if layer.linear_attn is not None)

    @property
    def recurrent_state_spec(self) -> RecurrentStateSpec | None:
        modules = self.linear_attention_modules
        if not modules:
            return None
        first = modules[0]
        return RecurrentStateSpec(
            num_layers=len(modules),
            conv_dim=first.conv_dim,
            conv_kernel_size=first.conv_kernel_size,
            num_heads=first.num_v_heads,
            head_k_dim=first.head_k_dim,
            head_v_dim=first.head_v_dim,
        )

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _decoder_layers(self):
        yield from self.layers
        if self.mtp is not None:
            yield from self.mtp.layers

    def set_moe_cuda_graph_static_dispatch(self, enabled: bool) -> None:
        """Select fixed assignment slots only while a MoE Decode graph is captured."""
        for layer in self._decoder_layers():
            if isinstance(layer.mlp, Qwen3SparseMoeBlock):
                layer.mlp.experts.cuda_graph_static_dispatch = enabled

    def reset_non_persistent_buffers(self, device: torch.device | str) -> None:
        for layer in self._decoder_layers():
            if layer.self_attn is not None:
                layer.self_attn.rope.reset_buffer(device)
        if self.visual is not None:
            self.visual.reset_buffers(device)

    @torch.no_grad()
    def encode_images(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: tuple[tuple[int, int, int], ...],
    ) -> torch.Tensor:
        """Vision features, one row per expanded image token, in the LLM hidden size."""
        if self.visual is None:
            raise RuntimeError("the model was loaded without its vision tower")
        device = self.embed_tokens.weight.device
        grid = torch.tensor(image_grid_thw, dtype=torch.long, device=device)
        features = self.visual(pixel_values.to(device), grid)
        return features.to(self.embed_tokens.weight.dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str = "cpu",
        attention_backend: AttentionBackend | None = None,
        tensor_parallel: TensorParallelContext | None = None,
        moe_dispatch_backend: str = "sorted",
        linear_attention_backend: str = "auto",
        enable_mtp: bool = False,
        fp8_weights: bool = True,
        enable_vision: bool = False,
    ) -> Qwen3ForCausalLM:
        from .loader import load_qwen3_model

        return load_qwen3_model(
            model_path,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
            tensor_parallel=tensor_parallel,
            moe_dispatch_backend=moe_dispatch_backend,
            linear_attention_backend=linear_attention_backend,
            enable_mtp=enable_mtp,
            fp8_weights=fp8_weights,
            enable_vision=enable_vision,
        )

    def validate_cache(self, cache: PagedKVCache) -> None:
        if cache.num_kv_heads != self.kv_cache_num_heads:
            raise ValueError(
                "KV cache head count does not match this TP rank: "
                f"{cache.num_kv_heads} != {self.kv_cache_num_heads}"
            )
        if cache.num_layers != self.num_kv_cache_layers:
            raise ValueError(
                "KV cache layer count does not match the full-attention layers: "
                f"{cache.num_layers} != {self.num_kv_cache_layers}"
            )
        if self.recurrent_state_spec is not None and getattr(cache, "recurrent", None) is None:
            raise ValueError("hybrid linear-attention models require recurrent state slots")
        self.attention_backend.validate_cache(cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache | None = None,
        cache_request_ids: tuple[str, ...] | None = None,
        mrope_positions: torch.Tensor | None = None,
        embedding_overrides: EmbeddingOverrides | None = None,
    ) -> torch.Tensor:
        """Uncached oracle forward, or a dense cached forward over ``cache_request_ids``.

        ``mrope_positions`` (``[3, batch, sequence]``) and ``embedding_overrides`` let the
        cached and uncached paths reproduce an image prompt end to end."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids sequence must not be empty")
        cache_batch = None
        if kv_cache is None:
            if cache_request_ids is not None:
                raise ValueError("cache selectors require a KV cache")
            max_total_length = input_ids.size(1)
            positions = torch.arange(max_total_length, device=input_ids.device)
        elif isinstance(kv_cache, PagedKVCache):
            if cache_request_ids is None:
                raise ValueError("cache_request_ids are required with PagedKVCache")
            if len(cache_request_ids) != input_ids.size(0):
                raise ValueError("cache_request_ids length must match input batch size")
            append_lengths = (input_ids.size(1),) * input_ids.size(0)
            cache_batch = kv_cache.prepare_batch(cache_request_ids, append_lengths)
            if mrope_positions is not None:
                cache_batch = replace(cache_batch, mrope_positions=mrope_positions)
            return self.forward_prepared(
                input_ids,
                kv_cache=kv_cache,
                cache_batch=cache_batch,
                embedding_overrides=embedding_overrides,
            )
        else:
            raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")

        if max_total_length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        return self._forward_tokens(
            input_ids,
            positions,
            kv_cache,
            cache_batch,
            mrope_positions=mrope_positions,
            embedding_overrides=embedding_overrides,
        )

    def forward_prepared(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        cache_batch: PagedKVBatch,
        logits_indices: tuple[int, ...] | None = None,
        state_batch: RecurrentStateBatch | None = None,
        return_hidden: bool = False,
        embedding_overrides: EmbeddingOverrides | None = None,
    ):
        """Cached forward. ``return_hidden`` also returns every final (normed) hidden row,
        which the MTP head consumes; ``state_batch`` overrides the default slot batch,
        e.g. to run speculative verification in log mode. ``embedding_overrides``
        replaces image-placeholder rows with vision features."""
        positions = self._validate_prepared(input_ids, cache_batch)
        self._validate_logits_indices(logits_indices, input_ids.numel())
        if state_batch is None and self.recurrent_state_spec is not None:
            recurrent = getattr(kv_cache, "recurrent", None)
            if recurrent is None:
                raise ValueError("hybrid linear-attention models require recurrent state slots")
            state_batch = recurrent.prepare_batch(
                cache_batch.request_ids, cache_batch.append_lengths
            )
        return self._forward_tokens(
            input_ids,
            positions,
            kv_cache,
            cache_batch,
            logits_indices,
            state_batch=state_batch,
            return_hidden=return_hidden,
            embedding_overrides=embedding_overrides,
        )

    def _validate_prepared(
        self,
        input_ids: torch.Tensor,
        cache_batch: PagedKVBatch,
    ) -> torch.Tensor:
        if input_ids.ndim == 2:
            if input_ids.size(0) != cache_batch.batch_size:
                raise ValueError("dense input batch size disagrees with cache metadata")
            if input_ids.size(1) == 0:
                raise ValueError("input_ids sequence must not be empty")
            if input_ids.size(1) != cache_batch.uniform_append_length:
                raise ValueError("dense input sequence length disagrees with cache metadata")
            positions = cache_batch.positions.view(input_ids.shape)
        elif input_ids.ndim == 1:
            if input_ids.numel() == 0:
                raise ValueError("packed input_ids must not be empty")
            positions = cache_batch.positions
        else:
            raise ValueError("input_ids must be dense rank 2 or packed rank 1")
        if input_ids.numel() != cache_batch.total_tokens:
            raise ValueError("input token count disagrees with cache metadata")
        if cache_batch.max_end > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        # 带图片的请求：RoPE 位置 = K/V 下标 + rope_delta（decode）或三维位置（prefill）。
        if cache_batch.rope_positions is not None:
            positions = cache_batch.rope_positions.view(positions.shape)
        return positions

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        cache_request_ids: tuple[str, ...],
        append_lengths: tuple[int, ...],
        logits_indices: tuple[int, ...] | None = None,
        state_batch: RecurrentStateBatch | None = None,
        return_hidden: bool = False,
        mrope_positions: torch.Tensor | None = None,
        embedding_overrides: EmbeddingOverrides | None = None,
    ):
        if input_ids.ndim != 1:
            raise ValueError("packed input_ids must have shape [total_tokens]")
        if input_ids.numel() == 0:
            raise ValueError("packed input_ids must not be empty")
        if len(cache_request_ids) != len(append_lengths):
            raise ValueError("request IDs and append lengths must have the same size")
        self._validate_logits_indices(logits_indices, input_ids.numel())
        cache_batch = kv_cache.prepare_batch(cache_request_ids, append_lengths)
        if mrope_positions is not None:
            cache_batch = replace(cache_batch, mrope_positions=mrope_positions)
        return self.forward_prepared(
            input_ids,
            kv_cache=kv_cache,
            cache_batch=cache_batch,
            logits_indices=logits_indices,
            state_batch=state_batch,
            return_hidden=return_hidden,
            embedding_overrides=embedding_overrides,
        )

    def forward_mtp(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        cache_batch: PagedKVBatch,
        logits_indices: tuple[int, ...] | None = None,
        embedding_overrides: EmbeddingOverrides | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the MTP head on packed ``(next token, target hidden)`` pairs.

        Row ``i`` pairs the hidden state at position ``p`` with the token at ``p + 1``
        and writes draft K/V at position ``p``. Returns logits at ``logits_indices`` and
        every normed MTP hidden row, which seeds the next recursive draft step.
        """
        if self.mtp is None:
            raise RuntimeError("the model was loaded without its MTP layers")
        positions = self._validate_prepared(input_ids, cache_batch)
        self._validate_logits_indices(logits_indices, input_ids.numel())
        if hidden_states.shape[:-1] != input_ids.shape:
            raise ValueError("MTP hidden states must align with the input tokens")
        embeddings = self._embed(input_ids, embedding_overrides)
        embeddings = self.mtp.pre_fc_norm_embedding(embeddings)
        hidden = self.mtp.pre_fc_norm_hidden(hidden_states.to(embeddings.dtype))
        hidden = self.mtp.fc(torch.cat((embeddings, hidden), dim=-1))
        mrope_positions = cache_batch.mrope_positions
        if mrope_positions is not None:
            mrope_positions = mrope_positions.view(3, *input_ids.shape)
        for layer in self.mtp.layers:
            hidden = layer(
                hidden,
                positions,
                kv_cache,
                cache_batch,
                mrope_positions=mrope_positions,
            )
        hidden = self.mtp.norm(hidden)
        return self._logits(hidden, logits_indices), hidden

    @torch.no_grad()
    def commit_speculative_states(
        self,
        state_batch: RecurrentStateBatch,
        accepted_lengths: tuple[int, ...] | torch.Tensor,
    ) -> None:
        """Advance DeltaNet slots by the accepted prefix of a verify forward.

        Equal-width verification uses the fixed-shape masked replay (one set of kernels,
        capturable); ragged n-gram verification replays only the accepted rows.
        """
        uniform = len(set(state_batch.append_lengths)) == 1
        if isinstance(accepted_lengths, torch.Tensor) or uniform:
            lengths = accepted_lengths
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long, device=state_batch.slots.device)
            for module in self.linear_attention_modules:
                module.commit_speculative_masked(state_batch, lengths)
            return
        for module in self.linear_attention_modules:
            module.commit_speculative(state_batch, accepted_lengths)

    @staticmethod
    def _validate_logits_indices(
        logits_indices: tuple[int, ...] | None,
        token_count: int,
    ) -> None:
        if logits_indices is not None and any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < token_count
            for index in logits_indices
        ):
            raise ValueError("logits_indices must contain valid token indices")

    def _embed(
        self,
        input_ids: torch.Tensor,
        overrides: EmbeddingOverrides | None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        if overrides is None:
            return hidden
        rows, values = overrides
        flat = hidden.reshape(-1, hidden.size(-1))
        # 图片占位 token 的 embedding 换成对应的视觉特征（与 HF 的 masked_scatter 等价）。
        flat.index_copy_(0, rows.to(flat.device), values.to(flat.device, flat.dtype))
        return flat.view_as(hidden)

    def _logits(
        self,
        normalized: torch.Tensor,
        logits_indices: tuple[int, ...] | None,
    ) -> torch.Tensor:
        if logits_indices is None:
            return self.lm_head(normalized).float()
        if not logits_indices:
            return normalized.new_empty((0, self.config.vocab_size), dtype=torch.float32)
        flat = normalized.reshape(-1, normalized.size(-1))
        indices = torch.tensor(logits_indices, dtype=torch.long, device=normalized.device)
        return self.lm_head(flat.index_select(0, indices)).float()

    def _forward_tokens(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
        logits_indices: tuple[int, ...] | None = None,
        *,
        state_batch: RecurrentStateBatch | None = None,
        return_hidden: bool = False,
        mrope_positions: torch.Tensor | None = None,
        embedding_overrides: EmbeddingOverrides | None = None,
    ):
        hidden_states = self._embed(input_ids, embedding_overrides)
        if mrope_positions is None and cache_batch is not None:
            mrope_positions = cache_batch.mrope_positions
        if mrope_positions is not None:
            mrope_positions = mrope_positions.view(3, *input_ids.shape)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                kv_cache,
                cache_batch,
                state_batch,
                mrope_positions,
            )
        if not return_hidden:
            if logits_indices is not None:
                if not logits_indices:
                    return hidden_states.new_empty(
                        (0, self.config.vocab_size), dtype=torch.float32
                    )
                indices = torch.tensor(
                    logits_indices, dtype=torch.long, device=hidden_states.device
                )
                hidden_states = hidden_states.index_select(0, indices)
            return self.lm_head(self.norm(hidden_states)).float()
        normalized = self.norm(hidden_states)
        return self._logits(normalized, logits_indices), normalized
