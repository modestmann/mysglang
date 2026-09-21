from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mysglang.cache import PagedKVBatch, PagedKVCache
from mysglang.config import ModelConfig

from .attention import AttentionBackend, TorchAttentionBackend
from .parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelContext


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Qwen3's default RoPE, computed only for positions used by this forward."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self, device: torch.device | str | None = None) -> torch.Tensor:
        indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        return 1.0 / self.theta ** (indices / self.head_dim)

    def reset_buffer(self, device: torch.device | str) -> None:
        # ``to_empty`` intentionally leaves buffers uninitialized; weights come from the
        # checkpoint, while this deterministic non-persistent RoPE buffer is rebuilt here.
        self.inv_freq = self._make_inv_freq(device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.float()
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
        return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        backend: AttentionBackend,
        tensor_parallel: TensorParallelContext,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.backend = backend
        self.tensor_parallel = tensor_parallel
        self.num_heads = tensor_parallel.local_size(
            config.num_attention_heads,
            "num_attention_heads",
        )
        self.num_kv_heads = tensor_parallel.local_size(
            config.num_key_value_heads,
            "num_key_value_heads",
        )
        assert config.head_dim is not None
        self.head_dim = config.head_dim
        self.global_q_size = config.num_attention_heads * self.head_dim
        self.global_kv_size = config.num_key_value_heads * self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Hugging Face stores Q/K/V separately. The loader independently takes this
        # rank's heads from each tensor, then packs them as [Q_rank, K_rank, V_rank].
        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            (self.global_q_size, self.global_kv_size, self.global_kv_size),
            tensor_parallel,
        )
        self.o_proj = RowParallelLinear(
            self.global_q_size,
            config.hidden_size,
            tensor_parallel,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
    ) -> torch.Tensor:
        if x.ndim == 3:
            batch_size, seq_len, _ = x.shape

            def split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
                return tensor.view(batch_size, seq_len, heads, self.head_dim).transpose(1, 2)

        elif x.ndim == 2:
            total_tokens = x.size(0)

            def split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
                return tensor.view(total_tokens, heads, self.head_dim)

        else:
            raise ValueError("hidden states must be dense rank 3 or packed rank 2")

        query, key, value = self.qkv_proj(x).split(
            (self.q_size, self.kv_size, self.kv_size),
            dim=-1,
        )
        query = self.q_norm(split_heads(query, self.num_heads))
        key = self.k_norm(split_heads(key, self.num_kv_heads))
        value = split_heads(value, self.num_kv_heads)
        query, key = self.rope(query, key, positions)
        output = self.backend.forward(
            query,
            key,
            value,
            layer_idx=self.layer_idx,
            kv_cache=kv_cache,
            cache_batch=cache_batch,
        )
        if output.ndim == 4:
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        else:
            output = output.reshape(output.size(0), -1)
        return self.o_proj(output)


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tensor_parallel: TensorParallelContext,
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
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tensor_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen3Experts(nn.Module):
    """Expert-sharded top-k dispatch with selectable communication and GEMM paths.

    ``naive``, ``sorted`` and ``grouped`` keep token hidden states replicated and combine
    rank-local expert contributions with all-reduce. ``all_to_all`` first gives each rank a
    contiguous token shard, dispatches its assignments to expert owners, returns expert
    outputs to token owners, then all-gathers token outputs because the following Attention
    TP layer still requires replicated tokens.
    """

    def __init__(
        self,
        config: ModelConfig,
        expert_parallel: TensorParallelContext,
        dispatch_backend: str,
    ) -> None:
        super().__init__()
        if dispatch_backend not in {"naive", "sorted", "grouped", "all_to_all"}:
            raise ValueError(
                "MoE dispatch backend must be 'naive', 'sorted', 'grouped' or 'all_to_all'"
            )
        self.num_experts = config.num_experts
        self.expert_parallel = expert_parallel
        self.dispatch_backend = dispatch_backend
        self.num_local_experts = expert_parallel.local_size(
            config.num_experts,
            "num_experts",
        )
        self.local_expert_start = expert_parallel.rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert_idx in range(self.num_local_experts):
            nn.init.kaiming_uniform_(self.gate_up_proj[expert_idx], a=5**0.5)
            nn.init.kaiming_uniform_(self.down_proj[expert_idx], a=5**0.5)

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
        if self.dispatch_backend == "grouped":
            result = self._forward_grouped(hidden_states, selected_experts, routing_weights)
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
            gate, up = F.linear(current, self.gate_up_proj[local_expert_idx]).chunk(2, dim=-1)
            current = F.linear(F.silu(gate) * up, self.down_proj[local_expert_idx])
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
                    self.gate_up_proj[expert_idx],
                ).chunk(2, dim=-1)
                sorted_output[start:end] = F.linear(
                    F.silu(gate) * up,
                    self.down_proj[expert_idx],
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
            self.gate_up_proj[active_experts].transpose(1, 2),
        )
        gate, up = gate_up.chunk(2, dim=-1)
        padded_output = torch.bmm(
            F.silu(gate) * up,
            self.down_proj[active_experts].transpose(1, 2),
        )
        sorted_output = padded_output[active_batch, slots]
        output = torch.empty_like(sorted_output)
        output[order] = sorted_output
        return output

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute this rank's replicated-token assignments with batched expert GEMMs."""
        top_k = selected_experts.size(1)
        flat_experts = selected_experts.reshape(-1)
        owned = (flat_experts >= self.local_expert_start) & (
            flat_experts < self.local_expert_end
        )
        assignment_indices = torch.where(owned)[0]
        token_indices = torch.div(assignment_indices, top_k, rounding_mode="floor")
        top_k_slots = assignment_indices.remainder(top_k)
        local_experts = flat_experts[assignment_indices] - self.local_expert_start
        current = self._grouped_expert_gemm(hidden_states[token_indices], local_experts)
        current = current * routing_weights[token_indices, top_k_slots, None]
        result = torch.zeros_like(hidden_states)
        result.index_add_(0, token_indices, current.to(result.dtype))
        return result

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
        send_token_indices = local_token_indices[order]

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

        local_result = torch.zeros_like(local_hidden)
        local_result.index_add_(
            0,
            send_token_indices,
            (returned_outputs * send_weights[:, None]).to(local_result.dtype),
        )
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
                    self.gate_up_proj[local_expert_idx],
                ).chunk(2, dim=-1)
                current = F.linear(
                    F.silu(gate) * up,
                    self.down_proj[local_expert_idx],
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
    ) -> None:
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3Experts(config, expert_parallel, dispatch_backend)

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
        return output.view(original_shape)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        backend: AttentionBackend,
        tensor_parallel: TensorParallelContext,
        moe_dispatch_backend: str,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx, backend, tensor_parallel)
        self.mlp = (
            Qwen3SparseMoeBlock(config, tensor_parallel, moe_dispatch_backend)
            if config.is_sparse_layer(layer_idx)
            else Qwen3MLP(config.hidden_size, config.intermediate_size, tensor_parallel)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            positions,
            kv_cache,
            cache_batch,
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3ForCausalLM(nn.Module):
    """Dense Qwen3 and Qwen3MoE inference model sharing the paged-attention path."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        attention_backend: AttentionBackend | None = None,
        tensor_parallel: TensorParallelContext | None = None,
        moe_dispatch_backend: str = "sorted",
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend or TorchAttentionBackend()
        self.tensor_parallel = tensor_parallel or TensorParallelContext()
        self.kv_cache_num_heads = self.tensor_parallel.local_size(
            config.num_key_value_heads,
            "num_key_value_heads",
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(
                config,
                layer_idx,
                self.attention_backend,
                self.tensor_parallel,
                moe_dispatch_backend,
            )
            for layer_idx in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def reset_non_persistent_buffers(self, device: torch.device | str) -> None:
        for layer in self.layers:
            layer.self_attn.rope.reset_buffer(device)

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
    ) -> Qwen3ForCausalLM:
        from .loader import load_qwen3_model

        return load_qwen3_model(
            model_path,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
            tensor_parallel=tensor_parallel,
            moe_dispatch_backend=moe_dispatch_backend,
        )

    def validate_cache(self, cache: PagedKVCache) -> None:
        if cache.num_kv_heads != self.kv_cache_num_heads:
            raise ValueError(
                "KV cache head count does not match this TP rank: "
                f"{cache.num_kv_heads} != {self.kv_cache_num_heads}"
            )
        self.attention_backend.validate_cache(cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache | None = None,
        cache_request_ids: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
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
            return self.forward_prepared(
                input_ids,
                kv_cache=kv_cache,
                cache_batch=cache_batch,
            )
        else:
            raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")

        if max_total_length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        return self._forward_tokens(input_ids, positions, kv_cache, cache_batch)

    def forward_prepared(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        cache_batch: PagedKVBatch,
        logits_indices: tuple[int, ...] | None = None,
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
        self._validate_logits_indices(logits_indices, input_ids.numel())
        if cache_batch.max_end > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        return self._forward_tokens(
            input_ids,
            positions,
            kv_cache,
            cache_batch,
            logits_indices,
        )

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        cache_request_ids: tuple[str, ...],
        append_lengths: tuple[int, ...],
        logits_indices: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("packed input_ids must have shape [total_tokens]")
        if input_ids.numel() == 0:
            raise ValueError("packed input_ids must not be empty")
        if len(cache_request_ids) != len(append_lengths):
            raise ValueError("request IDs and append lengths must have the same size")
        self._validate_logits_indices(logits_indices, input_ids.numel())
        cache_batch = kv_cache.prepare_batch(cache_request_ids, append_lengths)
        return self.forward_prepared(
            input_ids,
            kv_cache=kv_cache,
            cache_batch=cache_batch,
            logits_indices=logits_indices,
        )

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

    def _forward_tokens(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
        logits_indices: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, kv_cache, cache_batch)
        if logits_indices is not None:
            if not logits_indices:
                return hidden_states.new_empty((0, self.config.vocab_size), dtype=torch.float32)
            indices = torch.tensor(logits_indices, dtype=torch.long, device=hidden_states.device)
            hidden_states = hidden_states.index_select(0, indices)
        return self.lm_head(self.norm(hidden_states)).float()
