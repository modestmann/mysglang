from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mysglang.cache import PagedKVBatch, PagedKVCache
from mysglang.config import ModelConfig

from .attention import AttentionBackend, TorchAttentionBackend


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
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.backend = backend
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert config.head_dim is not None
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Q/K/V are separate in Hugging Face checkpoints but fused at runtime. Besides
        # reducing launches, this is the future column-parallel TP boundary.
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
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
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # gate/up share the same input and form the future column-parallel TP boundary.
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen3Experts(nn.Module):
    """Readable top-k expert dispatch; grouped GEMM can replace this narrow module later."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_slot, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current = hidden_states[token_idx]
            gate, up = F.linear(current, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            current = current * routing_weights[token_idx, top_k_slot, None]
            result.index_add_(0, token_idx, current.to(result.dtype))
        return result


class Qwen3SparseMoeBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3Experts(config)

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
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx, backend)
        self.mlp = (
            Qwen3SparseMoeBlock(config)
            if config.is_sparse_layer(layer_idx)
            else Qwen3MLP(config.hidden_size, config.intermediate_size)
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
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend or TorchAttentionBackend()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, layer_idx, self.attention_backend)
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
    ) -> Qwen3ForCausalLM:
        from .loader import load_qwen3_model

        return load_qwen3_model(
            model_path,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
        )

    def validate_cache(self, cache: PagedKVCache) -> None:
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
