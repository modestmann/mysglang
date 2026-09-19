from __future__ import annotations

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
        # Accumulate in fp32, then return to the activation dtype.
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        positions = torch.arange(max_positions).float()
        frequencies = torch.outer(positions, inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", embeddings.cos(), persistent=False)
        self.register_buffer("sin", embeddings.sin(), persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Dense Q/K are rank 4; packed Q/K are [total_tokens, heads, head_dim].
        embeddings = self.cos[positions]
        if query.ndim == 3 and positions.ndim == 1:
            cos = embeddings[:, None, :]
            sin = self.sin[positions][:, None, :]
        elif query.ndim == 4 and positions.ndim == 1:
            cos = embeddings[None, None, :, :]
            sin = self.sin[positions][None, None, :, :]
        elif query.ndim == 4 and positions.ndim == 2:
            cos = embeddings[:, None, :, :]
            sin = self.sin[positions][:, None, :, :]
        else:
            raise ValueError("positions shape does not match dense or packed Q/K")
        cos = cos.to(dtype=query.dtype, device=query.device)
        sin = sin.to(dtype=query.dtype, device=query.device)
        return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
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
        self.head_dim = config.head_dim
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

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

        query = self.q_norm(split_heads(self.q_proj(x), self.num_heads))
        key = self.k_norm(split_heads(self.k_proj(x), self.num_kv_heads))
        value = split_heads(self.v_proj(x), self.num_kv_heads)
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


class GatedMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx, backend)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)

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


class TinyCausalLM(nn.Module):
    """A tiny dense Qwen3 model used for Hugging Face correctness alignment."""

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
            DecoderLayer(config, layer_idx, self.attention_backend)
            for layer_idx in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

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
            max_total_length = max(cache_batch.ends)
            starts = cache_batch.cache_seqlens.to(dtype=torch.long)[:, None]
            offsets = torch.arange(input_ids.size(1), device=input_ids.device)[None, :]
            positions = starts + offsets
        else:
            raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")

        if max_total_length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")

        return self._forward_tokens(input_ids, positions, kv_cache, cache_batch)

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        cache_request_ids: tuple[str, ...],
        append_lengths: tuple[int, ...],
    ) -> torch.Tensor:
        """Run flattened variable-length chunks without padding query tokens."""
        if input_ids.ndim != 1:
            raise ValueError("packed input_ids must have shape [total_tokens]")
        if input_ids.numel() == 0:
            raise ValueError("packed input_ids must not be empty")
        if len(cache_request_ids) != len(append_lengths):
            raise ValueError("request IDs and append lengths must have the same size")

        cache_batch = kv_cache.prepare_batch(cache_request_ids, append_lengths)
        if input_ids.numel() != cache_batch.total_tokens:
            raise ValueError("packed input token count disagrees with append lengths")
        if cache_batch.max_end > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        return self._forward_tokens(
            input_ids,
            cache_batch.positions,
            kv_cache,
            cache_batch,
        )

    def _forward_tokens(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
    ) -> torch.Tensor:

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                kv_cache,
                cache_batch,
            )
        return self.lm_head(self.norm(hidden_states)).float()
