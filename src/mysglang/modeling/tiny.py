from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mysglang.cache import ContiguousKVCache, SlotKVCache
from mysglang.config import ModelConfig


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
        # query/key: [batch, heads, sequence, head_dim]
        embeddings = self.cos[positions]
        if positions.ndim == 1:
            cos = embeddings[None, None, :, :]
            sin = self.sin[positions][None, None, :, :]
        elif positions.ndim == 2:
            cos = embeddings[:, None, :, :]
            sin = self.sin[positions][:, None, :, :]
        else:
            raise ValueError("positions must have shape [sequence] or [batch, sequence]")
        cos = cos.to(dtype=query.dtype, device=query.device)
        sin = sin.to(dtype=query.dtype, device=query.device)
        return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x if repeats == 1 else x.repeat_interleave(repeats, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
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
        kv_cache: ContiguousKVCache | SlotKVCache | None,
        cache_slots: tuple[int, ...] | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        def split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, heads, self.head_dim).transpose(1, 2)

        query = self.q_norm(split_heads(self.q_proj(x), self.num_heads))
        key = self.k_norm(split_heads(self.k_proj(x), self.num_kv_heads))
        value = split_heads(self.v_proj(x), self.num_kv_heads)
        query, key = self.rope(query, key, positions)
        past_length = 0
        attention_mask = None
        if isinstance(kv_cache, ContiguousKVCache):
            past_length = kv_cache.layer_length(self.layer_idx)
            if past_length > 0 and seq_len != 1:
                raise ValueError("cached decode currently accepts exactly one new token")
            key, value = kv_cache.append(self.layer_idx, key, value)
        elif isinstance(kv_cache, SlotKVCache):
            if cache_slots is None:
                raise ValueError("cache_slots are required with SlotKVCache")
            key, value, attention_mask = kv_cache.append(
                self.layer_idx, key, value, cache_slots
            )
        elif kv_cache is not None:
            raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")

        repeats = self.num_heads // self.num_kv_heads
        key, value = _repeat_kv(key, repeats), _repeat_kv(value, repeats)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=kv_cache is None or (
                isinstance(kv_cache, ContiguousKVCache) and past_length == 0
            ),
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
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
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: ContiguousKVCache | SlotKVCache | None,
        cache_slots: tuple[int, ...] | None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), positions, kv_cache, cache_slots
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyCausalLM(nn.Module):
    """A tiny dense Qwen3 model used for Hugging Face correctness alignment."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: ContiguousKVCache | SlotKVCache | None = None,
        cache_slots: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids sequence must not be empty")
        if kv_cache is None:
            if cache_slots is not None:
                raise ValueError("cache_slots require a KV cache")
            total_length = input_ids.size(1)
            max_total_length = total_length
            positions = torch.arange(total_length, device=input_ids.device)
        elif isinstance(kv_cache, ContiguousKVCache):
            if cache_slots is not None:
                raise ValueError("cache_slots are only valid with SlotKVCache")
            past_length = kv_cache.length
            total_length = past_length + input_ids.size(1)
            max_total_length = total_length
            if kv_cache.batch_size != input_ids.size(0):
                raise ValueError("input batch size must match KV cache batch size")
            if total_length > kv_cache.max_length:
                raise ValueError("sequence exceeds KV cache capacity")
            if past_length > 0 and input_ids.size(1) != 1:
                raise ValueError("cached decode currently accepts exactly one new token")
            positions = torch.arange(past_length, total_length, device=input_ids.device)
        elif isinstance(kv_cache, SlotKVCache):
            if cache_slots is None:
                raise ValueError("cache_slots are required with SlotKVCache")
            if len(cache_slots) != input_ids.size(0):
                raise ValueError("cache_slots length must match input batch size")
            past_lengths = kv_cache.lengths(cache_slots)
            total_lengths = tuple(length + input_ids.size(1) for length in past_lengths)
            max_total_length = max(total_lengths)
            if max(total_lengths) > kv_cache.max_length:
                raise ValueError("sequence exceeds KV cache capacity")
            starts = torch.tensor(past_lengths, device=input_ids.device)[:, None]
            offsets = torch.arange(input_ids.size(1), device=input_ids.device)[None, :]
            positions = starts + offsets
        else:
            raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")

        if max_total_length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, kv_cache, cache_slots)
        return self.lm_head(self.norm(hidden_states)).float()
