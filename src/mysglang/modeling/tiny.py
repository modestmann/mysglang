from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query/key: [batch, heads, sequence, head_dim]
        cos = self.cos[positions][None, None, :, :].to(query.dtype)
        sin = self.sin[positions][None, None, :, :].to(query.dtype)
        return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x if repeats == 1 else x.repeat_interleave(repeats, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = self.qkv_proj(x).split((q_size, kv_size, kv_size), dim=-1)

        def split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, heads, self.head_dim).transpose(1, 2)

        query = split_heads(query, self.num_heads)
        key = split_heads(key, self.num_kv_heads)
        value = split_heads(value, self.num_kv_heads)
        positions = torch.arange(seq_len, device=x.device)
        query, key = self.rope(query, key, positions)
        repeats = self.num_heads // self.num_kv_heads
        key, value = _repeat_kv(key, repeats), _repeat_kv(value, repeats)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))


class TinyCausalLM(nn.Module):
    """A small Qwen/Llama-shaped model used as the correctness oracle.

    Milestone 1 intentionally has no KV cache: each generation step recomputes
    the complete prefix. Milestone 3 will optimize this without changing tokens.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.norm(hidden_states)).float()

