"""Engram 的词表压缩、分桶哈希及门控注入；可选依赖只在使用时导入。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .config import DeepSeekV41Config


def bucket_primes(config: DeepSeekV41Config) -> tuple[tuple[int, ...], ...]:
    def is_prime(value: int) -> bool:
        if value < 2 or value % 2 == 0:
            return value == 2
        return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))

    seen: set[int] = set()
    layers = []
    for _ in config.engram_layer_ids:
        row = []
        for _ in range(config.engram_max_ngram_size - 1):
            prime = config.engram_vocab_size - 1
            for _ in range(config.engram_n_heads):
                prime += 1
                while prime in seen or not is_prime(prime):
                    prime += 1
                row.append(prime)
                seen.add(prime)
        layers.append(tuple(row))
    return tuple(layers)


def compressed_token_map(tokenizer) -> list[int]:
    """严格遵守训练时的 Unicode/空白归一化，不用 Python lower 近似。"""
    try:
        from tokenizers import Regex, normalizers
    except ImportError as exc:
        raise RuntimeError("Engram tokenizer normalization requires the 'model' extra") from exc
    sentinel = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    vocabulary: dict[str, int] = {}
    result = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else normalize.normalize_str(text) or text
        )
        if key not in vocabulary:
            vocabulary[key] = len(vocabulary)
        result.append(vocabulary[key])
    return result


class EngramHasher(nn.Module):
    def __init__(self, config: DeepSeekV41Config, token_map: Sequence[int]):
        super().__init__()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Engram hash initialization requires numpy ('model' extra)") from exc
        if len(token_map) != config.vocab_size or set(token_map) != set(
            range(config.engram_compressed_vocab_size)
        ):
            raise ValueError("compressed token map does not match configured vocabulary")
        primes = bucket_primes(config)
        if tuple(map(sum, primes)) != config.engram_num_embeddings:
            raise ValueError("Engram table sizes do not match generated prime buckets")
        self.lookback = config.engram_max_ngram_size - 1
        self.pad_id = token_map[config.engram_pad_id]
        self.n_heads = config.engram_n_heads
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        offsets = self.primes.cumsum(-1) - self.primes
        self.register_buffer("offsets", offsets, persistent=False)
        bound = max(1, ((2**63 - 1) // config.engram_compressed_vocab_size) // 2)
        multipliers = []
        for layer_id in config.engram_layer_ids:
            rng = np.random.default_rng(10007 * layer_id)
            values = rng.integers(0, bound, size=self.lookback + 1, dtype=np.int64)
            multipliers.append((values * 2 + 1).tolist())
        self.register_buffer("multipliers", torch.tensor(multipliers), persistent=False)

    def forward(
        self, tokens: torch.Tensor, *, start: int = 0, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """tokens 包含完整前缀；返回 start 之后各位置的哈希，-1 截断 n-gram 历史。"""
        if not 0 <= start <= tokens.shape[1]:
            raise ValueError("invalid Engram start position")
        mapped = self.token_map[tokens]
        if token_mask is not None:
            if token_mask.shape != tokens.shape:
                raise ValueError("token mask must match tokens")
            mapped = mapped.masked_fill(~token_mask, -1)
        pos = torch.arange(start, tokens.shape[1], device=tokens.device)
        blocked = torch.zeros(tokens.shape[0], len(pos), dtype=torch.bool, device=tokens.device)
        history = []
        for shift in range(self.lookback + 1):
            source = mapped[:, (pos - shift).clamp_min(0)]
            blocked = blocked | (pos < shift) | (source == -1)
            history.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(history, -1).unsqueeze(-2) * self.multipliers
        rolling = products[..., 0]
        columns = []
        for shift in range(1, self.lookback + 1):
            rolling = rolling ^ products[..., shift]
            primes = self.primes[:, (shift - 1) * self.n_heads : shift * self.n_heads]
            columns.append(rolling.unsqueeze(-1) % primes)
        return torch.cat(columns, -1) + self.offsets


class Engram(nn.Module):
    def __init__(self, config: DeepSeekV41Config, layer: int):
        super().__init__()
        self.config = config
        self.hash_index = config.engram_layer_ids.index(layer)
        self.embed = nn.Embedding(
            config.engram_num_embeddings[self.hash_index], config.engram_head_dim
        )
        columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = nn.Linear(
            columns * config.engram_head_dim, config.dim * (config.hc_mult + 1), bias=False
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.dim))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.dim))

    def forward(self, x: torch.Tensor, hashes: torch.Tensor) -> torch.Tensor:
        c = self.config
        projected = self.wkv(self.embed(hashes).flatten(-2))
        key, value = projected.split((c.hc_mult * c.dim, c.dim), -1)
        key = key.unflatten(-1, (c.hc_mult, c.dim))
        dot = (x * key * self.q_weight * self.k_weight).sum(-1)
        dot = dot * torch.rsqrt(x.square().mean(-1) + c.norm_eps)
        dot = dot * torch.rsqrt(key.square().mean(-1) + c.norm_eps) / math.sqrt(c.dim)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return x + gate.unsqueeze(-1) * value.unsqueeze(-2)
