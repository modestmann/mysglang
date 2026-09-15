from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Architecture-only configuration for the tiny decoder model.

    It intentionally contains no serving options. Cache size, batch limits and
    devices will belong to an EngineConfig in a later milestone.
    """

    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        positive = {
            "vocab_size": self.vocab_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "max_position_embeddings": self.max_position_embeddings,
        }
        if invalid := [name for name, value in positive.items() if value <= 0]:
            raise ValueError(f"configuration values must be positive: {invalid}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

