from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Qwen3/Qwen3MoE architecture configuration, independent of serving limits."""

    model_type: str = "qwen3"
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int | None = None
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = None
    torch_dtype: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    # Qwen3MoE fields. Dense Qwen3 keeps num_experts=0.
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    norm_topk_prob: bool = False

    def __post_init__(self) -> None:
        if self.model_type not in {"qwen3", "qwen3_moe"}:
            raise ValueError("model_type must be 'qwen3' or 'qwen3_moe'")
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError(
                    "head_dim is required when hidden_size is not divisible by attention heads"
                )
            object.__setattr__(
                self,
                "head_dim",
                self.hidden_size // self.num_attention_heads,
            )
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "decoder_sparse_step": self.decoder_sparse_step,
        }
        if invalid := [name for name, value in positive.items() if value is None or value <= 0]:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.attention_bias:
            raise ValueError("attention_bias=True is not implemented")
        if self.use_sliding_window:
            raise ValueError("sliding-window attention is not implemented")
        if any(not 0 <= layer < self.num_layers for layer in self.mlp_only_layers):
            raise ValueError("mlp_only_layers contains an invalid layer index")

        if self.model_type == "qwen3":
            if self.num_experts or self.num_experts_per_tok or self.moe_intermediate_size:
                raise ValueError("dense Qwen3 must not configure MoE experts")
        else:
            moe_values = {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_intermediate_size": self.moe_intermediate_size,
            }
            if invalid := [name for name, value in moe_values.items() if value <= 0]:
                raise ValueError(f"Qwen3MoE values must be positive: {invalid}")
            if self.num_experts_per_tok > self.num_experts:
                raise ValueError("num_experts_per_tok cannot exceed num_experts")

    @property
    def is_moe(self) -> bool:
        return self.model_type == "qwen3_moe"

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return (
            self.is_moe
            and layer_idx not in self.mlp_only_layers
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> ModelConfig:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        rope_type = rope.get("rope_type") if isinstance(rope, dict) else None
        if rope_type not in {None, "default"}:
            raise ValueError(f"unsupported RoPE type: {rope_type}")
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError("only the Qwen3 silu activation is implemented")
        if raw.get("shared_expert_intermediate_size", 0):
            raise ValueError("Qwen3MoE shared experts are not implemented")
        eos = raw.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        return cls(
            model_type=raw.get("model_type", "qwen3"),
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim"),
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", rope.get("rope_theta", 10_000.0)),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", False),
            use_sliding_window=raw.get("use_sliding_window", False),
            sliding_window=raw.get("sliding_window"),
            torch_dtype=raw.get("torch_dtype") or raw.get("dtype"),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=eos,
            num_experts=raw.get("num_experts", 0),
            num_experts_per_tok=raw.get("num_experts_per_tok", 0),
            moe_intermediate_size=raw.get("moe_intermediate_size", 0),
            decoder_sparse_step=raw.get("decoder_sparse_step", 1),
            mlp_only_layers=tuple(raw.get("mlp_only_layers") or ()),
            norm_topk_prob=raw.get("norm_topk_prob", False),
        )
