from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

# Hugging Face 的纯文本子配置使用 *_text 后缀；内部统一成四种模型族。
_MODEL_TYPE_ALIASES = {
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3_moe",
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe",
}


@dataclass(frozen=True)
class Fp8QuantConfig:
    """Fine-grained block FP8 weights as stored by official Qwen FP8 checkpoints.

    Each ``[block_n, block_k]`` weight tile shares one float32 ``weight_scale_inv``;
    the real weight is ``fp8_weight * scale``. Activations are quantized dynamically, so
    the checkpoint carries no activation scales.
    """

    weight_block_size: tuple[int, int] = (128, 128)
    activation_scheme: str = "dynamic"
    modules_to_not_convert: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.weight_block_size) != 2 or any(size <= 0 for size in self.weight_block_size):
            raise ValueError("FP8 weight_block_size must contain two positive integers")
        if self.activation_scheme != "dynamic":
            raise ValueError("only dynamic FP8 activation scaling is implemented")

    def is_quantized(self, hf_module_name: str) -> bool:
        """Whether a checkpoint Linear (HF name, without ``.weight``) stores FP8."""
        # 与 HF 一致按模块路径片段匹配：既接受完整路径，也接受 "lm_head" 这类短名。
        name = f".{_normalize_hf_module_name(hf_module_name)}."
        for excluded in self.modules_to_not_convert:
            if f".{_normalize_hf_module_name(excluded)}." in name:
                return False
        return True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Fp8QuantConfig:
        if raw.get("quant_method") != "fp8":
            raise ValueError(f"unsupported quantization method: {raw.get('quant_method')}")
        if raw.get("fmt", "e4m3") != "e4m3":
            raise ValueError("only e4m3 FP8 checkpoints are implemented")
        block = raw.get("weight_block_size")
        if block is None:
            raise ValueError("only block-wise FP8 checkpoints are implemented")
        return cls(
            weight_block_size=tuple(block),
            activation_scheme=raw.get("activation_scheme", "dynamic"),
            modules_to_not_convert=tuple(raw.get("modules_to_not_convert") or ()),
        )


def _normalize_hf_module_name(name: str) -> str:
    # 同一个模块在不同导出里可能写成 model.language_model.x / model.x / x。
    for prefix in ("model.language_model.", "language_model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


@dataclass(frozen=True)
class ModelConfig:
    """Qwen3 / Qwen3MoE / Qwen3.5 hybrid architecture, independent of serving limits."""

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
    # generation_config.json 常给出多个停止 token（例如 <|im_end|> 和 <|endoftext|>）。
    eos_token_ids: tuple[int, ...] = ()

    # Qwen3MoE fields. Dense Qwen3 keeps num_experts=0.
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    norm_topk_prob: bool = False
    # Qwen3.5 MoE 在路由专家之外还有一个带 sigmoid 门的 shared expert。
    shared_expert_intermediate_size: int = 0

    # Qwen3.5 hybrid fields. An empty layer_types means every layer is full attention.
    layer_types: tuple[str, ...] = ()
    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = False
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    mtp_num_hidden_layers: int = 0

    quantization: Fp8QuantConfig | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}:
            raise ValueError(
                "model_type must be 'qwen3', 'qwen3_moe', 'qwen3_5' or 'qwen3_5_moe'"
            )
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
        if not self.layer_types:
            object.__setattr__(self, "layer_types", (FULL_ATTENTION,) * self.num_layers)
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        object.__setattr__(self, "eos_token_ids", tuple(self.eos_token_ids))
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
        if not 0 < self.partial_rotary_factor <= 1:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        if self.rotary_dim % 2 or self.rotary_dim <= 0:
            raise ValueError("partial rotary dimension must be a positive even number")
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
        if len(self.layer_types) != self.num_layers:
            raise ValueError("layer_types must contain one entry per layer")
        if unknown := set(self.layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION}:
            raise ValueError(f"unsupported layer types: {sorted(unknown)}")
        if FULL_ATTENTION not in self.layer_types:
            raise ValueError("at least one full-attention layer is required")
        if self.mtp_num_hidden_layers < 0:
            raise ValueError("mtp_num_hidden_layers must be non-negative")

        if not self.is_qwen3_5:
            if self.has_linear_attention or self.attn_output_gate or self.mtp_num_hidden_layers:
                raise ValueError("hybrid attention, output gates and MTP require Qwen3.5")
            if self.partial_rotary_factor != 1:
                raise ValueError("partial rotary embeddings require Qwen3.5")
            if self.shared_expert_intermediate_size:
                raise ValueError("Qwen3MoE shared experts are not implemented")
        if self.has_linear_attention:
            linear = {
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_num_value_heads": self.linear_num_value_heads,
            }
            if invalid := [name for name, value in linear.items() if value <= 0]:
                raise ValueError(f"linear-attention values must be positive: {invalid}")
            if self.linear_num_value_heads % self.linear_num_key_heads:
                raise ValueError("linear value heads must be a multiple of linear key heads")

        if not self.is_moe:
            if self.num_experts or self.num_experts_per_tok or self.moe_intermediate_size:
                raise ValueError("dense models must not configure MoE experts")
            if self.shared_expert_intermediate_size:
                raise ValueError("dense models must not configure a shared expert")
        else:
            moe_values = {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_intermediate_size": self.moe_intermediate_size,
            }
            if invalid := [name for name, value in moe_values.items() if value <= 0]:
                raise ValueError(f"MoE values must be positive: {invalid}")
            if self.num_experts_per_tok > self.num_experts:
                raise ValueError("num_experts_per_tok cannot exceed num_experts")
            if self.shared_expert_intermediate_size < 0:
                raise ValueError("shared_expert_intermediate_size must be non-negative")

    @property
    def is_moe(self) -> bool:
        return self.model_type in {"qwen3_moe", "qwen3_5_moe"}

    @property
    def is_qwen3_5(self) -> bool:
        """Qwen3.5 family: zero-centered RMSNorm, gated attention, optional DeltaNet/MTP."""
        return self.model_type in {"qwen3_5", "qwen3_5_moe"}

    @property
    def has_linear_attention(self) -> bool:
        return LINEAR_ATTENTION in self.layer_types

    @property
    def rotary_dim(self) -> int:
        assert self.head_dim is not None
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            index for index, kind in enumerate(self.layer_types) if kind == FULL_ATTENTION
        )

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            index for index, kind in enumerate(self.layer_types) if kind == LINEAR_ATTENTION
        )

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        """Every configured end-of-generation token, in a stable order."""
        values = ([] if self.eos_token_id is None else [self.eos_token_id]) + list(
            self.eos_token_ids
        )
        return tuple(dict.fromkeys(values))

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
            outer: dict[str, Any] = json.load(handle)
        # 多模态 Qwen3.5 把语言模型参数放在 text_config；量化配置仍在顶层。
        raw: dict[str, Any] = dict(outer.get("text_config") or outer)
        model_type = _MODEL_TYPE_ALIASES.get(raw.get("model_type", "qwen3"))
        if model_type is None:
            raise ValueError(f"unsupported model_type: {raw.get('model_type')}")
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        rope_type = rope.get("rope_type") if isinstance(rope, dict) else None
        if rope_type not in {None, "default"}:
            raise ValueError(f"unsupported RoPE type: {rope_type}")
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError("only the Qwen3 silu activation is implemented")
        if model_type == "qwen3_moe" and raw.get("shared_expert_intermediate_size", 0):
            raise ValueError("Qwen3MoE shared experts are not implemented")
        if raw.get("mtp_use_dedicated_embeddings"):
            raise ValueError("MTP layers with dedicated embeddings are not implemented")

        eos_ids = _token_list(raw.get("eos_token_id", outer.get("eos_token_id")))
        generation_path = (path if path.is_dir() else path.parent) / "generation_config.json"
        if generation_path.is_file():
            with generation_path.open(encoding="utf-8") as handle:
                eos_ids += _token_list(json.load(handle).get("eos_token_id"))
        eos_ids = list(dict.fromkeys(eos_ids))

        quantization = outer.get("quantization_config") or raw.get("quantization_config")
        partial_rotary = raw.get("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0))
        moe_intermediate = raw.get("moe_intermediate_size", 0)
        return cls(
            model_type=model_type,
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            # 纯 MoE 的 Qwen3.5 没有 dense MLP 层，也就不提供 intermediate_size。
            intermediate_size=raw.get("intermediate_size") or moe_intermediate,
            num_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim"),
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", rope.get("rope_theta", 10_000.0)),
            tie_word_embeddings=raw.get(
                "tie_word_embeddings", outer.get("tie_word_embeddings", False)
            ),
            attention_bias=raw.get("attention_bias", False),
            use_sliding_window=raw.get("use_sliding_window", False),
            sliding_window=raw.get("sliding_window"),
            torch_dtype=raw.get("torch_dtype") or raw.get("dtype"),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_id=eos_ids[0] if eos_ids else None,
            eos_token_ids=tuple(eos_ids[1:]),
            num_experts=raw.get("num_experts", 0) or 0,
            num_experts_per_tok=raw.get("num_experts_per_tok", 0) or 0,
            moe_intermediate_size=moe_intermediate or 0,
            decoder_sparse_step=raw.get("decoder_sparse_step", 1),
            mlp_only_layers=tuple(raw.get("mlp_only_layers") or ()),
            # Qwen3.5 MoE 总是对 top-k 路由权重重新归一化，配置里没有该字段。
            norm_topk_prob=raw.get("norm_topk_prob", model_type == "qwen3_5_moe"),
            shared_expert_intermediate_size=raw.get("shared_expert_intermediate_size", 0) or 0,
            layer_types=tuple(raw.get("layer_types") or ()),
            partial_rotary_factor=partial_rotary,
            attn_output_gate=raw.get("attn_output_gate", False),
            linear_conv_kernel_dim=raw.get("linear_conv_kernel_dim", 4),
            linear_key_head_dim=raw.get("linear_key_head_dim", 0),
            linear_value_head_dim=raw.get("linear_value_head_dim", 0),
            linear_num_key_heads=raw.get("linear_num_key_heads", 0),
            linear_num_value_heads=raw.get("linear_num_value_heads", 0),
            mtp_num_hidden_layers=raw.get("mtp_num_hidden_layers", 0),
            quantization=Fp8QuantConfig.from_dict(quantization) if quantization else None,
        )


def _token_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(token) for token in value]  # type: ignore[union-attr]
