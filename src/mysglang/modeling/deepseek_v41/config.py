"""DeepSeek V4.1 配置；与 Qwen 的 GQA/DeltaNet 配置分别维护。"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeepSeekV41Config:
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_heads: int = 64
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0,) * 3
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    dtype: str = "fp8"
    expert_dtype: str | None = "fp4"
    n_mtp_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3
    vision_n_layers: int = 32

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "dim",
            "moe_inter_dim",
            "n_layers",
            "n_heads",
            "n_routed_experts",
            "n_activated_experts",
            "q_lora_rank",
            "head_dim",
            "rope_head_dim",
            "o_groups",
            "o_lora_rank",
            "window_size",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "hc_mult",
            "hc_sinkhorn_iters",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_type != "deepseek_v41":
            raise ValueError(f"unsupported model_type: {self.model_type}")
        if self.n_shared_experts != 1 or self.n_activated_experts > self.n_routed_experts:
            raise ValueError("requires one shared expert and top-k <= routed experts")
        if self.score_func not in {"sqrtsoftplus", "sigmoid", "softmax"}:
            raise ValueError(f"unsupported score_func: {self.score_func}")
        if min(self.gate_temp, self.norm_eps, self.hc_eps, self.rope_factor) <= 0:
            raise ValueError("temperature, eps and RoPE factor must be positive")
        if min(self.rope_theta, self.compress_rope_theta) <= 1:
            raise ValueError("RoPE theta must exceed one")
        if self.beta_fast <= 0 or self.beta_slow <= 0 or self.original_seq_len < 0:
            raise ValueError("invalid YaRN configuration")
        if self.n_heads % self.o_groups:
            raise ValueError("n_heads must be divisible by o_groups")
        if self.rope_head_dim % 2 or self.rope_head_dim > min(self.head_dim, self.index_head_dim):
            raise ValueError("RoPE dimension must be even and fit both attention and indexer")
        if self.n_mtp_layers < 0 or self.vision_n_layers < 0:
            raise ValueError("negative layer count")
        if len(self.compress_ratios) != self.n_layers + self.n_mtp_layers:
            raise ValueError("compress_ratios must cover backbone and draft layers")
        if any(r not in (0, 1, 2) for r in self.compress_ratios):
            raise ValueError("V4.1 compression ratios must be 0, 1 or 2")
        for name in ("kv_source_layers", "index_source_layers", "engram_layer_ids"):
            ids = getattr(self, name)
            if tuple(sorted(set(ids))) != ids or any(i < 0 or i >= self.n_layers for i in ids):
                raise ValueError(f"{name} must contain sorted unique backbone layer indices")
        if not set(self.kv_source_layers) <= set(self.index_source_layers):
            raise ValueError("each KV source must also produce index keys")
        kv_source = index_source = None
        for layer, ratio in enumerate(self.compress_ratios[: self.n_layers]):
            if layer in self.kv_source_layers:
                kv_source = layer
            if layer in self.index_source_layers:
                index_source = layer
            if not ratio:
                if layer in self.index_source_layers:
                    raise ValueError("sliding-window-only layer cannot own compressed state")
                continue
            if kv_source is None or index_source is None:
                raise ValueError("compressed attention consumer precedes its source")
            if any(self.compress_ratios[i] != ratio for i in (kv_source, index_source)):
                raise ValueError(
                    "shared KV/index source must have the consumer's compression ratio"
                )
        if self.candidate_source_layer >= 0:
            if self.candidate_source_layer not in self.index_source_layers:
                raise ValueError("candidate source must be an index source")
            if min(self.candidate_topk_blocks, self.candidate_block_size) <= 0:
                raise ValueError("candidate block counts and size must be positive")
            candidate_ratio = self.compress_ratios[self.candidate_source_layer]
            if any(
                self.compress_ratios[i] != candidate_ratio
                for i in self.index_source_layers
                if i > self.candidate_source_layer
            ):
                raise ValueError("candidate consumers must share the source's position space")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("each Engram layer needs a table size")
        if self.engram_layer_ids:
            if (
                self.engram_max_ngram_size < 2
                or min(
                    self.engram_vocab_size,
                    self.engram_n_heads,
                    self.engram_head_dim,
                    self.engram_compressed_vocab_size,
                    *self.engram_num_embeddings,
                )
                <= 0
            ):
                raise ValueError("invalid Engram dimensions")
            if not 0 <= self.engram_pad_id < self.vocab_size:
                raise ValueError("Engram pad token is outside vocabulary")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DeepSeekV41Config:
        """读取发布版嵌套 HF 配置或官方 inference/config.json。"""
        kind = raw.get("model_type")
        if kind not in (None, "deepseek_v41", "deepseek_v41_text"):
            raise ValueError(f"unsupported model_type: {kind}")
        text = dict(raw.get("text_config", raw))
        if text.get("model_type") not in (None, "deepseek_v41", "deepseek_v41_text"):
            raise ValueError("text_config belongs to another model family")
        aliases = {
            "hidden_size": "dim",
            "moe_intermediate_size": "moe_inter_dim",
            "num_hidden_layers": "n_layers",
            "num_attention_heads": "n_heads",
            "qk_rope_head_dim": "rope_head_dim",
            "rms_norm_eps": "norm_eps",
            "num_experts_per_tok": "n_activated_experts",
            "scoring_func": "score_func",
            "routed_scaling_factor": "route_scale",
            "sliding_window": "window_size",
            "kv_source_layer_ids": "kv_source_layers",
            "index_source_layer_ids": "index_source_layers",
            "candidate_source_layer_id": "candidate_source_layer",
            "engram_pad_token_id": "engram_pad_id",
            "num_nextn_predict_layers": "n_mtp_layers",
            "dspark_num_experts_per_tok": "dspark_n_activated_experts",
        }
        allowed = {field.name for field in fields(cls)}
        values = {aliases.get(k, k): v for k, v in text.items() if aliases.get(k, k) in allowed}
        required = {
            "vocab_size",
            "dim",
            "n_layers",
            "n_heads",
            "compress_ratios",
            "kv_source_layers",
            "index_source_layers",
        }
        if missing := required - values.keys():
            raise ValueError(f"missing DeepSeek V4.1 config fields: {sorted(missing)}")
        values["model_type"] = "deepseek_v41"
        for source, target in {
            "factor": "rope_factor",
            "original_max_position_embeddings": "original_seq_len",
            "beta_fast": "beta_fast",
            "beta_slow": "beta_slow",
        }.items():
            if source in text.get("rope_scaling", {}):
                values[target] = text["rope_scaling"][source]
        quant = raw.get("quantization_config", {})
        if quant:
            if (
                quant.get("quant_method") != "fp8"
                or quant.get("scale_fmt") != "ue8m0"
                or quant.get("weight_block_size") != [32, 32]
                or quant.get("expert_dtype") != "fp4"
            ):
                raise ValueError("expected native MXFP8/MXFP4 weights with UE8M0 scales")
            values.update(dtype="fp8", expert_dtype="fp4")
        if "vision_config" in raw:
            values["vision_n_layers"] = raw["vision_config"].get("num_hidden_layers", 0)
        for field in fields(cls):
            if isinstance(field.default, tuple) and field.name in values:
                values[field.name] = tuple(values[field.name])
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> DeepSeekV41Config:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def tiny(cls) -> DeepSeekV41Config:
        """覆盖窗口、两种压缩率、跨层复用、候选过滤及 Engram 的 CPU 测试配置。"""
        return cls(
            vocab_size=64,
            dim=32,
            moe_inter_dim=24,
            n_layers=6,
            n_heads=4,
            n_routed_experts=4,
            n_activated_experts=2,
            q_lora_rank=16,
            head_dim=16,
            rope_head_dim=8,
            o_groups=2,
            o_lora_rank=8,
            window_size=4,
            compress_ratios=(0, 2, 2, 1, 1, 1),
            kv_source_layers=(1, 3),
            index_source_layers=(1, 3, 5),
            original_seq_len=16,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            candidate_source_layer=3,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            engram_layer_ids=(1, 4),
            engram_num_embeddings=(834, 1020),
            engram_max_ngram_size=4,
            engram_vocab_size=127,
            engram_n_heads=2,
            engram_head_dim=8,
            engram_compressed_vocab_size=64,
            dtype="bf16",
            expert_dtype=None,
            n_mtp_layers=0,
            dspark_block_size=0,
            dspark_target_layer_ids=(),
            vision_n_layers=0,
        )
