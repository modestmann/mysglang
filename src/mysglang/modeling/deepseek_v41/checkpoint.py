"""只读 SafeTensors 文件头，核对原生布局并估算 TP/EP 常驻权重。

估算采用官方 convert.py 的分片规则：attention TP、整专家 EP、Engram 按行切分；
wo_a 转 BF16，其余保持 checkpoint 存储类型。不是总显存或性能预测。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DeepSeekV41Config

_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_EXPERT = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


@dataclass(frozen=True)
class TensorHeader:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_shard_header(path: Path) -> tuple[dict[str, TensorHeader], bytes]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated SafeTensors header: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(128 * 1024**2, size - 8):
            raise ValueError(f"invalid SafeTensors header length: {path.name}")
        raw = handle.read(length)
    data = json.loads(raw, object_pairs_hook=_unique_object)
    tensors, intervals = {}, []
    for name, item in data.items():
        if name == "__metadata__":
            continue
        shape, dtype, offsets = item["shape"], item["dtype"], item["data_offsets"]
        if dtype not in _BYTES or any(type(s) is not int or s < 0 for s in shape):
            raise ValueError(f"invalid tensor type/shape: {name}")
        if (
            len(offsets) != 2
            or any(type(i) is not int for i in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length
        ):
            raise ValueError(f"invalid tensor offsets: {name}")
        nbytes = math.prod(shape) * _BYTES[dtype]
        if offsets[1] - offsets[0] != nbytes:
            raise ValueError(f"tensor byte count disagrees with shape/dtype: {name}")
        tensors[name] = TensorHeader(name, path.name, dtype, tuple(shape), nbytes)
        intervals.append((*offsets, name))
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise ValueError(f"overlap or hole in SafeTensors payload before {name}")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError(f"unindexed trailing payload: {path.name}")
    return tensors, prefix + raw


def read_checkpoint_headers(directory: str | Path):
    directory = Path(directory).resolve()
    index = json.loads(
        (directory / "model.safetensors.index.json").read_text(), object_pairs_hook=_unique_object
    )
    weight_map = index["weight_map"]
    if not weight_map:
        raise ValueError("empty checkpoint index")
    tensors: dict[str, TensorHeader] = {}
    digest = hashlib.sha256()
    read_bytes = 0
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        path = directory / shard
        if path.resolve().parent != directory or path.suffix != ".safetensors":
            raise ValueError(f"shard must be directly inside checkpoint directory: {shard}")
        header, raw = read_shard_header(path)
        digest.update(shard.encode())
        digest.update(raw)
        read_bytes += len(raw)
        for name, tensor in header.items():
            if name in tensors or weight_map.get(name) != shard:
                raise ValueError(f"duplicate, extra or misplaced tensor: {name}")
            tensors[name] = tensor
    if tensors.keys() != weight_map.keys():
        raise ValueError(f"missing tensors: {sorted(weight_map.keys() - tensors.keys())[:8]}")
    total = sum(t.nbytes for t in tensors.values())
    if index.get("metadata", {}).get("total_size") != total:
        raise ValueError("index total_size differs from tensor payload bytes")
    return tensors, {
        "shards": len(shards),
        "tensor_count": len(tensors),
        "checkpoint_bytes": total,
        "header_bytes_read": read_bytes,
        "headers_sha256": digest.hexdigest(),
    }


def validate_quantization(tensors: dict[str, TensorHeader]) -> None:
    for name, item in tensors.items():
        if name.endswith(".scale"):
            weight = tensors.get(name[:-6] + ".weight")
            if weight is None or weight.dtype not in {"I8", "F8_E4M3"}:
                raise ValueError(f"orphan or unsupported quantization scale: {name}")
            if item.dtype != "F8_E8M0":
                raise ValueError(f"expected UE8M0 scale: {name}")
        if item.dtype not in {"I8", "F8_E4M3"}:
            continue
        if not name.endswith(".weight") or len(item.shape) != 2:
            raise ValueError(f"unrecognized quantized tensor: {name}")
        scale = tensors.get(name[:-7] + ".scale")
        if scale is None:
            raise ValueError(f"missing quantization scale: {name}")
        rows, columns = item.shape
        if item.dtype == "I8":
            if not _EXPERT.fullmatch(name) or columns * 2 % 32:
                raise ValueError(f"invalid packed MXFP4 expert: {name}")
            expected = (rows, columns * 2 // 32)
        elif ".engram.embed." in name:
            if columns % 32:
                raise ValueError(f"invalid Engram row quantization: {name}")
            expected = (rows, columns // 32)
        else:
            expected = (math.ceil(rows / 32), math.ceil(columns / 32))
        if scale.shape != expected:
            raise ValueError(f"scale layout mismatch for {name}: {scale.shape} != {expected}")


def expected_backbone_shapes(c: DeepSeekV41Config) -> dict[str, tuple[int, ...]]:
    """无需构造模型即可核对全部 backbone 权重的逻辑形状（FP4 的 K 维需解包）。"""
    shapes = {
        "embed.weight": (c.vocab_size, c.dim),
        "head.weight": (c.vocab_size, c.dim),
        "norm.weight": (c.dim,),
    }
    for layer in range(c.n_layers):
        root = f"layers.{layer}."
        current = {
            "attn.attn_sink": (c.n_heads,),
            "attn.wq_a.weight": (c.q_lora_rank, c.dim),
            "attn.q_norm.weight": (c.q_lora_rank,),
            "attn.wq_b.weight": (c.n_heads * c.head_dim, c.q_lora_rank),
            "attn.wkv.weight": (c.head_dim, c.dim),
            "attn.kv_norm.weight": (c.head_dim,),
            "attn.wo_a.weight": (c.o_groups * c.o_lora_rank, c.n_heads * c.head_dim // c.o_groups),
            "attn.wo_b.weight": (c.dim, c.o_groups * c.o_lora_rank),
            "attn_norm.weight": (c.dim,),
            "ffn_norm.weight": (c.dim,),
            "ffn.gate.weight": (c.n_routed_experts, c.dim),
            "ffn.gate.bias": (c.n_routed_experts,),
        }
        if c.vision_n_layers:
            current["ffn.gate.bias_vl"] = (c.n_routed_experts,)
        for part in ("attn", "ffn"):
            count = c.hc_mult * (c.hc_mult + 2)
            current.update(
                {
                    f"hc_{part}_fn": (count, c.hc_mult * c.dim),
                    f"hc_{part}_base": (count,),
                    f"hc_{part}_scale": (3,),
                }
            )
        for expert in ["shared_experts", *(f"experts.{i}" for i in range(c.n_routed_experts))]:
            for proj in ("w1", "w2", "w3"):
                current[f"ffn.{expert}.{proj}.weight"] = (
                    (c.dim, c.moe_inter_dim) if proj == "w2" else (c.moe_inter_dim, c.dim)
                )
        if layer in c.kv_source_layers:
            current.update(
                {
                    "attn.compressor.wkv.weight": (c.head_dim, c.dim),
                    "attn.compressor.norm.weight": (c.head_dim,),
                    "attn.indexer.wk.weight": (c.index_head_dim, c.head_dim),
                    "attn.indexer.k_norm.weight": (c.index_head_dim,),
                }
            )
            if c.compress_ratios[layer] > 1:
                current["attn.compressor.wgate.weight"] = (c.head_dim, c.dim)
        if layer in c.index_source_layers:
            current.update(
                {
                    "attn.indexer.wq_b.weight": (c.index_n_heads * c.index_head_dim, c.q_lora_rank),
                    "attn.indexer.weights_proj.weight": (c.index_n_heads, c.dim),
                }
            )
        if layer in c.engram_layer_ids:
            rows = c.engram_num_embeddings[c.engram_layer_ids.index(layer)]
            width = (c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim
            current.update(
                {
                    "engram.embed.weight": (rows, c.engram_head_dim),
                    "engram.wkv.weight": ((c.hc_mult + 1) * c.dim, width),
                    "engram.q_weight": (c.hc_mult, c.dim),
                    "engram.k_weight": (c.hc_mult, c.dim),
                }
            )
        shapes.update({root + name: shape for name, shape in current.items()})
    return shapes


def validate_backbone(tensors: dict[str, TensorHeader], config: DeepSeekV41Config) -> None:
    expected = expected_backbone_shapes(config)
    for name, shape in expected.items():
        if name not in tensors:
            raise ValueError(f"missing backbone parameter: {name}")
        item = tensors[name]
        actual = item.shape
        if item.dtype == "I8":
            actual = (actual[0], actual[1] * 2)
        if actual != shape:
            raise ValueError(f"backbone shape mismatch: {name}: {actual} != {shape}")
    for name in tensors:
        if name.startswith(("layers.", "embed.", "head.", "norm.")):
            weight_name = name[:-6] + ".weight" if name.endswith(".scale") else name
            if weight_name not in expected:
                raise ValueError(f"unexpected backbone parameter: {name}")


def tensor_group(name: str) -> str:
    if name.startswith(("vision.", "aligner.", "image_")):
        return "vision"
    if name.startswith("mtp."):
        return "draft"
    if ".engram.embed." in name:
        return "engram_tables"
    if ".ffn.experts." in name:
        return "routed_experts"
    return "backbone_other"


def rank_storage(
    item: TensorHeader, config: DeepSeekV41Config, tp_size: int
) -> tuple[list[int], str]:
    name, shape = item.name, item.shape
    match = _EXPERT.fullmatch(name)
    if match:
        count = config.n_routed_experts if match[1] == "layers" else config.dspark_n_routed_experts
        if count <= 0 or count % tp_size or int(match[3]) >= count:
            raise ValueError(f"expert cannot be assigned to rank: {name}")
        owner = int(match[3]) // (count // tp_size)
        return [item.nbytes if rank == owner else 0 for rank in range(tp_size)], "expert_parallel"
    if ".engram.embed." in name:
        # 官方转换器会给最后一个 rank 补行；scale 的补充值为 1。
        rows = math.ceil(shape[0] / tp_size)
        nbytes = rows * math.prod(shape[1:]) * _BYTES[item.dtype]
        return [nbytes] * tp_size, "engram_row_shard"
    if name.startswith("mtp.") and name.split(".", 2)[-1] in {"embed.weight", "head.weight"}:
        return [0] * tp_size, "tied_draft_weight"
    key = name.split(".")[-2] if name.endswith((".weight", ".scale")) else name.split(".")[-1]
    axis = {
        "embed": 0,
        "head": 0,
        "wq_b": 0,
        "wo_a": 0,
        "wo_b": 1,
        "attn_sink": 0,
        "weights_proj": 0,
    }.get(key)
    if axis is None:
        return [item.nbytes] * tp_size, "replicated"
    if len(shape) <= axis or shape[axis] % tp_size:
        raise ValueError(f"tensor axis {axis} cannot be divided by TP={tp_size}: {name}")
    nbytes = item.nbytes // tp_size
    if key == "wo_a":
        nbytes = 0 if name.endswith(".scale") else math.prod(shape) * 2 // tp_size
    return [nbytes] * tp_size, "tensor_parallel"


def audit_checkpoint(
    directory: str | Path,
    config: DeepSeekV41Config,
    *,
    tp_size: int = 8,
    text_only: bool = False,
    include_draft: bool = True,
) -> dict:
    if tp_size <= 0 or any(
        value % tp_size
        for value in (
            config.vocab_size,
            config.n_heads,
            config.index_n_heads,
            config.o_groups,
            config.n_routed_experts,
        )
    ):
        raise ValueError(
            "TP must divide vocabulary, attention/index heads, output groups and experts"
        )
    tensors, report = read_checkpoint_headers(directory)
    validate_quantization(tensors)
    validate_backbone(tensors, config)
    groups, dtypes, rules = Counter(), Counter(), Counter()
    ranks = [Counter() for _ in range(tp_size)]
    included_bytes = 0
    for name, item in tensors.items():
        group = tensor_group(name)
        groups[group] += item.nbytes
        dtypes[item.dtype] += item.nbytes
        if (text_only and group == "vision") or (not include_draft and group == "draft"):
            continue
        included_bytes += item.nbytes
        sizes, rule = rank_storage(item, config, tp_size)
        rules[rule] += 1
        for rank, size in zip(ranks, sizes, strict=True):
            rank[group] += size
    report.update(
        {
            "model_path": str(Path(directory).resolve()),
            "tp_size": tp_size,
            "text_only": text_only,
            "include_draft": include_draft,
            "source_bytes_by_group": dict(groups),
            "source_bytes_by_dtype": dict(dtypes),
            "included_source_bytes": included_bytes,
            "tensor_count_by_rule": dict(rules),
            "estimated_rank_weight_bytes": [sum(rank.values()) for rank in ranks],
            "estimated_rank_weight_bytes_by_group": [dict(rank) for rank in ranks],
            "scope": "headers_only; backbone/scale shapes checked; tensor payloads not read",
            "memory_policy": "official-reference TP/EP + padded Engram row shards + BF16 wo_a",
            "excluded_memory": [
                "KV/index/Engram history caches",
                "activations",
                "dequantization workspace",
                "communication buffers",
                "CUDA graphs",
                "allocator overhead",
            ],
            "limitations": [
                "vision/draft topology is not fully shape-validated",
                "estimated resident weights are not measured GPU memory",
            ],
        }
    )
    return report
