"""只读流式加载发布版 SafeTensors；TP/EP 分片直接写入 rank 本地 buffer。"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import (
    TensorHeader,
    rank_storage,
    read_checkpoint_headers,
    tensor_group,
    validate_backbone,
    validate_quantization,
)
from .config import DeepSeekV41Config
from .quantization import dequantize_weight


@dataclass(frozen=True)
class TensorSlice:
    shape: tuple[int, ...]
    axis: int | None = None
    start: int = 0
    stop: int = 0


def local_slice(
    item: TensorHeader, config: DeepSeekV41Config, rank: int, world_size: int
) -> TensorSlice | None:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world size")
    if tensor_group(item.name) in {"vision", "draft"}:
        return None
    sizes, rule = rank_storage(item, config, world_size)
    if rule == "expert_parallel" and sizes[rank] == 0:
        return None
    shape = list(item.shape)
    if rule == "engram_row_shard":
        rows = math.ceil(shape[0] / world_size)
        start = rank * rows
        shape[0] = rows
        return TensorSlice(tuple(shape), 0, start, min(start + rows, item.shape[0]))
    key = (
        item.name.split(".")[-2]
        if item.name.endswith((".weight", ".scale"))
        else (item.name.split(".")[-1])
    )
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
        return TensorSlice(item.shape)
    shape[axis] //= world_size
    start = rank * shape[axis]
    return TensorSlice(tuple(shape), axis, start, start + shape[axis])


def validate_parallel(config: DeepSeekV41Config, world_size: int) -> None:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    for name in ("vocab_size", "n_heads", "o_groups", "index_n_heads", "n_routed_experts"):
        if getattr(config, name) % world_size:
            raise ValueError(f"{name} must be divisible by TP={world_size}")
    # MX 的 scale tile 不得跨 rank；wo_a 在加载后转为 BF16。
    for width in (
        config.n_heads * config.head_dim,
        config.index_n_heads * config.index_head_dim,
        config.o_groups * config.o_lora_rank,
    ):
        if width // world_size % 32:
            raise ValueError("local quantized projection width must align to 32")


def load_rank_weights(
    directory: str | Path,
    config: DeepSeekV41Config,
    *,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device | str = "cpu",
    chunk_bytes: int = 64 << 20,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """每次最多读取 chunk_bytes 的连续行；不复制整张 Engram 表到 CPU/GPU。"""
    from safetensors import safe_open

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    validate_parallel(config, world_size)
    directory, device = Path(directory), torch.device(device)
    headers, report = read_checkpoint_headers(directory)
    validate_quantization(headers)
    validate_backbone(headers, config)
    plan = {
        name: spec
        for name, item in headers.items()
        if (spec := local_slice(item, config, rank, world_size)) is not None
    }
    expected_bytes = sum(rank_storage(headers[name], config, world_size)[0][rank] for name in plan)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if expected_bytes + (4 << 30) > free:
            raise RuntimeError(
                f"rank {rank} needs {expected_bytes / 2**30:.2f} GiB weights "
                f"plus 4 GiB reserve; only {free / 2**30:.2f} GiB free"
            )
    by_shard = defaultdict(list)
    for name in plan:
        by_shard[headers[name].shard].append(name)
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "I8": torch.uint8,
        "F8_E8M0": torch.uint8,
        "F8_E4M3": torch.float8_e4m3fn,
    }
    result = {}
    copied_bytes = 0
    for number, (shard, names) in enumerate(sorted(by_shard.items()), 1):
        with safe_open(directory / shard, framework="pt", device="cpu") as source:
            for name in sorted(names):
                item, spec = headers[name], plan[name]
                if item.dtype not in dtypes:
                    raise ValueError(f"unsupported runtime dtype: {name}: {item.dtype}")
                target = torch.empty(spec.shape, dtype=dtypes[item.dtype], device=device)
                sliced = source.get_slice(name)
                source_rows = spec.stop - spec.start if spec.axis == 0 else item.shape[0]
                row_bytes = math.prod(spec.shape[1:]) * target.element_size()
                step = max(1, chunk_bytes // row_bytes)
                for begin in range(0, source_rows, step):
                    end = min(begin + step, source_rows)
                    indices = [slice(None)] * len(item.shape)
                    indices[0] = slice(begin, end)
                    if spec.axis == 0:
                        indices[0] = slice(begin + spec.start, end + spec.start)
                    elif spec.axis is not None:
                        indices[spec.axis] = slice(spec.start, spec.stop)
                    chunk = sliced[tuple(indices)]
                    if item.dtype in {"I8", "F8_E8M0"}:
                        chunk = chunk.view(torch.uint8)
                    target[begin:end].copy_(chunk)
                    copied_bytes += chunk.numel() * chunk.element_size()
                    del chunk
                if source_rows < spec.shape[0]:
                    target[source_rows:].fill_(127 if name.endswith(".scale") else 0)
                result[name] = target
        if progress is not None:
            progress(
                f"rank {rank}: shard {number}/{len(by_shard)}, "
                f"{copied_bytes / 2**30:.2f} GiB copied"
            )
    for name in list(result):
        if name.endswith(".wo_a.weight") and result[name].dtype == torch.float8_e4m3fn:
            scale = result.pop(name[:-7] + ".scale")
            result[name] = dequantize_weight(result[name], scale, "fp8")
    actual_bytes = sum(t.numel() * t.element_size() for t in result.values())
    if actual_bytes != expected_bytes:
        raise ValueError(f"rank storage audit mismatch: {actual_bytes} != {expected_bytes}")
    report.update(
        rank=rank,
        world_size=world_size,
        loaded_tensors=len(result),
        loaded_bytes=actual_bytes,
        source_bytes_copied=copied_bytes,
        excluded_groups=["vision", "draft"],
        chunk_bytes=chunk_bytes,
    )
    return result, report


class WeightStore:
    """消费一次并检查完整性，避免遗漏参数被随机初始化掩盖。"""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors.copy()

    def pop(self, name: str) -> torch.Tensor:
        try:
            return self.tensors.pop(name)
        except KeyError as exc:
            raise ValueError(f"missing local weight: {name}") from exc

    def finish(self) -> None:
        if self.tensors:
            raise ValueError(f"unconsumed local weights: {sorted(self.tensors)[:8]}")
