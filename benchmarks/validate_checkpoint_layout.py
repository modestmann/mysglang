"""Check that every tensor of a (possibly remote) checkpoint maps onto each TP rank.

Only SafeTensors headers are read: for a Hugging Face repo they are fetched with HTTP
Range requests, so a 35 GB checkpoint can be validated without downloading weights.
For every TP rank the model is built on the meta device; each checkpoint tensor is run
through the real loader mapping and the rank-local slice shape is compared with the
target parameter. Finally every parameter must be fully covered.

    PYTHONPATH=src python benchmarks/validate_checkpoint_layout.py \\
        --repo Qwen/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size 4 --enable-mtp
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist

from mysglang.config import ModelConfig
from mysglang.modeling import Qwen3ForCausalLM, TensorParallelContext
from mysglang.modeling.fp8 import FP8_DTYPE
from mysglang.modeling.loader import (
    _map_checkpoint_name,
    _ParameterIndex,
    _parameters,
    cast_unquantized_parameters,
)

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": FP8_DTYPE,
}


def _fetch(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _remote_headers(
    repo: str,
    revision: str,
    directory: Path,
) -> dict[str, tuple[str, list[int]]]:
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        try:
            (directory / name).write_bytes(_fetch(f"{base}/{name}"))
        except Exception:
            if name != "generation_config.json":
                raise
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    tensors: dict[str, tuple[str, list[int]]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        url = f"{base}/{shard}"
        (length,) = struct.unpack("<Q", _fetch(url, {"Range": "bytes=0-7"}))
        header = json.loads(_fetch(url, {"Range": f"bytes=8-{7 + length}"}))
        header.pop("__metadata__", None)
        tensors.update({name: (item["dtype"], item["shape"]) for name, item in header.items()})
    return tensors


def _local_headers(model_path: Path) -> dict[str, tuple[str, list[int]]]:
    tensors: dict[str, tuple[str, list[int]]] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with shard.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        header.pop("__metadata__", None)
        tensors.update({name: (item["dtype"], item["shape"]) for name, item in header.items()})
    return tensors


def validate_rank(
    config: ModelConfig,
    tensors: dict[str, tuple[str, list[int]]],
    rank: int,
    world_size: int,
    *,
    enable_mtp: bool,
) -> tuple[int, int]:
    parallel = TensorParallelContext(rank=rank, world_size=world_size)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config, tensor_parallel=parallel, enable_mtp=enable_mtp)
    cast_unquantized_parameters(model, torch.bfloat16)
    model.tie_weights()
    with _ParameterIndex(model):
        return _validate_model(model, config, tensors, rank)


def _validate_model(model, config, tensors, rank: int) -> tuple[int, int]:
    parameters = _parameters(model)
    coverage: dict[str, int] = {}
    ignored = 0
    for name, (dtype, shape) in tensors.items():
        assignments = _map_checkpoint_name(model, name)
        if assignments is None:
            ignored += 1
            continue
        source = torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
        for assignment in assignments:
            if assignment.target_name is None:
                continue
            target = parameters[assignment.target_name]
            if target.dtype == FP8_DTYPE and source.dtype != FP8_DTYPE:
                raise TypeError(f"{name}: FP8 parameter fed by a {dtype} checkpoint tensor")
            view = target[assignment.target_index]
            source_view = source[assignment.source_index]
            if tuple(view.shape) != tuple(source_view.shape):
                raise ValueError(
                    f"rank {rank}: {name} -> {assignment.target_name}: "
                    f"{tuple(source_view.shape)} != {tuple(view.shape)}"
                )
            target_name = assignment.target_name
            coverage[target_name] = coverage.get(target_name, 0) + view.numel()
    if config.tie_word_embeddings:
        coverage.setdefault("lm_head.weight", parameters["lm_head.weight"].numel())
    missing = [
        name for name, parameter in parameters.items() if coverage.get(name, 0) != parameter.numel()
    ]
    if missing:
        raise KeyError(f"rank {rank}: parameters not exactly covered: {missing[:8]}")
    local_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters.values()
    )
    return ignored, local_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Hugging Face repo, read through HTTP Range requests")
    source.add_argument("--model", type=Path, help="local checkpoint directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-mtp", action="store_true")
    args = parser.parse_args()

    if not dist.is_initialized():
        # TensorParallelContext only requires an initialized group; a local one suffices
        # for meta construction of every simulated rank.
        store = dist.HashStore()
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    with tempfile.TemporaryDirectory() as directory:
        if args.repo:
            tensors = _remote_headers(args.repo, args.revision, Path(directory))
            config = ModelConfig.from_pretrained(Path(directory))
        else:
            tensors = _local_headers(args.model)
            config = ModelConfig.from_pretrained(args.model)
    print(
        f"{len(tensors)} tensors; model_type={config.model_type}; "
        f"fp8={config.quantization is not None}; stop tokens={config.stop_token_ids}"
    )
    for rank in range(args.tensor_parallel_size):
        ignored, local_bytes = validate_rank(
            config,
            tensors,
            rank,
            args.tensor_parallel_size,
            enable_mtp=args.enable_mtp,
        )
        print(
            f"TP rank {rank}/{args.tensor_parallel_size}: every parameter covered; "
            f"{ignored} tensors ignored (vision/MTP); weights {local_bytes / 2**30:.2f} GiB"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
