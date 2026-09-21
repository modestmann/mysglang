from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from mysglang.config import ModelConfig

from .attention import AttentionBackend
from .parallel import RowParallelLinear, TensorParallelContext

_EXPERT_WEIGHT = re.compile(
    r"^(layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_files: tuple[Path, ...]
    loaded_tensors: int
    ignored_tensors: tuple[str, ...]


@dataclass(frozen=True)
class _Assignment:
    target_name: str | None
    target_index: object
    part: str
    source_index: object = Ellipsis


def resolve_dtype(dtype: torch.dtype | str | None, config: ModelConfig) -> torch.dtype:
    value = dtype or config.torch_dtype or torch.float32
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[value.removeprefix("torch.").lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported checkpoint dtype: {value}") from exc


def load_qwen3_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype | str | None,
    device: torch.device | str,
    attention_backend: AttentionBackend | None,
    tensor_parallel: TensorParallelContext | None,
    moe_dispatch_backend: str,
):
    from .qwen3 import Qwen3ForCausalLM

    path = Path(model_path)
    config = ModelConfig.from_pretrained(path)
    parameter_dtype = resolve_dtype(dtype, config)

    # Meta construction avoids first allocating a full fp32 model. ``to(dtype)`` only
    # changes meta tensor descriptors; ``to_empty`` then allocates the final storage once.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(
            config,
            attention_backend=attention_backend,
            tensor_parallel=tensor_parallel,
            moe_dispatch_backend=moe_dispatch_backend,
        )
    model.to(dtype=parameter_dtype)
    model.to_empty(device=device)
    model.tie_weights()
    model.reset_non_persistent_buffers(device)
    report = load_safetensors(model, path)
    model.checkpoint_load_report = report
    return model.eval()


@torch.no_grad()
def load_safetensors(model, model_path: str | Path) -> CheckpointLoadReport:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("loading Qwen3 checkpoints requires the 'safetensors' package") from exc

    files = _checkpoint_files(Path(model_path))
    loaded = 0
    ignored: list[str] = []
    coverage: dict[str, int] = {}
    parts: set[tuple[str, str]] = set()

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for source_name in checkpoint.keys():
                assignment = _map_weight(model, source_name)
                if assignment is None:
                    if source_name.endswith("rotary_emb.inv_freq"):
                        ignored.append(source_name)
                        continue
                    raise KeyError(f"unexpected checkpoint tensor: {source_name}")
                if assignment.target_name is None:
                    # Legacy MoE checkpoints store every expert separately. Do not even
                    # materialize another EP rank's tensor from SafeTensors into host RAM.
                    continue
                source, source_is_view = _read_safetensors_assignment(
                    checkpoint,
                    source_name,
                    assignment,
                )
                _copy_assignment(
                    model,
                    assignment,
                    source,
                    coverage,
                    parts,
                    source_is_view=source_is_view,
                )
                loaded += 1

    _validate_coverage(model, coverage)
    return CheckpointLoadReport(files, loaded, tuple(ignored))


@torch.no_grad()
def load_huggingface_state_dict(
    model,
    state_dict: Mapping[str, torch.Tensor],
) -> CheckpointLoadReport:
    """Load an in-memory HF state dict; used for small dense/MoE oracle tests."""
    coverage: dict[str, int] = {}
    parts: set[tuple[str, str]] = set()
    ignored: list[str] = []
    loaded = 0
    for source_name, tensor in state_dict.items():
        assignment = _map_weight(model, source_name)
        if assignment is None:
            if source_name.endswith("rotary_emb.inv_freq"):
                ignored.append(source_name)
                continue
            raise KeyError(f"unexpected checkpoint tensor: {source_name}")
        _copy_assignment(model, assignment, tensor, coverage, parts)
        loaded += 1
    _validate_coverage(model, coverage)
    return CheckpointLoadReport((), loaded, tuple(ignored))


def _checkpoint_files(model_path: Path) -> tuple[Path, ...]:
    if model_path.is_file() and model_path.suffix == ".safetensors":
        return (model_path,)
    single = model_path / "model.safetensors"
    if single.is_file():
        return (single,)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no SafeTensors checkpoint found under {model_path}")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    return tuple(sorted({model_path / filename for filename in index["weight_map"].values()}))


def _parameters(model) -> dict[str, torch.nn.Parameter]:
    return dict(model.named_parameters(remove_duplicate=False))


def _map_weight(model, source_name: str) -> _Assignment | None:
    parameters = _parameters(model)
    name = source_name.removeprefix("model.")
    if name in parameters:
        module_path, _, parameter_name = name.rpartition(".")
        module = model.get_submodule(module_path) if module_path else model
        if parameter_name in {"gate_up_proj", "down_proj"} and hasattr(
            module, "source_experts"
        ):
            return _Assignment(
                name,
                Ellipsis,
                "full",
                (module.source_experts, slice(None), slice(None)),
            )
        if parameter_name == "weight" and isinstance(module, RowParallelLinear):
            return _Assignment(
                name,
                Ellipsis,
                "full",
                (slice(None), module.source_columns),
            )
        return _Assignment(name, Ellipsis, "full")

    for projection, part in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
        suffix = f".{projection}.weight"
        if name.endswith(suffix):
            target_name = name.removesuffix(suffix) + ".qkv_proj.weight"
            if target_name not in parameters:
                return None
            module_path = name.removesuffix(suffix)
            attention = model.get_submodule(module_path)
            part_index = {"q": 0, "k": 1, "v": 2}[part]
            target_rows = attention.qkv_proj.local_segment(part_index)
            source_rows = attention.qkv_proj.source_segment(part_index)
            return _Assignment(
                target_name,
                (target_rows, slice(None)),
                part,
                (source_rows, slice(None)),
            )

    expert = _EXPERT_WEIGHT.match(name)
    if expert:
        prefix, expert_index_text, projection = expert.groups()
        expert_index = int(expert_index_text)
        experts = model.get_submodule(prefix)
        local_expert_index = experts.local_index(expert_index)
        if local_expert_index is None:
            # The tensor belongs to another EP rank. Recognize it without allocating
            # permanent model storage or treating it as an unknown checkpoint key.
            return _Assignment(None, Ellipsis, f"remote-expert-{expert_index}-{projection}")
        if projection == "down_proj":
            target_name = f"{prefix}.down_proj"
            index = (local_expert_index, slice(None), slice(None))
        else:
            target_name = f"{prefix}.gate_up_proj"
            intermediate_size = experts.intermediate_size
            offset = 0 if projection == "gate_proj" else intermediate_size
            index = (
                local_expert_index,
                slice(offset, offset + intermediate_size),
                slice(None),
            )
        if target_name not in parameters:
            return None
        return _Assignment(target_name, index, f"expert-{expert_index}-{projection}")

    for projection, part in (("gate_proj", "gate"), ("up_proj", "up")):
        suffix = f".{projection}.weight"
        if name.endswith(suffix) and ".experts." not in name:
            target_name = name.removesuffix(suffix) + ".gate_up_proj.weight"
            if target_name not in parameters:
                return None
            mlp = model.get_submodule(name.removesuffix(suffix))
            part_index = 0 if projection == "gate_proj" else 1
            return _Assignment(
                target_name,
                (mlp.gate_up_proj.local_segment(part_index), slice(None)),
                part,
                (mlp.gate_up_proj.source_segment(part_index), slice(None)),
            )
    return None


def _copy_assignment(
    model,
    assignment: _Assignment,
    source: torch.Tensor,
    coverage: dict[str, int],
    parts: set[tuple[str, str]],
    *,
    source_is_view: bool = False,
) -> None:
    if assignment.target_name is None:
        return
    marker = (assignment.target_name, assignment.part)
    if marker in parts:
        raise ValueError(f"checkpoint initializes {assignment.target_name} part twice")
    target = _parameters(model)[assignment.target_name]
    target_view = target[assignment.target_index]
    source_view = source if source_is_view else source[assignment.source_index]
    if tuple(target_view.shape) != tuple(source_view.shape):
        raise ValueError(
            f"shape mismatch for {assignment.target_name}: "
            f"checkpoint shard {tuple(source_view.shape)} != model {tuple(target_view.shape)}"
        )
    target_view.copy_(source_view.to(device=target.device, dtype=target.dtype))
    coverage[assignment.target_name] = coverage.get(assignment.target_name, 0) + target_view.numel()
    parts.add(marker)


def _read_safetensors_assignment(
    checkpoint,
    source_name: str,
    assignment: _Assignment,
) -> tuple[torch.Tensor, bool]:
    if assignment.source_index is Ellipsis:
        return checkpoint.get_tensor(source_name), False
    # Apply the rank-local slice while reading the memory-mapped file. Host RAM never
    # holds a full packed expert or global TP projection only to discard most of it.
    return checkpoint.get_slice(source_name)[assignment.source_index], True


def _validate_coverage(model, coverage: dict[str, int]) -> None:
    parameters = _parameters(model)
    if model.config.tie_word_embeddings and coverage.get("embed_tokens.weight", 0):
        coverage.setdefault("lm_head.weight", parameters["lm_head.weight"].numel())
    missing = [
        name for name, parameter in parameters.items() if coverage.get(name, 0) < parameter.numel()
    ]
    if missing:
        raise KeyError(f"checkpoint did not initialize model parameters: {missing}")
