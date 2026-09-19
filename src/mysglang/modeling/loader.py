from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from mysglang.config import ModelConfig

from .attention import AttentionBackend

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
    target_name: str
    index: object
    part: str


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
):
    from .qwen3 import Qwen3ForCausalLM

    path = Path(model_path)
    config = ModelConfig.from_pretrained(path)
    parameter_dtype = resolve_dtype(dtype, config)

    # Meta construction avoids first allocating a full fp32 model. ``to(dtype)`` only
    # changes meta tensor descriptors; ``to_empty`` then allocates the final storage once.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config, attention_backend=attention_backend)
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
                _copy_assignment(
                    model,
                    assignment,
                    checkpoint.get_tensor(source_name),
                    coverage,
                    parts,
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
        return _Assignment(name, Ellipsis, "full")

    for projection, part in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
        suffix = f".{projection}.weight"
        if name.endswith(suffix):
            target_name = name.removesuffix(suffix) + ".qkv_proj.weight"
            if target_name not in parameters:
                return None
            module_path = name.removesuffix(suffix)
            attention = model.get_submodule(module_path)
            offsets = {
                "q": (0, attention.q_size),
                "k": (attention.q_size, attention.q_size + attention.kv_size),
                "v": (
                    attention.q_size + attention.kv_size,
                    attention.q_size + 2 * attention.kv_size,
                ),
            }
            start, end = offsets[part]
            return _Assignment(target_name, (slice(start, end), slice(None)), part)

    expert = _EXPERT_WEIGHT.match(name)
    if expert:
        prefix, expert_index_text, projection = expert.groups()
        expert_index = int(expert_index_text)
        if projection == "down_proj":
            target_name = f"{prefix}.down_proj"
            index = (expert_index, slice(None), slice(None))
        else:
            target_name = f"{prefix}.gate_up_proj"
            intermediate_size = model.get_submodule(prefix).intermediate_size
            offset = 0 if projection == "gate_proj" else intermediate_size
            index = (
                expert_index,
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
            target = parameters[target_name]
            intermediate_size = target.size(0) // 2
            offset = 0 if projection == "gate_proj" else intermediate_size
            return _Assignment(
                target_name,
                (slice(offset, offset + intermediate_size), slice(None)),
                part,
            )
    return None


def _copy_assignment(
    model,
    assignment: _Assignment,
    source: torch.Tensor,
    coverage: dict[str, int],
    parts: set[tuple[str, str]],
) -> None:
    marker = (assignment.target_name, assignment.part)
    if marker in parts:
        raise ValueError(f"checkpoint initializes {assignment.target_name} part twice")
    target = _parameters(model)[assignment.target_name]
    target_view = target[assignment.index]
    if tuple(target_view.shape) != tuple(source.shape):
        raise ValueError(
            f"shape mismatch for {assignment.target_name}: "
            f"checkpoint {tuple(source.shape)} != model {tuple(target_view.shape)}"
        )
    target_view.copy_(source.to(device=target.device, dtype=target.dtype))
    coverage[assignment.target_name] = coverage.get(assignment.target_name, 0) + target_view.numel()
    parts.add(marker)


def _validate_coverage(model, coverage: dict[str, int]) -> None:
    parameters = _parameters(model)
    if model.config.tie_word_embeddings and coverage.get("embed_tokens.weight", 0):
        coverage.setdefault("lm_head.weight", parameters["lm_head.weight"].numel())
    missing = [
        name for name, parameter in parameters.items() if coverage.get(name, 0) < parameter.numel()
    ]
    if missing:
        raise KeyError(f"checkpoint did not initialize model parameters: {missing}")
