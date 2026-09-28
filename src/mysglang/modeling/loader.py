from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mysglang.config import ModelConfig

from .attention import AttentionBackend
from .deltanet import Qwen3_5GatedDeltaNet
from .fp8 import FP8_DTYPE, dequantize_block_fp8
from .parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelContext

_EXPERT_WEIGHT = re.compile(
    r"^((?:mtp\.)?layers\.\d+\.mlp\.experts)\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_SCALE_SUFFIX = "_scale_inv"


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


def cast_unquantized_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    """``model.to(dtype)`` that leaves FP8 weights and their float32 scales intact."""
    for module in model.modules():
        keep = getattr(module, "keep_dtype_parameters", ())
        for name, parameter in list(module._parameters.items()):
            if parameter is None or name in keep or not parameter.is_floating_point():
                continue
            module._parameters[name] = nn.Parameter(
                parameter.detach().to(dtype),
                requires_grad=parameter.requires_grad,
            )


def load_qwen3_model(
    model_path: str | Path,
    *,
    dtype: torch.dtype | str | None,
    device: torch.device | str,
    attention_backend: AttentionBackend | None,
    tensor_parallel: TensorParallelContext | None,
    moe_dispatch_backend: str,
    linear_attention_backend: str = "auto",
    enable_mtp: bool = False,
    fp8_weights: bool = True,
    enable_vision: bool = False,
):
    from .qwen3 import Qwen3ForCausalLM

    path = Path(model_path)
    config = ModelConfig.from_pretrained(path)
    parameter_dtype = resolve_dtype(dtype, config)

    # Meta construction avoids first allocating a full fp32 model. The dtype cast only
    # changes meta tensor descriptors; ``to_empty`` then allocates the final storage once.
    with torch.device("meta"):
        model = Qwen3ForCausalLM(
            config,
            attention_backend=attention_backend,
            tensor_parallel=tensor_parallel,
            moe_dispatch_backend=moe_dispatch_backend,
            linear_attention_backend=linear_attention_backend,
            enable_mtp=enable_mtp,
            fp8_weights=fp8_weights,
            enable_vision=enable_vision,
        )
    cast_unquantized_parameters(model, parameter_dtype)
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
    with _ParameterIndex(model):
        return _load_files(model, files, safe_open)


def _load_files(model, files: tuple[Path, ...], safe_open) -> CheckpointLoadReport:
    loaded = 0
    ignored: list[str] = []
    coverage: dict[str, int] = {}
    parts: set[tuple[str, str]] = set()
    scale_files = _scale_locations(files, safe_open)

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for source_name in checkpoint.keys():
                assignments = _map_checkpoint_name(model, source_name)
                if assignments is None:
                    ignored.append(source_name)
                    continue
                for assignment in assignments:
                    if assignment.target_name is None:
                        # Legacy MoE checkpoints store every expert separately. Do not even
                        # materialize another EP rank's tensor from SafeTensors into host RAM.
                        continue
                    target = _parameters(model)[assignment.target_name]
                    if _needs_dequantization(checkpoint, source_name, target):
                        source = _dequantized_source(
                            checkpoint, source_name, scale_files, safe_open, model, target
                        )
                        source_is_view = False
                    else:
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
    """Load an in-memory HF state dict; used for small dense/MoE/hybrid oracle tests."""
    with _ParameterIndex(model):
        return _load_state_dict(model, state_dict)


def _load_state_dict(model, state_dict: Mapping[str, torch.Tensor]) -> CheckpointLoadReport:
    coverage: dict[str, int] = {}
    parts: set[tuple[str, str]] = set()
    ignored: list[str] = []
    loaded = 0
    for source_name, tensor in state_dict.items():
        assignments = _map_checkpoint_name(model, source_name)
        if assignments is None:
            ignored.append(source_name)
            continue
        for assignment in assignments:
            if assignment.target_name is None:
                continue
            target = _parameters(model)[assignment.target_name]
            source = tensor
            if tensor.dtype == FP8_DTYPE and target.dtype != FP8_DTYPE:
                scale = state_dict[source_name + _SCALE_SUFFIX]
                source = dequantize_block_fp8(
                    tensor, scale, model.config.quantization.weight_block_size, target.dtype
                )
            _copy_assignment(model, assignment, source, coverage, parts)
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


def _scale_locations(files: tuple[Path, ...], safe_open) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.endswith(_SCALE_SUFFIX):
                    locations[name] = file
    return locations


def _needs_dequantization(checkpoint, source_name: str, target: torch.Tensor) -> bool:
    if target.dtype == FP8_DTYPE or source_name.endswith(_SCALE_SUFFIX):
        return False
    return checkpoint.get_slice(source_name).get_dtype() == "F8_E4M3"


def _dequantized_source(
    checkpoint,
    source_name: str,
    scale_files: dict[str, Path],
    safe_open,
    model,
    target: torch.Tensor,
) -> torch.Tensor:
    scale_name = source_name + _SCALE_SUFFIX
    if scale_name not in scale_files:
        raise KeyError(f"FP8 checkpoint tensor has no scale: {source_name}")
    with safe_open(scale_files[scale_name], framework="pt", device="cpu") as scale_file:
        scale = scale_file.get_tensor(scale_name)
    return dequantize_block_fp8(
        checkpoint.get_tensor(source_name),
        scale,
        model.config.quantization.weight_block_size,
        target.dtype,
    )


def _parameters(model) -> dict[str, torch.nn.Parameter]:
    # 35B checkpoint 有六万多个张量；加载期间复用同一份参数索引，避免每个张量都遍历模型。
    index = model.__dict__.get("_loader_parameter_index")
    if index is not None:
        return index
    return dict(model.named_parameters(remove_duplicate=False))


class _ParameterIndex:
    """Cache ``named_parameters`` while a load runs; parameters are fixed by then."""

    def __init__(self, model) -> None:
        self.model = model

    def __enter__(self):
        self.model.__dict__["_loader_parameter_index"] = dict(
            self.model.named_parameters(remove_duplicate=False)
        )
        return self

    def __exit__(self, *exc) -> None:
        self.model.__dict__.pop("_loader_parameter_index", None)


def _normalize_checkpoint_name(model, source_name: str) -> str | None:
    """Map HF names to this model's parameter namespace, or None for ignored tensors."""
    if source_name.startswith(("model.visual.", "visual.")):
        # 未启用视觉输入时不加载视觉塔；启用时它在所有 TP rank 上复制。
        if model.visual is None:
            return None
        return "visual." + source_name.split("visual.", 1)[1]
    if source_name.startswith("mtp."):
        return source_name if model.mtp is not None else None
    for prefix in ("model.language_model.", "language_model.model.", "model."):
        if source_name.startswith(prefix):
            return source_name[len(prefix) :]
    return source_name


def _map_checkpoint_name(model, source_name: str) -> list[_Assignment] | None:
    name = _normalize_checkpoint_name(model, source_name)
    if name is None or name.endswith("rotary_emb.inv_freq"):
        return None
    assignments = _map_weight(model, name)
    if assignments is None:
        raise KeyError(f"unexpected checkpoint tensor: {source_name}")
    return assignments


def _block_slice(rows: slice, block: int) -> slice:
    if rows.start % block:
        raise ValueError("FP8 shard does not start on a scale block boundary")
    return slice(rows.start // block, -(-rows.stop // block))


def _column_segment(
    linear: ColumnParallelLinear,
    target_name: str,
    part: int,
    source_rows: slice,
    is_scale: bool,
    label: str,
) -> _Assignment:
    target_rows = linear.local_segment(part)
    if is_scale:
        block_n = linear.quant.weight_block_size[0]
        target_rows = _block_slice(target_rows, block_n)
        source_rows = _block_slice(source_rows, block_n)
    return _Assignment(target_name, (target_rows, slice(None)), label, (source_rows, slice(None)))


def _split_parameter_name(name: str) -> tuple[str, str, bool]:
    """``x.q_proj.weight_scale_inv`` -> (``x.q_proj``, ``weight_scale_inv``, True)."""
    module_path, _, parameter_name = name.rpartition(".")
    return module_path, parameter_name, parameter_name.endswith(_SCALE_SUFFIX)


def _map_weight(model, name: str) -> list[_Assignment] | None:
    parameters = _parameters(model)
    module_path, parameter_name, is_scale = _split_parameter_name(name)
    if is_scale and model.fp8 is None:
        # 以 fp8_weights=False 加载 FP8 checkpoint：scale 只在反量化对应权重时使用。
        return []

    if name in parameters:
        module = model.get_submodule(module_path) if module_path else model
        if parameter_name.removesuffix(_SCALE_SUFFIX) in {"gate_up_proj", "down_proj"} and hasattr(
            module, "source_experts"
        ):
            return [
                _Assignment(
                    name,
                    Ellipsis,
                    "full",
                    (module.source_experts, slice(None), slice(None)),
                )
            ]
        if isinstance(module, RowParallelLinear) and parameter_name in {
            "weight",
            "weight" + _SCALE_SUFFIX,
        }:
            columns = module.source_columns
            if is_scale:
                columns = _block_slice(columns, module.quant.weight_block_size[1])
            return [_Assignment(name, Ellipsis, "full", (slice(None), columns))]
        parent_path, _, leaf = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        if isinstance(module, Qwen3_5GatedDeltaNet) and parameter_name in {"A_log", "dt_bias"}:
            return [_Assignment(name, Ellipsis, "full", module.head_shard())]
        if leaf == "conv1d" and isinstance(parent, Qwen3_5GatedDeltaNet):
            # conv1d 通道与 in_proj_qkv 的 [q|k|v] 输出一一对应，按同样的三段切分。
            return [
                _Assignment(
                    name,
                    (parent.conv_local_rows(part), slice(None), slice(None)),
                    f"conv-{part}",
                    (parent.qkv_source_rows(part), slice(None), slice(None)),
                )
                for part in range(3)
            ]
        return [_Assignment(name, Ellipsis, "full")]

    suffix = "weight" + (_SCALE_SUFFIX if is_scale else "")
    parent_path, _, projection = module_path.rpartition(".")

    if projection in {"q_proj", "k_proj", "v_proj"}:
        target_name = f"{parent_path}.qkv_proj.{suffix}"
        if target_name not in parameters:
            return None
        attention = model.get_submodule(parent_path)
        part = {"q_proj": 0, "k_proj": 1, "v_proj": 2}[projection]
        source_rows = attention.qkv_proj.source_segment(part)
        return [
            _column_segment(
                attention.qkv_proj, target_name, part, source_rows, is_scale, projection
            )
        ]

    if projection in {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"}:
        mixer = model.get_submodule(parent_path)
        if not isinstance(mixer, Qwen3_5GatedDeltaNet):
            return None
        if projection == "in_proj_qkv":
            target_name = f"{parent_path}.in_proj_qkvz.{suffix}"
            return [
                _column_segment(
                    mixer.in_proj_qkvz,
                    target_name,
                    part,
                    mixer.qkv_source_rows(part),
                    is_scale,
                    f"qkv-{part}",
                )
                for part in range(3)
            ]
        if projection == "in_proj_z":
            target_name = f"{parent_path}.in_proj_qkvz.{suffix}"
            return [
                _column_segment(
                    mixer.in_proj_qkvz,
                    target_name,
                    3,
                    mixer.in_proj_qkvz.source_segment(3),
                    is_scale,
                    "z",
                )
            ]
        part = 0 if projection == "in_proj_b" else 1
        target_name = f"{parent_path}.in_proj_ba.{suffix}"
        return [
            _column_segment(
                mixer.in_proj_ba,
                target_name,
                part,
                mixer.in_proj_ba.source_segment(part),
                is_scale,
                projection,
            )
        ]

    expert = _EXPERT_WEIGHT.match(name)
    if expert:
        prefix, expert_index_text, projection, tensor_kind = expert.groups()
        expert_index = int(expert_index_text)
        experts = model.get_submodule(prefix)
        local_expert_index = experts.local_index(expert_index)
        if local_expert_index is None:
            # The tensor belongs to another EP rank. Recognize it without allocating
            # permanent model storage or treating it as an unknown checkpoint key.
            return [_Assignment(None, Ellipsis, f"remote-expert-{expert_index}-{projection}")]
        scale_suffix = _SCALE_SUFFIX if tensor_kind != "weight" else ""
        rows_per_part = experts.intermediate_size
        if scale_suffix:
            rows_per_part //= experts.quant.weight_block_size[0]
        if projection == "down_proj":
            target_name = f"{prefix}.down_proj{scale_suffix}"
            index = (local_expert_index, slice(None), slice(None))
        else:
            target_name = f"{prefix}.gate_up_proj{scale_suffix}"
            offset = 0 if projection == "gate_proj" else rows_per_part
            index = (
                local_expert_index,
                slice(offset, offset + rows_per_part),
                slice(None),
            )
        if target_name not in parameters:
            return None
        label = f"expert-{expert_index}-{projection}-{tensor_kind}"
        return [_Assignment(target_name, index, label)]

    if projection in {"gate_proj", "up_proj"} and ".experts." not in name:
        target_name = f"{parent_path}.gate_up_proj.{suffix}"
        if target_name not in parameters:
            return None
        mlp = model.get_submodule(parent_path)
        part = 0 if projection == "gate_proj" else 1
        return [
            _column_segment(
                mlp.gate_up_proj,
                target_name,
                part,
                mlp.gate_up_proj.source_segment(part),
                is_scale,
                projection,
            )
        ]

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
