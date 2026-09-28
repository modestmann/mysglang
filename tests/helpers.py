from __future__ import annotations

from collections.abc import Iterable

import torch

from mysglang import ModelConfig, Qwen3ForCausalLM, Request, SamplingParams


def make_model(
    *,
    seed: int = 2026,
    vocab_size: int = 64,
    max_position_embeddings: int = 64,
) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(
        ModelConfig(
            vocab_size=vocab_size,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max_position_embeddings,
        )
    ).eval()


def make_request(
    request_id: str,
    prompt_token_ids: Iterable[int],
    max_new_tokens: int = 4,
) -> Request:
    return Request.from_token_ids(
        request_id,
        prompt_token_ids,
        SamplingParams(max_new_tokens=max_new_tokens),
    )


@torch.inference_mode()
def reference_generate(
    model: Qwen3ForCausalLM,
    prompt_token_ids: Iterable[int],
    max_new_tokens: int,
) -> list[int]:
    """Small full-recomputation oracle kept outside the production package."""

    parameter = next(model.parameters())
    tokens = torch.tensor(
        [tuple(prompt_token_ids)],
        dtype=torch.long,
        device=parameter.device,
    )
    output: list[int] = []
    for _ in range(max_new_tokens):
        token_id = int(model(tokens)[:, -1].argmax(dim=-1).item())
        output.append(token_id)
        next_token = torch.tensor([[token_id]], dtype=torch.long, device=parameter.device)
        tokens = torch.cat((tokens, next_token), dim=1)
    return output


def drain_scheduler(scheduler) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    while scheduler.has_work:
        step = scheduler.step()
        if step is None:
            raise AssertionError("scheduler reported work but made no progress")
        for event in step.outputs:
            outputs.setdefault(event.request_id, []).append(event.token_id)
        scheduler.check_integrity()
    return outputs


def export_hf_state_dict(model) -> dict[str, torch.Tensor]:
    """Invert the loader's fusion for a TP=1 model: produce Hugging Face tensor names.

    FP8 layers export their e4m3 weights plus ``weight_scale_inv`` split along the same
    segment boundaries, exactly like an official block-FP8 checkpoint.
    """
    from mysglang.modeling.deltanet import Qwen3_5GatedDeltaNet
    from mysglang.modeling.parallel import ColumnParallelLinear

    if model.tensor_parallel.enabled:
        raise ValueError("export requires a TP=1 model")
    result: dict[str, torch.Tensor] = {}

    def hf_name(name: str) -> str:
        if name.startswith("mtp.") or name.startswith("lm_head."):
            return name
        return f"model.{name}"

    def split_segments(module: ColumnParallelLinear, names: tuple[str, ...], prefix: str) -> None:
        tensors = [("weight", module.weight)]
        if module.quant is not None:
            tensors.append(("weight_scale_inv", module.weight_scale_inv))
        for suffix, tensor in tensors:
            block = 1 if suffix == "weight" else module.quant.weight_block_size[0]
            offset = 0
            for index, target in enumerate(names):
                rows = module.local_output_sizes[index] // block
                if target is not None:
                    result[hf_name(f"{prefix}.{target}.{suffix}")] = tensor[
                        offset : offset + rows
                    ].detach().clone()
                offset += rows

    handled: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            qkvz = module.in_proj_qkvz
            weights = [("weight", qkvz.weight, 1)]
            if qkvz.quant is not None:
                block_n = qkvz.quant.weight_block_size[0]
                weights.append(("weight_scale_inv", qkvz.weight_scale_inv, block_n))
            for suffix, tensor, block in weights:
                boundary = module.conv_dim // block
                qkv, z = tensor[:boundary], tensor[boundary:]
                result[hf_name(f"{module_name}.in_proj_qkv.{suffix}")] = qkv.detach().clone()
                result[hf_name(f"{module_name}.in_proj_z.{suffix}")] = z.detach().clone()
            split_segments(module.in_proj_ba, ("in_proj_b", "in_proj_a"), module_name)
            handled.update({f"{module_name}.in_proj_qkvz", f"{module_name}.in_proj_ba"})
        elif isinstance(module, ColumnParallelLinear) and module_name not in handled:
            parent, _, leaf = module_name.rpartition(".")
            if leaf == "qkv_proj":
                split_segments(module, ("q_proj", "k_proj", "v_proj"), parent)
            elif leaf == "gate_up_proj":
                split_segments(module, ("gate_proj", "up_proj"), parent)
            handled.add(module_name)

    for module_name, module in model.named_modules():
        if getattr(module, "quant", None) is None or not hasattr(module, "source_experts"):
            continue
        # 官方 FP8 checkpoint 逐个专家存 gate/up/down 及其 scale。
        handled.add(module_name)
        block_n = module.quant.weight_block_size[0]
        for expert in range(module.num_local_experts):
            prefix = f"{module_name}.{expert}"
            for suffix, gate_up, down, rows in (
                ("weight", module.gate_up_proj, module.down_proj, module.intermediate_size),
                (
                    "weight_scale_inv",
                    module.gate_up_proj_scale_inv,
                    module.down_proj_scale_inv,
                    module.intermediate_size // block_n,
                ),
            ):
                gate, up = gate_up[expert, :rows], gate_up[expert, rows:]
                result[hf_name(f"{prefix}.gate_proj.{suffix}")] = gate.detach().clone()
                result[hf_name(f"{prefix}.up_proj.{suffix}")] = up.detach().clone()
                result[hf_name(f"{prefix}.down_proj.{suffix}")] = down[expert].detach().clone()

    fused_prefixes = tuple(f"{name}." for name in handled)
    for name, tensor in model.state_dict().items():
        if name.startswith(fused_prefixes):
            continue
        if model.config.tie_word_embeddings and name == "lm_head.weight":
            continue
        result[hf_name(name)] = tensor.detach().clone()
    return result
