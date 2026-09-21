from __future__ import annotations

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mysglang import (
    ModelConfig,
    Qwen3ForCausalLM,
    TensorParallelContext,
    load_huggingface_state_dict,
)


def _hugging_face_layout(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    """Expand this project's fused dense weights into the Qwen3 checkpoint layout."""

    result: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".self_attn.qkv_proj.weight"):
            module_name = name.removesuffix(".qkv_proj.weight")
            attention = model.get_submodule(module_name)
            query, key, value = tensor.split(
                (attention.global_q_size, attention.global_kv_size, attention.global_kv_size),
                dim=0,
            )
            result[f"model.{module_name}.q_proj.weight"] = query.clone()
            result[f"model.{module_name}.k_proj.weight"] = key.clone()
            result[f"model.{module_name}.v_proj.weight"] = value.clone()
        elif name.endswith(".mlp.gate_up_proj.weight"):
            module_name = name.removesuffix(".gate_up_proj.weight")
            gate, up = tensor.chunk(2, dim=0)
            result[f"model.{module_name}.gate_proj.weight"] = gate.clone()
            result[f"model.{module_name}.up_proj.weight"] = up.clone()
        elif name.startswith("layers.") or name in {"embed_tokens.weight", "norm.weight"}:
            result[f"model.{name}"] = tensor.clone()
        else:
            result[name] = tensor.clone()
    return result


def _legacy_moe_layout(
    state_dict: dict[str, torch.Tensor],
    config: ModelConfig,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for expert_idx in range(config.num_experts):
                result[f"{prefix}.{expert_idx}.gate_proj.weight"] = tensor[
                    expert_idx, : config.moe_intermediate_size
                ]
                result[f"{prefix}.{expert_idx}.up_proj.weight"] = tensor[
                    expert_idx, config.moe_intermediate_size :
                ]
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for expert_idx in range(config.num_experts):
                result[f"{prefix}.{expert_idx}.down_proj.weight"] = tensor[expert_idx]
        else:
            result[name] = tensor
    return result


def _dense_tp_worker(
    rank: int,
    world_size: int,
    init_method: str,
    config: ModelConfig,
    state_dict: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        tensor_parallel = TensorParallelContext.from_distributed()
        model = Qwen3ForCausalLM(config, tensor_parallel=tensor_parallel).eval()
        load_huggingface_state_dict(model, state_dict)
        with torch.inference_mode():
            actual = model(input_ids)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        attention = model.layers[0].self_attn
        assert attention.num_heads == config.num_attention_heads // world_size
        assert attention.num_kv_heads == config.num_key_value_heads // world_size
        assert attention.qkv_proj.weight.size(0) == (
            config.num_attention_heads + 2 * config.num_key_value_heads
        ) * config.head_dim // world_size
        assert model.layers[0].mlp.intermediate_size == config.intermediate_size // world_size
    finally:
        dist.destroy_process_group()


def _moe_tp_worker(
    rank: int,
    world_size: int,
    init_method: str,
    config: ModelConfig,
    state_dict: dict[str, torch.Tensor],
    legacy_state_dict: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        parallel = TensorParallelContext.from_distributed()
        model = Qwen3ForCausalLM(config, tensor_parallel=parallel).eval()
        load_huggingface_state_dict(model, state_dict)
        with torch.inference_mode():
            actual = model(input_ids)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        experts = model.layers[0].mlp.experts
        assert experts.num_local_experts == config.num_experts // world_size
        assert experts.local_expert_start == rank * experts.num_local_experts
        assert experts.gate_up_proj.size(0) == experts.num_local_experts

        legacy_model = Qwen3ForCausalLM(config, tensor_parallel=parallel).eval()
        load_huggingface_state_dict(legacy_model, legacy_state_dict)
        with torch.inference_mode():
            legacy_actual = legacy_model(input_ids)
        torch.testing.assert_close(legacy_actual, expected, atol=1e-5, rtol=1e-5)

        for backend in ("grouped", "all_to_all"):
            optimized_model = Qwen3ForCausalLM(
                config,
                tensor_parallel=parallel,
                moe_dispatch_backend=backend,
            ).eval()
            load_huggingface_state_dict(optimized_model, state_dict)
            with torch.inference_mode():
                optimized_actual = optimized_model(input_ids)
            torch.testing.assert_close(optimized_actual, expected, atol=1e-5, rtol=1e-5)
    finally:
        dist.destroy_process_group()


class TensorParallelTest(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "requires torch.distributed with Gloo",
    )
    def test_dense_tp2_logits_match_tp1(self) -> None:
        torch.manual_seed(2026)
        config = ModelConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        reference = Qwen3ForCausalLM(config).eval()
        state_dict = _hugging_face_layout(reference)
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        with torch.inference_mode():
            expected = reference(input_ids)

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _dense_tp_worker,
                args=(
                    2,
                    f"file://{directory}/process-group",
                    config,
                    state_dict,
                    input_ids,
                    expected,
                ),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "requires torch.distributed with Gloo",
    )
    def test_moe_tp2_expert_sharding_matches_tp1(self) -> None:
        torch.manual_seed(2027)
        config = ModelConfig(
            model_type="qwen3_moe",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            norm_topk_prob=True,
        )
        reference = Qwen3ForCausalLM(config).eval()
        state_dict = _hugging_face_layout(reference)
        legacy_state_dict = _legacy_moe_layout(state_dict, config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with torch.inference_mode():
            expected = reference(input_ids)

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _moe_tp_worker,
                args=(
                    2,
                    f"file://{directory}/process-group",
                    config,
                    state_dict,
                    legacy_state_dict,
                    input_ids,
                    expected,
                ),
                nprocs=2,
                join=True,
            )

    def test_tp_context_rejects_invalid_rank(self) -> None:
        context = TensorParallelContext()
        self.assertEqual(context.local_size(4, "heads"), 4)
        with self.assertRaisesRegex(ValueError, "rank must be within"):
            TensorParallelContext(rank=1, world_size=1)


if __name__ == "__main__":
    unittest.main()
