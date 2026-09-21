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
    Request,
    SamplingParams,
    SchedulerConfig,
    TensorParallelContext,
    TensorParallelScheduler,
    load_huggingface_state_dict,
)


def _hugging_face_layout(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
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


@torch.inference_mode()
def _reference_generate(
    model: Qwen3ForCausalLM,
    prompt: tuple[int, ...],
    max_new_tokens: int,
) -> tuple[int, ...]:
    tokens = torch.tensor([prompt], dtype=torch.long)
    output = []
    for _ in range(max_new_tokens):
        token_id = int(model(tokens)[:, -1].argmax(dim=-1).item())
        output.append(token_id)
        tokens = torch.cat((tokens, torch.tensor([[token_id]], dtype=torch.long)), dim=1)
    return tuple(output)


def _distributed_scheduler_worker(
    rank: int,
    world_size: int,
    init_method: str,
    config: ModelConfig,
    state_dict: dict[str, torch.Tensor],
    expected: dict[str, tuple[int, ...]],
    moe_dispatch_backend: str,
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
        model = Qwen3ForCausalLM(
            config,
            tensor_parallel=tensor_parallel,
            moe_dispatch_backend=moe_dispatch_backend,
        ).eval()
        load_huggingface_state_dict(model, state_dict)
        scheduler = TensorParallelScheduler(
            model,
            SchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=3,
                num_pages=16,
                page_size=2,
            ),
        )

        if rank != 0:
            scheduler.run_worker_loop()
            scheduler.check_integrity()
            return

        requests = (
            Request("left", (1, 2, 3, 4, 5), SamplingParams(max_new_tokens=3)),
            Request("right", (6, 7), SamplingParams(max_new_tokens=2)),
        )
        for request in requests:
            scheduler.add(request)

        actual: dict[str, list[int]] = {}
        phases = []
        while scheduler.has_work:
            step = scheduler.step()
            assert step is not None
            phases.append(step.phase)
            for event in step.outputs:
                assert event.token_id is not None
                actual.setdefault(event.request_id, []).append(event.token_id)
            scheduler.check_integrity()

        assert "prefill" in phases
        assert "mixed" in phases
        assert "decode" in phases
        assert {key: tuple(value) for key, value in actual.items()} == expected
        assert scheduler.stats.finished_requests == 2
        scheduler.shutdown()
        scheduler.check_integrity()
    finally:
        dist.destroy_process_group()


class TensorParallelSchedulerTest(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "requires torch.distributed with Gloo",
    )
    def test_tp2_continuous_batching_matches_tp1(self) -> None:
        torch.manual_seed(321)
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
        expected = {
            "left": _reference_generate(reference, (1, 2, 3, 4, 5), 3),
            "right": _reference_generate(reference, (6, 7), 2),
        }

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_scheduler_worker,
                args=(
                    2,
                    f"file://{directory}/process-group",
                    config,
                    state_dict,
                    expected,
                    "sorted",
                ),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "requires torch.distributed with Gloo",
    )
    def test_moe_tp2_continuous_batching_matches_tp1(self) -> None:
        torch.manual_seed(322)
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
        expected = {
            "left": _reference_generate(reference, (1, 2, 3, 4, 5), 3),
            "right": _reference_generate(reference, (6, 7), 2),
        }

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_scheduler_worker,
                args=(
                    2,
                    f"file://{directory}/process-group",
                    config,
                    state_dict,
                    expected,
                    "all_to_all",
                ),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
