"""Qwen3.5 hybrid (Gated DeltaNet + gated attention) model, cache and scheduler tests."""

from __future__ import annotations

import importlib.util
import os
import random
import tempfile
import unittest
from unittest.mock import patch

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
from mysglang.cache import HybridPagedKVCache
from mysglang.config import Fp8QuantConfig
from mysglang.modeling.deltanet import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from mysglang.scheduler import Scheduler
from tests.helpers import drain_scheduler, export_hf_state_dict, make_request, reference_generate

try:
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf_qwen3_5
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as hf_qwen3_5_moe
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
        Qwen3_5TextConfig,
    )
except ImportError:
    hf_qwen3_5 = None

LAYER_TYPES = ("linear_attention", "linear_attention", "full_attention", "linear_attention")


def hybrid_config(
    *,
    moe: bool = False,
    mtp: bool = False,
    num_key_value_heads: int = 2,
    quantization: Fp8QuantConfig | None = None,
    head_dim: int = 16,
    **overrides,
) -> ModelConfig:
    moe_fields = (
        dict(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=24,
            norm_topk_prob=True,
        )
        if moe
        else {}
    )
    fields = dict(
        model_type="qwen3_5_moe" if moe else "qwen3_5",
        vocab_size=48,
        hidden_size=32,
        intermediate_size=16 if moe else 48,
        num_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=160,
        partial_rotary_factor=0.25,
        attn_output_gate=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=LAYER_TYPES,
        mtp_num_hidden_layers=1 if mtp else 0,
        quantization=quantization,
        **moe_fields,
    )
    fields.update(overrides)
    return ModelConfig(**fields)


def make_hybrid(seed: int = 11, **kwargs) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(hybrid_config(**kwargs), enable_mtp=kwargs.get("mtp", False)).eval()
    with torch.no_grad():
        # Zero-centered norms start at zero; perturb them so (1 + w) is exercised.
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") and not parameter.abs().sum():
                parameter.add_(torch.randn_like(parameter) * 0.1)
    return model


class _TorchFallbackKernels:
    """HF resolves fla kernels at import time; force its PyTorch path for CPU oracles."""

    names = (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )

    def __enter__(self):
        self._patches = [
            patch.object(module, name, getattr(getattr(module, name), "__wrapped__", None))
            for module in (hf_qwen3_5, hf_qwen3_5_moe)
            for name in self.names
            if getattr(getattr(module, name), "__wrapped__", None) is not None
        ]
        for current in self._patches:
            current.start()
        return self

    def __exit__(self, *exc):
        for current in self._patches:
            current.stop()


class DeltaRuleKernelTest(unittest.TestCase):
    def test_chunked_algorithm_matches_the_per_token_recurrence(self) -> None:
        torch.manual_seed(1)
        length, heads, key_dim, value_dim = 150, 3, 8, 6
        query = torch.nn.functional.normalize(torch.randn(length, heads, key_dim), dim=-1)
        key = torch.nn.functional.normalize(torch.randn(length, heads, key_dim), dim=-1)
        value = torch.randn(length, heads, value_dim)
        decay = -torch.rand(length, heads)
        beta = torch.rand(length, heads)
        state = torch.randn(heads, key_dim, value_dim)

        chunked, chunked_state = torch_chunk_gated_delta_rule(
            query, key, value, decay, beta, state, chunk_size=16
        )
        recurrent, recurrent_state = torch_recurrent_gated_delta_rule(
            query[None], key[None], value[None], decay[None], beta[None], state[None]
        )
        torch.testing.assert_close(chunked, recurrent[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(chunked_state, recurrent_state[0], atol=1e-4, rtol=1e-4)


@unittest.skipUnless(hf_qwen3_5 is not None, "requires transformers with Qwen3.5")
class HuggingFaceHybridOracleTest(unittest.TestCase):
    def _reference(self, moe: bool):
        torch.manual_seed(7)
        common = dict(
            vocab_size=48,
            hidden_size=32,
            num_hidden_layers=len(LAYER_TYPES),
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=160,
            rms_norm_eps=1e-6,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
            linear_conv_kernel_dim=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=list(LAYER_TYPES),
            tie_word_embeddings=False,
            use_cache=False,
        )
        if moe:
            hf_config = Qwen3_5MoeTextConfig(
                **common,
                moe_intermediate_size=16,
                shared_expert_intermediate_size=24,
                num_experts=4,
                num_experts_per_tok=2,
            )
            reference = Qwen3_5MoeForCausalLM(hf_config)
        else:
            hf_config = Qwen3_5TextConfig(**common, intermediate_size=48)
            reference = Qwen3_5ForCausalLM(hf_config)
        hf_config._attn_implementation = "eager"
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.add_(torch.randn_like(parameter) * 0.1)
        model = Qwen3ForCausalLM(hybrid_config(moe=moe)).eval()
        report = load_huggingface_state_dict(model, reference.state_dict())
        self.assertEqual(report.ignored_tensors, ())
        return reference.eval(), model

    @torch.inference_mode()
    def test_dense_and_moe_logits_match_hugging_face(self) -> None:
        for moe in (False, True):
            with self.subTest(moe=moe), _TorchFallbackKernels():
                reference, model = self._reference(moe)
                # 150 tokens span three DeltaNet chunks, including a padded final chunk.
                input_ids = torch.randint(0, 48, (2, 150))
                expected = reference(input_ids, use_cache=False).logits.float()
                torch.testing.assert_close(model(input_ids), expected, atol=2e-5, rtol=2e-5)


class HybridCacheTest(unittest.TestCase):
    @torch.inference_mode()
    def test_chunked_prefill_and_decode_match_full_recomputation(self) -> None:
        model = make_hybrid(moe=True)
        cache = HybridPagedKVCache(
            recurrent=model.recurrent_state_spec,
            num_recurrent_slots=2,
            num_layers=model.num_kv_cache_layers,
            num_pages=16,
            page_size=4,
            num_kv_heads=model.kv_cache_num_heads,
            head_dim=model.config.head_dim,
            dtype=torch.float32,
            device="cpu",
        )
        self.assertFalse(cache.enable_prefix_cache)
        tokens = torch.randint(0, 48, (1, 30))
        expected = model(tokens)
        self.assertEqual(cache.reserve_request_with_prefix("a", 40, tokens[0].tolist()), 0)
        pieces = []
        for start, end in ((0, 7), (7, 20), (20, 21), (21, 30)):
            cache.ensure_capacity("a", end)
            pieces.append(
                model.forward_packed(
                    tokens[0, start:end],
                    kv_cache=cache,
                    cache_request_ids=("a",),
                    append_lengths=(end - start,),
                )
            )
        torch.testing.assert_close(torch.cat(pieces), expected[0], atol=1e-5, rtol=1e-5)
        cache.check_integrity()
        cache.release_request("a")
        self.assertEqual(cache.recurrent.stats.free_slots, 2)
        cache.check_integrity()


class HybridSchedulerTest(unittest.TestCase):
    def _run(self, model, *, prompts, lengths, **config) -> tuple[Scheduler, dict]:
        expected = {
            f"r{index}": reference_generate(model, prompt, length)
            for index, (prompt, length) in enumerate(zip(prompts, lengths))
        }
        scheduler = Scheduler(
            model,
            SchedulerConfig(
                max_running_requests=3,
                prefill_token_budget=16,
                num_pages=64,
                page_size=4,
                **config,
            ),
        )
        for index, (prompt, length) in enumerate(zip(prompts, lengths)):
            scheduler.add(make_request(f"r{index}", prompt, length))
        self.assertEqual(drain_scheduler(scheduler), expected)
        scheduler.check_integrity()
        return scheduler, expected

    @staticmethod
    def _prompts(seed: int = 5) -> list[list[int]]:
        rng = random.Random(seed)
        return [[rng.randrange(48) for _ in range(length)] for length in (5, 23, 40, 9)]

    @torch.inference_mode()
    def test_continuous_batching_matches_greedy_oracle(self) -> None:
        for moe in (False, True):
            with self.subTest(moe=moe):
                scheduler, _ = self._run(
                    make_hybrid(moe=moe), prompts=self._prompts(), lengths=[12, 7, 15, 20]
                )
                self.assertEqual(scheduler.stats.finished_requests, 4)

    @torch.inference_mode()
    def test_abort_releases_the_recurrent_slot(self) -> None:
        model = make_hybrid()
        scheduler = Scheduler(model, SchedulerConfig(max_running_requests=1, num_pages=16))
        scheduler.add(make_request("a", [1, 2, 3, 4, 5], 10))
        scheduler.step()
        scheduler.abort("a")
        self.assertEqual(scheduler.cache.recurrent.stats.free_slots, 1)
        scheduler.check_integrity()
        expected = reference_generate(model, [6, 7, 8], 4)
        scheduler.add(make_request("b", [6, 7, 8], 4))
        self.assertEqual(drain_scheduler(scheduler)["b"], expected)

    @torch.inference_mode()
    def test_ngram_verification_replays_only_the_accepted_prefix(self) -> None:
        model = make_hybrid(seed=21)
        prompts = self._prompts(seed=8)[:2]
        expected = {f"r{i}": reference_generate(model, p, 9) for i, p in enumerate(prompts)}

        def propose(history, max_tokens):
            index = next(
                i for i, prompt in enumerate(prompts) if history[: len(prompt)] == tuple(prompt)
            )
            generated = len(history) - len(prompts[index])
            oracle = list(expected[f"r{index}"][generated : generated + max_tokens])
            # Accept one draft, then force a rejection so the DeltaNet replay must stop early.
            if len(oracle) >= 2 and (generated // 2) % 2 == 0:
                oracle[1] = (oracle[1] + 1) % 48
            return tuple(oracle)

        scheduler = Scheduler(
            model,
            SchedulerConfig(num_pages=32, page_size=4, speculative_ngram_max_tokens=3),
        )
        scheduler._ngram_proposer.propose = propose
        for index, prompt in enumerate(prompts):
            scheduler.add(make_request(f"r{index}", prompt, 9))
        self.assertEqual(drain_scheduler(scheduler), expected)
        stats = scheduler.stats
        self.assertGreater(stats.speculative_accepted_tokens, 0)
        self.assertLess(stats.speculative_accepted_tokens, stats.speculative_draft_tokens)

    @torch.inference_mode()
    def test_mtp_speculation_matches_greedy_oracle(self) -> None:
        for moe in (False, True):
            for steps in (1, 3):
                with self.subTest(moe=moe, steps=steps):
                    scheduler, _ = self._run(
                        make_hybrid(moe=moe, mtp=True),
                        prompts=self._prompts(),
                        lengths=[12, 7, 15, 20],
                        speculative_mtp_steps=steps,
                    )
                    stats = scheduler.stats
                    self.assertGreater(stats.speculative_verify_forwards, 0)
                    self.assertGreater(stats.speculative_draft_forwards, 0)

    @torch.inference_mode()
    def test_prefix_cache_restores_deltanet_state_snapshots(self) -> None:
        rng = random.Random(31)
        shared = [rng.randrange(48) for _ in range(13)]  # 13 个 token：快照边界 12（3 页）
        prompts = {
            "first": shared + [5, 6, 7],
            "second": shared + [9, 1],
            "third": shared[:9] + [40, 41, 42, 43, 44],  # 只共享 2 页：要在中间切节点
        }
        for moe, steps in ((False, 0), (True, 0), (False, 2)):
            with self.subTest(moe=moe, mtp_steps=steps):
                model = make_hybrid(moe=moe, mtp=bool(steps))
                expected = {key: reference_generate(model, p, 6) for key, p in prompts.items()}
                scheduler = Scheduler(
                    model,
                    SchedulerConfig(
                        max_running_requests=2,
                        prefill_token_budget=16,
                        num_pages=40,
                        page_size=4,
                        speculative_mtp_steps=steps,
                        state_snapshots=4,
                    ),
                )
                outputs = {}
                for key, prompt in prompts.items():
                    scheduler.add(make_request(key, prompt, 6))
                    outputs.update(drain_scheduler(scheduler))
                self.assertEqual(outputs, expected)
                # first 读全部 16 个，在 12 处存快照；second（15 个）命中这 12 个、只读 3 个；
                # third 只和 first 共享 9 个 token：radix 在 8 处切出的节点没有快照，命中 0，
                # 读全部 14 个，并在自己的边界 12 处再存一份快照。
                self.assertEqual(scheduler.prefill_input_tokens, 16 + 3 + 14)
                self.assertEqual(scheduler.cache.recurrent.free_snapshots, 4 - 2)
                scheduler.check_integrity()

    @torch.inference_mode()
    def test_state_snapshot_pool_full_and_eviction_keep_outputs_exact(self) -> None:
        rng = random.Random(32)
        prompts = {f"r{i}": [rng.randrange(48) for _ in range(9)] for i in range(4)}
        prompts["again"] = list(prompts["r0"])
        model = make_hybrid()
        expected = {key: reference_generate(model, p, 5) for key, p in prompts.items()}
        scheduler = Scheduler(
            model,
            SchedulerConfig(
                max_running_requests=1,
                prefill_token_budget=16,
                num_pages=8,  # 页很少：缓存的前缀会被 LRU 淘汰，快照要跟着还回去
                page_size=4,
                state_snapshots=2,  # 快照池比前缀少：满了要丢最久没用的快照
            ),
        )
        outputs = {}
        for key, prompt in prompts.items():
            scheduler.add(make_request(key, prompt, 5))
            outputs.update(drain_scheduler(scheduler))
        self.assertEqual(outputs, expected)
        self.assertGreater(scheduler.cache.prefix_cache.stats.evicted_pages, 0)
        scheduler.check_integrity()
        scheduler.cache.reset_prefix_cache()
        self.assertEqual(scheduler.cache.recurrent.free_snapshots, 2)

    @torch.inference_mode()
    def test_preemption_recomputes_state_slots_and_mtp_draft_kv(self) -> None:
        prompts, lengths = self._prompts(), [12, 7, 15, 20]
        for steps in (0, 1, 3):
            with self.subTest(mtp_steps=steps):
                model = make_hybrid(mtp=bool(steps))
                expected = {
                    f"r{index}": reference_generate(model, prompt, length)
                    for index, (prompt, length) in enumerate(zip(prompts, lengths))
                }
                scheduler = Scheduler(
                    model,
                    SchedulerConfig(
                        max_running_requests=3,
                        prefill_token_budget=16,
                        # 三个请求最坏情况要约 30 页；只给 18 页，逼出抢占。
                        num_pages=18,
                        page_size=4,
                        speculative_mtp_steps=steps,
                        enable_preemption=True,
                    ),
                )
                for index, (prompt, length) in enumerate(zip(prompts, lengths)):
                    scheduler.add(make_request(f"r{index}", prompt, length))
                self.assertEqual(drain_scheduler(scheduler), expected)
                self.assertGreater(scheduler.stats.preempted_requests, 0)
                self.assertEqual(scheduler.cache.recurrent.stats.free_slots, 3)
                scheduler.check_integrity()

    @torch.inference_mode()
    def test_mtp_accepted_drafts_keep_state_and_draft_cache_aligned(self) -> None:
        model = make_hybrid(seed=33, mtp=True)
        prompts = self._prompts(seed=9)[:2]
        expected = {f"r{i}": reference_generate(model, p, 10) for i, p in enumerate(prompts)}
        scheduler = Scheduler(
            model,
            SchedulerConfig(num_pages=32, page_size=4, speculative_mtp_steps=3),
        )
        original = scheduler._mtp_drafts

        def oracle_drafts(entries):
            # Run the real MTP draft forwards, then replace their tokens with a mostly
            # correct oracle chain so accepted and rejected suffixes both occur.
            drafts, probs = original(entries)
            result = []
            for entry, draft in zip(entries, drafts):
                request = entry.request
                generated = request.num_output_tokens
                oracle = list(expected[request.request_id][generated : generated + len(draft)])
                if len(oracle) == 3 and generated % 3 == 1:
                    oracle[1] = (oracle[1] + 1) % 48
                result.append(tuple(oracle))
            # Greedy drafts are deterministic, so their q stays None.
            return tuple(result), probs

        scheduler._mtp_drafts = oracle_drafts
        for index, prompt in enumerate(prompts):
            scheduler.add(make_request(f"r{index}", prompt, 10))
        self.assertEqual(drain_scheduler(scheduler), expected)
        stats = scheduler.stats
        self.assertGreater(stats.speculative_accepted_tokens, stats.speculative_draft_tokens // 2)
        self.assertLess(stats.speculative_accepted_tokens, stats.speculative_draft_tokens)

    def test_mtp_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "either MTP or n-gram"):
            SchedulerConfig(speculative_mtp_steps=2, speculative_ngram_max_tokens=2)
        with self.assertRaisesRegex(ValueError, "enable_mtp"):
            Scheduler(make_hybrid(), SchedulerConfig(speculative_mtp_steps=1))

    @torch.inference_mode()
    def test_multiple_stop_tokens_finish_generation(self) -> None:
        model = make_hybrid()
        expected = reference_generate(model, [1, 2, 3], 8)
        stop = expected[3]
        scheduler = Scheduler(model, SchedulerConfig(num_pages=16))
        request = Request.from_token_ids(
            "a",
            [1, 2, 3],
            SamplingParams(max_new_tokens=8, eos_token_id=47, stop_token_ids=(stop,)),
        )
        scheduler.add(request)
        drain_scheduler(scheduler)
        first_stop = expected.index(stop)
        self.assertEqual(request.output_token_ids, tuple(expected[: first_stop + 1]))


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None,
    "requires CUDA and flash-attn",
)
class HybridCudaGraphTest(unittest.TestCase):
    """Graph replay must reproduce eager Decode, MTP drafts, verify and state commits."""

    @torch.inference_mode()
    def test_graph_paths_match_eager_generation(self) -> None:
        from mysglang import FlashAttentionBackend, FlashInferBackend

        cases = [
            ("flash", 256, 32, moe, steps) for moe, steps in ((False, 0), (False, 3), (True, 2))
        ]
        if importlib.util.find_spec("flashinfer") is not None:
            # FlashInfer：小页（16）+ 它支持的 head_dim 64；verify / 草稿 / 追赶都走它的 kernel。
            cases += [("flashinfer", 16, 64, False, 0), ("flashinfer", 16, 64, False, 3)]
        for backend_name, page_size, head_dim, moe, steps in cases:
            with self.subTest(backend=backend_name, moe=moe, steps=steps):
                torch.manual_seed(50)
                config = hybrid_config(
                    moe=moe,
                    mtp=True,
                    head_dim=head_dim,
                    linear_key_head_dim=16,
                    linear_value_head_dim=16,
                    max_position_embeddings=512,
                    vocab_size=128,
                )
                model = Qwen3ForCausalLM(
                    config,
                    attention_backend=(
                        FlashInferBackend()
                        if backend_name == "flashinfer"
                        else FlashAttentionBackend()
                    ),
                    moe_dispatch_backend="triton_grouped",
                    enable_mtp=True,
                )
                model = model.to(device="cuda", dtype=torch.bfloat16).eval()
                prompts = [list(range(3, 40)), list(range(10, 17)), list(range(20, 61))]
                outputs = []
                # ()：eager；(1, 2, 3, 4)：精确大小；(1, 4)：batch 2、3 向上补齐到 4。
                for sizes in ((), (1, 2, 3, 4), (1, 4)):
                    graph = bool(sizes)
                    scheduler = Scheduler(
                        model,
                        SchedulerConfig(
                            max_running_requests=4,
                            prefill_token_budget=256,
                            num_pages=2048 // page_size,
                            page_size=page_size,
                            speculative_mtp_steps=steps,
                            decode_cuda_graph_batch_sizes=sizes,
                        ),
                    )
                    for index, prompt in enumerate(prompts):
                        scheduler.add(make_request(f"r{index}", prompt, 12 + 5 * index))
                    # A sampled request exercises graph logits + speculative sampling.
                    scheduler.add(
                        Request.from_token_ids(
                            "sampled",
                            list(range(5, 30)),
                            SamplingParams(
                                max_new_tokens=15, temperature=0.8, top_p=0.95, seed=9
                            ),
                        )
                    )
                    outputs.append(drain_scheduler(scheduler))
                    if graph:
                        self.assertGreater(scheduler.stats.cuda_graph_replays, 0)
                    scheduler.close()
                self.assertEqual(outputs[0], outputs[1])
                self.assertEqual(outputs[0], outputs[2])


def _hybrid_tp_worker(
    rank: int,
    world_size: int,
    init_method: str,
    config: ModelConfig,
    state_dict: dict[str, torch.Tensor],
    prompts: dict[str, tuple[int, ...]],
    expected: dict[str, tuple[int, ...]],
    mtp_steps: int,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        model = Qwen3ForCausalLM(
            config,
            tensor_parallel=TensorParallelContext.from_distributed(),
            moe_dispatch_backend="sorted",
            enable_mtp=bool(mtp_steps),
        ).eval()
        load_huggingface_state_dict(model, state_dict)
        scheduler = TensorParallelScheduler(
            model,
            SchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=5,
                num_pages=32,
                page_size=2,
                speculative_mtp_steps=mtp_steps,
                state_snapshots=4,  # 各 rank 在同样的边界存快照、按同样顺序分配编号
            ),
        )
        if rank != 0:
            scheduler.run_worker_loop()
            scheduler.check_integrity()
            return
        for request_id, prompt in prompts.items():
            scheduler.add(Request(request_id, prompt, SamplingParams(len(expected[request_id]))))
        # Sampled drafts and acceptance run on every rank; rank 0's outcome is broadcast.
        sampled = SamplingParams(7, temperature=0.9, top_p=0.95, seed=4, presence_penalty=0.5)
        scheduler.add(Request("sampled", (11, 12, 13, 14), sampled))
        actual: dict[str, list[int]] = {}
        while scheduler.has_work:
            step = scheduler.step()
            assert step is not None
            for event in step.outputs:
                actual.setdefault(event.request_id, []).append(event.token_id)
            scheduler.check_integrity()
        assert len(actual.pop("sampled")) == 7
        assert {key: tuple(value) for key, value in actual.items()} == expected, actual
        if mtp_steps:
            assert scheduler.stats.speculative_verify_forwards > 0
        scheduler.shutdown()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(),
    "requires torch.distributed with Gloo",
)
class HybridTensorParallelTest(unittest.TestCase):
    def _spawn(self, world_size: int, mtp_steps: int, **config_kwargs) -> None:
        reference = make_hybrid(
            seed=40 + world_size, moe=True, mtp=bool(mtp_steps), **config_kwargs
        )
        state_dict = export_hf_state_dict(reference)
        prompts = {"left": (1, 2, 3, 4, 5, 6, 7), "right": (8, 9, 10)}
        with torch.inference_mode():
            expected = {
                "left": tuple(reference_generate(reference, prompts["left"], 6)),
                "right": tuple(reference_generate(reference, prompts["right"], 4)),
            }
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _hybrid_tp_worker,
                args=(
                    world_size,
                    f"file://{directory}/process-group",
                    reference.config,
                    state_dict,
                    prompts,
                    expected,
                    mtp_steps,
                ),
                nprocs=world_size,
                join=True,
            )

    def test_tp2_hybrid_moe_with_replicated_kv_head_and_mtp(self) -> None:
        # One KV head is replicated on both ranks; DeltaNet heads and experts are split.
        self._spawn(2, 2, num_key_value_heads=1)

    def test_tp4_hybrid_moe_with_mtp(self) -> None:
        # Each rank owns one query head, one DeltaNet key head and one expert.
        self._spawn(4, 1, linear_num_key_heads=4, linear_num_value_heads=8)

    def test_tp2_fp8_hybrid_moe(self) -> None:
        self._spawn(
            2,
            1,
            head_dim=8,
            # Like official checkpoints: per-head DeltaNet b/a projections stay unquantized.
            quantization=Fp8QuantConfig(
                weight_block_size=(4, 4),
                modules_to_not_convert=("mtp.fc", "in_proj_a", "in_proj_b"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
