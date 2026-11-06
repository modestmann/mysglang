import importlib.util
import unittest
from unittest.mock import patch

import torch

from mysglang import (
    FlashInferBackend,
    ModelConfig,
    Qwen3ForCausalLM,
    Scheduler,
    SchedulerConfig,
    TorchAttentionBackend,
)
from mysglang.cache import PagedKVCache
from tests.helpers import drain_scheduler, make_request

FLASHINFER_AVAILABLE = importlib.util.find_spec("flashinfer") is not None
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None


class AttentionBackendTest(unittest.TestCase):
    @unittest.skipUnless(
        FLASHINFER_AVAILABLE and torch.cuda.is_available(),
        "requires CUDA and flashinfer",
    )
    @torch.inference_mode()
    def test_flashinfer_backend_matches_torch_with_paged_kv(self) -> None:
        config = ModelConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=512,
        )
        torch.manual_seed(2027)
        reference = (
            Qwen3ForCausalLM(
                config,
                attention_backend=TorchAttentionBackend(),
            )
            .cuda()
            .half()
        )
        flashinfer = (
            Qwen3ForCausalLM(
                config,
                attention_backend=FlashInferBackend(),
            )
            .cuda()
            .half()
        )
        flashinfer.load_state_dict(reference.state_dict())

        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            device="cuda",
        )
        # FlashInfer 只接收 paged KV；以 Torch 完整重算校验 dense cached prefill。
        initial_cache = self._make_cache(flashinfer, config)
        initial_ids = ("dense-a", "dense-b")
        for request_id in initial_ids:
            self.assertTrue(initial_cache.reserve_request(request_id, input_ids.size(1)))
            initial_cache.ensure_capacity(request_id, input_ids.size(1))
        torch.testing.assert_close(
            flashinfer(input_ids, kv_cache=initial_cache, cache_request_ids=initial_ids),
            reference(input_ids),
            atol=3e-3,
            rtol=3e-3,
        )
        for request_id in initial_ids:
            initial_cache.release_request(request_id)
        initial_cache.check_integrity()

        packed_reference_cache = self._make_cache(reference, config)
        packed_flash_cache = self._make_cache(flashinfer, config)
        packed_request_ids = ("packed-short", "packed-long")
        for cache in (packed_reference_cache, packed_flash_cache):
            for request_id, length in zip(packed_request_ids, (3, 5)):
                self.assertTrue(cache.reserve_request(request_id, 32))
                cache.ensure_capacity(request_id, length)
        packed_input = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device="cuda")
        expected = reference.forward_packed(
            packed_input,
            kv_cache=packed_reference_cache,
            cache_request_ids=packed_request_ids,
            append_lengths=(3, 5),
        )
        actual = flashinfer.forward_packed(
            packed_input,
            kv_cache=packed_flash_cache,
            cache_request_ids=packed_request_ids,
            append_lengths=(3, 5),
        )
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)

        for cache in (packed_reference_cache, packed_flash_cache):
            cache.ensure_capacity("packed-short", 5)
            cache.ensure_capacity("packed-long", 6)
        packed_suffix = torch.tensor([9, 10, 11], device="cuda")
        expected = reference.forward_packed(
            packed_suffix,
            kv_cache=packed_reference_cache,
            cache_request_ids=packed_request_ids,
            append_lengths=(2, 1),
        )
        actual = flashinfer.forward_packed(
            packed_suffix,
            kv_cache=packed_flash_cache,
            cache_request_ids=packed_request_ids,
            append_lengths=(2, 1),
            logits_indices=(1, 2),
        )
        torch.testing.assert_close(actual, expected[[1, 2]], atol=3e-3, rtol=3e-3)

        reference_cache = self._make_cache(reference, config)
        flash_cache = self._make_cache(flashinfer, config)
        prompts = {
            "short": torch.tensor([[1, 2, 3]], device="cuda"),
            "long": torch.tensor([[4, 5, 6, 7, 8]], device="cuda"),
        }
        prompt_logits: dict[str, torch.Tensor] = {}
        for request_id, prompt in prompts.items():
            for cache in (reference_cache, flash_cache):
                self.assertTrue(cache.reserve_request(request_id, 32))
                cache.ensure_capacity(request_id, prompt.size(1))
            expected = reference(
                prompt,
                kv_cache=reference_cache,
                cache_request_ids=(request_id,),
            )
            actual = flashinfer(
                prompt,
                kv_cache=flash_cache,
                cache_request_ids=(request_id,),
            )
            torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
            prompt_logits[request_id] = expected

        request_ids = tuple(prompts)
        next_tokens = torch.tensor(
            [[int(prompt_logits[request_id][:, -1].argmax().item())] for request_id in request_ids],
            device="cuda",
        )
        for cache in (reference_cache, flash_cache):
            for request_id in request_ids:
                current_length = cache.lengths((request_id,))[0]
                cache.ensure_capacity(request_id, current_length + 1)

        expected = reference(
            next_tokens,
            kv_cache=reference_cache,
            cache_request_ids=request_ids,
        )
        actual = flashinfer(
            next_tokens,
            kv_cache=flash_cache,
            cache_request_ids=request_ids,
        )
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
        reference_cache.check_integrity()
        flash_cache.check_integrity()

        scheduler_config = SchedulerConfig(
            max_running_requests=2,
            prefill_token_budget=128,
            num_pages=96,
            page_size=16,
        )
        reference_scheduler = Scheduler(reference, scheduler_config)
        flash_scheduler = Scheduler(flashinfer, scheduler_config)
        seed_prompt = [index % config.vocab_size for index in range(300)]
        expected_tokens = self._run_request(reference_scheduler, "seed", seed_prompt)
        actual_tokens = self._run_request(flash_scheduler, "seed", seed_prompt)
        self.assertEqual(actual_tokens, expected_tokens)

        shared_prompt = seed_prompt[:280] + [17] * 20
        prefill_before = flash_scheduler.prefill_input_tokens
        expected_tokens = self._run_request(reference_scheduler, "shared", shared_prompt)
        actual_tokens = self._run_request(flash_scheduler, "shared", shared_prompt)
        self.assertEqual(actual_tokens, expected_tokens)
        self.assertEqual(flash_scheduler.prefill_input_tokens - prefill_before, 28)

        ragged_config = SchedulerConfig(
            max_running_requests=3,
            prefill_token_budget=128,
            num_pages=96,
            page_size=16,
        )
        reference_ragged = Scheduler(reference, ragged_config)
        flash_ragged = Scheduler(flashinfer, ragged_config)
        ragged_prompts = {
            "ragged-short": [index % config.vocab_size for index in range(20)],
            "ragged-medium": [index % config.vocab_size for index in range(70)],
            "ragged-long": [index % config.vocab_size for index in range(150)],
        }
        for scheduler in (reference_ragged, flash_ragged):
            for request_id, prompt in ragged_prompts.items():
                scheduler.add(make_request(request_id, prompt, max_new_tokens=3))
        expected = drain_scheduler(reference_ragged)
        with patch.object(flashinfer, "forward_packed", wraps=flashinfer.forward_packed) as packed:
            actual = drain_scheduler(flash_ragged)
        self.assertTrue(
            any(
                1 in call.kwargs["append_lengths"] and max(call.kwargs["append_lengths"]) > 1
                for call in packed.call_args_list
            )
        )
        self.assertEqual(actual, expected)
        self.assertEqual(flash_ragged.stats.max_prefill_batch_size, 3)

        graph_base = dict(
            max_running_requests=2,
            prefill_token_budget=512,
            num_pages=64,
            page_size=16,
        )
        reference_graph = Scheduler(reference, SchedulerConfig(**graph_base))
        flash_graph = Scheduler(
            flashinfer,
            SchedulerConfig(
                **graph_base,
                decode_cuda_graph_batch_sizes=(2,),
            ),
        )
        graph_prompts = {
            "graph-a": [index % config.vocab_size for index in range(255)],
            "graph-b": [(index + 11) % config.vocab_size for index in range(250)],
        }
        for scheduler in (reference_graph, flash_graph):
            for request_id, prompt in graph_prompts.items():
                scheduler.add(make_request(request_id, prompt, max_new_tokens=3))
        expected = drain_scheduler(reference_graph)
        actual = drain_scheduler(flash_graph)
        self.assertEqual(actual, expected)
        self.assertEqual(flash_graph.stats.cuda_graph_captures, 1)
        self.assertEqual(flash_graph.stats.cuda_graph_replays, 2)

        # 同一个 bucket 换成全新的请求和长度，必须只改静态 buffer 的内容而不重捕获。
        replacement_prompts = {
            "graph-c": [31] * 20,
            "graph-d": [47] * 30,
        }
        for scheduler in (reference_graph, flash_graph):
            for request_id, prompt in replacement_prompts.items():
                scheduler.add(make_request(request_id, prompt, max_new_tokens=2))
        expected = drain_scheduler(reference_graph)
        actual = drain_scheduler(flash_graph)
        self.assertEqual(actual, expected)
        self.assertEqual(flash_graph.stats.cuda_graph_captures, 1)
        self.assertEqual(flash_graph.stats.cuda_graph_replays, 3)
        flash_graph.check_integrity()

        # 只录 batch 4：3 个请求补 1 个假请求、之后 2 个请求补 2 个，结果必须与 eager 一致，
        # 假请求只写垃圾页，真请求的页与长度不受影响。
        padded_base = dict(
            max_running_requests=4, prefill_token_budget=1024, num_pages=96, page_size=16
        )
        eager = Scheduler(reference, SchedulerConfig(**padded_base))
        padded = Scheduler(
            flashinfer, SchedulerConfig(**padded_base, decode_cuda_graph_batch_sizes=(4,))
        )
        self.assertEqual(padded.cache.scratch_page, 96)
        # 启动时就用全是假请求的 batch 录好，之后跑真请求不再捕获。
        self.assertEqual(padded.stats.cuda_graph_captures, 1)
        self.assertEqual(padded.stats.cuda_graph_replays, 0)
        padded_prompts = {
            "pad-a": ([5] * 40, 3),
            "pad-b": ([9] * 70, 6),
            "pad-c": ([13] * 25, 6),
        }
        for scheduler in (eager, padded):
            for request_id, (prompt, new_tokens) in padded_prompts.items():
                scheduler.add(make_request(request_id, prompt, max_new_tokens=new_tokens))
        expected = drain_scheduler(eager)
        actual = drain_scheduler(padded)
        self.assertEqual(actual, expected)
        self.assertEqual(padded.stats.cuda_graph_captures, 1)
        self.assertGreaterEqual(padded.stats.cuda_graph_replays, 4)
        padded.check_integrity()

    @unittest.skipUnless(
        FLASHINFER_AVAILABLE and TRITON_AVAILABLE and torch.cuda.is_available(),
        "requires CUDA, flashinfer and Triton",
    )
    @torch.inference_mode()
    def test_moe_decode_cuda_graph_matches_eager(self) -> None:
        config = ModelConfig(
            model_type="qwen3_moe",
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=512,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            norm_topk_prob=True,
        )
        torch.manual_seed(2030)
        eager_model = (
            Qwen3ForCausalLM(
                config,
                attention_backend=FlashInferBackend(),
                moe_dispatch_backend="triton_grouped",
            )
            .cuda()
            .half()
        )
        graph_model = (
            Qwen3ForCausalLM(
                config,
                attention_backend=FlashInferBackend(),
                moe_dispatch_backend="triton_grouped",
            )
            .cuda()
            .half()
        )
        graph_model.load_state_dict(eager_model.state_dict())
        scheduler_base = dict(
            max_running_requests=2,
            prefill_token_budget=512,
            num_pages=64,
            page_size=16,
        )
        eager = Scheduler(eager_model, SchedulerConfig(**scheduler_base))
        graph = Scheduler(
            graph_model,
            SchedulerConfig(
                **scheduler_base,
                decode_cuda_graph_batch_sizes=(2,),
            ),
        )
        prompts = {
            "moe-graph-left": list(range(1, 33)),
            "moe-graph-right": list(range(20, 49)),
        }
        for scheduler in (eager, graph):
            for request_id, prompt in prompts.items():
                scheduler.add(make_request(request_id, prompt, max_new_tokens=4))

        self.assertEqual(drain_scheduler(graph), drain_scheduler(eager))
        self.assertEqual(graph.stats.cuda_graph_captures, 1)
        self.assertEqual(graph.stats.cuda_graph_replays, 3)
        graph.check_integrity()

    @staticmethod
    def _make_cache(model: Qwen3ForCausalLM, config: ModelConfig) -> PagedKVCache:
        parameter = next(model.parameters())
        return PagedKVCache.from_config(
            config,
            num_pages=64,
            page_size=16,
            dtype=parameter.dtype,
            device=parameter.device,
        )

    @staticmethod
    def _run_request(
        scheduler: Scheduler,
        request_id: str,
        prompt_token_ids: list[int],
    ) -> list[int]:
        scheduler.add(make_request(request_id, prompt_token_ids, max_new_tokens=3))
        return drain_scheduler(scheduler)[request_id]


class FlashInferBackendTest(unittest.TestCase):
    """FlashInfer must reproduce the torch oracle at any page size, eager and under Graph."""

    def test_paged_kv_csr(self) -> None:
        from mysglang.modeling.attention import paged_kv_csr

        self.assertEqual(
            paged_kv_csr([[3, 0, 5, 8], [7], [2, 9]], [10, 4, 6], 4),
            ([0, 3, 4, 6], [3, 0, 5, 7, 2, 9], [2, 4, 2]),
        )

    @unittest.skipUnless(
        FLASHINFER_AVAILABLE and torch.cuda.is_available(), "requires CUDA and flashinfer"
    )
    def test_matches_torch_backend_at_small_page_sizes(self) -> None:
        self._check_against_torch(num_attention_heads=4, num_key_value_heads=2)

    @unittest.skipUnless(
        FLASHINFER_AVAILABLE and torch.cuda.is_available(), "requires CUDA and flashinfer"
    )
    def test_tensor_core_decode_for_unsupported_group_size(self) -> None:
        # 组大小 5（如 Qwen3-14B 的 40/8）不在非 TC decode kernel 的编译列表里，必须走 tensor core
        self._check_against_torch(num_attention_heads=5, num_key_value_heads=1)

    @torch.inference_mode()
    def _check_against_torch(self, num_attention_heads: int, num_key_value_heads: int) -> None:
        config = ModelConfig(
            vocab_size=64,
            hidden_size=256,
            intermediate_size=256,
            num_layers=2,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=64,
            max_position_embeddings=512,
        )
        torch.manual_seed(2028)
        reference = Qwen3ForCausalLM(config, attention_backend=TorchAttentionBackend())
        reference = reference.cuda().half().eval()
        shared = [index % 64 for index in range(37)]
        prompts = {
            "a": shared + [1, 2, 3],
            "b": shared + [4, 5],  # 与 a 共享 37 个 token：小页时能命中前缀
            "c": [(index * 7) % 64 for index in range(90)],
            "d": [9] * 5,
        }
        for page_size in (16, 1):
            for graph_sizes in ((), (1, 2, 4)):
                with self.subTest(page_size=page_size, graph=graph_sizes):
                    base = dict(
                        max_running_requests=4,
                        prefill_token_budget=24,  # 分块 prefill，并与 decode 拼成混合步
                        num_pages=9600 // page_size,
                        page_size=page_size,
                    )
                    # 每个子测试用全新的 backend：图在任何 eager forward 之前录制
                    # （与线上启动顺序一致）。之前共用 backend 时 eager 子测试先建好
                    # workspace，掩盖了 wrapper 被释放的 bug。
                    backend = FlashInferBackend()
                    flashinfer = Qwen3ForCausalLM(config, attention_backend=backend)
                    flashinfer.load_state_dict(reference.state_dict())
                    flashinfer = flashinfer.cuda().half().eval()
                    outputs = []
                    schedulers = (
                        Scheduler(reference, SchedulerConfig(**base)),
                        Scheduler(
                            flashinfer,
                            SchedulerConfig(**base, decode_cuda_graph_batch_sizes=graph_sizes),
                        ),
                    )
                    captured = dict(backend._graph_wrappers)
                    self.assertEqual(len(captured), len(graph_sizes))
                    for scheduler in schedulers:
                        for request_id, prompt in prompts.items():
                            scheduler.add(make_request(request_id, prompt, max_new_tokens=6))
                        outputs.append(drain_scheduler(scheduler))
                        scheduler.check_integrity()
                    self.assertEqual(outputs[1], outputs[0])
                    if graph_sizes:
                        self.assertGreater(schedulers[1].stats.cuda_graph_replays, 0)
                    # 图里固化了录制时 wrapper 缓冲区的地址：eager prefill 之后必须还是同一批对象
                    for key, wrapper in captured.items():
                        self.assertIs(backend._graph_wrappers[key], wrapper)
                    schedulers[1].close()
                    self.assertEqual(backend._graph_wrappers, {})


if __name__ == "__main__":
    unittest.main()
