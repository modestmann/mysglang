import importlib.util
import unittest
from unittest.mock import patch

import torch

from mysglang import (
    FlashAttentionBackend,
    ModelConfig,
    Scheduler,
    SchedulerConfig,
    TinyCausalLM,
    TorchAttentionBackend,
)
from mysglang.cache import PagedKVCache
from tests.helpers import drain_scheduler, make_request

FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None


class AttentionBackendTest(unittest.TestCase):
    @unittest.skipUnless(FLASH_ATTN_AVAILABLE, "requires flash-attn")
    def test_flash_backend_rejects_incompatible_page_size_before_serving(self) -> None:
        model = TinyCausalLM(
            ModelConfig(),
            attention_backend=FlashAttentionBackend(),
        )

        with self.assertRaisesRegex(ValueError, "multiple of 256"):
            Scheduler(model, SchedulerConfig(page_size=4))

    @unittest.skipUnless(
        FLASH_ATTN_AVAILABLE and torch.cuda.is_available(),
        "requires CUDA and flash-attn",
    )
    @torch.inference_mode()
    def test_flash_backend_matches_torch_with_paged_kv(self) -> None:
        config = ModelConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        torch.manual_seed(2027)
        reference = (
            TinyCausalLM(
                config,
                attention_backend=TorchAttentionBackend(),
            )
            .cuda()
            .half()
        )
        flash = (
            TinyCausalLM(
                config,
                attention_backend=FlashAttentionBackend(),
            )
            .cuda()
            .half()
        )
        flash.load_state_dict(reference.state_dict())

        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            device="cuda",
        )
        torch.testing.assert_close(
            flash(input_ids),
            reference(input_ids),
            atol=3e-3,
            rtol=3e-3,
        )

        packed_reference_cache = self._make_cache(reference, config)
        packed_flash_cache = self._make_cache(flash, config)
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
        actual = flash.forward_packed(
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
        actual = flash.forward_packed(
            packed_suffix,
            kv_cache=packed_flash_cache,
            cache_request_ids=packed_request_ids,
            append_lengths=(2, 1),
        )
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)

        reference_cache = self._make_cache(reference, config)
        flash_cache = self._make_cache(flash, config)
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
            actual = flash(
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
        actual = flash(
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
            num_pages=6,
            page_size=256,
        )
        reference_scheduler = Scheduler(reference, scheduler_config)
        flash_scheduler = Scheduler(flash, scheduler_config)
        seed_prompt = [index % config.vocab_size for index in range(300)]
        expected_tokens = self._run_request(reference_scheduler, "seed", seed_prompt)
        actual_tokens = self._run_request(flash_scheduler, "seed", seed_prompt)
        self.assertEqual(actual_tokens, expected_tokens)

        shared_prompt = seed_prompt[:280] + [17] * 20
        prefill_before = flash_scheduler.prefill_input_tokens
        expected_tokens = self._run_request(reference_scheduler, "shared", shared_prompt)
        actual_tokens = self._run_request(flash_scheduler, "shared", shared_prompt)
        self.assertEqual(actual_tokens, expected_tokens)
        self.assertEqual(flash_scheduler.prefill_input_tokens - prefill_before, 44)

        ragged_config = SchedulerConfig(
            max_running_requests=3,
            prefill_token_budget=128,
            num_pages=6,
            page_size=256,
        )
        reference_ragged = Scheduler(reference, ragged_config)
        flash_ragged = Scheduler(flash, ragged_config)
        ragged_prompts = {
            "ragged-short": [index % config.vocab_size for index in range(20)],
            "ragged-medium": [index % config.vocab_size for index in range(70)],
            "ragged-long": [index % config.vocab_size for index in range(150)],
        }
        for scheduler in (reference_ragged, flash_ragged):
            for request_id, prompt in ragged_prompts.items():
                scheduler.add(make_request(request_id, prompt, max_new_tokens=3))
        expected = drain_scheduler(reference_ragged)
        with patch.object(flash, "forward_packed", wraps=flash.forward_packed) as packed:
            actual = drain_scheduler(flash_ragged)
        self.assertTrue(
            any(
                1 in call.kwargs["append_lengths"] and max(call.kwargs["append_lengths"]) > 1
                for call in packed.call_args_list
            )
        )
        self.assertEqual(actual, expected)
        self.assertEqual(flash_ragged.stats.max_prefill_batch_size, 3)

    @staticmethod
    def _make_cache(model: TinyCausalLM, config: ModelConfig) -> PagedKVCache:
        parameter = next(model.parameters())
        return PagedKVCache.from_config(
            config,
            num_pages=4,
            page_size=256,
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


if __name__ == "__main__":
    unittest.main()
