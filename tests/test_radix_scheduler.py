import unittest

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM
from mysglang.core import FinishReason
from mysglang.generation import greedy_generate_cached
from mysglang.scheduler import RadixBatchScheduler, RadixSchedulerConfig


def make_model() -> TinyCausalLM:
    torch.manual_seed(808)
    return TinyCausalLM(
        ModelConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval()


def make_request(request_id: str, prompt: list[int], max_new_tokens: int = 4) -> Request:
    return Request.from_token_ids(
        request_id,
        prompt,
        SamplingParams(max_new_tokens=max_new_tokens),
    )


def run_request(scheduler: RadixBatchScheduler, request: Request) -> list[int]:
    output: list[int] = []
    scheduler.add(request)
    while scheduler.has_work:
        step = scheduler.step()
        output.extend(event.token_id for event in step.outputs)
        scheduler.check_integrity()
    return output


class RadixBatchSchedulerTest(unittest.TestCase):
    @torch.inference_mode()
    def test_shared_prefix_reduces_prefill_without_changing_tokens(self) -> None:
        model = make_model()
        scheduler = RadixBatchScheduler(
            model,
            RadixSchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=16,
                num_pages=16,
                page_size=2,
            ),
        )
        first_prompt = [1, 2, 3, 4, 5, 6]
        second_prompt = [1, 2, 3, 4, 9, 10]

        first = run_request(scheduler, make_request("first", first_prompt))
        first_prefill = scheduler.prefill_input_tokens
        second = run_request(scheduler, make_request("second", second_prompt))
        second_prefill = scheduler.prefill_input_tokens - first_prefill

        expected_first = greedy_generate_cached(
            model, torch.tensor([first_prompt]), max_new_tokens=4
        )[0, len(first_prompt) :].tolist()
        expected_second = greedy_generate_cached(
            model, torch.tensor([second_prompt]), max_new_tokens=4
        )[0, len(second_prompt) :].tolist()
        self.assertEqual(first, expected_first)
        self.assertEqual(second, expected_second)
        self.assertEqual(first_prefill, 6)
        self.assertEqual(second_prefill, 2)
        self.assertGreaterEqual(scheduler.cache.prefix_cache.stats.matched_tokens, 4)

    @torch.inference_mode()
    def test_two_active_requests_share_and_protect_the_same_pages(self) -> None:
        scheduler = RadixBatchScheduler(
            make_model(),
            RadixSchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=16,
                num_pages=12,
                page_size=2,
            ),
        )
        run_request(scheduler, make_request("seed", [1, 2, 3, 4, 5, 6], 2))
        scheduler.add(make_request("left", [1, 2, 3, 4, 7, 8], 3))
        scheduler.add(make_request("right", [1, 2, 3, 4, 9, 10], 3))

        scheduler.step()
        scheduler.step()
        scheduler.step()
        self.assertGreaterEqual(scheduler.cache.prefix_cache.stats.protected_pages, 2)
        scheduler.check_integrity()

        while scheduler.has_work:
            scheduler.step()
            scheduler.check_integrity()

    @torch.inference_mode()
    def test_reset_returns_cached_pages_to_the_pool(self) -> None:
        scheduler = RadixBatchScheduler(
            make_model(),
            RadixSchedulerConfig(num_pages=8, page_size=2),
        )
        run_request(scheduler, make_request("one", [1, 2, 3, 4], 2))
        self.assertGreater(scheduler.cache.allocator.stats.cached_pages, 0)

        scheduler.reset_prefix_cache()
        stats = scheduler.cache.allocator.stats
        self.assertEqual(stats.cached_pages, 0)
        self.assertEqual(stats.free_pages, stats.total_pages)
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_lru_prefix_is_evicted_when_the_physical_pool_is_full(self) -> None:
        scheduler = RadixBatchScheduler(
            make_model(),
            RadixSchedulerConfig(
                max_running_requests=2,
                prefill_token_budget=16,
                num_pages=5,
                page_size=2,
            ),
        )
        run_request(scheduler, make_request("oldest", [1, 2, 3, 4], 1))
        run_request(scheduler, make_request("newer", [5, 6, 7, 8], 1))
        run_request(scheduler, make_request("trigger", [9, 10, 11, 12], 1))

        prefix = scheduler.cache.prefix_cache
        self.assertGreaterEqual(prefix.stats.evicted_pages, 2)
        self.assertEqual(prefix.match_prefix([1, 2, 3, 4], record=False).cached_len, 0)
        self.assertEqual(prefix.match_prefix([5, 6, 7, 8], record=False).cached_len, 4)
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_abort_unlocks_matched_prefix_and_releases_private_pages(self) -> None:
        scheduler = RadixBatchScheduler(
            make_model(),
            RadixSchedulerConfig(num_pages=10, page_size=2),
        )
        run_request(scheduler, make_request("seed", [1, 2, 3, 4, 5, 6], 2))
        scheduler.add(make_request("abort-me", [1, 2, 3, 4, 9, 10], 5))
        scheduler.step()
        self.assertGreater(scheduler.cache.prefix_cache.stats.protected_pages, 0)

        event = scheduler.abort("abort-me")
        self.assertEqual(event.finish_reason, FinishReason.ABORTED)
        self.assertEqual(scheduler.cache.prefix_cache.stats.protected_pages, 0)
        self.assertEqual(scheduler.cache.allocator.stats.request_count, 0)
        scheduler.check_integrity()


if __name__ == "__main__":
    unittest.main()
