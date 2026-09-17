import unittest

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM, greedy_generate_cached
from mysglang.core import FinishReason, RequestState
from mysglang.scheduler import PagedBatchScheduler, PagedSchedulerConfig


def make_model() -> TinyCausalLM:
    torch.manual_seed(1357)
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


def make_request(request_id: str, prompt: list[int], max_new_tokens: int) -> Request:
    return Request.from_token_ids(
        request_id,
        prompt,
        SamplingParams(max_new_tokens=max_new_tokens),
    )


class PagedBatchSchedulerTest(unittest.TestCase):
    @torch.inference_mode()
    def test_tokens_match_isolated_generation_and_all_pages_return(self) -> None:
        model = make_model()
        scheduler = PagedBatchScheduler(
            model,
            PagedSchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=16,
                num_pages=8,
                page_size=4,
            ),
        )
        first_prompt = [1, 2, 3]
        second_prompt = [4, 5, 6, 7, 8]
        first = make_request("first", first_prompt, 6)
        second = make_request("second", second_prompt, 5)
        actual = {"first": [], "second": []}

        scheduler.add(first)
        first_step = scheduler.step()
        for event in first_step.outputs:
            actual[event.request_id].append(event.token_id)
        scheduler.add(second)

        while scheduler.has_work:
            step = scheduler.step()
            for event in step.outputs:
                actual[event.request_id].append(event.token_id)
            scheduler.check_integrity()

        expected_first = greedy_generate_cached(
            model,
            torch.tensor([first_prompt]),
            max_new_tokens=6,
        )[0, len(first_prompt) :].tolist()
        expected_second = greedy_generate_cached(
            model,
            torch.tensor([second_prompt]),
            max_new_tokens=5,
        )[0, len(second_prompt) :].tolist()
        self.assertEqual(actual["first"], expected_first)
        self.assertEqual(actual["second"], expected_second)
        self.assertEqual(scheduler.stats.max_decode_batch_size, 2)
        memory = scheduler.cache.allocator.stats
        self.assertEqual(memory.free_pages, memory.total_pages)
        self.assertEqual(memory.allocated_pages, 0)
        self.assertEqual(memory.reserved_pages, 0)

    @torch.inference_mode()
    def test_request_waits_until_reserved_capacity_is_released(self) -> None:
        scheduler = PagedBatchScheduler(
            make_model(),
            PagedSchedulerConfig(
                max_running_requests=3,
                prefill_token_budget=16,
                num_pages=4,
                page_size=4,
            ),
        )
        first = make_request("first", [1, 2, 3], 5)
        second = make_request("second", [4, 5, 6], 5)
        third = make_request("third", [7, 8, 9], 2)
        scheduler.add(first)
        scheduler.add(second)
        scheduler.add(third)

        scheduler.step()  # first prefill
        scheduler.step()  # protect first decode
        scheduler.step()  # second prefill; all four pages are now reserved
        self.assertEqual(third.state, RequestState.WAITING)
        self.assertEqual(scheduler.cache.allocator.stats.reserved_pages, 4)

        while scheduler.has_work:
            step = scheduler.step()
            self.assertIsNotNone(step)
            scheduler.check_integrity()

        self.assertEqual(third.state, RequestState.FINISHED)
        self.assertEqual(scheduler.cache.allocator.stats.free_pages, 4)

    def test_impossible_request_is_rejected_before_entering_scheduler(self) -> None:
        scheduler = PagedBatchScheduler(
            make_model(),
            PagedSchedulerConfig(
                max_running_requests=2,
                num_pages=2,
                page_size=4,
            ),
        )
        request = make_request("too-large", [1, 2, 3, 4, 5], 4)

        with self.assertRaisesRegex(ValueError, "total paged KV cache capacity"):
            scheduler.add(request)
        self.assertEqual(request.state, RequestState.WAITING)
        self.assertEqual(scheduler.stats.active_requests, 0)
        self.assertEqual(scheduler.cache.allocator.stats.reserved_pages, 0)

    @torch.inference_mode()
    def test_abort_returns_allocated_and_reserved_pages(self) -> None:
        scheduler = PagedBatchScheduler(
            make_model(),
            PagedSchedulerConfig(
                max_running_requests=2,
                num_pages=4,
                page_size=4,
            ),
        )
        request = make_request("abort-me", [1, 2, 3], 5)
        scheduler.add(request)
        scheduler.step()
        self.assertGreater(scheduler.cache.allocator.stats.allocated_pages, 0)

        event = scheduler.abort("abort-me")
        self.assertEqual(event.finish_reason, FinishReason.ABORTED)
        stats = scheduler.cache.allocator.stats
        self.assertEqual(stats.allocated_pages, 0)
        self.assertEqual(stats.reserved_pages, 0)
        self.assertEqual(stats.free_pages, stats.total_pages)
        scheduler.check_integrity()


if __name__ == "__main__":
    unittest.main()
