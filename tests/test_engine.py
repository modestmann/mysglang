import unittest
from unittest.mock import patch

import torch

from mysglang.core import FinishReason, RequestState
from mysglang.scheduler import Scheduler, SchedulerConfig
from tests.helpers import drain_scheduler, make_model, make_request, reference_generate


class SchedulerTest(unittest.TestCase):
    @torch.inference_mode()
    def test_batching_matches_reference_and_reuses_prefix(self) -> None:
        model = make_model(seed=808)
        scheduler = Scheduler(
            model,
            SchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=16,
                num_pages=20,
                page_size=2,
            ),
        )
        seed_prompt = [1, 2, 3, 4, 5, 6]
        seed = make_request("seed", seed_prompt, 2)
        seed_output = drain_scheduler_after_add(scheduler, seed)["seed"]
        self.assertEqual(seed_output, reference_generate(model, seed_prompt, 2))

        prefill_before = scheduler.prefill_input_tokens
        prompts = {
            "left": [1, 2, 3, 4, 7, 8],
            "right": [1, 2, 3, 4, 9, 10],
        }
        for request_id, prompt in prompts.items():
            scheduler.add(make_request(request_id, prompt, 4))
        outputs = drain_scheduler(scheduler)

        for request_id, prompt in prompts.items():
            self.assertEqual(outputs[request_id], reference_generate(model, prompt, 4))
        self.assertEqual(scheduler.prefill_input_tokens - prefill_before, 4)
        self.assertGreaterEqual(scheduler.cache.prefix_cache.stats.matched_tokens, 8)
        self.assertEqual(scheduler.stats.max_decode_batch_size, 2)
        self.assertEqual(scheduler.cache.allocator.stats.request_count, 0)

    @torch.inference_mode()
    def test_prefill_budget_chunks_a_long_prompt(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(prefill_token_budget=2, num_pages=8, page_size=2),
        )
        request = make_request("chunked", [1, 2, 3, 4, 5], 1)
        scheduler.add(request)

        first = scheduler.step()
        second = scheduler.step()
        third = scheduler.step()

        self.assertEqual((first.input_tokens, second.input_tokens, third.input_tokens), (2, 2, 1))
        self.assertEqual(third.outputs[0].finish_reason, FinishReason.LENGTH)
        self.assertEqual(request.state, RequestState.FINISHED)
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_prefill_cannot_starve_an_existing_decode(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=16,
                max_consecutive_prefill_steps=1,
                num_pages=24,
                page_size=2,
            ),
        )
        scheduler.add(make_request("first", [1, 2, 3], 8))
        scheduler.step()
        for index in range(3):
            scheduler.add(make_request(f"later-{index}", [4, 5, 6 + index], 3))

        phases = [scheduler.step().phase for _ in range(5)]

        self.assertEqual(phases, ["decode", "prefill", "decode", "decode", "decode"])
        self.assertEqual(scheduler.stats.max_prefill_batch_size, 3)

    @torch.inference_mode()
    def test_prefill_packs_different_chunk_lengths_into_one_forward(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(
                max_running_requests=3,
                prefill_token_budget=7,
                num_pages=16,
                page_size=2,
            ),
        )
        scheduler.add(make_request("short", [1, 2], 1))
        scheduler.add(make_request("medium", [3, 4, 5, 6], 1))
        scheduler.add(make_request("long", [7, 8, 9, 10, 11, 12, 13], 1))

        with patch.object(
            scheduler.model,
            "forward_packed",
            wraps=scheduler.model.forward_packed,
        ) as forward_packed:
            step = scheduler.step()

        self.assertEqual(step.request_ids, ("short", "medium", "long"))
        self.assertEqual(step.input_tokens, 7)
        self.assertEqual(forward_packed.call_count, 1)
        self.assertEqual(forward_packed.call_args.kwargs["append_lengths"], (2, 2, 3))
        self.assertEqual(scheduler.stats.max_prefill_batch_size, 3)
        scheduler.check_integrity()
        drain_scheduler(scheduler)

    @torch.inference_mode()
    def test_prefill_prefix_is_reusable_before_publisher_finishes(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(num_pages=20, page_size=2),
        )
        first = make_request("first", [1, 2, 3, 4, 5, 6], 8)
        scheduler.add(first)
        scheduler.step()
        self.assertEqual(first.state, RequestState.DECODING)

        second = make_request("second", [1, 2, 3, 4, 5, 6, 7, 8], 2)
        scheduler.add(second)
        self.assertEqual(scheduler.step().phase, "decode")
        second_prefill = scheduler.step()

        self.assertEqual(second_prefill.request_ids, ("second",))
        self.assertEqual(second_prefill.input_tokens, 2)
        self.assertEqual(first.state, RequestState.DECODING)
        drain_scheduler(scheduler)
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_admission_waits_and_impossible_request_is_atomic(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(
                max_running_requests=3,
                prefill_token_budget=16,
                num_pages=4,
                page_size=4,
            ),
        )
        impossible = make_request("impossible", [1, 2, 3, 4, 5], 12)
        with self.assertRaises(ValueError):
            scheduler.add(impossible)
        self.assertEqual(scheduler.stats.active_requests, 0)
        self.assertEqual(scheduler.cache.allocator.stats.reserved_pages, 0)

        first = make_request("first", [1, 2, 3], 5)
        second = make_request("second", [4, 5, 6], 5)
        third = make_request("third", [7, 8, 9], 2)
        scheduler.add(first)
        scheduler.add(second)
        scheduler.add(third)
        scheduler.step()
        scheduler.step()
        scheduler.step()

        self.assertEqual(third.state, RequestState.WAITING)
        self.assertEqual(scheduler.cache.allocator.stats.reserved_pages, 4)
        drain_scheduler(scheduler)
        self.assertEqual(third.state, RequestState.FINISHED)
        self.assertEqual(scheduler.cache.allocator.stats.request_count, 0)

    @torch.inference_mode()
    def test_abort_unlocks_prefix_and_reset_returns_every_page(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(num_pages=10, page_size=2),
        )
        drain_scheduler_after_add(
            scheduler,
            make_request("seed", [1, 2, 3, 4, 5, 6], 2),
        )
        scheduler.add(make_request("abort-me", [1, 2, 3, 4, 9, 10], 5))
        scheduler.step()
        self.assertGreater(scheduler.cache.prefix_cache.stats.protected_pages, 0)

        event = scheduler.abort("abort-me")

        self.assertEqual(event.finish_reason, FinishReason.ABORTED)
        self.assertEqual(scheduler.cache.prefix_cache.stats.protected_pages, 0)
        self.assertEqual(scheduler.cache.allocator.stats.request_count, 0)
        scheduler.reset_prefix_cache()
        stats = scheduler.cache.allocator.stats
        self.assertEqual(stats.cached_pages, 0)
        self.assertEqual(stats.free_pages, stats.total_pages)
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_reset_rejects_active_request_with_root_handle(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(num_pages=8, page_size=4),
        )
        scheduler.add(make_request("short", [1, 2], 3))
        scheduler.step()
        self.assertEqual(scheduler.cache.prefix_cache.stats.protected_pages, 0)

        with self.assertRaisesRegex(RuntimeError, "requests are active"):
            scheduler.reset_prefix_cache()

        drain_scheduler(scheduler)
        scheduler.reset_prefix_cache()
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_oldest_prefix_is_evicted_under_physical_memory_pressure(self) -> None:
        scheduler = Scheduler(
            make_model(),
            SchedulerConfig(
                max_running_requests=2,
                prefill_token_budget=16,
                num_pages=5,
                page_size=2,
            ),
        )
        drain_scheduler_after_add(scheduler, make_request("oldest", [1, 2, 3, 4], 1))
        drain_scheduler_after_add(scheduler, make_request("newer", [5, 6, 7, 8], 1))
        drain_scheduler_after_add(scheduler, make_request("trigger", [9, 10, 11, 12], 1))

        prefix = scheduler.cache.prefix_cache
        self.assertGreaterEqual(prefix.stats.evicted_pages, 2)
        self.assertEqual(prefix.match_prefix([1, 2, 3, 4], record=False).cached_len, 0)
        self.assertEqual(prefix.match_prefix([5, 6, 7, 8], record=False).cached_len, 4)
        scheduler.check_integrity()


def drain_scheduler_after_add(scheduler: Scheduler, request) -> dict[str, list[int]]:
    scheduler.add(request)
    return drain_scheduler(scheduler)


if __name__ == "__main__":
    unittest.main()
