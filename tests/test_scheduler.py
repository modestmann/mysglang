import unittest

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM, greedy_generate_cached
from mysglang.core import FinishReason, RequestState
from mysglang.scheduler import ContinuousBatchScheduler, SchedulerConfig


def make_model() -> TinyCausalLM:
    torch.manual_seed(987)
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


class ContinuousBatchSchedulerTest(unittest.TestCase):
    @torch.inference_mode()
    def test_late_request_joins_variable_length_decode_batch(self) -> None:
        model = make_model()
        scheduler = ContinuousBatchScheduler(
            model,
            SchedulerConfig(max_running_requests=4, prefill_token_budget=16),
        )
        first_prompt = [1, 2, 3]
        second_prompt = [4, 5, 6, 7, 8]
        first = make_request("first", first_prompt, 6)
        second = make_request("second", second_prompt, 5)
        actual = {"first": [], "second": []}

        scheduler.add(first)
        first_step = scheduler.step()
        self.assertEqual(first_step.phase, "prefill")
        for event in first_step.outputs:
            actual[event.request_id].append(event.token_id)

        scheduler.add(second)
        while scheduler.has_work:
            step = scheduler.step()
            self.assertIsNotNone(step)
            for event in step.outputs:
                actual[event.request_id].append(event.token_id)

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
        self.assertTrue(
            any(step.phase == "decode" and step.batch_size == 2 for step in scheduler.history)
        )
        self.assertEqual(scheduler.stats.max_decode_batch_size, 2)
        self.assertEqual(scheduler.stats.active_requests, 0)

    @torch.inference_mode()
    def test_prefill_token_budget_chunks_a_long_prompt(self) -> None:
        scheduler = ContinuousBatchScheduler(
            make_model(),
            SchedulerConfig(max_running_requests=2, prefill_token_budget=2),
        )
        request = make_request("chunked", [1, 2, 3, 4, 5], 1)
        scheduler.add(request)

        first = scheduler.step()
        self.assertEqual(first.input_tokens, 2)
        self.assertEqual(request.state, RequestState.PREFILL)
        self.assertEqual(scheduler.cache.lengths((0,)), (2,))

        second = scheduler.step()
        self.assertEqual(second.input_tokens, 2)
        self.assertEqual(scheduler.cache.lengths((0,)), (4,))

        third = scheduler.step()
        self.assertEqual(third.input_tokens, 1)
        self.assertEqual(third.outputs[0].finish_reason, FinishReason.LENGTH)
        self.assertEqual(request.state, RequestState.FINISHED)
        self.assertEqual(scheduler.cache.lengths((0,)), (0,))

    @torch.inference_mode()
    def test_prefill_cannot_starve_an_existing_decode(self) -> None:
        scheduler = ContinuousBatchScheduler(
            make_model(),
            SchedulerConfig(
                max_running_requests=4,
                prefill_token_budget=16,
                max_consecutive_prefill_steps=1,
            ),
        )
        scheduler.add(make_request("first", [1, 2, 3], 8))
        scheduler.step()
        for index in range(3):
            scheduler.add(make_request(f"later-{index}", [4, 5, 6], 3))

        phases = [scheduler.step().phase for _ in range(5)]
        self.assertEqual(phases, ["decode", "prefill", "decode", "prefill", "decode"])

    @torch.inference_mode()
    def test_abort_releases_a_slot_for_the_oldest_waiter(self) -> None:
        scheduler = ContinuousBatchScheduler(
            make_model(),
            SchedulerConfig(max_running_requests=1, prefill_token_budget=16),
        )
        first = make_request("first", [1, 2, 3], 8)
        second = make_request("second", [4, 5, 6], 1)
        scheduler.add(first)
        scheduler.add(second)
        scheduler.step()

        aborted = scheduler.abort("first")
        self.assertEqual(aborted.finish_reason, FinishReason.ABORTED)
        self.assertEqual(first.state, RequestState.ABORTED)
        self.assertEqual(scheduler.cache.lengths((0,)), (0,))

        step = scheduler.step()
        self.assertEqual(step.request_ids, ("second",))
        self.assertEqual(second.state, RequestState.FINISHED)
        self.assertEqual(scheduler.stats.aborted_requests, 1)
        self.assertEqual(scheduler.stats.finished_requests, 1)


if __name__ == "__main__":
    unittest.main()
