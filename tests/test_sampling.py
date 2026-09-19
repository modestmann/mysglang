import unittest

import torch

from mysglang import Request, SamplingParams, Scheduler, SchedulerConfig
from mysglang.scheduler.sampler import sample_token
from tests.helpers import drain_scheduler, make_model


class SamplingTest(unittest.TestCase):
    def test_top_k_and_seed_are_reproducible(self) -> None:
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
        params = SamplingParams(max_new_tokens=1, temperature=0.8, top_k=2, seed=123)
        first = torch.Generator().manual_seed(123)
        second = torch.Generator().manual_seed(123)

        left = [sample_token(logits, params, generator=first) for _ in range(20)]
        right = [sample_token(logits, params, generator=second) for _ in range(20)]

        self.assertEqual(left, right)
        self.assertTrue(set(left) <= {2, 3})

    def test_top_p_keeps_the_token_that_crosses_threshold(self) -> None:
        params = SamplingParams(max_new_tokens=1, temperature=1.0, top_p=0.5, seed=7)
        generator = torch.Generator().manual_seed(7)
        logits = torch.log(torch.tensor([0.45, 0.35, 0.15, 0.05]))

        sampled = {sample_token(logits, params, generator=generator) for _ in range(100)}

        self.assertTrue(sampled <= {0, 1})
        self.assertEqual(sampled, {0, 1})

    @torch.inference_mode()
    def test_same_seed_is_independent_of_continuous_batching(self) -> None:
        scheduler = Scheduler(
            make_model(seed=975),
            SchedulerConfig(max_running_requests=2, num_pages=16, page_size=2),
        )
        params = SamplingParams(
            max_new_tokens=8,
            temperature=0.8,
            top_k=8,
            top_p=0.9,
            seed=2029,
        )
        scheduler.add(Request.from_token_ids("left", [1, 2, 3], params))
        scheduler.add(Request.from_token_ids("right", [1, 2, 3], params))

        outputs = drain_scheduler(scheduler)

        self.assertEqual(outputs["left"], outputs["right"])


if __name__ == "__main__":
    unittest.main()
