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


class SpeculativeSamplingTest(unittest.TestCase):
    """Speculative sampling must reproduce the target distribution exactly."""

    def _first_token_frequencies(self, q: torch.Tensor | None, trials: int) -> torch.Tensor:
        from mysglang.scheduler.sampler import speculative_accept

        target = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25]]))
        params = SamplingParams(max_new_tokens=4, temperature=1.0, seed=3)
        generator = torch.Generator().manual_seed(3)
        counts = torch.zeros(4)
        for _ in range(trials):
            if q is None:
                draft = 3  # A deterministic (n-gram style) draft of the least likely token.
            else:
                draft = int(torch.multinomial(q, 1, generator=generator))
            output = speculative_accept(
                (draft,), (q,), target, params, (), generator=generator
            )
            counts[output[0]] += 1
        return counts / trials

    def test_deterministic_and_sampled_drafts_preserve_the_target_distribution(self) -> None:
        expected = torch.tensor([0.5, 0.3, 0.15, 0.05])
        for q in (None, torch.tensor([0.1, 0.2, 0.3, 0.4])):
            with self.subTest(sampled_draft=q is not None):
                frequencies = self._first_token_frequencies(q, 20000)
                torch.testing.assert_close(frequencies, expected, atol=0.015, rtol=0)

    @torch.inference_mode()
    def test_penalized_greedy_speculation_matches_plain_decoding(self) -> None:
        from tests.test_hybrid import make_hybrid

        model = make_hybrid(seed=61, mtp=True)
        params = SamplingParams(max_new_tokens=12, presence_penalty=1.5, frequency_penalty=0.3)
        outputs = []
        for steps in (0, 3):
            scheduler = Scheduler(
                model,
                SchedulerConfig(num_pages=32, page_size=4, speculative_mtp_steps=steps),
            )
            scheduler.add(Request.from_token_ids("a", [1, 2, 3, 4, 5], params))
            outputs.append(drain_scheduler(scheduler)["a"])
            if steps:
                self.assertGreater(scheduler.stats.speculative_verify_forwards, 0)
            scheduler.check_integrity()
        self.assertEqual(outputs[0], outputs[1])

    @torch.inference_mode()
    def test_sampled_mtp_speculation_is_seed_reproducible(self) -> None:
        from tests.test_hybrid import make_hybrid

        model = make_hybrid(seed=62, mtp=True)
        params = SamplingParams(
            max_new_tokens=10, temperature=0.9, top_k=12, top_p=0.9, seed=11
        )
        scheduler = Scheduler(
            model,
            SchedulerConfig(
                max_running_requests=2, num_pages=32, page_size=4, speculative_mtp_steps=2
            ),
        )
        scheduler.add(Request.from_token_ids("left", [1, 2, 3], params))
        scheduler.add(Request.from_token_ids("right", [1, 2, 3], params))
        outputs = drain_scheduler(scheduler)
        self.assertEqual(outputs["left"], outputs["right"])
        self.assertEqual(len(outputs["left"]), 10)
        self.assertGreater(scheduler.stats.speculative_draft_tokens, 0)
        scheduler.check_integrity()
