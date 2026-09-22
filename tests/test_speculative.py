from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import torch

from mysglang import FlashAttentionBackend, ModelConfig, Qwen3ForCausalLM
from mysglang.scheduler import Scheduler, SchedulerConfig
from mysglang.scheduler.ngram import NGramProposer
from tests.helpers import drain_scheduler, make_model, make_request, reference_generate

FLASH_ATTN_AVAILABLE = importlib.util.find_spec("flash_attn") is not None


class NGramProposerTest(unittest.TestCase):
    def test_prefers_longest_suffix_and_copies_only_observed_tokens(self) -> None:
        proposer = NGramProposer(min_match=2, max_match=4)

        self.assertEqual(proposer.propose([1, 2, 3, 1, 2], 4), (3, 1, 2))
        self.assertEqual(proposer.propose([1, 2, 3, 1, 2], 1), (3,))
        self.assertEqual(proposer.propose([1, 2, 3, 4], 4), ())

    def test_uses_most_recent_occurrence_for_an_equal_length_match(self) -> None:
        proposer = NGramProposer(min_match=2, max_match=2)

        self.assertEqual(proposer.propose([1, 2, 7, 1, 2, 8, 1, 2], 2), (8, 1))


class SpeculativeSchedulerTest(unittest.TestCase):
    @torch.inference_mode()
    def test_accepts_draft_prefix_rejects_suffix_and_matches_greedy_oracle(self) -> None:
        model = make_model(seed=9876)
        prompt = [1, 2, 3, 4]
        expected = reference_generate(model, prompt, 6)
        scheduler = Scheduler(
            model,
            SchedulerConfig(
                num_pages=12,
                page_size=2,
                speculative_ngram_max_tokens=3,
            ),
        )
        request = make_request("speculative", prompt, 6)
        scheduler.add(request)
        first = scheduler.step()
        self.assertEqual(first.outputs[0].token_id, expected[0])

        wrong = (expected[2] + 1) % model.config.vocab_size
        with patch.object(
            scheduler._ngram_proposer,
            "propose",
            side_effect=[(expected[1], wrong), (), (), ()],
        ):
            verified = scheduler.step()
            remaining = drain_scheduler(scheduler)

        self.assertEqual([event.token_id for event in verified.outputs], expected[1:3])
        self.assertEqual(
            [first.outputs[0].token_id]
            + [event.token_id for event in verified.outputs]
            + remaining["speculative"],
            expected,
        )
        self.assertEqual(verified.input_tokens, 3)
        self.assertEqual(scheduler.stats.speculative_verify_forwards, 1)
        self.assertEqual(scheduler.stats.speculative_draft_tokens, 2)
        self.assertEqual(scheduler.stats.speculative_accepted_tokens, 1)
        scheduler.check_integrity()

    @torch.inference_mode()
    def test_all_accepted_drafts_emit_a_bonus_token(self) -> None:
        model = make_model(seed=5432)
        prompt = [5, 6, 7]
        expected = reference_generate(model, prompt, 5)
        scheduler = Scheduler(
            model,
            SchedulerConfig(
                num_pages=10,
                page_size=2,
                speculative_ngram_max_tokens=2,
            ),
        )
        request = make_request("bonus", prompt, 5)
        scheduler.add(request)
        scheduler.step()

        with patch.object(
            scheduler._ngram_proposer,
            "propose",
            side_effect=[tuple(expected[1:3]), ()],
        ):
            verified = scheduler.step()
            tail = drain_scheduler(scheduler)["bonus"]

        self.assertEqual([event.token_id for event in verified.outputs], expected[1:4])
        actual = [expected[0], *[event.token_id for event in verified.outputs], *tail]
        self.assertEqual(actual, expected)
        self.assertEqual(scheduler.stats.speculative_draft_tokens, 2)
        self.assertEqual(scheduler.stats.speculative_accepted_tokens, 2)

    def test_configuration_rejects_invalid_ngram_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            SchedulerConfig(
                speculative_ngram_min_match=4,
                speculative_ngram_max_match=3,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SchedulerConfig(speculative_ngram_max_tokens=-1)

    @unittest.skipUnless(
        FLASH_ATTN_AVAILABLE and torch.cuda.is_available(),
        "requires CUDA and flash-attn",
    )
    @torch.inference_mode()
    def test_flash_paged_verification_matches_one_token_decode(self) -> None:
        torch.manual_seed(8642)
        model = (
            Qwen3ForCausalLM(
                ModelConfig(
                    vocab_size=64,
                    hidden_size=128,
                    intermediate_size=256,
                    num_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=512,
                ),
                attention_backend=FlashAttentionBackend(),
            )
            .cuda()
            .half()
        )
        base = dict(num_pages=2, page_size=256)
        eager = Scheduler(model, SchedulerConfig(**base))
        speculative = Scheduler(
            model,
            SchedulerConfig(**base, speculative_ngram_max_tokens=2),
        )
        prompt = list(range(1, 17))
        eager.add(make_request("eager", prompt, 5))
        expected = drain_scheduler(eager)["eager"]
        speculative.add(make_request("speculative-fa", prompt, 5))
        first = speculative.step()

        with patch.object(
            speculative._ngram_proposer,
            "propose",
            return_value=tuple(expected[1:3]),
        ):
            actual = [first.outputs[0].token_id]
            actual.extend(
                token
                for events in drain_scheduler(speculative).values()
                for token in events
            )

        self.assertEqual(actual, expected)
        self.assertEqual(speculative.stats.speculative_accepted_tokens, 2)
        speculative.check_integrity()


if __name__ == "__main__":
    unittest.main()
