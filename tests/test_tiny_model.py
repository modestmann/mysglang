import unittest

import torch

from mysglang import ModelConfig, TinyCausalLM, greedy_generate


class TinyModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        config = ModelConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        self.model = TinyCausalLM(config).eval()

    def test_forward_shape_and_finite_logits(self) -> None:
        logits = self.model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(logits.shape, (2, 3, self.model.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_causal_prefix_is_unchanged_by_future_tokens(self) -> None:
        prefix = torch.tensor([[1, 2, 3]])
        extended = torch.tensor([[1, 2, 3, 4, 5]])
        torch.testing.assert_close(
            self.model(prefix), self.model(extended)[:, :3], atol=1e-5, rtol=1e-5
        )

    def test_greedy_generation_is_deterministic_and_bounded(self) -> None:
        prompt = torch.tensor([[1, 2, 3]])
        first = greedy_generate(self.model, prompt, max_new_tokens=4)
        second = greedy_generate(self.model, prompt, max_new_tokens=4)
        self.assertEqual(first.shape, (1, 7))
        self.assertTrue(torch.equal(first, second))

    def test_zero_new_tokens_returns_prompt(self) -> None:
        prompt = torch.tensor([[1, 2, 3]])
        self.assertTrue(torch.equal(greedy_generate(self.model, prompt, max_new_tokens=0), prompt))

    def test_invalid_gqa_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            ModelConfig(hidden_size=24, num_attention_heads=6, num_key_value_heads=4)


if __name__ == "__main__":
    unittest.main()
