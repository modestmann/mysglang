import unittest

import torch

from mysglang import ModelConfig, TinyCausalLM, greedy_generate
from mysglang.modeling.tiny import RMSNorm, RotaryEmbedding


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

    def test_rms_norm_matches_its_reference_equation(self) -> None:
        norm = RMSNorm(hidden_size=3, eps=1e-6)
        norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5]))
        inputs = torch.tensor([[[1.0, -2.0, 3.0]]])
        expected = norm.weight * inputs * torch.rsqrt(inputs.square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(norm(inputs), expected)

    def test_rope_preserves_vector_norm(self) -> None:
        rope = RotaryEmbedding(head_dim=6, max_positions=8, theta=10_000.0)
        query = torch.randn(1, 4, 3, 6)
        key = torch.randn(1, 2, 3, 6)
        positions = torch.arange(3)
        rotated_query, rotated_key = rope(query, key, positions)
        torch.testing.assert_close(
            rotated_query.square().sum(-1), query.square().sum(-1), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            rotated_key.square().sum(-1), key.square().sum(-1), atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
