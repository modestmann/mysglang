import unittest

import torch

from mysglang import (
    ContiguousKVCache,
    ModelConfig,
    TinyCausalLM,
    greedy_generate,
    greedy_generate_cached,
)


class ContiguousKVCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(456)
        self.config = ModelConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        self.model = TinyCausalLM(self.config).eval()

    def make_cache(self, *, max_length: int | None = None) -> ContiguousKVCache:
        parameter = next(self.model.parameters())
        return ContiguousKVCache.from_config(
            self.config,
            batch_size=1,
            dtype=parameter.dtype,
            device=parameter.device,
            max_length=max_length,
        )

    @torch.inference_mode()
    def test_cached_decode_logits_match_full_prefix_recomputation(self) -> None:
        full_ids = torch.tensor([[1, 5, 9]])
        cache = self.make_cache()
        cached_logits = self.model(full_ids, kv_cache=cache)
        torch.testing.assert_close(cached_logits, self.model(full_ids), atol=1e-5, rtol=1e-5)

        for _ in range(4):
            next_token = self.model(full_ids)[:, -1].argmax(dim=-1, keepdim=True)
            full_ids = torch.cat((full_ids, next_token), dim=1)
            cached_logits = self.model(next_token, kv_cache=cache)
            expected_logits = self.model(full_ids)[:, -1:]
            torch.testing.assert_close(cached_logits, expected_logits, atol=1e-5, rtol=1e-5)

        self.assertEqual(cache.length, full_ids.size(1))

    @torch.inference_mode()
    def test_cached_and_uncached_generation_produce_the_same_tokens(self) -> None:
        prompt = torch.tensor([[1, 5, 9]])
        expected = greedy_generate(self.model, prompt, max_new_tokens=5)
        actual = greedy_generate_cached(self.model, prompt, max_new_tokens=5)
        self.assertTrue(torch.equal(actual, expected))

    @torch.inference_mode()
    def test_cached_generation_projects_only_prompt_then_single_tokens(self) -> None:
        prompt = torch.tensor([[1, 5, 9]])
        projected_tokens = 0

        def count_projection_tokens(_module, args) -> None:
            nonlocal projected_tokens
            projected_tokens += args[0].shape[0] * args[0].shape[1]

        handle = self.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
            count_projection_tokens
        )
        try:
            greedy_generate_cached(self.model, prompt, max_new_tokens=5)
        finally:
            handle.remove()

        self.assertEqual(projected_tokens, 3 + 4)

    @torch.inference_mode()
    def test_cache_rejects_multi_token_decode_after_prefill(self) -> None:
        cache = self.make_cache()
        self.model(torch.tensor([[1, 2, 3]]), kv_cache=cache)
        with self.assertRaisesRegex(ValueError, "exactly one new token"):
            self.model(torch.tensor([[4, 5]]), kv_cache=cache)
        self.assertEqual(cache.length, 3)

    def test_cache_reset_clears_lengths(self) -> None:
        cache = self.make_cache()
        key = torch.randn(1, 2, 2, self.config.head_dim)
        value = torch.randn_like(key)
        for layer_idx in range(self.config.num_layers):
            cache.append(layer_idx, key, value)
        self.assertEqual(cache.length, 2)

        cache.reset()
        self.assertEqual(cache.length, 0)


if __name__ == "__main__":
    unittest.main()
