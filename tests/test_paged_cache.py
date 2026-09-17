import random
import unittest

import torch

from mysglang import ModelConfig, PageAllocator, PagedKVCache, TinyCausalLM


class PageAllocatorTest(unittest.TestCase):
    def test_failed_reservation_is_atomic(self) -> None:
        allocator = PageAllocator(num_pages=4, page_size=4)
        self.assertTrue(allocator.reserve_request("first", max_tokens=12))
        before = allocator.stats

        self.assertFalse(allocator.reserve_request("second", max_tokens=8))
        self.assertEqual(allocator.stats, before)
        self.assertEqual(allocator.request_ids, frozenset({"first"}))
        allocator.check_integrity()

    def test_random_allocate_grow_and_release_preserves_invariants(self) -> None:
        rng = random.Random(1234)
        allocator = PageAllocator(num_pages=12, page_size=4)
        max_tokens: dict[str, int] = {}
        next_id = 0

        for _ in range(500):
            operation = rng.choice(("reserve", "grow", "release"))
            if operation == "reserve":
                request_id = f"req-{next_id}"
                requested = rng.randint(1, 12)
                if allocator.reserve_request(request_id, requested):
                    max_tokens[request_id] = requested
                    next_id += 1
            elif operation == "grow" and max_tokens:
                request_id = rng.choice(list(max_tokens))
                allocator.ensure_capacity(
                    request_id,
                    rng.randint(0, max_tokens[request_id]),
                )
            elif operation == "release" and max_tokens:
                request_id = rng.choice(list(max_tokens))
                allocator.release(request_id)
                max_tokens.pop(request_id)
            allocator.check_integrity()

        for request_id in list(max_tokens):
            allocator.release(request_id)
        allocator.check_integrity()
        self.assertEqual(allocator.stats.free_pages, allocator.stats.total_pages)


class PagedKVCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2468)
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
        parameter = next(self.model.parameters())
        self.cache = PagedKVCache.from_config(
            self.config,
            num_pages=8,
            page_size=2,
            dtype=parameter.dtype,
            device=parameter.device,
        )

    @torch.inference_mode()
    def test_variable_length_requests_decode_in_one_batch(self) -> None:
        first = torch.tensor([[1, 2, 3]])
        second = torch.tensor([[4, 5, 6, 7, 8]])
        self.assertTrue(self.cache.reserve_request("first", 8))
        self.assertTrue(self.cache.reserve_request("second", 8))
        self.cache.ensure_capacity("first", first.size(1))
        self.cache.ensure_capacity("second", second.size(1))
        first_logits = self.model(
            first,
            kv_cache=self.cache,
            cache_request_ids=("first",),
        )
        second_logits = self.model(
            second,
            kv_cache=self.cache,
            cache_request_ids=("second",),
        )
        first_token = first_logits[:, -1].argmax(dim=-1, keepdim=True)
        second_token = second_logits[:, -1].argmax(dim=-1, keepdim=True)

        self.cache.ensure_capacity("first", 4)
        self.cache.ensure_capacity("second", 6)
        actual = self.model(
            torch.cat((first_token, second_token), dim=0),
            kv_cache=self.cache,
            cache_request_ids=("first", "second"),
        )
        expected_first = self.model(torch.cat((first, first_token), dim=1))[:, -1]
        expected_second = self.model(torch.cat((second, second_token), dim=1))[:, -1]

        torch.testing.assert_close(actual[0, -1], expected_first[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual[1, -1], expected_second[0], atol=1e-5, rtol=1e-5)
        self.cache.check_integrity()

    @torch.inference_mode()
    def test_chunked_prefill_crosses_page_boundaries(self) -> None:
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        self.assertTrue(self.cache.reserve_request("chunked", 8))
        self.cache.ensure_capacity("chunked", 2)
        self.model(
            prompt[:, :2],
            kv_cache=self.cache,
            cache_request_ids=("chunked",),
        )
        self.cache.ensure_capacity("chunked", 5)
        actual = self.model(
            prompt[:, 2:],
            kv_cache=self.cache,
            cache_request_ids=("chunked",),
        )
        expected = self.model(prompt)[:, 2:]

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(self.cache.lengths(("chunked",)), (5,))
        self.assertEqual(len(self.cache.allocator.page_table("chunked")), 3)

    @torch.inference_mode()
    def test_release_and_reuse_does_not_expose_stale_kv(self) -> None:
        first = torch.tensor([[1, 1, 1, 1]])
        self.assertTrue(self.cache.reserve_request("old", 8))
        self.cache.ensure_capacity("old", 4)
        self.model(first, kv_cache=self.cache, cache_request_ids=("old",))
        released = self.cache.release_request("old")

        second = torch.tensor([[7, 6, 5]])
        self.assertTrue(self.cache.reserve_request("new", 8))
        self.cache.ensure_capacity("new", 3)
        self.assertTrue(set(released) & set(self.cache.allocator.page_table("new")))
        actual = self.model(second, kv_cache=self.cache, cache_request_ids=("new",))

        torch.testing.assert_close(actual, self.model(second), atol=1e-5, rtol=1e-5)
        self.cache.check_integrity()


if __name__ == "__main__":
    unittest.main()
