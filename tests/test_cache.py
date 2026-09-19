import random
import unittest
from unittest.mock import patch

import torch

from mysglang.cache import PageAllocator, PagedKVCache, RadixPrefixCache
from tests.helpers import make_model


class PageAllocatorTest(unittest.TestCase):
    def test_failed_reservation_is_atomic(self) -> None:
        allocator = PageAllocator(num_pages=4, page_size=4)
        self.assertTrue(allocator.reserve_request("first", max_tokens=12))
        before = allocator.stats

        self.assertFalse(allocator.reserve_request("second", max_tokens=8))

        self.assertEqual(allocator.stats, before)
        self.assertEqual(allocator.request_ids, frozenset({"first"}))
        allocator.check_integrity()

    def test_random_lifecycle_preserves_allocator_invariants(self) -> None:
        rng = random.Random(1234)
        allocator = PageAllocator(num_pages=12, page_size=4)
        limits: dict[str, int] = {}
        next_id = 0

        for _ in range(500):
            operation = rng.choice(("reserve", "grow", "release"))
            if operation == "reserve":
                request_id = f"req-{next_id}"
                limit = rng.randint(1, 12)
                if allocator.reserve_request(request_id, limit):
                    limits[request_id] = limit
                    next_id += 1
            elif operation == "grow" and limits:
                request_id = rng.choice(tuple(limits))
                allocator.ensure_capacity(request_id, rng.randint(0, limits[request_id]))
            elif operation == "release" and limits:
                request_id = rng.choice(tuple(limits))
                allocator.release(request_id)
                limits.pop(request_id)
            allocator.check_integrity()

        for request_id in tuple(limits):
            allocator.release(request_id)
        allocator.check_integrity()
        self.assertEqual(allocator.stats.free_pages, allocator.stats.total_pages)


class PagedKVCacheTest(unittest.TestCase):
    @torch.inference_mode()
    def test_decode_metadata_buffer_keeps_addresses_and_commits_once(self) -> None:
        for request_id, prompt in (("a", [1, 2, 3]), ("b", [4])):
            self.assertTrue(self.cache.reserve_request(request_id, 8))
            self.cache.ensure_capacity(request_id, len(prompt))
            self.model(
                torch.tensor([prompt]),
                kv_cache=self.cache,
                cache_request_ids=(request_id,),
            )

        buffer = self.cache.allocate_decode_buffer(2, max_blocks=4)
        pointers = tuple(
            tensor.data_ptr()
            for tensor in (
                buffer.block_table,
                buffer.cache_seqlens,
                buffer.cu_seqlens_q,
                buffer.cu_seqlens_k,
                buffer.positions,
                buffer.slot_mapping,
            )
        )
        self.cache.ensure_capacity("a", 4)
        self.cache.ensure_capacity("b", 2)
        batch = self.cache.prepare_decode_batch(
            ("a", "b"),
            buffer,
            commit_lengths=False,
        )
        self.model.forward_prepared(
            torch.tensor([[6], [7]]),
            kv_cache=self.cache,
            cache_batch=batch,
        )
        self.assertEqual(self.cache.lengths(("a", "b")), (3, 1))
        self.cache.commit_batch(batch)
        self.assertEqual(self.cache.lengths(("a", "b")), (4, 2))

        self.cache.ensure_capacity("a", 5)
        self.cache.ensure_capacity("b", 3)
        reused = self.cache.prepare_decode_batch(("b", "a"), buffer)
        self.assertEqual(reused.starts, (2, 4))
        self.assertEqual(
            pointers,
            tuple(
                tensor.data_ptr()
                for tensor in (
                    buffer.block_table,
                    buffer.cache_seqlens,
                    buffer.cu_seqlens_q,
                    buffer.cu_seqlens_k,
                    buffer.positions,
                    buffer.slot_mapping,
                )
            ),
        )
        self.assertTrue(torch.all(buffer.block_table[:, 3] == -1))
        self.cache.check_integrity()

    @torch.inference_mode()
    def test_selected_logits_skip_head_but_preserve_all_kv(self) -> None:
        self.cache.reserve_request("a", 8)
        self.cache.reserve_request("b", 8)
        self.cache.ensure_capacity("a", 2)
        self.cache.ensure_capacity("b", 3)
        with patch.object(self.model.lm_head, "forward", wraps=self.model.lm_head.forward) as head:
            empty = self.model.forward_packed(
                torch.tensor([1, 2, 3, 4, 5]),
                kv_cache=self.cache,
                cache_request_ids=("a", "b"),
                append_lengths=(2, 3),
                logits_indices=(),
            )
            head.assert_not_called()
        self.assertEqual(empty.shape, (0, self.model.config.vocab_size))
        self.assertEqual(self.cache.lengths(("a", "b")), (2, 3))
        self.cache.ensure_capacity("a", 4)
        self.cache.ensure_capacity("b", 4)
        with patch.object(self.model.lm_head, "forward", wraps=self.model.lm_head.forward) as head:
            actual = self.model.forward_packed(
                torch.tensor([6, 7, 8]),
                kv_cache=self.cache,
                cache_request_ids=("a", "b"),
                append_lengths=(2, 1),
                logits_indices=(2, 1),
            )
            self.assertEqual(head.call_args.args[0].shape[0], 2)
        expected = torch.stack(
            (
                self.model(torch.tensor([[3, 4, 5, 8]]))[0, -1],
                self.model(torch.tensor([[1, 2, 6, 7]]))[0, -1],
            )
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        before = self.cache.lengths(("a", "b"))
        with self.assertRaisesRegex(ValueError, "logits_indices"):
            self.model.forward_packed(
                torch.tensor([9, 10]),
                kv_cache=self.cache,
                cache_request_ids=("a", "b"),
                append_lengths=(1, 1),
                logits_indices=(2,),
            )
        self.assertEqual(self.cache.lengths(("a", "b")), before)
        self.cache.check_integrity()

    def setUp(self) -> None:
        self.model = make_model(seed=2468)
        parameter = next(self.model.parameters())
        self.cache = PagedKVCache.from_config(
            self.model.config,
            num_pages=8,
            page_size=2,
            dtype=parameter.dtype,
            device=parameter.device,
        )

    @torch.inference_mode()
    def test_batch_metadata_is_built_once_and_shared_across_layers(self) -> None:
        self.assertTrue(self.cache.reserve_request("shared-metadata", 8))
        self.cache.ensure_capacity("shared-metadata", 3)

        with (
            patch.object(
                self.cache,
                "prepare_batch",
                wraps=self.cache.prepare_batch,
            ) as prepare_batch,
            patch.object(
                self.cache,
                "prepare_append",
                wraps=self.cache.prepare_append,
            ) as prepare_append,
        ):
            self.model(
                torch.tensor([[1, 2, 3]]),
                kv_cache=self.cache,
                cache_request_ids=("shared-metadata",),
            )

        self.assertEqual(prepare_batch.call_count, 1)
        self.assertEqual(prepare_append.call_count, self.model.config.num_layers)
        layer_batches = [call.args[3] for call in prepare_append.call_args_list]
        self.assertTrue(all(batch is layer_batches[0] for batch in layer_batches))
        self.assertEqual(self.cache.lengths(("shared-metadata",)), (3,))

    @torch.inference_mode()
    def test_direct_append_remains_usable_one_layer_at_a_time(self) -> None:
        self.assertTrue(self.cache.reserve_request("direct", 4))
        self.cache.ensure_capacity("direct", 1)
        shape = (1, self.model.config.num_key_value_heads, 1, self.model.config.head_dim)
        key = torch.randn(shape)
        value = torch.randn(shape)

        for layer_idx in range(self.model.config.num_layers):
            self.cache.append(layer_idx, key, value, ("direct",))

        self.assertEqual(self.cache.lengths(("direct",)), (1,))
        self.cache.check_integrity()

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
    def test_packed_ragged_prefill_matches_full_recomputation(self) -> None:
        first = torch.tensor([1, 2])
        second = torch.tensor([3, 4, 5])
        self.assertTrue(self.cache.reserve_request("first-packed", 8))
        self.assertTrue(self.cache.reserve_request("second-packed", 8))
        self.cache.ensure_capacity("first-packed", 2)
        self.cache.ensure_capacity("second-packed", 3)

        actual = self.model.forward_packed(
            torch.cat((first, second)),
            kv_cache=self.cache,
            cache_request_ids=("first-packed", "second-packed"),
            append_lengths=(2, 3),
        )
        expected = torch.cat((self.model(first[None])[0], self.model(second[None])[0]))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        first_suffix = torch.tensor([6, 7])
        second_suffix = torch.tensor([8])
        self.cache.ensure_capacity("first-packed", 4)
        self.cache.ensure_capacity("second-packed", 4)
        actual = self.model.forward_packed(
            torch.cat((first_suffix, second_suffix)),
            kv_cache=self.cache,
            cache_request_ids=("first-packed", "second-packed"),
            append_lengths=(2, 1),
        )
        expected_first = self.model(torch.cat((first, first_suffix))[None])[0, -2:]
        expected_second = self.model(torch.cat((second, second_suffix))[None])[0, -1:]
        torch.testing.assert_close(
            actual,
            torch.cat((expected_first, expected_second)),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(self.cache.lengths(("first-packed", "second-packed")), (4, 4))
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

        torch.testing.assert_close(actual, self.model(prompt)[:, 2:], atol=1e-5, rtol=1e-5)
        self.assertEqual(self.cache.lengths(("chunked",)), (5,))
        self.assertEqual(len(self.cache.allocator.page_table("chunked")), 3)

    @torch.inference_mode()
    def test_reused_pages_do_not_expose_stale_kv(self) -> None:
        self.assertTrue(self.cache.reserve_request("old", 8))
        self.cache.ensure_capacity("old", 4)
        self.model(
            torch.tensor([[1, 1, 1, 1]]),
            kv_cache=self.cache,
            cache_request_ids=("old",),
        )
        released = self.cache.release_request("old")

        prompt = torch.tensor([[7, 6, 5]])
        self.assertTrue(self.cache.reserve_request("new", 8))
        self.cache.ensure_capacity("new", 3)
        self.assertTrue(set(released) & set(self.cache.allocator.page_table("new")))
        actual = self.model(
            prompt,
            kv_cache=self.cache,
            cache_request_ids=("new",),
        )

        torch.testing.assert_close(actual, self.model(prompt), atol=1e-5, rtol=1e-5)
        self.cache.check_integrity()


class RadixPrefixCacheTest(unittest.TestCase):
    def test_reset_handles_root_and_rejects_stale_cached_handle(self) -> None:
        cache = RadixPrefixCache(page_size=2)
        root_handle = cache.match_prefix((1, 2))
        cache.lock(root_handle)
        cache.reset()
        cache.unlock(root_handle)

        stale = cache.insert_prefix((1, 2), (0,)).handle
        cache.reset()
        with self.assertRaisesRegex(RuntimeError, "current tree"):
            cache.lock(stale)

        evicted = cache.insert_prefix((3, 4), (1,)).handle
        cache.evict_pages(1)
        with self.assertRaisesRegex(RuntimeError, "current tree"):
            cache.lock(evicted)

    def test_page_aligned_match_splits_compressed_edge(self) -> None:
        cache = RadixPrefixCache(page_size=2)
        inserted = cache.insert_prefix(range(1, 9), (10, 11, 12, 13))
        self.assertEqual(inserted.already_cached_len, 0)

        match = cache.match_prefix((1, 2, 3, 4, 99, 100))
        self.assertEqual(match.cached_len, 4)
        self.assertEqual(match.pages, (10, 11))
        branch = cache.insert_prefix((1, 2, 3, 4, 7, 7), (10, 11, 20))
        self.assertEqual(branch.already_cached_len, 4)
        cache.check_integrity()

    def test_locked_prefix_is_protected_from_lru_eviction(self) -> None:
        cache = RadixPrefixCache(page_size=2)
        protected = cache.insert_prefix((1, 2, 3, 4), (0, 1)).handle
        cache.insert_prefix((8, 8, 9, 9), (2, 3))
        cache.lock(protected)

        self.assertEqual(set(cache.evict_pages(1)), {2, 3})
        self.assertEqual(cache.match_prefix((1, 2, 3, 4)).pages, (0, 1))
        self.assertEqual(cache.match_prefix((8, 8, 9, 9)).cached_len, 0)

        cache.unlock(protected)
        self.assertEqual(set(cache.evict_pages(1)), {0, 1})
        cache.check_integrity()

    def test_random_insert_match_and_evict_preserve_tree_invariants(self) -> None:
        rng = random.Random(2026)
        cache = RadixPrefixCache(page_size=2)
        next_page = 0

        for _ in range(200):
            if cache.stats.cached_pages and rng.random() < 0.25:
                cache.evict_pages(rng.randint(1, cache.stats.evictable_pages))
            else:
                page_count = rng.randint(1, 4)
                tokens = tuple(rng.randrange(8) for _ in range(page_count * 2))
                pages = list(cache.match_prefix(tokens).pages)
                while len(pages) < page_count:
                    pages.append(next_page)
                    next_page += 1
                cache.insert_prefix(tokens, pages)
            cache.check_integrity()

        returned = cache.reset()
        self.assertEqual(len(returned), len(set(returned)))
        self.assertEqual(cache.stats.cached_pages, 0)
        cache.check_integrity()


if __name__ == "__main__":
    unittest.main()
