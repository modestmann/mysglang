import random
import unittest

from mysglang import RadixPrefixCache


class RadixPrefixCacheTest(unittest.TestCase):
    def test_page_aligned_match_splits_compressed_edge(self) -> None:
        cache = RadixPrefixCache(page_size=2)
        inserted = cache.insert_prefix(range(1, 9), (10, 11, 12, 13))
        self.assertEqual(inserted.already_cached_len, 0)

        match = cache.match_prefix((1, 2, 3, 4, 99, 100))
        self.assertEqual(match.cached_len, 4)
        self.assertEqual(match.pages, (10, 11))

        branch = cache.insert_prefix((1, 2, 3, 4, 7, 7), (10, 11, 20))
        self.assertEqual(branch.already_cached_len, 4)
        self.assertEqual(cache.stats.cached_pages, 5)
        cache.check_integrity()

    def test_locked_prefix_is_protected_from_lru_eviction(self) -> None:
        cache = RadixPrefixCache(page_size=2)
        first = cache.insert_prefix((1, 2, 3, 4), (0, 1)).handle
        cache.insert_prefix((8, 8, 9, 9), (2, 3))
        cache.lock(first)

        evicted = cache.evict_pages(1)
        self.assertEqual(set(evicted), {2, 3})
        self.assertEqual(cache.match_prefix((1, 2, 3, 4)).pages, (0, 1))
        self.assertEqual(cache.match_prefix((8, 8, 9, 9)).cached_len, 0)

        cache.unlock(first)
        self.assertEqual(set(cache.evict_pages(1)), {0, 1})
        cache.check_integrity()

    def test_reset_returns_every_page_and_clears_stats_size(self) -> None:
        cache = RadixPrefixCache(page_size=2)
        cache.insert_prefix((1, 2, 3, 4), (4, 5))
        cache.insert_prefix((8, 9), (7,))

        self.assertEqual(cache.reset(), (4, 5, 7))
        self.assertEqual(cache.stats.cached_pages, 0)
        self.assertEqual(cache.stats.node_count, 0)
        cache.check_integrity()

    def test_random_insert_match_evict_preserves_tree_invariants(self) -> None:
        rng = random.Random(2026)
        cache = RadixPrefixCache(page_size=2)
        next_page = 0

        for _ in range(200):
            if cache.stats.cached_pages and rng.random() < 0.25:
                cache.evict_pages(rng.randint(1, cache.stats.evictable_pages))
            else:
                page_count = rng.randint(1, 4)
                tokens = tuple(rng.randrange(8) for _ in range(page_count * 2))
                match = cache.match_prefix(tokens)
                pages = list(match.pages)
                while len(pages) < page_count:
                    pages.append(next_page)
                    next_page += 1
                cache.insert_prefix(tokens, pages)
            cache.check_integrity()


if __name__ == "__main__":
    unittest.main()
