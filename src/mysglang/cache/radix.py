from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .paged import PagedKVCache


@dataclass(frozen=True)
class RadixCacheStats:
    cached_pages: int
    protected_pages: int
    evictable_pages: int
    node_count: int
    match_requests: int
    matched_tokens: int
    insert_requests: int
    evicted_pages: int


class _RadixNode:
    def __init__(
        self,
        *,
        key: tuple[int, ...] = (),
        pages: tuple[int, ...] = (),
        parent: _RadixNode | None = None,
        timestamp: int = 0,
        node_id: int = 0,
    ) -> None:
        self.key = key
        self.pages = pages
        self.parent = parent
        self.children: dict[tuple[int, ...], _RadixNode] = {}
        self.ref_count = 0
        self.timestamp = timestamp
        self.node_id = node_id


@dataclass(frozen=True)
class RadixCacheHandle:
    cached_len: int
    node: _RadixNode

    @property
    def pages(self) -> tuple[int, ...]:
        parts: list[tuple[int, ...]] = []
        node = self.node
        while node.parent is not None:
            parts.append(node.pages)
            node = node.parent
        return tuple(page for part in reversed(parts) for page in part)


@dataclass(frozen=True)
class RadixInsertResult:
    already_cached_len: int
    handle: RadixCacheHandle


class RadixPrefixCache:
    """A page-aligned compressed radix tree from token IDs to physical pages."""

    def __init__(self, page_size: int) -> None:
        if not isinstance(page_size, int) or isinstance(page_size, bool):
            raise TypeError("page_size must be an integer")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self._clock = 0
        self._node_ids = itertools.count()
        self.root = self._new_node()
        self.root.ref_count = 1
        self._match_requests = 0
        self._matched_tokens = 0
        self._insert_requests = 0
        self._evicted_pages = 0

    @property
    def stats(self) -> RadixCacheStats:
        nodes = list(self._nodes())
        cached = sum(len(node.pages) for node in nodes if node is not self.root)
        protected = sum(
            len(node.pages) for node in nodes if node is not self.root and node.ref_count > 0
        )
        return RadixCacheStats(
            cached_pages=cached,
            protected_pages=protected,
            evictable_pages=cached - protected,
            node_count=len(nodes) - 1,
            match_requests=self._match_requests,
            matched_tokens=self._matched_tokens,
            insert_requests=self._insert_requests,
            evicted_pages=self._evicted_pages,
        )

    @property
    def pages(self) -> frozenset[int]:
        return frozenset(
            page for node in self._nodes() if node is not self.root for page in node.pages
        )

    def match_prefix(self, token_ids: Sequence[int], *, record: bool = True) -> RadixCacheHandle:
        tokens = self._aligned_tokens(token_ids)
        node, matched = self._walk(tokens, touch=record)
        if record:
            self._match_requests += 1
            self._matched_tokens += matched
        return RadixCacheHandle(matched, node)

    def insert_prefix(self, token_ids: Sequence[int], pages: Sequence[int]) -> RadixInsertResult:
        tokens = self._aligned_tokens(token_ids)
        physical_pages = tuple(pages)
        if len(tokens) != len(physical_pages) * self.page_size:
            raise ValueError("one physical page is required for each aligned token page")
        if len(physical_pages) != len(set(physical_pages)):
            raise ValueError("a prefix cannot contain duplicate physical pages")

        self._insert_requests += 1
        node, matched = self._walk(tokens, touch=True)
        if matched < len(tokens):
            page_offset = matched // self.page_size
            child = self._new_node(
                key=tokens[matched:],
                pages=physical_pages[page_offset:],
                parent=node,
            )
            node.children[self._edge_key(child.key)] = child
            node = child
        return RadixInsertResult(matched, RadixCacheHandle(len(tokens), node))

    def lock(self, handle: RadixCacheHandle) -> None:
        node = handle.node
        while node is not self.root:
            node.ref_count += 1
            node = self._parent(node)

    def unlock(self, handle: RadixCacheHandle) -> None:
        node = handle.node
        while node is not self.root:
            if node.ref_count <= 0:
                raise RuntimeError("radix handle is not locked")
            node.ref_count -= 1
            node = self._parent(node)

    def evict_pages(self, minimum_pages: int) -> tuple[int, ...]:
        if minimum_pages < 0:
            raise ValueError("minimum_pages must be non-negative")
        if minimum_pages == 0:
            return ()
        if minimum_pages > self.stats.evictable_pages:
            raise RuntimeError("not enough evictable prefix pages")

        leaves = [
            (node.timestamp, node.node_id, node)
            for node in self._nodes()
            if node is not self.root and not node.children and node.ref_count == 0
        ]
        heapq.heapify(leaves)
        evicted: list[int] = []
        while len(evicted) < minimum_pages:
            if not leaves:
                raise RuntimeError("radix tree could not find enough evictable leaves")
            _, _, node = heapq.heappop(leaves)
            if node.children or node.ref_count != 0:
                continue
            parent = self._parent(node)
            del parent.children[self._edge_key(node.key)]
            evicted.extend(node.pages)
            if parent is not self.root and not parent.children and parent.ref_count == 0:
                heapq.heappush(leaves, (parent.timestamp, parent.node_id, parent))

        self._evicted_pages += len(evicted)
        return tuple(evicted)

    def reset(self) -> tuple[int, ...]:
        if self.stats.protected_pages:
            raise RuntimeError("cannot reset while prefix handles are locked")
        pages = tuple(sorted(self.pages))
        self.root = self._new_node()
        self.root.ref_count = 1
        return pages

    def check_integrity(self) -> None:
        if self.root.parent is not None or self.root.key or self.root.pages:
            raise RuntimeError("radix root must not store an edge")
        if self.root.ref_count != 1:
            raise RuntimeError("radix root must remain permanently protected")

        seen_nodes: set[int] = set()
        seen_pages: set[int] = set()
        for node in self._nodes():
            if node.node_id in seen_nodes:
                raise RuntimeError("radix tree contains a node cycle")
            seen_nodes.add(node.node_id)
            if node.ref_count < 0:
                raise RuntimeError("radix node has a negative reference count")
            if node is self.root:
                continue
            if not node.key or len(node.key) % self.page_size:
                raise RuntimeError("radix edges must contain complete pages")
            if len(node.key) != len(node.pages) * self.page_size:
                raise RuntimeError("radix edge tokens and pages disagree")
            parent = self._parent(node)
            if parent.children.get(self._edge_key(node.key)) is not node:
                raise RuntimeError("radix parent/child index is inconsistent")
            if node.ref_count > parent.ref_count and parent is not self.root:
                raise RuntimeError("a child cannot have more locks than its parent")
            overlap = seen_pages.intersection(node.pages)
            if overlap:
                raise RuntimeError("a physical page appears twice in the radix tree")
            seen_pages.update(node.pages)

    def _walk(self, tokens: tuple[int, ...], *, touch: bool) -> tuple[_RadixNode, int]:
        node = self.root
        matched = 0
        while matched < len(tokens):
            child = node.children.get(self._edge_key(tokens[matched:]))
            if child is None:
                break
            common = self._common_prefix(child.key, tokens[matched:])
            common -= common % self.page_size
            if common == 0:
                break
            matched += common
            if common < len(child.key):
                node = self._split(child, common)
                if touch:
                    self._touch(node)
                return node, matched
            node = child
            if touch:
                self._touch(node)
        return node, matched

    def _split(self, node: _RadixNode, offset: int) -> _RadixNode:
        if not 0 < offset < len(node.key) or offset % self.page_size:
            raise ValueError("radix split must occur at an internal page boundary")
        parent = self._parent(node)
        del parent.children[self._edge_key(node.key)]
        page_offset = offset // self.page_size
        prefix = self._new_node(
            key=node.key[:offset],
            pages=node.pages[:page_offset],
            parent=parent,
            timestamp=node.timestamp,
        )
        prefix.ref_count = node.ref_count
        parent.children[self._edge_key(prefix.key)] = prefix
        node.key = node.key[offset:]
        node.pages = node.pages[page_offset:]
        node.parent = prefix
        prefix.children[self._edge_key(node.key)] = node
        return prefix

    def _new_node(
        self,
        *,
        key: tuple[int, ...] = (),
        pages: tuple[int, ...] = (),
        parent: _RadixNode | None = None,
        timestamp: int | None = None,
    ) -> _RadixNode:
        return _RadixNode(
            key=key,
            pages=pages,
            parent=parent,
            timestamp=self._tick() if timestamp is None else timestamp,
            node_id=next(self._node_ids),
        )

    def _touch(self, node: _RadixNode) -> None:
        node.timestamp = self._tick()

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _edge_key(self, tokens: Sequence[int]) -> tuple[int, ...]:
        return tuple(tokens[: self.page_size])

    def _aligned_tokens(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        tokens = tuple(int(token) for token in token_ids)
        length = len(tokens) - len(tokens) % self.page_size
        return tokens[:length]

    @staticmethod
    def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
        matched = 0
        for lhs, rhs in zip(left, right):
            if lhs != rhs:
                break
            matched += 1
        return matched

    @staticmethod
    def _parent(node: _RadixNode) -> _RadixNode:
        if node.parent is None:
            raise RuntimeError("radix node unexpectedly has no parent")
        return node.parent

    def _nodes(self) -> Iterable[_RadixNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())


class RadixPagedKVCache(PagedKVCache):
    """Paged KV storage whose complete pages can outlive and serve requests."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix_cache = RadixPrefixCache(self.page_size)
        self._handles: dict[str, RadixCacheHandle] = {}

    def can_reserve_request(self, max_tokens: int, prompt_token_ids: Sequence[int]) -> bool:
        handle = self.prefix_cache.match_prefix(prompt_token_ids[:-1], record=False)
        total_pages = self.allocator.pages_for_tokens(max_tokens)
        private_pages = total_pages - len(handle.pages)
        newly_protected = self._newly_protected_pages(handle)
        stats = self.prefix_cache.stats
        committed = self.allocator.stats.reserved_pages + stats.protected_pages + newly_protected
        return committed + private_pages <= self.num_pages

    def reserve_request_with_prefix(
        self,
        request_id: str,
        max_tokens: int,
        prompt_token_ids: Sequence[int],
    ) -> int | None:
        handle = self.prefix_cache.match_prefix(prompt_token_ids[:-1])
        self.prefix_cache.lock(handle)
        total_pages = self.allocator.pages_for_tokens(max_tokens)
        private_pages = total_pages - len(handle.pages)
        committed = (
            self.allocator.stats.reserved_pages
            + self.prefix_cache.stats.protected_pages
            + private_pages
        )
        if committed > self.num_pages:
            self.prefix_cache.unlock(handle)
            return None
        reserved = super().reserve_request(
            request_id,
            max_tokens,
            initial_pages=handle.pages,
            initial_length=handle.cached_len,
            reserved_pages=private_pages,
        )
        if not reserved:
            self.prefix_cache.unlock(handle)
            return None
        self._handles[request_id] = handle
        return handle.cached_len

    def ensure_capacity(self, request_id: str, num_tokens: int) -> tuple[int, ...]:
        table = self.allocator.page_table(request_id)
        needed = self.allocator.pages_for_tokens(num_tokens)
        additional = max(0, needed - len(table))
        missing = max(0, additional - self.allocator.stats.free_pages)
        if missing:
            evicted = self.prefix_cache.evict_pages(missing)
            self.allocator.evict_cached(evicted)
        return super().ensure_capacity(request_id, num_tokens)

    def finish_request(self, request_id: str, token_ids: Sequence[int]) -> tuple[int, ...]:
        self._validate_request(request_id)
        length = self.lengths((request_id,))[0]
        cacheable_length = min(length, len(token_ids))
        cacheable_length -= cacheable_length % self.page_size
        table = self.allocator.page_table(request_id)
        pages = table[: cacheable_length // self.page_size]
        result = self.prefix_cache.insert_prefix(tuple(token_ids)[:cacheable_length], pages)
        self.allocator.mark_cached(result.handle.pages)
        self.allocator.replace_prefix(request_id, result.handle.pages)
        self.prefix_cache.unlock(self._handles.pop(request_id))
        return super().release_request(request_id)

    def release_request(self, request_id: str) -> tuple[int, ...]:
        handle = self._handles.pop(request_id, None)
        if handle is not None:
            self.prefix_cache.unlock(handle)
        return super().release_request(request_id)

    def reset_prefix_cache(self) -> None:
        pages = self.prefix_cache.reset()
        self.allocator.evict_cached(pages)

    def check_integrity(self) -> None:
        super().check_integrity()
        self.prefix_cache.check_integrity()
        if self.prefix_cache.pages != self.allocator.cached_pages:
            raise RuntimeError("radix tree and allocator disagree about cached pages")
        if set(self._handles) != set(self.allocator.request_ids):
            raise RuntimeError("radix handles and active cache requests disagree")
        committed = self.allocator.stats.reserved_pages + self.prefix_cache.stats.protected_pages
        if committed > self.num_pages:
            raise RuntimeError("protected prefixes and request reservations exceed the pool")
        for request_id, handle in self._handles.items():
            table = self.allocator.page_table(request_id)
            if table[: len(handle.pages)] != handle.pages:
                raise RuntimeError("request page table does not start with matched pages")
            if self.lengths((request_id,))[0] < handle.cached_len:
                raise RuntimeError("request length is shorter than its matched prefix")

    def _newly_protected_pages(self, handle: RadixCacheHandle) -> int:
        pages = 0
        node = handle.node
        while node is not self.prefix_cache.root:
            if node.ref_count == 0:
                pages += len(node.pages)
            node = RadixPrefixCache._parent(node)
        return pages
