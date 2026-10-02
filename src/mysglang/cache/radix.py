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
        key: tuple[int, ...] = (),#这条边上的 token，长度一定是 page_size 的倍数
        pages: tuple[int, ...] = (),#这些 token 对应的物理页，一页对应 key 里的 4 个 token
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


class RadixPrefixCache:#只管"树"
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
########入口
    def match_prefix(self, token_ids: Sequence[int], *, record: bool = True) -> RadixCacheHandle:
        tokens = self._aligned_tokens(token_ids) # ① 截到整页
        node, matched = self._walk(tokens, touch=record) # ② 往下走
        if record:
            self._match_requests += 1
            self._matched_tokens += matched
        return RadixCacheHandle(matched, node)   # ③ 返回"走到哪个节点、匹配几个 token"

    def insert_prefix(self, token_ids: Sequence[int], pages: Sequence[int]) -> RadixInsertResult:#插入
        tokens = self._aligned_tokens(token_ids)
        physical_pages = tuple(pages)
        if len(tokens) != len(physical_pages) * self.page_size:
            raise ValueError("one physical page is required for each aligned token page")
        if len(physical_pages) != len(set(physical_pages)):
            raise ValueError("a prefix cannot contain duplicate physical pages")

        self._insert_requests += 1
        node, matched = self._walk(tokens, touch=True)#走一遍
        if matched < len(tokens):#挂新节点
            page_offset = matched // self.page_size
            child = self._new_node(
                key=tokens[matched:],
                pages=physical_pages[page_offset:],
                parent=node,
            )
            node.children[self._edge_key(child.key)] = child
            node = child
        return RadixInsertResult(matched, RadixCacheHandle(len(tokens), node))

    def lock(self, handle: RadixCacheHandle) -> None:#加锁要一路走到 root，handle.node 是匹配到的最深那个节点
        if self._is_root_handle(handle):
            return
        self._validate_current_handle(handle)
        node = handle.node
        while node is not self.root:
            node.ref_count += 1
            node = self._parent(node)

    def unlock(self, handle: RadixCacheHandle) -> None:
        if self._is_root_handle(handle):
            return
        self._validate_current_handle(handle)
        node = handle.node
        while node is not self.root:
            if node.ref_count <= 0:
                raise RuntimeError("radix handle is not locked")
            node.ref_count -= 1
            node = self._parent(node)
#只淘汰叶子，也就是没有 children 的节点。只淘汰 ref_count == 0 的，有人在用的不能动。
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
        heapq.heapify(leaves)# 做成一个最小堆：每次能直接取出 timestamp 最小的
        evicted: list[int] = []
        while len(evicted) < minimum_pages:# 还没腾够
            if not leaves:
                raise RuntimeError("radix tree could not find enough evictable leaves")
            _, _, node = heapq.heappop(leaves)  # 取出最久没用的叶子
            if node.children or node.ref_count != 0:
                continue
            parent = self._parent(node)
            del parent.children[self._edge_key(node.key)]# 把它从父节点上摘掉
            node.parent = None
            evicted.extend(node.pages)  # 它的页记为已淘汰
            if parent is not self.root and not parent.children and parent.ref_count == 0:
                heapq.heappush(leaves, (parent.timestamp, parent.node_id, parent))  # 父节点刚变成叶子，也加入候选

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
            child = node.children.get(self._edge_key(tokens[matched:]))# _edge_key 取前 4 个 = (A,B,C,D)
            if child is None:
                break
            common = self._common_prefix(child.key, tokens[matched:])#匹配最长
            common -= common % self.page_size# （向下对齐到整页）
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
#摘下 N1 → 建前半段挂到 root → N1 改成后半段 → N1 挂到前半段下面
    def _split(self, node: _RadixNode, offset: int) -> _RadixNode:
        if not 0 < offset < len(node.key) or offset % self.page_size:
            raise ValueError("radix split must occur at an internal page boundary")
        parent = self._parent(node)#找父节点
        del parent.children[self._edge_key(node.key)]# 先把 N1 从 root 下面摘掉
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
#全局时钟，在 _walk 里，每走过一个节点就 _touch 一次（查询和插入时都会走）。
#所以 timestamp 记录的是这个节点最后一次被用到的时间：数字越小，越久没用
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

    @staticmethod
    def _is_root_handle(handle: RadixCacheHandle) -> bool:
        node = handle.node
        return handle.cached_len == 0 and node.parent is None and not node.key and not node.pages

    def _validate_current_handle(self, handle: RadixCacheHandle) -> None:
        node = handle.node
        while node.parent is not None:
            node = node.parent
        if node is not self.root:
            raise RuntimeError("radix handle does not belong to the current tree")


class RadixPagedKVCache(PagedKVCache):#继承 PagedKVCache，把"树"和"allocator"粘在一起
    """Paged KV storage whose complete pages can outlive and serve requests."""

    def __init__(self, *, enable_prefix_cache: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix_cache = RadixPrefixCache(self.page_size)
        # 递归状态（DeltaNet）无法从页面前缀恢复，混合模型因此关闭前缀复用；
        # 关闭后仍复用同一套 handle/reservation 路径，只是永远匹配根节点。
        self.enable_prefix_cache = enable_prefix_cache
        self._handles: dict[str, RadixCacheHandle] = {}#请求 id → 这个请求的借书凭证（handle）

    def _match(self, token_ids: Sequence[int], *, record: bool) -> RadixCacheHandle:
        if not self.enable_prefix_cache:
            return RadixCacheHandle(0, self.prefix_cache.root)
        return self.prefix_cache.match_prefix(token_ids, record=record)

    def can_reserve_request(self, max_tokens: int, prompt_token_ids: Sequence[int]) -> bool:
        handle = self._match(prompt_token_ids[:-1], record=False)
        total_pages = self.allocator.pages_for_tokens(max_tokens)
        private_pages = total_pages - len(handle.pages)
        newly_protected = self._newly_protected_pages(handle)
        stats = self.prefix_cache.stats
        committed = self.allocator.stats.reserved_pages + stats.protected_pages + newly_protected
        return committed + private_pages <= self.num_pages

    def reserve_request_with_prefix(#查树，lock(handle)，锁住复用的前缀
        self,
        request_id: str,
        max_tokens: int,
        prompt_token_ids: Sequence[int],
    ) -> int | None:
        handle = self._match(prompt_token_ids[:-1], record=True)
        self.prefix_cache.lock(handle)
        total_pages = self.allocator.pages_for_tokens(max_tokens)
        private_pages = total_pages - len(handle.pages)
        committed = (
            self.allocator.stats.reserved_pages
            + self.prefix_cache.stats.protected_pages
            + private_pages
        )#检验是否超过总页数
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
        if missing:#空闲页不足
            evicted = self.prefix_cache.evict_pages(missing) # ① 从树里淘汰 missing 页
            self.allocator.evict_cached(evicted) # ② 这些页从"缓存"挪到"空闲"
        return super().ensure_capacity(request_id, num_tokens)

    def finish_request(self, request_id: str, token_ids: Sequence[int]) -> tuple[int, ...]:#正常结束解锁
        self.publish_prefix(request_id, token_ids)   # ① 再试着缓存一次
        self.prefix_cache.unlock(self._handles.pop(request_id)) # ② 解锁并扔掉凭证
        return super().release_request(request_id)    # ③ 父类：归还页
##页表是 [3, 0, 1]。页3、页0 是缓存页，留在树里；页1 不是缓存页，也没人在用，回到空闲页
    def publish_prefix(self, request_id: str, token_ids: Sequence[int]) -> int:#	prompt 刚读完	换锁：锁新节点，解旧节点
        """Publish all complete, already-computed pages of an active request.
####R 的 prompt 已经算完了，立刻把它放进树里，别的请求就能复用
        The request keeps a lock on the canonical prefix, so another request can
        reuse it immediately while the publisher is still decoding.
        """
        self._validate_request(request_id)
        if not self.enable_prefix_cache:
            return 0
        length = self.lengths((request_id,))[0]
        cacheable_length = min(length, len(token_ids))
        cacheable_length -= cacheable_length % self.page_size
        previous = self._handles[request_id]
        if cacheable_length <= previous.cached_len:
            return previous.cached_len

        table = self.allocator.page_table(request_id)
        pages = table[: cacheable_length // self.page_size]
        result = self.prefix_cache.insert_prefix(tuple(token_ids)[:cacheable_length], pages)#缓存写满的页
        self.allocator.mark_cached(result.handle.pages)
        self.allocator.replace_prefix(request_id, result.handle.pages)
        self.prefix_cache.lock(result.handle)#加锁
        self._handles[request_id] = result.handle
        self.prefix_cache.unlock(previous)#解锁（适配新凭证）
        return result.handle.cached_len

    def release_request(self, request_id: str) -> tuple[int, ...]:#abort解锁
        handle = self._handles.pop(request_id, None)
        if handle is not None:
            self.prefix_cache.unlock(handle)#解锁
        return super().release_request(request_id)#归还页

    def reset_prefix_cache(self) -> None:
        if self._handles or self.allocator.request_ids:
            raise RuntimeError("cannot reset the prefix cache while requests are active")
        pages = self.prefix_cache.reset()
        self.allocator.evict_cached(pages)

    def check_integrity(self) -> None:
        super().check_integrity()
        self.prefix_cache.check_integrity()
        if self.prefix_cache.pages != self.allocator.cached_pages:
            raise RuntimeError("radix tree and allocator disagree about cached pages")
        if set(self._handles) != set(self.allocator.request_ids):
            raise RuntimeError("radix handles and active cache requests disagree")
        tree_nodes = set(self.prefix_cache._nodes())
        if any(handle.node not in tree_nodes for handle in self._handles.values()):
            raise RuntimeError("an active radix handle points outside the current tree")
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
