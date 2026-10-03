from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .paged import PagedKVCompanion
from .radix import RadixCacheHandle, RadixPagedKVCache, RadixPrefixCache
from .recurrent import RecurrentStatePool


@dataclass(frozen=True)
class RecurrentStateSpec:
    """Rank-local DeltaNet state shape; the scheduler chooses the slot count."""

    num_layers: int
    conv_dim: int
    conv_kernel_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int


class HybridPagedKVCache(RadixPagedKVCache):
    """One owner for every per-request cache resource of a Qwen3.5 hybrid model.

    - paged K/V for the full-attention layers (inherited);
    - optional MTP draft K/V that shares the same page tables;
    - one recurrent state slot per request for the DeltaNet layers.

    Admission, completion and abort go through this class, so a request can never keep a
    state slot after its pages are released, or the other way round.
    """

    def __init__(
        self,
        *,
        recurrent: RecurrentStateSpec | None,
        num_recurrent_slots: int,
        num_draft_layers: int = 0,
        num_state_snapshots: int = 0,
        **kwargs,
    ) -> None:
        # DeltaNet 状态无法从页面恢复：混合模型只在开了状态快照时才做前缀复用，
        # 并且只承认挂着快照的 radix 节点（在那里能把 S 一起恢复出来）。
        kwargs.setdefault("enable_prefix_cache", recurrent is None or num_state_snapshots > 0)
        super().__init__(**kwargs)
        self.recurrent: RecurrentStatePool | None = None
        if recurrent is not None:
            self.recurrent = RecurrentStatePool(
                num_layers=recurrent.num_layers,
                num_slots=num_recurrent_slots,
                conv_dim=recurrent.conv_dim,
                conv_kernel_size=recurrent.conv_kernel_size,
                num_heads=recurrent.num_heads,
                head_k_dim=recurrent.head_k_dim,
                head_v_dim=recurrent.head_v_dim,
                conv_dtype=self._keys.dtype,
                device=self._keys.device,
                scratch_slots=0 if self.scratch_page is None else 1,
                num_snapshots=num_state_snapshots if self.enable_prefix_cache else 0,
            )
        self.draft: PagedKVCompanion | None = None#MTP的KVcache
        if num_draft_layers:
            self.draft = PagedKVCompanion(
                allocator=self.allocator,
                num_layers=num_draft_layers,
                num_pages=self.num_pages,
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=self._keys.dtype,
                device=self._keys.device,
                scratch_pages=0 if self.scratch_page is None else 1,
            )

    @property
    def memory_bytes(self) -> int:
        total = super().memory_bytes
        if self.draft is not None:
            total += self.draft.memory_bytes
        if self.recurrent is not None:
            total += self.recurrent.memory_bytes
        return total

    def can_reserve_request(
        self,
        max_tokens: int,
        prompt_token_ids: Sequence[int],
        reserved_tokens: int | None = None,
    ) -> bool:
        if self.recurrent is not None and not self.recurrent.can_reserve():
            return False
        return super().can_reserve_request(max_tokens, prompt_token_ids, reserved_tokens)

    def reserve_request_with_prefix(
        self,
        request_id: str,
        max_tokens: int,
        prompt_token_ids: Sequence[int],
        reserved_tokens: int | None = None,
    ) -> int | None:
        if self.recurrent is not None and not self.recurrent.can_reserve():
            return None
        matched = super().reserve_request_with_prefix(
            request_id, max_tokens, prompt_token_ids, reserved_tokens
        )
        if matched is None:
            return None
        if self.recurrent is not None:
            self.recurrent.reserve(request_id)  # 新格子先清零
            if matched:
                # 命中的前缀一定止于挂着快照的节点：把快照复制进格子，等于"已读完这段前缀"。
                snapshot = self._handles[request_id].node.snapshot
                assert snapshot is not None
                self.recurrent.load_snapshot(snapshot, request_id)
        if self.draft is not None:
            # 草稿层与目标层写同一批页面；命中的前缀页也已含草稿 K/V。
            self.draft.attach_request(request_id, matched)
        return matched

    @property
    def supports_state_snapshots(self) -> bool:
        return self.recurrent is not None and self.recurrent.num_snapshots > 0

    def _match(self, token_ids: Sequence[int], *, record: bool) -> RadixCacheHandle:
        handle = super()._match(token_ids, record=record)
        if self.recurrent is None:
            return handle
        # 只有挂着快照的节点能把 DeltaNet 状态一起恢复：退到路径上最深的那个。
        node, length = handle.node, handle.cached_len
        while node is not self.prefix_cache.root and node.snapshot is None:
            length -= len(node.key)
            node = RadixPrefixCache._parent(node)
        return RadixCacheHandle(length, node)

    def publish_prefix(self, request_id: str, token_ids: Sequence[int]) -> int:
        if self.recurrent is None:
            return super().publish_prefix(request_id, token_ids)
        # 混合模型：只在 prefill 停在快照边界时发布（publish_with_snapshot），
        # prompt 读完 / 请求结束时的普通发布没有配套的 S，发布了也命中不了。
        return self._handles[request_id].cached_len

    def publish_with_snapshot(self, request_id: str, token_ids: Sequence[int]) -> bool:
        """prefill 刚好停在整页边界 len(token_ids) 时调用：发布这段前缀并挂上当前 S 的快照。"""
        if not self.supports_state_snapshots:
            return False
        if len(token_ids) % self.page_size or self.lengths((request_id,))[0] != len(token_ids):
            raise ValueError("a state snapshot must be taken exactly at a cached page boundary")
        super().publish_prefix(request_id, token_ids)
        node = self._handles[request_id].node
        if node.snapshot is not None:
            return True  # 别的请求已经存过同一段前缀的快照
        snapshot = self.recurrent.save_snapshot(request_id)
        if snapshot is None and self._drop_lru_snapshot():
            snapshot = self.recurrent.save_snapshot(request_id)
        node.snapshot = snapshot
        return snapshot is not None

    def _drop_lru_snapshot(self) -> bool:
        """快照池满：丢掉最久没用、且没人锁着的节点上的快照（节点的 KV 页照样留在树里）。"""
        candidates = [
            node
            for node in self.prefix_cache._nodes()
            if node.snapshot is not None and node.ref_count == 0
        ]
        if not candidates:
            return False
        oldest = min(candidates, key=lambda node: (node.timestamp, node.node_id))
        self.recurrent.release_snapshot(oldest.snapshot)
        oldest.snapshot = None
        return True

    def _drain_released_snapshots(self) -> None:
        released = self.prefix_cache.released_snapshots
        while released:
            self.recurrent.release_snapshot(released.pop())

    def ensure_capacity(self, request_id: str, num_tokens: int) -> tuple[int, ...]:
        table = super().ensure_capacity(request_id, num_tokens)  # 缺页时会按 LRU 淘汰节点
        self._drain_released_snapshots()
        return table

    def reset_prefix_cache(self) -> None:
        super().reset_prefix_cache()
        self._drain_released_snapshots()

    def finish_request(self, request_id: str, token_ids: Sequence[int]) -> tuple[int, ...]:
        self._release_extras(request_id)
        return super().finish_request(request_id, token_ids)

    def release_request(self, request_id: str) -> tuple[int, ...]:
        self._release_extras(request_id)
        return super().release_request(request_id)

    def _release_extras(self, request_id: str) -> None:
        if self.recurrent is not None and request_id in self.recurrent.request_ids:
            self.recurrent.release(request_id)
        if self.draft is not None and request_id in self.draft._lengths:
            self.draft.detach_request(request_id)

    def check_integrity(self) -> None:
        super().check_integrity()
        active = set(self.allocator.request_ids)
        if self.recurrent is not None:
            self.recurrent.check_integrity()
            if set(self.recurrent.request_ids) != active:
                raise RuntimeError("recurrent state slots disagree with active cache requests")
            used = [
                node.snapshot
                for node in self.prefix_cache._nodes()
                if node.snapshot is not None
            ]
            if len(used) != len(set(used)) or len(used) + self.recurrent.free_snapshots != (
                self.recurrent.num_snapshots
            ):
                raise RuntimeError("recurrent snapshots disagree with the radix tree")
        if self.draft is not None:
            self.draft.check_integrity()
            for request_id in active:
                draft_length = self.draft.lengths((request_id,))[0]
                if draft_length > self.lengths((request_id,))[0]:
                    raise RuntimeError("draft K/V is ahead of the target cache between steps")
