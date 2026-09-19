from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mysglang.config import ModelConfig


class PageAllocationError(RuntimeError):
    """Raised when a request exceeds its reservation or physical page capacity."""


@dataclass(frozen=True)
class PageAllocatorStats:
    total_pages: int
    page_size: int
    free_pages: int
    allocated_pages: int
    reserved_pages: int
    admission_available_pages: int
    request_count: int
    cached_pages: int = 0


@dataclass(frozen=True)
class PagedKVAppendPlan:
    """One layer's immutable metadata for appending and attending to paged KV."""

    layer_idx: int
    request_ids: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor


@dataclass
class _Allocation:
    page_table: list[int]
    max_tokens: int
    reserved_pages: int
    shared_prefix_pages: int


class PageAllocator:
    """Own physical page IDs, per-request block tables, and admission reservations."""

    def __init__(self, *, num_pages: int, page_size: int) -> None:
        for name, value in (("num_pages", num_pages), ("page_size", page_size)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.num_pages = num_pages
        self.page_size = page_size
        self._free_pages = set(range(num_pages))  # 空闲物理页编号
        self._cached_pages: set[int] = set()
        # 请求页表、最大 token 容量和 admission 预留承诺必须一起增删。
        # 例如："request-A" -> page_table=[0, 2]，"request-B" -> page_table=[1]。
        self._allocations: dict[str, _Allocation] = {}

    @property
    def request_ids(self) -> frozenset[str]:
        return frozenset(self._allocations)

    @property
    def cached_pages(self) -> frozenset[int]:
        return frozenset(self._cached_pages)

    @property
    def stats(self) -> PageAllocatorStats:
        allocated = self.num_pages - len(self._free_pages)
        reserved = sum(allocation.reserved_pages for allocation in self._allocations.values())
        return PageAllocatorStats(
            total_pages=self.num_pages,
            page_size=self.page_size,
            free_pages=len(self._free_pages),
            allocated_pages=allocated,
            reserved_pages=reserved,
            admission_available_pages=self.num_pages - reserved,
            request_count=len(self._allocations),
            cached_pages=len(self._cached_pages),
        )

    def pages_for_tokens(self, num_tokens: int) -> int:
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise TypeError("num_tokens must be an integer")
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return (num_tokens + self.page_size - 1) // self.page_size

    def can_reserve(self, max_tokens: int) -> bool:
        needed = self.pages_for_tokens(max_tokens)
        return needed <= self.stats.admission_available_pages

    def reserve_request(
        self,
        request_id: str,
        max_tokens: int,
        *,
        initial_pages: Sequence[int] = (),
        reserved_pages: int | None = None,
    ) -> bool:
        self._validate_new_request_id(request_id)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        prefix = tuple(initial_pages)
        if len(prefix) != len(set(prefix)):
            raise ValueError("initial_pages must be unique")
        if any(page not in self._cached_pages for page in prefix):
            raise ValueError("initial_pages must already belong to the prefix cache")
        total_pages = self.pages_for_tokens(max_tokens)
        if len(prefix) > total_pages:
            raise ValueError("initial_pages exceed the request's maximum page count")
        needed = total_pages - len(prefix) if reserved_pages is None else reserved_pages
        if needed < 0 or needed + len(prefix) != total_pages:
            raise ValueError("reserved_pages must cover the non-prefix request pages")
        if needed > self.stats.admission_available_pages:
            return False
        self._allocations[request_id] = _Allocation(
            page_table=list(prefix),
            max_tokens=max_tokens,
            reserved_pages=needed,
            shared_prefix_pages=len(prefix),
        )
        return True

    def ensure_capacity(self, request_id: str, num_tokens: int) -> tuple[int, ...]:
        self._validate_known_request(request_id)
        allocation = self._allocations[request_id]
        if num_tokens > allocation.max_tokens:
            raise PageAllocationError(
                f"request {request_id!r} exceeds its token reservation: "
                f"{num_tokens} > {allocation.max_tokens}"
            )
        needed = self.pages_for_tokens(num_tokens)
        table = allocation.page_table
        additional = needed - len(table)
        if additional <= 0:
            return tuple(table)
        if additional > len(self._free_pages):
            raise PageAllocationError("reserved request cannot obtain its physical pages")

        allocated = sorted(self._free_pages)[:additional]
        table.extend(allocated)
        self._free_pages.difference_update(allocated)
        return tuple(table)

    def page_table(self, request_id: str) -> tuple[int, ...]:
        self._validate_known_request(request_id)
        return tuple(self._allocations[request_id].page_table)

    def max_tokens(self, request_id: str) -> int:
        self._validate_known_request(request_id)
        return self._allocations[request_id].max_tokens

    def release(self, request_id: str) -> tuple[int, ...]:
        self._validate_known_request(request_id)
        pages = tuple(self._allocations.pop(request_id).page_table)
        still_referenced = {
            page for allocation in self._allocations.values() for page in allocation.page_table
        }
        released = tuple(
            page
            for page in pages
            if page not in self._cached_pages and page not in still_referenced
        )
        self._free_pages.update(released)
        return released

    def mark_cached(self, pages: Sequence[int]) -> None:
        cached = tuple(pages)
        if len(cached) != len(set(cached)):
            raise ValueError("cached pages must be unique")
        if any(not 0 <= page < self.num_pages for page in cached):
            raise ValueError("cached page is outside the physical pool")
        if set(cached) & self._free_pages:
            raise ValueError("a free page cannot be added to the prefix cache")
        self._cached_pages.update(cached)

    def replace_prefix(self, request_id: str, prefix_pages: Sequence[int]) -> tuple[int, ...]:
        """Replace a request's prefix with canonical shared pages."""
        self._validate_known_request(request_id)
        prefix = tuple(prefix_pages)
        if any(page not in self._cached_pages for page in prefix):
            raise ValueError("replacement prefix pages must be cached")
        allocation = self._allocations[request_id]
        table = allocation.page_table
        if len(prefix) > len(table):
            raise ValueError("replacement prefix exceeds the request page table")
        displaced = tuple(table[: len(prefix)])
        table[: len(prefix)] = prefix
        allocation.shared_prefix_pages = len(prefix)
        allocation.reserved_pages = self.pages_for_tokens(allocation.max_tokens) - len(prefix)
        still_referenced = {
            page for allocation in self._allocations.values() for page in allocation.page_table
        }
        released = tuple(
            page
            for page in displaced
            if page not in self._cached_pages and page not in still_referenced
        )
        self._free_pages.update(released)
        return released

    def evict_cached(self, pages: Sequence[int]) -> None:
        evicted = set(pages)
        if not evicted <= self._cached_pages:
            raise ValueError("cannot evict pages that are not prefix-cached")
        referenced = {
            page for allocation in self._allocations.values() for page in allocation.page_table
        }
        if evicted & referenced:
            raise RuntimeError("cannot evict a prefix page used by an active request")
        self._cached_pages.difference_update(evicted)
        self._free_pages.update(evicted)

    def check_integrity(self) -> None:
        tables = [allocation.page_table for allocation in self._allocations.values()]
        allocated = [page for table in tables for page in table]
        duplicate_pages = {page for page in set(allocated) if allocated.count(page) > 1}
        if duplicate_pages - self._cached_pages:
            raise RuntimeError("a non-cached physical page has multiple request owners")
        owned = set(allocated) | self._cached_pages
        if owned & self._free_pages:
            raise RuntimeError("physical page is both allocated and free")
        if owned | self._free_pages != set(range(self.num_pages)):
            raise RuntimeError("physical page accounting does not cover the pool")
        reserved_pages = sum(allocation.reserved_pages for allocation in self._allocations.values())
        if reserved_pages > self.num_pages:
            raise RuntimeError("reserved pages exceed physical pool capacity")
        for allocation in self._allocations.values():
            total_pages = self.pages_for_tokens(allocation.max_tokens)
            if len(allocation.page_table) > total_pages:
                raise RuntimeError("allocated pages exceed request maximum length")
            if not 0 <= allocation.shared_prefix_pages <= len(allocation.page_table):
                raise RuntimeError("shared prefix exceeds the request page table")
            if allocation.reserved_pages != total_pages - allocation.shared_prefix_pages:
                raise RuntimeError("request reservation disagrees with its shared prefix")
            shared_prefix = allocation.page_table[: allocation.shared_prefix_pages]
            if any(page not in self._cached_pages for page in shared_prefix):
                raise RuntimeError("request shared prefix contains a non-cached page")

    def _validate_new_request_id(self, request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._allocations:
            raise ValueError(f"request already has a page reservation: {request_id}")

    def _validate_known_request(self, request_id: str) -> None:
        if request_id not in self._allocations:
            raise KeyError(f"unknown paged-cache request: {request_id}")


class PagedKVCache:
    """Physical paged K/V tensors with request-specific logical block tables."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "num_pages": num_pages,
            "page_size": page_size,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        if invalid := [name for name, value in dimensions.items() if value <= 0]:
            raise ValueError(f"cache dimensions must be positive: {invalid}")
        shape = (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        self._keys = torch.empty(shape, dtype=dtype, device=device)
        self._values = torch.empty_like(self._keys)
        self.allocator = PageAllocator(num_pages=num_pages, page_size=page_size)
        self._lengths: dict[str, list[int]] = {}
        # 全局物理KV池
        """
  layer_idx                         决定访问哪一层
  page_table[position // page_size] 决定访问哪个物理页
  position % page_size              决定页内偏移
  _keys[layer_idx, physical_page, offset]
        """

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        *,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> PagedKVCache:
        return cls(
            num_layers=config.num_layers,
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )

    @property
    def num_layers(self) -> int:
        return self._keys.size(0)

    @property
    def num_pages(self) -> int:
        return self._keys.size(1)

    @property
    def page_size(self) -> int:
        return self._keys.size(2)

    @property
    def memory_bytes(self) -> int:
        return (self._keys.numel() + self._values.numel()) * self._keys.element_size()

    def reserve_request(
        self,
        request_id: str,
        max_tokens: int,
        *,
        initial_pages: Sequence[int] = (),
        initial_length: int = 0,
        reserved_pages: int | None = None,
    ) -> bool:
        prefix = tuple(initial_pages)
        if initial_length < 0 or initial_length % self.page_size != 0:
            raise ValueError("initial_length must be a non-negative page multiple")
        if len(prefix) * self.page_size != initial_length:
            raise ValueError("initial_pages and initial_length disagree")
        reserved = self.allocator.reserve_request(
            request_id,
            max_tokens,
            initial_pages=prefix,
            reserved_pages=reserved_pages,
        )
        if reserved:
            self._lengths[request_id] = [initial_length] * self.num_layers
        return reserved

    def can_reserve(self, max_tokens: int) -> bool:
        return self.allocator.can_reserve(max_tokens)

    def ensure_capacity(self, request_id: str, num_tokens: int) -> tuple[int, ...]:
        self._validate_request(request_id)
        return self.allocator.ensure_capacity(request_id, num_tokens)

    def lengths(self, request_ids: Sequence[str]) -> tuple[int, ...]:
        requests = self._validate_requests(request_ids)
        result = tuple(self._lengths[request_id][0] for request_id in requests)
        for layer_idx in range(1, self.num_layers):
            current = tuple(self._lengths[request_id][layer_idx] for request_id in requests)
            if current != result:
                raise RuntimeError("KV cache layers have inconsistent request lengths")
        return result

    def layer_lengths(self, layer_idx: int, request_ids: Sequence[str]) -> tuple[int, ...]:
        self._validate_layer(layer_idx)
        requests = self._validate_requests(request_ids)
        return tuple(self._lengths[request_id][layer_idx] for request_id in requests)

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        request_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plan = self.prepare_append(layer_idx, key, value, request_ids)
        result = self.stage_append(plan, key, value)
        self.commit_append(plan)
        return result

    def stage_append(
        self,
        plan: PagedKVAppendPlan,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write K/V and gather the reference tensors without advancing lengths."""
        chunk_length = key.size(2)

        with torch.no_grad():
            for batch_index, (request_id, start, end) in enumerate(
                zip(plan.request_ids, plan.starts, plan.ends)
            ):
                page_table = self.allocator.page_table(request_id)
                for chunk_index, position in enumerate(range(start, end)):
                    page = page_table[position // self.page_size]
                    offset = position % self.page_size
                    plan.key_cache[page, offset].copy_(key[batch_index, :, chunk_index])
                    plan.value_cache[page, offset].copy_(value[batch_index, :, chunk_index])

        max_end = max(plan.ends)
        keys = key.new_zeros((len(plan.request_ids), key.size(1), max_end, key.size(3)))
        values = value.new_zeros(keys.shape)
        flat_keys = plan.key_cache.view(-1, key.size(1), key.size(3))
        flat_values = plan.value_cache.view(-1, key.size(1), key.size(3))
        for batch_index, (request_id, length) in enumerate(zip(plan.request_ids, plan.ends)):
            table = self.allocator.page_table(request_id)
            physical_indices = [
                table[position // self.page_size] * self.page_size + position % self.page_size
                for position in range(length)
            ]
            index = torch.tensor(physical_indices, dtype=torch.long, device=key.device)
            keys[batch_index, :, :length] = flat_keys.index_select(0, index).transpose(0, 1)
            values[batch_index, :, :length] = flat_values.index_select(0, index).transpose(0, 1)

        query_positions = torch.tensor(plan.starts, device=key.device)[:, None]
        query_positions = query_positions + torch.arange(chunk_length, device=key.device)[None, :]
        key_positions = torch.arange(max_end, device=key.device)
        attention_mask = key_positions[None, None, :] <= query_positions[:, :, None]
        return keys, values, attention_mask[:, None, :, :]

    def prepare_append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        request_ids: Sequence[str],
    ) -> PagedKVAppendPlan:
        """Validate one append and build GPU-friendly block-table metadata."""
        self._validate_layer(layer_idx)
        requests = self._validate_requests(request_ids)
        if len(set(requests)) != len(requests):
            raise ValueError("request_ids must be unique within a batch")

        expected = (len(requests), self._keys.size(3), self._keys.size(4))
        actual = (key.size(0), key.size(1), key.size(3)) if key.ndim == 4 else None

        if actual != expected:
            raise ValueError(
                "key must have shape "
                f"[{len(requests)}, {self._keys.size(3)}, sequence, {self._keys.size(4)}]"
            )
        if value.shape != key.shape:
            raise ValueError("key and value must have the same shape")
        if key.device != self._keys.device or value.device != self._values.device:
            raise ValueError("key/value device must match the cache device")
        if key.dtype != self._keys.dtype or value.dtype != self._values.dtype:
            raise ValueError("key/value dtype must match the cache dtype")
        if key.size(2) == 0:
            raise ValueError("key/value sequence must not be empty")

        starts = self.layer_lengths(layer_idx, requests)
        chunk_length = key.size(2)
        ends = tuple(start + chunk_length for start in starts)
        for request_id, end in zip(requests, ends):
            allocated_tokens = len(self.allocator.page_table(request_id)) * self.page_size
            if end > allocated_tokens:
                raise PageAllocationError(
                    f"request {request_id!r} needs {end} token slots but only "
                    f"{allocated_tokens} were allocated"
                )

        tables = tuple(self.allocator.page_table(request_id) for request_id in requests)
        max_blocks = max(map(len, tables))
        block_table = torch.full(
            (len(requests), max_blocks),
            -1,
            dtype=torch.int32,
            device=key.device,
        )
        for batch_index, table in enumerate(tables):
            block_table[batch_index, : len(table)] = torch.tensor(
                table,
                dtype=torch.int32,
                device=key.device,
            )

        return PagedKVAppendPlan(
            layer_idx=layer_idx,
            request_ids=requests,
            starts=starts,
            ends=ends,
            key_cache=self._keys[layer_idx],
            value_cache=self._values[layer_idx],
            block_table=block_table,
            cache_seqlens=torch.tensor(starts, dtype=torch.int32, device=key.device),
        )

    def commit_append(self, plan: PagedKVAppendPlan) -> None:
        """Advance logical lengths after a backend has written every K/V item."""
        self._validate_layer(plan.layer_idx)
        requests = self._validate_requests(plan.request_ids)
        if self.layer_lengths(plan.layer_idx, requests) != plan.starts:
            raise RuntimeError("paged KV append plan is stale")
        for request_id, end in zip(requests, plan.ends):
            self._lengths[request_id][plan.layer_idx] = end

    def release_request(self, request_id: str) -> tuple[int, ...]:
        self._validate_request(request_id)
        self._lengths.pop(request_id)
        return self.allocator.release(request_id)

    def check_integrity(self) -> None:
        self.allocator.check_integrity()
        if set(self._lengths) != set(self.allocator.request_ids):
            raise RuntimeError("cache lengths and allocator requests disagree")
        for request_id, layer_lengths in self._lengths.items():
            if len(layer_lengths) != self.num_layers:
                raise RuntimeError("cache request has the wrong number of layer lengths")
            allocated_tokens = len(self.allocator.page_table(request_id)) * self.page_size
            if any(not 0 <= length <= allocated_tokens for length in layer_lengths):
                raise RuntimeError("cache length exceeds allocated physical pages")

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")

    def _validate_request(self, request_id: str) -> None:
        if request_id not in self._lengths:
            raise KeyError(f"unknown paged-cache request: {request_id}")

    def _validate_requests(self, request_ids: Sequence[str]) -> tuple[str, ...]:
        requests = tuple(request_ids)
        if not requests:
            raise ValueError("request_ids must not be empty")
        for request_id in requests:
            self._validate_request(request_id)
        return requests
