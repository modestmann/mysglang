from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .paged import PagedKVCompanion
from .radix import RadixPagedKVCache
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
        **kwargs,
    ) -> None:
        # 当前 radix 只记录 token -> K/V 页；DeltaNet 状态无法从页面恢复，所以混合模型关闭复用。
        kwargs.setdefault("enable_prefix_cache", recurrent is None)
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
            self.recurrent.reserve(request_id)
        if self.draft is not None:
            # 草稿层与目标层写同一批页面；命中的前缀页也已含草稿 K/V。
            self.draft.attach_request(request_id, matched)
        return matched

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
        if self.draft is not None:
            self.draft.check_integrity()
            for request_id in active:
                draft_length = self.draft.lengths((request_id,))[0]
                if draft_length > self.lengths((request_id,))[0]:
                    raise RuntimeError("draft K/V is ahead of the target cache between steps")
