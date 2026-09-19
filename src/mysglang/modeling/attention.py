from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from mysglang.cache import PagedKVBatch, PagedKVCache


class AttentionBackend(ABC):
    """Execution boundary between model projections and an attention kernel."""

    name: str

    def validate_cache(self, cache: PagedKVCache) -> None:
        """Reject cache layouts unsupported by this backend before serving starts."""

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        layer_idx: int,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
    ) -> torch.Tensor:
        """Return dense rank-4 or packed rank-3 attention output."""


class TorchAttentionBackend(AttentionBackend):
    """Readable correctness backend that gathers paged K/V into dense tensors."""

    name = "torch"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        layer_idx: int,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
    ) -> torch.Tensor:
        attention_mask = None
        plan = None
        if kv_cache is not None:
            if cache_batch is None:
                raise ValueError("cache_batch is required with PagedKVCache")
            plan = kv_cache.prepare_append(
                layer_idx,
                key,
                value,
                cache_batch,
            )
            if query.ndim == 3:
                keys, values = kv_cache.stage_packed_append(plan, key, value)
                output = self._packed_attention(query, keys, values, cache_batch)
                if cache_batch.commit_lengths:
                    kv_cache.commit_append(plan)
                return output
            key, value, attention_mask = kv_cache.stage_append(plan, key, value)

        repeats = query.size(1) // key.size(1)
        key = key if repeats == 1 else key.repeat_interleave(repeats, dim=1)
        value = value if repeats == 1 else value.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=kv_cache is None,
        )
        if plan is not None and cache_batch.commit_lengths:
            kv_cache.commit_append(plan)
        return output

    @staticmethod
    def _packed_attention(
        query: torch.Tensor,
        keys: tuple[torch.Tensor, ...],
        values: tuple[torch.Tensor, ...],
        batch: PagedKVBatch,
    ) -> torch.Tensor:
        outputs = []
        query_offset = 0
        for start, end, query_length, key, value in zip(
            batch.starts,
            batch.ends,
            batch.append_lengths,
            keys,
            values,
        ):
            request_query = query[query_offset : query_offset + query_length]
            request_query = request_query.transpose(0, 1).unsqueeze(0)
            request_key = key.transpose(0, 1).unsqueeze(0)
            request_value = value.transpose(0, 1).unsqueeze(0)
            repeats = request_query.size(1) // request_key.size(1)
            if repeats != 1:
                request_key = request_key.repeat_interleave(repeats, dim=1)
                request_value = request_value.repeat_interleave(repeats, dim=1)

            query_positions = torch.arange(start, end, device=query.device)[:, None]
            key_positions = torch.arange(end, device=query.device)[None, :]
            attention_mask = key_positions <= query_positions
            request_output = F.scaled_dot_product_attention(
                request_query,
                request_key,
                request_value,
                attn_mask=attention_mask[None, None, :, :],
            )
            outputs.append(request_output.squeeze(0).transpose(0, 1))
            query_offset += query_length
        return torch.cat(outputs, dim=0)


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention 2 backend with direct paged-KV reads and in-place writes."""

    name = "flash-attn-2"
    page_size_multiple = 256

    def __init__(self) -> None:
        try:
            from flash_attn import (
                flash_attn_func,
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
            )
        except ImportError as exc:
            raise RuntimeError("FlashAttentionBackend requires the 'flash-attn' package") from exc
        self._flash_attn_func = flash_attn_func
        self._flash_attn_varlen_func = flash_attn_varlen_func
        self._flash_attn_with_kvcache = flash_attn_with_kvcache

    def validate_cache(self, cache: PagedKVCache) -> None:
        if cache.page_size % self.page_size_multiple:
            raise ValueError(
                "FlashAttention paged KV requires page_size to be a multiple of "
                f"{self.page_size_multiple}"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        layer_idx: int,
        kv_cache: PagedKVCache | None,
        cache_batch: PagedKVBatch | None,
    ) -> torch.Tensor:
        self._validate_inputs(query, key, value)
        packed = query.ndim == 3
        query_fa = query if packed else query.transpose(1, 2)
        key_fa = key if packed else key.transpose(1, 2)
        value_fa = value if packed else value.transpose(1, 2)

        if kv_cache is None:
            if packed:
                raise ValueError("packed attention requires PagedKVCache metadata")
            output = self._flash_attn_func(
                query_fa,
                key_fa,
                value_fa,
                causal=True,
            )
        else:
            if cache_batch is None:
                raise ValueError("cache_batch is required with PagedKVCache")
            self.validate_cache(kv_cache)
            plan = kv_cache.prepare_append(layer_idx, key, value, cache_batch)
            if packed:
                kv_cache.write_append(plan, key, value)
                output = self._flash_attn_varlen_func(
                    query_fa,
                    plan.key_cache,
                    plan.value_cache,
                    cache_batch.cu_seqlens_q,
                    cache_batch.cu_seqlens_k,
                    cache_batch.max_append_length,
                    cache_batch.max_end,
                    causal=True,
                    block_table=cache_batch.block_table,
                )
            else:
                output = self._flash_attn_with_kvcache(
                    query_fa,
                    plan.key_cache,
                    plan.value_cache,
                    k=key_fa,
                    v=value_fa,
                    cache_seqlens=cache_batch.cache_seqlens,
                    block_table=cache_batch.block_table,
                    causal=True,
                )
            if cache_batch.commit_lengths:
                kv_cache.commit_append(plan)

        return output if packed else output.transpose(1, 2)

    @staticmethod
    def _validate_inputs(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if query.ndim not in {3, 4} or key.ndim != query.ndim or value.ndim != query.ndim:
            raise ValueError("query, key and value must all have rank 3 or rank 4")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        compatible = query.shape[:1] + query.shape[2:] == key.shape[:1] + key.shape[2:]
        if not compatible:
            raise ValueError("query and key/value token and head_dim dimensions must match")
        if query.size(1) % key.size(1):
            raise ValueError("query heads must be divisible by key/value heads")
        if not query.is_cuda:
            raise RuntimeError("FlashAttentionBackend requires CUDA tensors")
        if key.device != query.device or value.device != query.device:
            raise ValueError("query, key and value must be on the same CUDA device")
        major, _minor = torch.cuda.get_device_capability(query.device)
        if major < 8:
            raise RuntimeError("FlashAttentionBackend requires an Ampere-or-newer GPU")
        if query.dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError("FlashAttentionBackend requires float16 or bfloat16 tensors")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise TypeError("query, key and value must use the same dtype")
        if query.size(-1) % 8:
            raise ValueError("FlashAttention head_dim must be divisible by 8")
        if query.size(-1) > 256:
            raise ValueError("FlashAttention head_dim must not exceed 256")
