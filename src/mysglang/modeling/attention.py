from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from mysglang.cache import PagedKVBatch, PagedKVCache


class AttentionBackend(ABC):
    """Execution boundary between model projections and an attention kernel."""

    name: str
    # 能否被 Decode CUDA Graph 捕获（kernel 只读固定地址的 metadata）。
    supports_cuda_graph = False

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
    supports_cuda_graph = True
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


def paged_kv_csr(
    page_tables: list[list[int]] | tuple[tuple[int, ...], ...],
    lengths: tuple[int, ...] | list[int],
    page_size: int,
) -> tuple[list[int], list[int], list[int]]:
    """把每个请求的页表压成 FlashInfer 用的 CSR 三元组。

    ``indices`` 是所有请求实际用到的页首尾相接；第 i 个请求的页在
    ``indices[indptr[i]:indptr[i + 1]]``；``last_page_len`` 是最后一页用了几个位置（1..page_size）。
    """
    indptr = [0]
    indices: list[int] = []
    last_page_len: list[int] = []
    for table, length in zip(page_tables, lengths):
        num_pages = (length + page_size - 1) // page_size  # 向上取整
        indices.extend(table[:num_pages])  # 只取当前请求真正用到的页
        indptr.append(indptr[-1] + num_pages)
        last_page_len.append((length - 1) % page_size + 1)  # 写满时是 page_size，不是 0
    return indptr, indices, last_page_len


class FlashInferBackend(AttentionBackend):
    """FlashInfer backend: any page size, CSR page metadata planned once per forward.

    K/V are written through the shared ``slot_mapping`` (``write_append``); attention reads the
    paged pool via ``indptr`` / ``indices`` / ``last_page_len``. Uniform one-token Decode uses the
    decode kernel, everything else (prefill, mixed, speculative verify) the prefill kernel with
    causal masking aligned to each request's end. CUDA Graph replays use per-bucket wrappers with
    fixed metadata buffers that ``prepare_graph`` re-plans outside the graph before each replay.
    """

    name = "flashinfer"
    supports_cuda_graph = True
    workspace_bytes = 128 << 20  # 下限；实际按 split-KV 临时缓冲的最坏情况放大
    # 非 tensor-core decode kernel 只为这些 GQA 组大小编译（FlashInfer DISPATCH_GQA_GROUP_SIZE）
    decode_group_sizes = frozenset((1, 2, 3, 4, 6, 8))

    def __init__(self) -> None:
        try:
            import flashinfer
        except ImportError as exc:
            raise RuntimeError(
                "FlashInferBackend requires the 'flashinfer-python' package"
            ) from exc
        self._flashinfer = flashinfer
        self._workspace: torch.Tensor | None = None
        self._prefill = None
        self._decode: dict[bool, object] = {}
        self._graph_wrappers: dict[object, object] = {}
        # 一次 forward 的所有层共用同一个 cache_batch：第一层 plan，之后各层只 run。
        self._planned_batch: PagedKVBatch | None = None
        self._planned_wrapper = None

    def validate_cache(self, cache: PagedKVCache) -> None:
        if cache.page_size <= 0:
            raise ValueError("FlashInfer paged KV requires a positive page_size")

    def _workspace_on(
        self, device: torch.device, num_qo_heads: int, num_kv_heads: int, head_dim: int
    ) -> torch.Tensor:
        """所有 wrapper（eager 与各图桶）共用一块 float workspace，只分配一次。

        不能在这里重建：已录的 CUDA Graph 固化了 workspace 和各 wrapper 缓冲区的地址，
        重建等于把图引用的显存还给分配器（use-after-free）。大小按 split-KV 临时缓冲
        tmp_v / tmp_s 的上界算：最多 max(2·SM 数 / kv 头数, 64) 个块，每块每个 q 头
        cta_tile_q·(head_dim + 1) 个 float（head_dim < 256 时 cta_tile_q 最大 128，否则 64）。
        """
        if self._workspace is None:
            num_sm = torch.cuda.get_device_properties(device).multi_processor_count
            blocks = max(2 * num_sm // num_kv_heads, 64)
            per_head = (128 if head_dim < 256 else 64) * (head_dim + 1)
            worst = num_qo_heads * blocks * per_head * 4 + (16 << 20)
            size = max(self.workspace_bytes, worst)
            self._workspace = torch.empty(size, dtype=torch.uint8, device=device)
        elif self._workspace.device != device:
            raise RuntimeError(
                f"FlashInferBackend is bound to {self._workspace.device}, got {device}"
            )
        return self._workspace

    def _use_tensor_cores(self, num_qo_heads: int, num_kv_heads: int) -> bool:
        return num_qo_heads // num_kv_heads not in self.decode_group_sizes

    def release_graphs(self, owner: object) -> None:
        """图被 reset 之后再调用：丢掉该 runner 各桶的 wrapper 及其固定缓冲区。"""
        for key in [key for key in self._graph_wrappers if key[0] == owner]:
            wrapper = self._graph_wrappers.pop(key)
            if self._planned_wrapper is wrapper:
                self._planned_batch = None
                self._planned_wrapper = None

    @staticmethod
    def _csr(kv_cache: PagedKVCache, batch: PagedKVBatch):
        tables = [kv_cache.allocator.page_table(request_id) for request_id in batch.request_ids]
        lengths = list(batch.ends)
        width = batch.append_lengths[-1] if batch.append_lengths else 1
        for _ in range(batch.padding_rows):  # CUDA Graph 补齐的假请求：只看垃圾页
            tables.append((kv_cache.scratch_page,))
            lengths.append(width)
        return paged_kv_csr(tables, lengths, kv_cache.page_size)

    def _plan(
        self, wrapper, decode: bool, kv_cache, batch, num_qo_heads: int, dtype
    ) -> None:
        indptr, indices, last_page_len = self._csr(kv_cache, batch)
        device = kv_cache._keys.device
        as_int = lambda values: torch.tensor(values, dtype=torch.int32, device=device)  # noqa: E731
        common = dict(
            num_qo_heads=num_qo_heads,
            num_kv_heads=kv_cache.num_kv_heads,
            page_size=kv_cache.page_size,
            q_data_type=dtype,
            kv_data_type=kv_cache._keys.dtype,
        )
        if decode:
            wrapper.plan(
                as_int(indptr),
                as_int(indices),
                as_int(last_page_len),
                head_dim=kv_cache.head_dim,
                **common,
            )
        else:
            qo_indptr = [0]
            for length in batch.append_lengths:
                qo_indptr.append(qo_indptr[-1] + length)
            wrapper.plan(
                as_int(qo_indptr),
                as_int(indptr),
                as_int(indices),
                as_int(last_page_len),
                head_dim_qk=kv_cache.head_dim,
                causal=True,
                **common,
            )

    def prepare_graph(
        self, key: object, kv_cache: PagedKVCache, batch: PagedKVBatch, num_qo_heads: int
    ) -> None:
        """Graph 捕获 / 重放前（图外）为这一批规划 metadata，写进该桶固定地址的缓冲区。

        ``key`` 是 ``(runner 编号, 桶大小)``：同一个桶每次都必须拿回同一个 wrapper，
        因为图里的 kernel 读的是它录制时那几块缓冲区的地址。不关 split-KV：图模式下
        FlashInfer 的切分与网格只由 batch 大小决定，多余的 block 由 block_valid_mask 屏蔽
        （与 SGLang 一致）；关掉反而会让 tensor-core / prefill 路径失去这道屏蔽。
        """
        decode = batch.uniform_append_length == 1
        wrapper = self._graph_wrappers.get(key)
        if wrapper is None:
            device = kv_cache._keys.device
            workspace = self._workspace_on(
                device, num_qo_heads, kv_cache.num_kv_heads, kv_cache.head_dim
            )
            rows = batch.batch_size
            max_pages = rows * batch.block_table.size(1)
            buffers = dict(
                indptr=torch.zeros(rows + 1, dtype=torch.int32, device=device),
                indices=torch.zeros(max_pages, dtype=torch.int32, device=device),
                last=torch.zeros(rows, dtype=torch.int32, device=device),
            )
            if decode:
                # 与 eager decode 同一个 kernel；组大小不在非 TC kernel 支持列表里才用 tensor core。
                wrapper = self._flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=self._use_tensor_cores(num_qo_heads, kv_cache.num_kv_heads),
                    paged_kv_indptr_buffer=buffers["indptr"],
                    paged_kv_indices_buffer=buffers["indices"],
                    paged_kv_last_page_len_buffer=buffers["last"],
                )
            else:
                wrapper = self._flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=torch.zeros(rows + 1, dtype=torch.int32, device=device),
                    paged_kv_indptr_buf=buffers["indptr"],
                    paged_kv_indices_buf=buffers["indices"],
                    paged_kv_last_page_len_buf=buffers["last"],
                )
            self._graph_wrappers[key] = wrapper
        self._plan(wrapper, decode, kv_cache, batch, num_qo_heads, kv_cache._keys.dtype)
        self._planned_batch = batch
        self._planned_wrapper = wrapper

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
        if kv_cache is None or cache_batch is None:
            raise ValueError("FlashInferBackend requires a PagedKVCache and its batch metadata")
        dense = query.ndim == 4
        if dense:  # [B, H, S, D] -> [B * S, H, D]
            batch_size, heads, length, head_dim = query.shape
            flat_query = query.transpose(1, 2).reshape(batch_size * length, heads, head_dim)
        else:
            flat_query = query
        plan = kv_cache.prepare_append(layer_idx, key, value, cache_batch)
        kv_cache.write_append(plan, key, value)
        if self._planned_batch is not cache_batch:
            num_qo_heads = flat_query.size(1)
            self._workspace_on(query.device, num_qo_heads, kv_cache.num_kv_heads, kv_cache.head_dim)
            decode = dense and cache_batch.uniform_append_length == 1
            if decode:
                tensor_cores = self._use_tensor_cores(num_qo_heads, kv_cache.num_kv_heads)
                if tensor_cores not in self._decode:
                    wrapper_type = self._flashinfer.BatchDecodeWithPagedKVCacheWrapper
                    self._decode[tensor_cores] = wrapper_type(
                        self._workspace, "NHD", use_tensor_cores=tensor_cores
                    )
                wrapper = self._decode[tensor_cores]
            else:
                if self._prefill is None:
                    self._prefill = self._flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                        self._workspace, "NHD"
                    )
                wrapper = self._prefill
            self._plan(wrapper, decode, kv_cache, cache_batch, num_qo_heads, query.dtype)
            self._planned_batch = cache_batch
            self._planned_wrapper = wrapper
        output = self._planned_wrapper.run(flat_query, (plan.key_cache, plan.value_cache))
        if cache_batch.commit_lengths:
            kv_cache.commit_append(plan)
        if dense:
            return output.view(batch_size, length, heads, head_dim).transpose(1, 2)
        return output
