from __future__ import annotations

from dataclasses import dataclass

import torch

from mysglang.cache import PagedKVBatch, PagedKVCache, PagedKVDecodeBuffer

from .attention import FlashAttentionBackend
from .qwen3 import Qwen3ForCausalLM


@dataclass
class _DecodeBucket:
    metadata: PagedKVDecodeBuffer
    input_ids: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    output_token_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class DecodeCudaGraphStats:
    captures: int
    replays: int


class DecodeCudaGraphRunner:
    """Capture and replay exact-size, greedy, one-token Decode forwards."""

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        cache: PagedKVCache,
        batch_sizes: tuple[int, ...],
        *,
        max_blocks: int,
        warmup_steps: int = 2,
    ) -> None:
        parameter = next(model.parameters())
        if parameter.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Decode CUDA Graph requires a CUDA model")
        if not isinstance(model.attention_backend, FlashAttentionBackend):
            raise RuntimeError("Decode CUDA Graph requires FlashAttentionBackend")
        if model.config.is_moe:
            raise RuntimeError("Decode CUDA Graph for dynamic MoE dispatch is not implemented")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        self.model = model
        self.cache = cache
        self._batch_sizes = frozenset(batch_sizes)
        self._warmup_steps = warmup_steps

        self._buckets = {
            size: _DecodeBucket(
                metadata=cache.allocate_decode_buffer(size, max_blocks=max_blocks),
                input_ids=torch.empty((size, 1), dtype=torch.long, device=parameter.device),
            )
            for size in batch_sizes
        }

        self._captures = 0
        self._replays = 0

    @property
    def stats(self) -> DecodeCudaGraphStats:
        return DecodeCudaGraphStats(captures=self._captures, replays=self._replays)

    def supports(self, batch_size: int) -> bool:
        return batch_size in self._batch_sizes

    @torch.inference_mode()
    def run(
        self,
        input_token_ids: tuple[int, ...],
        request_ids: tuple[str, ...],
    ) -> tuple[int, ...]:
        batch_size = len(request_ids)
        if batch_size not in self._buckets:
            raise ValueError(f"no CUDA Graph bucket for Decode batch size {batch_size}")
        if len(input_token_ids) != batch_size:
            raise ValueError("Decode input tokens must match the request batch size")

        bucket = self._buckets[batch_size]
        batch = self.cache.prepare_decode_batch(
            request_ids,
            bucket.metadata,
            commit_lengths=False,
        )
        # Graph bucket 的地址保持不变，每轮只更新其中的内容。
        bucket.input_ids[:, 0].copy_(torch.tensor(input_token_ids, dtype=torch.long))

        if bucket.graph is None:
            self._capture(bucket, batch)

        assert bucket.graph is not None
        assert bucket.output_token_ids is not None

        # replay 直接执行捕获的 GPU 命令图，不再进入模型的 Python forward。
        bucket.graph.replay()

        # CPU 需要 token，同时这个同步点保证所有层 KV 已写完，之后才能提交逻辑长度。
        output_token_ids = tuple(int(token) for token in bucket.output_token_ids.tolist())
        self.cache.commit_batch(batch)
        self._replays += 1
        return output_token_ids

    def _capture(self, bucket: _DecodeBucket, batch: PagedKVBatch) -> None:
        """记录 CUDA stream 上的 kernel、内存操作和依赖，而非 Python 模型函数。"""
        device = bucket.input_ids.device
        capture_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            # 预热阶段完成 lazy 初始化、workspace 分配和 kernel 选择。
            for _ in range(self._warmup_steps):
                logits = self.model.forward_prepared(
                    bucket.input_ids,
                    kv_cache=self.cache,
                    cache_batch=batch,
                )
                logits[:, -1].argmax(dim=-1)
        capture_stream.synchronize()

        # 捕获稳定的 GPU 执行流。
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            logits = self.model.forward_prepared(
                bucket.input_ids,
                kv_cache=self.cache,
                cache_batch=batch,
            )
            output_token_ids = logits[:, -1].argmax(dim=-1)
        current_stream.wait_stream(capture_stream)
        bucket.graph = graph
        bucket.output_token_ids = output_token_ids
        self._captures += 1
