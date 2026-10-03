from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from itertools import count

import torch
import torch.distributed as dist

from mysglang.cache import (
    PagedKVBatch,
    PagedKVCache,
    PagedKVDecodeBuffer,
    RecurrentDecodeBuffer,
    RecurrentStateBatch,
)

from .qwen3 import Qwen3ForCausalLM

_RUNNER_IDS = count()


@dataclass
class _DecodeBucket:
    metadata: PagedKVDecodeBuffer
    input_ids: torch.Tensor
    state: RecurrentDecodeBuffer | None = None
    # MTP 图额外需要固定地址的目标 hidden 输入。
    input_hidden: torch.Tensor | None = None
    # verify 之后的 DeltaNet 掩码提交也按 bucket 捕获；接受长度写入固定地址 buffer。
    commit_lengths: torch.Tensor | None = None
    commit_graph: torch.cuda.CUDAGraph | None = None
    graph: torch.cuda.CUDAGraph | None = None
    output_token_ids: torch.Tensor | None = None
    output_hidden: torch.Tensor | None = None
    output_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class DecodeCudaGraphStats:
    captures: int
    replays: int


@dataclass(frozen=True)
class GraphDecodeResult:
    """Greedy tokens for every input position (``[batch, append_length]``).

    ``logits`` (graph-owned, valid until the next replay) lets the scheduler sample or
    apply penalties outside the graph; ``hidden`` feeds the MTP head.
    """

    token_ids: tuple[tuple[int, ...], ...]
    hidden: torch.Tensor | None
    state_batch: RecurrentStateBatch | None
    logits: torch.Tensor | None = None


class DecodeCudaGraphRunner:
    """Capture exact-size greedy Decode, including NCCL TP and static-shape MoE.

    ``append_length > 1`` captures speculative verification instead: every request feeds
    ``1 + draft_length`` tokens and the graph returns the greedy token at each position.
    Hybrid models read DeltaNet states through fixed slot buffers; verification runs them
    in log mode so the scheduler can replay only the accepted prefix after the replay.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        cache: PagedKVCache,
        batch_sizes: tuple[int, ...],
        *,
        max_blocks: int,
        warmup_steps: int = 2,
        append_length: int = 1,
        return_hidden: bool = False,
        pad_to_bucket: bool = False,
    ) -> None:
        parameter = model.embed_tokens.weight
        if parameter.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Decode CUDA Graph requires a CUDA model")
        if not getattr(model.attention_backend, "supports_cuda_graph", False):
            raise RuntimeError(
                "Decode CUDA Graph requires a graph-capable attention backend (flash / flashinfer)"
            )
        # FlashInfer 在图外按每批 metadata 规划，需要知道本 rank 的 query 头数。
        self._graph_owner = next(_RUNNER_IDS)  # 不用 id(self)：runner 回收后 id 可能被复用
        self._num_qo_heads = next(
            layer.self_attn.num_heads for layer in model.layers if layer.self_attn is not None
        )
        if model.config.is_moe and model.moe_dispatch_backend != "triton_grouped":
            raise RuntimeError(
                "MoE Decode CUDA Graph requires --moe-dispatch triton_grouped"
            )
        if model.tensor_parallel.enabled:
            backend = str(dist.get_backend(model.tensor_parallel.process_group))
            if backend != "nccl":
                raise RuntimeError("TP Decode CUDA Graph requires an NCCL model process group")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        self.model = model
        self.cache = cache
        self.append_length = append_length
        self.return_hidden = return_hidden
        self._batch_sizes = frozenset(batch_sizes)
        # 补齐：实际 batch 用第一个不小于它的桶，多出的行是只写垃圾页 / 垃圾格子的假请求。
        self._sorted_sizes = tuple(sorted(batch_sizes))
        self._pad_to_bucket = pad_to_bucket
        if pad_to_bucket and getattr(cache, "scratch_page", None) is None:
            raise ValueError("padded Decode CUDA Graphs require a cache with a scratch page")
        self._warmup_steps = warmup_steps
        self._recurrent = getattr(cache, "recurrent", None)
        speculative = append_length > 1

        self._buckets = {
            size: _DecodeBucket(
                metadata=cache.allocate_decode_buffer(
                    size,
                    max_blocks=max_blocks,
                    append_length=append_length,
                ),
                input_ids=torch.empty(
                    (size, append_length), dtype=torch.long, device=parameter.device
                ),
                state=(
                    None
                    if self._recurrent is None
                    else self._recurrent.allocate_decode_buffer(
                        size,
                        append_length=append_length,
                        speculative=speculative,
                    )
                ),
            )
            for size in batch_sizes
        }

        self._captures = 0
        self._replays = 0

    @property
    def stats(self) -> DecodeCudaGraphStats:
        return DecodeCudaGraphStats(captures=self._captures, replays=self._replays)

    def supports(self, batch_size: int) -> bool:
        return self.bucket_size(batch_size) is not None

    def bucket_size(self, batch_size: int) -> int | None:
        """这个 batch 用哪个桶；None 表示没有合适的桶，退回 eager。"""
        if not self._pad_to_bucket:
            return batch_size if batch_size in self._batch_sizes else None
        # self._sorted_sizes 是从小到大排好的桶大小，例如 (1, 2, 4, 8)；
        # 二分找第一个 >= batch_size 的（相当于 C++ std::lower_bound）。
        index = bisect_left(self._sorted_sizes, batch_size)
        return self._sorted_sizes[index] if index < len(self._sorted_sizes) else None

    def close(self) -> None:
        """Release captured graphs before their NCCL process group is destroyed."""
        if not any(bucket.graph is not None for bucket in self._buckets.values()):
            return
        device = next(iter(self._buckets.values())).input_ids.device
        torch.cuda.synchronize(device)
        for bucket in self._buckets.values():
            for graph in (bucket.graph, bucket.commit_graph):
                if graph is not None:
                    graph.reset()
            bucket.graph = None
            bucket.commit_graph = None
            bucket.output_token_ids = None
            bucket.output_hidden = None
            bucket.output_logits = None
        torch.cuda.synchronize(device)
        release = getattr(self.model.attention_backend, "release_graphs", None)
        if release is not None:
            release(self._graph_owner)

    @torch.inference_mode()
    def run(
        self,
        input_token_ids: tuple[int, ...],
        request_ids: tuple[str, ...],
    ) -> tuple[int, ...]:
        """Plain greedy Decode: one next token per request."""
        result = self.run_positions(input_token_ids, request_ids)
        return tuple(tokens[-1] for tokens in result.token_ids)

    @torch.inference_mode()
    def capture_all(self) -> None:
        """启动时预录所有桶：每个桶用一个全是假请求的 batch 录一次，避免服务中途懒捕获卡顿。"""
        if not self._pad_to_bucket:
            return  # 精确大小的图（MTP 那几张）没有补齐，仍在第一次用到时录
        for size in self._sorted_sizes:
            bucket = self._buckets[size]
            if bucket.graph is not None:
                continue
            # 0 个真请求：整个桶都是只读写垃圾页 / 垃圾格子的假请求，
            # 所以不用像 run_positions 那样存档、读档 S，录完也没有长度要 commit。
            batch = self.cache.prepare_decode_batch((), bucket.metadata, commit_lengths=False)
            state_batch = (
                None
                if bucket.state is None
                else self._recurrent.prepare_decode_batch((), bucket.state)
            )
            bucket.input_ids.zero_()
            self._prepare_attention(size, batch)
            self._capture(bucket, batch, state_batch)

    @torch.inference_mode()#主函数
    def run_positions(
        self,
        input_token_ids: tuple[int, ...],
        request_ids: tuple[str, ...],
        input_hidden: torch.Tensor | None = None,
        rope_deltas: tuple[int, ...] | None = None,
    ) -> GraphDecodeResult:
        batch_size = len(request_ids)
        size = self.bucket_size(batch_size)
        if size is None:
            raise ValueError(f"no CUDA Graph bucket for Decode batch size {batch_size}")
        if len(input_token_ids) != batch_size * self.append_length:
            raise ValueError("Decode input tokens must match the bucket shape")
        padding = size - batch_size
        if padding and input_hidden is not None:
            raise ValueError("MTP graphs are captured at exact batch sizes only")

        bucket = self._buckets[size] # 取出这个 batch 大小的桶
        batch = self.cache.prepare_decode_batch( # ① 把这一步的数据复制进桶里的固定地址
            request_ids,
            bucket.metadata,
            commit_lengths=False,  # forward 里不要推进 _lengths（不幂等！）
            rope_deltas=rope_deltas,
        )
        state_batch = (
            None
            if bucket.state is None
            else self._recurrent.prepare_decode_batch(request_ids, bucket.state)
        )
        # Graph bucket 的地址保持不变，每轮只更新其中的内容。
        # 规划 attention metadata 必须在图外、捕获和每次重放之前（FlashInfer；FA2 什么都不做）。
        self._prepare_attention(size, batch)
        # 假请求输入 token 0：算出来什么都无所谓，结果不会被读。
        padded_tokens = tuple(input_token_ids) + (0,) * (padding * self.append_length)
        bucket.input_ids.copy_(
            torch.tensor(padded_tokens, dtype=torch.long).view(size, self.append_length)
        )
        if bucket.input_hidden is not None:
            if input_hidden is None:
                raise ValueError("this graph requires input hidden states")
            bucket.input_hidden.copy_(input_hidden.view_as(bucket.input_hidden))
# ② 第一次用这个桶：先录图
        if bucket.graph is None:
            # 预热会真实执行前向并推进 DeltaNet 状态；捕获后恢复，replay 才从正确状态开始。
            snapshot = (
                None if state_batch is None else self._recurrent.snapshot(state_batch.slots)
            ) # 存档：把这批请求的格子复制一份
            self._capture(bucket, batch, state_batch)#预热、捕获  # 预热 2 遍 + 录 1 遍：S 被多写了 3 次
            if snapshot is not None:
                self._recurrent.restore(state_batch.slots, snapshot)  # 读档：把格子恢复成存档时的样子

        assert bucket.graph is not None
        assert bucket.output_token_ids is not None

        # replay 直接执行捕获的 GPU 命令图，不再进入模型的 Python forward。
        bucket.graph.replay()

        # CPU 需要 token，同时这个同步点保证所有层 KV 已写完，之后才能提交逻辑长度。
        token_ids, hidden, logits = self._real_rows(bucket, batch_size)
        self.cache.commit_batch(batch) # ⑤ GPU 确定写完了，CPU 才把每个请求的 _lengths 推进一步（只做这一次）
        self._replays += 1
        return GraphDecodeResult(token_ids, hidden, state_batch, logits)

    def _prepare_attention(self, size: int, batch: PagedKVBatch) -> None:
        prepare = getattr(self.model.attention_backend, "prepare_graph", None)
        if prepare is not None:
            prepare((self._graph_owner, size), self.cache, batch, self._num_qo_heads)

    @staticmethod
    def _real_rows(bucket: _DecodeBucket, batch_size: int):
        """只取前 batch_size 行（真请求）的 token、hidden、logits；后面是假请求，扔掉。"""
        rows = bucket.output_token_ids.tolist()
        token_ids = tuple(tuple(row) for row in rows[:batch_size])
        hidden = None if bucket.output_hidden is None else bucket.output_hidden[:batch_size]
        logits = None if bucket.output_logits is None else bucket.output_logits[:batch_size]
        return token_ids, hidden, logits

    @torch.inference_mode()
    def commit_states(
        self,
        state_batch: RecurrentStateBatch,
        accepted_lengths: tuple[int, ...],
    ) -> None:
        """Replay the masked DeltaNet commit for a verify that ran through this graph."""
        bucket = self._buckets[len(accepted_lengths)]
        if bucket.commit_lengths is None:
            bucket.commit_lengths = torch.zeros(
                len(accepted_lengths), dtype=torch.long, device=bucket.input_ids.device
            )
        bucket.commit_lengths.copy_(torch.tensor(accepted_lengths, dtype=torch.long))
        if bucket.commit_graph is None:
            device = bucket.input_ids.device
            stream = torch.cuda.Stream(device=device)
            current = torch.cuda.current_stream(device)
            # 提交会写 slot；预热前后快照/恢复，真正的提交只由下面的 replay 完成。
            snapshot = self._recurrent.snapshot(state_batch.slots)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                self.model.commit_speculative_states(state_batch, bucket.commit_lengths)
            stream.synchronize()
            self._recurrent.restore(state_batch.slots, snapshot)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.model.commit_speculative_states(state_batch, bucket.commit_lengths)
            current.wait_stream(stream)
            bucket.commit_graph = graph
            self._captures += 1
        bucket.commit_graph.replay()
        self._replays += 1

    def _forward(self, bucket: _DecodeBucket, batch: PagedKVBatch, state_batch):
        result = self.model.forward_prepared(
            bucket.input_ids,
            kv_cache=self.cache,
            cache_batch=batch,
            state_batch=state_batch,
            return_hidden=self.return_hidden,
        )
        logits, hidden = result if self.return_hidden else (result, None)
        return logits.argmax(dim=-1), hidden, logits

    def _capture(
        self,
        bucket: _DecodeBucket,
        batch: PagedKVBatch,
        state_batch: RecurrentStateBatch | None,
    ) -> None:
        """记录 CUDA stream 上的 kernel、内存操作和依赖，而非 Python 模型函数。"""
        device = bucket.input_ids.device
        capture_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device)
        capture_stream.wait_stream(current_stream)
        # Eager Prefill/Decode continues to compact only rank-local assignments. Fixed
        # batch * top-k slots are selected solely for the forward recorded below.
        self.model.set_moe_cuda_graph_static_dispatch(True)
        try:
            with torch.cuda.stream(capture_stream):
                # 预热阶段完成 lazy 初始化、workspace 分配和 kernel 选择。
                for _ in range(self._warmup_steps):
                    self._forward(bucket, batch, state_batch)
            capture_stream.synchronize()

            # Every TP rank must begin capturing the same NCCL collective sequence. The
            # scheduler already mirrors the batch; this barrier only aligns first capture.
            if self.model.tensor_parallel.enabled:
                device_index = device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                dist.barrier(
                    group=self.model.tensor_parallel.process_group,
                    device_ids=[device_index],
                )
                capture_stream.wait_stream(current_stream)

            # 捕获稳定的 GPU 执行流。
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                output_token_ids, output_hidden, output_logits = self._forward(
                    bucket, batch, state_batch
                )
        finally:
            self.model.set_moe_cuda_graph_static_dispatch(False)
        current_stream.wait_stream(capture_stream)
        bucket.graph = graph
        bucket.output_token_ids = output_token_ids
        bucket.output_hidden = output_hidden
        bucket.output_logits = output_logits
        self._captures += 1


class MtpCudaGraphRunner(DecodeCudaGraphRunner):
    """Capture the MTP head for fixed ``[batch, append_length]`` draft-cache shapes.

    ``append_length=1`` is one recursive draft step. ``append_length=1 + k`` is the
    catch-up after a full-width verification: every verified row is absorbed and the
    scheduler afterwards truncates the draft K/V to each request's accepted length.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        draft_cache: PagedKVCache,
        batch_sizes: tuple[int, ...],
        *,
        max_blocks: int,
        append_length: int = 1,
        warmup_steps: int = 2,
    ) -> None:
        if model.mtp is None:
            raise ValueError("MTP CUDA Graph requires a model loaded with enable_mtp")
        super().__init__(
            model,
            draft_cache,
            batch_sizes,
            max_blocks=max_blocks,
            warmup_steps=warmup_steps,
            append_length=append_length,
            return_hidden=True,
        )
        hidden_size = model.config.hidden_size
        dtype = model.embed_tokens.weight.dtype
        for size, bucket in self._buckets.items():
            bucket.state = None
            bucket.input_hidden = torch.zeros(
                (size, append_length, hidden_size),
                dtype=dtype,
                device=bucket.input_ids.device,
            )

    def _forward(self, bucket: _DecodeBucket, batch: PagedKVBatch, state_batch):
        logits, hidden = self.model.forward_mtp(
            bucket.input_ids,
            bucket.input_hidden,
            kv_cache=self.cache,
            cache_batch=batch,
        )
        return logits.argmax(dim=-1), hidden, logits
