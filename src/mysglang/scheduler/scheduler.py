"""
每个 rank 都有一个自己的 Scheduler 实例；rank 0 的 TensorParallelScheduler
负责向其他 rank 广播命令，让这些 Scheduler 保持镜像一致。
rank 0                                      rank 1
  ──────────────────────                      ──────────────────────
  GenerationService                           不创建 Service
          │
  TensorParallelScheduler                     run_worker_loop()
          │                                          │
  广播 add / step / abort ──────────────────────────>│
          │                                          │
  本地 Scheduler 0                           本地 Scheduler 1
  本地 Radix/页表                            本地 Radix/页表
  GPU 0 的局部 KV pool                       GPU 1 的局部 KV pool
          │                                          │
          └──────── 同步进入 model forward ──────────┘
                              │
                        TP all-reduce
其他 rank 不需要这些东西，只进入：

tp_scheduler.run_worker_loop()

等待 rank 0 发命令
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from mysglang.cache import PagedKVDecodeBuffer, RadixPagedKVCache
from mysglang.core import IncrementalOutput, Request, RequestState
from mysglang.modeling.cuda_graph import DecodeCudaGraphRunner
from mysglang.modeling.qwen3 import Qwen3ForCausalLM

from .config import SchedulerConfig
from .coordination import SchedulerBatchPlan, SchedulerCoordinator
from .sampler import sample_token


@dataclass(frozen=True)
class SchedulerStep:
    index: int
    phase: str
    request_ids: tuple[str, ...]
    input_tokens: int
    outputs: tuple[IncrementalOutput, ...]

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)


@dataclass(frozen=True)
class SchedulerStats:
    active_requests: int
    waiting_requests: int
    running_requests: int
    max_active_requests: int
    finished_requests: int
    aborted_requests: int
    model_forwards: int
    max_prefill_batch_size: int
    max_decode_batch_size: int
    max_wait_steps: int
    prefill_input_tokens: int
    cuda_graph_captures: int
    cuda_graph_replays: int


@dataclass
class _Entry:
    request: Request
    # 请求进入 Scheduler 的 step，用来统计排队时间。
    enqueue_step: int
    # Prompt 已经 Prefill 到哪里；长 Prompt 会按 budget 分块处理。
    prefill_offset: int = 0


class Scheduler:
    """The single continuous-batching scheduler used by MySGLang.

    It owns admission, chunked prefill, batched decode and the paged/radix KV
    cache. There is deliberately no selectable legacy backend anymore.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        config: SchedulerConfig,
        *,
        coordinator: SchedulerCoordinator | None = None,
    ) -> None:
        if model.tensor_parallel.enabled and coordinator is None:
            raise RuntimeError(
                "Tensor-parallel models require the distributed worker runtime; "
                "the single-process Scheduler cannot drive collectives safely"
            )
        if model.tensor_parallel.enabled and config.decode_cuda_graph_batch_sizes:
            raise RuntimeError(
                "TP Decode CUDA Graph capture is not implemented; use eager Decode first"
            )
        self.model = model.eval()
        self.config = config
        self._coordinator = coordinator
        parameter = next(model.parameters())
        self.cache = RadixPagedKVCache.from_config(
            model.config,
            num_pages=config.num_pages,
            page_size=config.page_size,
            dtype=parameter.dtype,
            device=parameter.device,
            num_kv_heads=model.kv_cache_num_heads,
        )
        self.model.validate_cache(self.cache)
        self._device = parameter.device
        self._decode_max_blocks = min(
            self.cache.num_pages,
            self.cache.allocator.pages_for_tokens(model.config.max_position_embeddings),
        )
        self._decode_buffers: dict[int, PagedKVDecodeBuffer] = {}
        self._cuda_graph_runner = (
            DecodeCudaGraphRunner(
                self.model,
                self.cache,
                config.decode_cuda_graph_batch_sizes,
                max_blocks=self._decode_max_blocks,
            )
            if config.decode_cuda_graph_batch_sizes
            else None
        )
        self._waiting: deque[_Entry] = deque()
        self._prefilling: dict[str, _Entry] = {}
        self._running: dict[str, _Entry] = {}
        # 所有未结束请求的总索引：waiting + prefilling + running。
        self._entries: dict[str, _Entry] = {}
        self._sampling_generators: dict[str, torch.Generator] = {}
        self._step_index = 0
        self._max_active_requests = 0
        self._finished_requests = 0
        self._aborted_requests = 0
        self._model_forwards = 0
        self._max_prefill_batch_size = 0
        self._max_decode_batch_size = 0
        self._max_wait_steps = 0
        # 只保留累计量，不保存无限增长的逐步 history。
        self._prefill_input_tokens = 0

    @property
    def has_work(self) -> bool:
        return bool(self._entries)

    @property
    def prefill_input_tokens(self) -> int:
        return self._prefill_input_tokens

    @property
    def stats(self) -> SchedulerStats:
        graph_stats = self._cuda_graph_runner.stats if self._cuda_graph_runner else None
        return SchedulerStats(
            active_requests=len(self._entries),
            waiting_requests=len(self._waiting) + len(self._prefilling),
            running_requests=len(self._running),
            max_active_requests=self._max_active_requests,
            finished_requests=self._finished_requests,
            aborted_requests=self._aborted_requests,
            model_forwards=self._model_forwards,
            max_prefill_batch_size=self._max_prefill_batch_size,
            max_decode_batch_size=self._max_decode_batch_size,
            max_wait_steps=self._max_wait_steps,
            prefill_input_tokens=self._prefill_input_tokens,
            cuda_graph_captures=graph_stats.captures if graph_stats else 0,
            cuda_graph_replays=graph_stats.replays if graph_stats else 0,
        )

    def validate(self, request: Request) -> None:
        """Validate a request without enqueueing or allocating cache pages."""
        if request.state is not RequestState.WAITING:
            raise ValueError("scheduler only accepts requests in WAITING state")
        total_length = self._max_request_tokens(request)
        if total_length > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus max_new_tokens exceeds max_position_embeddings")
        if any(token_id >= self.model.config.vocab_size for token_id in request.prompt_token_ids):
            raise ValueError("prompt token IDs must be smaller than model vocab_size")
        eos_token_id = request.sampling_params.eos_token_id
        if eos_token_id is not None and eos_token_id >= self.model.config.vocab_size:
            raise ValueError("eos_token_id must be smaller than model vocab_size")
        needed_pages = self.cache.allocator.pages_for_tokens(total_length)
        if needed_pages > self.cache.num_pages:
            raise ValueError("request exceeds total paged KV cache capacity")

    def add(self, request: Request) -> None:
        self.validate(request)
        if request.request_id in self._entries:
            raise ValueError(f"duplicate request_id: {request.request_id}")

        # add 只进入逻辑等待队列；真正的 reservation/page allocation 在被调度时发生。
        entry = _Entry(request=request, enqueue_step=self._step_index)
        self._waiting.append(entry)
        self._entries[request.request_id] = entry
        self._max_active_requests = max(self._max_active_requests, len(self._entries))

    def abort(self, request_id: str) -> IncrementalOutput | None:
        entry = self._entries.get(request_id)
        if entry is None or entry.request.is_terminal:
            return None
        if request_id in self._prefilling:
            self._prefilling.pop(request_id)
        else:
            try:
                self._waiting.remove(entry)
            except ValueError:
                pass
        self._running.pop(request_id, None)
        event = entry.request.abort()
        self._release(entry, finished=False)
        return event

    @torch.inference_mode()
    def step(self) -> SchedulerStep | None:
        """Include every running decode in a prefill step, or run dense decode."""
        if self._can_prefill():
            result = self._prefill_step()
        elif self._running:
            result = self._decode_step()
        else:
            return None

        self._step_index += 1
        return result

    def reset_prefix_cache(self) -> None:
        self.cache.reset_prefix_cache()

    def check_integrity(self) -> None:
        self.cache.check_integrity()
        cache_requests = set(self.cache.allocator.request_ids)
        scheduled_requests = set(self._running) | set(self._prefilling)
        if cache_requests != scheduled_requests:
            raise RuntimeError("scheduler and paged cache disagree about active owners")
        waiting_ids = {entry.request.request_id for entry in self._waiting}
        if waiting_ids & cache_requests:
            raise RuntimeError("a waiting request already owns physical cache pages")
        if set(self._entries) != waiting_ids | scheduled_requests:
            raise RuntimeError("scheduler request indexes disagree")
        if any(entry.request.state is not RequestState.WAITING for entry in self._waiting):
            raise RuntimeError("waiting queue contains a request in the wrong state")
        if any(
            entry.request.state is not RequestState.PREFILL for entry in self._prefilling.values()
        ):
            raise RuntimeError("prefill set contains a request in the wrong state")
        if any(
            entry.request.state is not RequestState.DECODING for entry in self._running.values()
        ):
            raise RuntimeError("running map contains a request in the wrong state")

    def _can_prefill(self) -> bool:
        if self._prefilling:
            return True
        if not self._waiting:
            return False
        if self.cache.allocator.stats.request_count >= self.config.max_running_requests:
            return False
        request = self._waiting[0].request
        return self.cache.can_reserve_request(
            self._max_request_tokens(request), request.prompt_token_ids
        )

    def _prefill_step(self) -> SchedulerStep:
        # 只快照本轮开始时的 Decode；刚完成 Prompt 的请求下轮才输入首个输出 token。
        decoding = list(self._running.values())
        self._admit_prefill_requests()
        entries = list(self._prefilling.values())
        selected = entries[: min(len(entries), self.config.prefill_token_budget)]
        remaining_budget = self.config.prefill_token_budget
        chunks: list[tuple[_Entry, int, int]] = []
        for index, entry in enumerate(selected):
            requests_left = len(selected) - index
            fair_share = max(1, remaining_budget // requests_left)
            start = entry.prefill_offset
            end = min(start + fair_share, len(entry.request.prompt_token_ids))
            chunks.append((entry, start, end))
            remaining_budget -= end - start

        decode_ids = tuple(entry.request.request_id for entry in decoding)
        request_ids = decode_ids + tuple(entry.request.request_id for entry, _, _ in chunks)
        append_lengths = (1,) * len(decoding) + tuple(end - start for _, start, end in chunks)
        packed_tokens = [entry.request.last_output_token_id for entry in decoding]
        if decoding:
            for request_id, length in zip(decode_ids, self.cache.lengths(decode_ids)):
                self.cache.ensure_capacity(request_id, length + 1)
        for entry, start, end in chunks:
            request_id = entry.request.request_id
            # reservation 是容量承诺；这里才从 free pages 取物理页并写入请求页表。
            self.cache.ensure_capacity(request_id, end)
            packed_tokens.extend(entry.request.prompt_token_ids[start:end])

        input_ids = torch.tensor(
            packed_tokens,
            dtype=torch.long,
            device=self._device,
        )
        # Decode 每个位置都采样；Prefill 仅在完整 Prompt 结束时采样。
        logits_indices = list(range(len(decoding)))
        packed_offset = len(decoding)
        for entry, start, end in chunks:
            packed_offset += end - start
            if end == len(entry.request.prompt_token_ids):
                logits_indices.append(packed_offset - 1)
        phase = "mixed" if decoding else "prefill"
        self._validate_batch_plan(
            phase=phase,
            execution="packed",
            request_ids=request_ids,
            input_token_ids=tuple(packed_tokens),
            append_lengths=append_lengths,
            logits_indices=tuple(logits_indices),
        )
        logits = self.model.forward_packed(
            input_ids,
            kv_cache=self.cache,
            cache_request_ids=request_ids,
            append_lengths=append_lengths,
            logits_indices=tuple(logits_indices),
        )
        self._model_forwards += 1
        input_token_count = sum(append_lengths)
        self._prefill_input_tokens += input_token_count - len(decoding)
        self._max_prefill_batch_size = max(self._max_prefill_batch_size, len(chunks))
        self._max_decode_batch_size = max(self._max_decode_batch_size, len(decoding))

        completed_prefills = [
            entry for entry, _start, end in chunks if end == len(entry.request.prompt_token_ids)
        ]
        sampled_entries = decoding + completed_prefills
        next_token_ids = self._sync_token_ids(
            tuple(
                self._sample(logits[index], entry.request)
                for index, entry in enumerate(sampled_entries)
            )
        )

        outputs = []
        for entry, next_token in zip(decoding, next_token_ids[: len(decoding)]):
            event = entry.request.record_token(next_token)
            outputs.append(event)
            if event.finished:
                self._running.pop(entry.request.request_id)
                self._release(entry, finished=True)
        token_offset = len(decoding)
        for entry, start, end in chunks:
            request = entry.request
            request_id = request.request_id
            entry.prefill_offset = end
            self._prefilling.pop(request_id)

            if end == len(request.prompt_token_ids):
                # Prompt 的完整页现在就发布并锁住；无需等长 Decode 全部结束即可复用。
                self.cache.publish_prefix(request_id, request.prompt_token_ids)
                request.start_decode()
                next_token = next_token_ids[token_offset]
                token_offset += 1
                event = request.record_token(next_token)
                outputs.append(event)
                if event.finished:
                    self._release(entry, finished=True)
                else:
                    self._running[request_id] = entry
            else:
                # 被选中但尚未完成的长请求移到队尾，避免小 budget 下独占 Prefill。
                self._prefilling[request_id] = entry

        return SchedulerStep(
            index=self._step_index,
            phase=phase,
            request_ids=request_ids,
            input_tokens=input_token_count,
            outputs=tuple(outputs),
        )

    def _admit_prefill_requests(self) -> None:
        while (
            self._waiting
            and self.cache.allocator.stats.request_count < self.config.max_running_requests
        ):
            entry = self._waiting[0]
            request = entry.request
            if not self.cache.can_reserve_request(
                self._max_request_tokens(request),
                request.prompt_token_ids,
            ):
                break
            matched = self.cache.reserve_request_with_prefix(
                request.request_id,
                self._max_request_tokens(request),
                request.prompt_token_ids,
            )
            if matched is None:
                raise RuntimeError("prefill selected without enough radix cache capacity")
            self._waiting.popleft()
            request.start_prefill()
            entry.prefill_offset = matched
            self._prefilling[request.request_id] = entry
            self._max_wait_steps = max(self._max_wait_steps, self._step_index - entry.enqueue_step)

    def _decode_step(self) -> SchedulerStep:
        entries = list(self._running.values())
        request_ids = tuple(entry.request.request_id for entry in entries)
        lengths = self.cache.lengths(request_ids)
        for request_id, length in zip(request_ids, lengths):
            self.cache.ensure_capacity(request_id, length + 1)
        input_token_ids = tuple(entry.request.last_output_token_id for entry in entries)
        graph_eligible = all(entry.request.sampling_params.is_greedy for entry in entries)
        use_cuda_graph = (
            graph_eligible
            and self._cuda_graph_runner
            and self._cuda_graph_runner.supports(len(entries))
        )
        self._validate_batch_plan(
            phase="decode",
            execution="decode-graph" if use_cuda_graph else "decode-eager",
            request_ids=request_ids,
            input_token_ids=input_token_ids,
            append_lengths=(1,) * len(entries),
            logits_indices=None,
        )
        if use_cuda_graph:
            # 满足固定 Decode bucket 条件后，从这里进入 Graph replay。
            next_token_ids = self._cuda_graph_runner.run(input_token_ids, request_ids)
        else:
            input_ids = torch.tensor(
                input_token_ids,
                dtype=torch.long,
                device=self._device,
            ).view(len(entries), 1)
            buffer = self._decode_buffers.get(len(entries))
            if buffer is None:
                buffer = self.cache.allocate_decode_buffer(
                    len(entries),
                    max_blocks=self._decode_max_blocks,
                )
                self._decode_buffers[len(entries)] = buffer
            batch = self.cache.prepare_decode_batch(request_ids, buffer)
            logits = self.model.forward_prepared(
                input_ids,
                kv_cache=self.cache,
                cache_batch=batch,
            )
            next_token_ids = tuple(
                self._sample(logits[index, -1], entry.request)
                for index, entry in enumerate(entries)
            )
        next_token_ids = self._sync_token_ids(next_token_ids)
        self._model_forwards += 1
        self._max_decode_batch_size = max(self._max_decode_batch_size, len(entries))

        outputs = []
        for entry, next_token in zip(entries, next_token_ids):
            event = entry.request.record_token(next_token)
            outputs.append(event)
            if event.finished:
                self._running.pop(entry.request.request_id)
                self._release(entry, finished=True)

        return SchedulerStep(
            index=self._step_index,
            phase="decode",
            request_ids=request_ids,
            input_tokens=len(entries),
            outputs=tuple(outputs),
        )

    def _release(self, entry: _Entry, *, finished: bool) -> None:
        request = entry.request
        request_id = request.request_id
        if request_id in self.cache.allocator.request_ids:
            if finished:
                self.cache.finish_request(request_id, request.all_token_ids)
            else:
                self.cache.release_request(request_id)
        self._entries.pop(request_id, None)
        self._sampling_generators.pop(request_id, None)
        if finished:
            self._finished_requests += 1
        else:
            self._aborted_requests += 1

    @staticmethod
    def _max_request_tokens(request: Request) -> int:
        return len(request.prompt_token_ids) + request.sampling_params.max_new_tokens

    def _sample(self, logits: torch.Tensor, request: Request) -> int:
        params = request.sampling_params
        generator = None
        if not params.is_greedy:
            generator = self._sampling_generators.get(request.request_id)
            if generator is None:
                generator = torch.Generator(device=self._device)
                if params.seed is None:
                    generator.seed()
                else:
                    generator.manual_seed(params.seed)
                self._sampling_generators[request.request_id] = generator
        return sample_token(logits, params, generator=generator)

    def _validate_batch_plan(
        self,
        *,
        phase: str,
        execution: str,
        request_ids: tuple[str, ...],
        input_token_ids: tuple[int, ...],
        append_lengths: tuple[int, ...],
        logits_indices: tuple[int, ...] | None,
    ) -> None:
        if self._coordinator is None:
            return
        plan = SchedulerBatchPlan(
            phase=phase,
            execution=execution,
            request_ids=request_ids,
            input_token_ids=input_token_ids,
            append_lengths=append_lengths,
            logits_indices=logits_indices,
            starts=self.cache.lengths(request_ids),
            page_tables=tuple(
                self.cache.allocator.page_table(request_id) for request_id in request_ids
            ),
        )
        self._coordinator.validate_batch(plan)

    def _sync_token_ids(self, token_ids: tuple[int, ...]) -> tuple[int, ...]:
        if self._coordinator is None:
            return token_ids
        result = self._coordinator.sync_token_ids(token_ids)
        if len(result) != len(token_ids):
            raise RuntimeError("coordinator returned the wrong number of sampled tokens")
        return result
