from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from mysglang.cache import RadixPagedKVCache
from mysglang.core import IncrementalOutput, Request, RequestState
from mysglang.modeling.tiny import TinyCausalLM

from .config import SchedulerConfig


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
    max_decode_batch_size: int
    max_wait_steps: int
    prefill_input_tokens: int


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

    def __init__(self, model: TinyCausalLM, config: SchedulerConfig) -> None:
        self.model = model.eval()
        self.config = config
        parameter = next(model.parameters())
        self.cache = RadixPagedKVCache.from_config(
            model.config,
            num_pages=config.num_pages,
            page_size=config.page_size,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self._device = parameter.device
        self._waiting: deque[_Entry] = deque()
        self._prefilling: _Entry | None = None
        self._running: dict[str, _Entry] = {}
        # 所有未结束请求的总索引：waiting + prefilling + running。
        self._entries: dict[str, _Entry] = {}
        self._step_index = 0
        self._consecutive_prefill_steps = 0
        self._max_active_requests = 0
        self._finished_requests = 0
        self._aborted_requests = 0
        self._model_forwards = 0
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
        return SchedulerStats(
            active_requests=len(self._entries),
            waiting_requests=len(self._waiting) + (self._prefilling is not None),
            running_requests=len(self._running),
            max_active_requests=self._max_active_requests,
            finished_requests=self._finished_requests,
            aborted_requests=self._aborted_requests,
            model_forwards=self._model_forwards,
            max_decode_batch_size=self._max_decode_batch_size,
            max_wait_steps=self._max_wait_steps,
            prefill_input_tokens=self._prefill_input_tokens,
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
        if self._prefilling is entry:
            self._prefilling = None
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
        """Run one scheduling decision: one prefill chunk or one decode batch."""
        can_prefill = self._can_prefill()
        should_prefill = can_prefill and (
            not self._running
            or self._consecutive_prefill_steps < self.config.max_consecutive_prefill_steps
        )

        if should_prefill:
            result = self._prefill_step()
            self._consecutive_prefill_steps += 1
        elif self._running:
            result = self._decode_step()
            self._consecutive_prefill_steps = 0
        elif can_prefill:
            result = self._prefill_step()
            self._consecutive_prefill_steps += 1
        else:
            return None

        self._step_index += 1
        return result

    def reset_prefix_cache(self) -> None:
        self.cache.reset_prefix_cache()

    def check_integrity(self) -> None:
        self.cache.check_integrity()
        cache_requests = set(self.cache.allocator.request_ids)
        scheduled_requests = set(self._running)
        if self._prefilling is not None:
            scheduled_requests.add(self._prefilling.request.request_id)
        if cache_requests != scheduled_requests:
            raise RuntimeError("scheduler and paged cache disagree about active owners")
        waiting_ids = {entry.request.request_id for entry in self._waiting}
        if waiting_ids & cache_requests:
            raise RuntimeError("a waiting request already owns physical cache pages")
        if set(self._entries) != waiting_ids | scheduled_requests:
            raise RuntimeError("scheduler request indexes disagree")
        if any(entry.request.state is not RequestState.WAITING for entry in self._waiting):
            raise RuntimeError("waiting queue contains a request in the wrong state")
        if self._prefilling is not None:
            if self._prefilling.request.state is not RequestState.PREFILL:
                raise RuntimeError("prefill slot contains a request in the wrong state")
        if any(
            entry.request.state is not RequestState.DECODING for entry in self._running.values()
        ):
            raise RuntimeError("running map contains a request in the wrong state")

    def _can_prefill(self) -> bool:
        if self._prefilling is not None:
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
        entry = self._prefilling
        if entry is None:
            entry = self._waiting[0]
            request = entry.request
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
            self._prefilling = entry
            self._max_wait_steps = max(self._max_wait_steps, self._step_index - entry.enqueue_step)

        start = entry.prefill_offset
        end = min(
            start + self.config.prefill_token_budget,
            len(entry.request.prompt_token_ids),
        )
        request_id = entry.request.request_id

        # reservation 是容量承诺；这里才从 free pages 取物理页并写入请求页表。
        self.cache.ensure_capacity(request_id, end)
        input_ids = torch.tensor(
            [entry.request.prompt_token_ids[start:end]],
            dtype=torch.long,
            device=self._device,
        )
        logits = self.model(
            input_ids,
            kv_cache=self.cache,
            cache_request_ids=(request_id,),
        )
        self._model_forwards += 1
        self._prefill_input_tokens += end - start
        entry.prefill_offset = end
        outputs: tuple[IncrementalOutput, ...] = ()

        if end == len(entry.request.prompt_token_ids):
            # Prompt 的完整页现在就发布并锁住；无需等长 Decode 全部结束，后来的请求即可复用。
            self.cache.publish_prefix(request_id, entry.request.prompt_token_ids)
            entry.request.start_decode()
            next_token = int(logits[:, -1].argmax(dim=-1).item())
            event = entry.request.record_token(next_token)
            outputs = (event,)
            self._prefilling = None
            if event.finished:
                self._release(entry, finished=True)
            else:
                self._running[request_id] = entry

        return SchedulerStep(
            index=self._step_index,
            phase="prefill",
            request_ids=(request_id,),
            input_tokens=end - start,
            outputs=outputs,
        )

    def _decode_step(self) -> SchedulerStep:
        entries = list(self._running.values())
        request_ids = tuple(entry.request.request_id for entry in entries)
        lengths = self.cache.lengths(request_ids)
        for request_id, length in zip(request_ids, lengths):
            self.cache.ensure_capacity(request_id, length + 1)
        input_ids = torch.tensor(
            [[entry.request.last_output_token_id] for entry in entries],
            dtype=torch.long,
            device=self._device,
        )
        logits = self.model(
            input_ids,
            kv_cache=self.cache,
            cache_request_ids=request_ids,
        )
        self._model_forwards += 1
        self._max_decode_batch_size = max(self._max_decode_batch_size, len(entries))

        outputs = []
        for batch_index, entry in enumerate(entries):
            next_token = int(logits[batch_index, -1].argmax().item())
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
        if finished:
            self._finished_requests += 1
        else:
            self._aborted_requests += 1

    @staticmethod
    def _max_request_tokens(request: Request) -> int:
        return len(request.prompt_token_ids) + request.sampling_params.max_new_tokens
