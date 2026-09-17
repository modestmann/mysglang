from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

import torch

from mysglang.cache import SlotKVCache
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
    active_requests: int #active =waiting+ prefilling+ running
    waiting_requests: int
    running_requests: int
    max_active_requests: int#从 Scheduler 创建以来，同时 active 的请求数量最大值。
    finished_requests: int
    aborted_requests: int
    model_forwards: int#Scheduler 累计调用模型多少次
    max_decode_batch_size: int
    max_wait_steps: int # 所有请求中，从加入 Scheduler 到开始 Prefill，等待调度 step 数的最大值。


@dataclass
class _Entry:
    request: Request
    enqueue_step: int #请求什么时候进入队列
    prefill_offset: int = 0#Prompt 已经 Prefill 到哪里
    slot_id: int | None = None#KV Cache 分配到了哪个 slot


class ContinuousBatchScheduler:
    """A synchronous step scheduler with chunked prefill and batched decode."""

    def __init__(self, model: TinyCausalLM, config: SchedulerConfig) -> None:
        self.model = model.eval()
        self.config = config
        parameter = next(model.parameters())
        self.cache = SlotKVCache.from_config(
            model.config,
            num_slots=config.max_running_requests,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self._device = parameter.device
        self._waiting: deque[_Entry] = deque()
        self._prefilling: _Entry | None = None
        self._running: OrderedDict[str, _Entry] = OrderedDict()
        self._entries: dict[str, _Entry] = {}#它是所有未结束请求的总索引：_entries = waiting + prefilling + running
        self._free_slots = set(range(config.max_running_requests))
        self._step_index = 0
        self._consecutive_prefill_steps = 0
        self._max_active_requests = 0
        self._finished_requests = 0
        self._aborted_requests = 0
        self._model_forwards = 0
        self._max_decode_batch_size = 0
        self._max_wait_steps = 0
        self.history: list[SchedulerStep] = []

    @property
    def has_work(self) -> bool:
        return bool(self._entries)

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
        )

    def add(self, request: Request) -> None:
        if request.state is not RequestState.WAITING:
            raise ValueError("scheduler only accepts requests in WAITING state")
        if request.request_id in self._entries:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        total_length = len(request.prompt_token_ids) + request.sampling_params.max_new_tokens
        if total_length > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus max_new_tokens exceeds max_position_embeddings")

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

    #实际调度
    @torch.inference_mode()
    def step(self) -> SchedulerStep | None:
        can_prefill = self._prefilling is not None or bool(self._waiting and self._free_slots)
        should_prefill = can_prefill and (
            not self._running
            or self._consecutive_prefill_steps
            < self.config.max_consecutive_prefill_steps
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

        self.history.append(result)
        self._step_index += 1
        return result

    def _prefill_step(self) -> SchedulerStep:
        entry = self._prefilling
        if entry is None:
            entry = self._waiting.popleft()
            entry.slot_id = min(self._free_slots)
            self._free_slots.remove(entry.slot_id)
            entry.request.start_prefill()
            self._prefilling = entry
            self._max_wait_steps = max(
                self._max_wait_steps, self._step_index - entry.enqueue_step
            )

        #这里做chunked
        start = entry.prefill_offset
        end = min(
            start + self.config.prefill_token_budget,
            len(entry.request.prompt_token_ids),
        )
        input_ids = torch.tensor(
            [entry.request.prompt_token_ids[start:end]],
            dtype=torch.long,
            device=self._device,
        )


        assert entry.slot_id is not None

        logits = self.model(input_ids, kv_cache=self.cache, cache_slots=(entry.slot_id,))

        self._model_forwards += 1
        entry.prefill_offset = end
        outputs: tuple[IncrementalOutput, ...] = ()

        if end == len(entry.request.prompt_token_ids):
            entry.request.start_decode()
            next_token = int(logits[:, -1].argmax(dim=-1).item())
            event = entry.request.record_token(next_token)
            outputs = (event,)
            self._prefilling = None
            if event.finished:
                self._release(entry, finished=True)
            else:
                self._running[entry.request.request_id] = entry

        return SchedulerStep(
            index=self._step_index,
            phase="prefill",
            request_ids=(entry.request.request_id,),
            input_tokens=end - start,
            outputs=outputs,
        )

    def _decode_step(self) -> SchedulerStep:
        entries = list(self._running.values())
        input_ids = torch.tensor(
            [[entry.request.output_token_ids[-1]] for entry in entries],
            dtype=torch.long,
            device=self._device,
        )
        slot_ids = tuple(entry.slot_id for entry in entries)
        assert all(slot_id is not None for slot_id in slot_ids)
        logits = self.model(
            input_ids,
            kv_cache=self.cache,
            cache_slots=slot_ids,
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
            request_ids=tuple(entry.request.request_id for entry in entries),
            input_tokens=len(entries),
            outputs=tuple(outputs),
        )

    def _release(self, entry: _Entry, *, finished: bool) -> None:
        if entry.slot_id is not None:
            self.cache.reset_slot(entry.slot_id)
            self._free_slots.add(entry.slot_id)
            entry.slot_id = None
        self._entries.pop(entry.request.request_id, None)
        if finished:
            self._finished_requests += 1
        else:
            self._aborted_requests += 1
