from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

import torch

from mysglang.cache import PagedKVCache
from mysglang.core import IncrementalOutput, Request, RequestState
from mysglang.modeling.tiny import TinyCausalLM

from .config import PagedSchedulerConfig
from .scheduler import SchedulerStats, SchedulerStep


@dataclass
class _PagedEntry:
    request: Request
    enqueue_step: int
    prefill_offset: int = 0


class PagedBatchScheduler:
    """Continuous batching backed by a shared, admission-safe physical page pool."""

    def __init__(self, model: TinyCausalLM, config: PagedSchedulerConfig) -> None:
        self.model = model.eval()
        self.config = config
        parameter = next(model.parameters())
        self.cache = PagedKVCache.from_config(
            model.config,
            num_pages=config.num_pages,
            page_size=config.page_size,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self._device = parameter.device
        self._waiting: deque[_PagedEntry] = deque()
        self._prefilling: _PagedEntry | None = None
        self._running: OrderedDict[str, _PagedEntry] = OrderedDict()
        self._entries: dict[str, _PagedEntry] = {}
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

    #add的时候还没有分配物理页面
    def add(self, request: Request) -> None:
        if request.state is not RequestState.WAITING:
            raise ValueError("scheduler only accepts requests in WAITING state")
        if request.request_id in self._entries:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        total_length = self._max_request_tokens(request)
        if total_length > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus max_new_tokens exceeds max_position_embeddings")
        needed_pages = self.cache.allocator.pages_for_tokens(total_length)
        if needed_pages > self.cache.num_pages:
            raise ValueError("request exceeds total paged KV cache capacity")

        entry = _PagedEntry(request=request, enqueue_step=self._step_index)
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
        can_prefill = self._can_prefill()
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
            raise RuntimeError("waiting request owns physical cache pages")

    def _can_prefill(self) -> bool:
        if self._prefilling is not None:
            return True
        if not self._waiting:
            return False
        if self.cache.allocator.stats.request_count >= self.config.max_running_requests:
            return False
        return self.cache.can_reserve(self._max_request_tokens(self._waiting[0].request))


    #真正分配物理页面在调度到prefill或者decode
    def _prefill_step(self) -> SchedulerStep:
        entry = self._prefilling
        if entry is None:
            entry = self._waiting[0]
            request_id = entry.request.request_id
            #分配页面，预约了最多可以使用 3 页，还没有拿到物理页编号
            if not self.cache.reserve_request(
                request_id, self._max_request_tokens(entry.request)
            ):
                raise RuntimeError("prefill selected without enough reserved page capacity")
            self._waiting.popleft()
            entry.request.start_prefill()
            self._prefilling = entry
            self._max_wait_steps = max(
                self._max_wait_steps, self._step_index - entry.enqueue_step
            )

        start = entry.prefill_offset
        end = min(
            start + self.config.prefill_token_budget,
            len(entry.request.prompt_token_ids),
        )
        request_id = entry.request.request_id

        #从 _free_pages 取出真正的物理页编号，追加到页表
        #循环追加
        self.cache.ensure_capacity(request_id, end)
        """
        reserved = 容量承诺
        allocated = 页表里已经绑定的物理页
        """

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
            [[entry.request.output_token_ids[-1]] for entry in entries],
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

    def _release(self, entry: _PagedEntry, *, finished: bool) -> None:
        request_id = entry.request.request_id
        if request_id in self.cache.allocator.request_ids:
            self.cache.release_request(request_id)
        self._entries.pop(request_id, None)
        if finished:
            self._finished_requests += 1
        else:
            self._aborted_requests += 1

    @staticmethod
    def _max_request_tokens(request: Request) -> int:
        return len(request.prompt_token_ids) + request.sampling_params.max_new_tokens
