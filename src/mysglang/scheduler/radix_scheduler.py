from __future__ import annotations

from mysglang.cache import RadixPagedKVCache
from mysglang.modeling.tiny import TinyCausalLM

from .config import RadixSchedulerConfig
from .paged_scheduler import PagedBatchScheduler, _PagedEntry
from .scheduler import SchedulerStep


class RadixBatchScheduler(PagedBatchScheduler):
    """Paged continuous batching with page-aligned prompt KV reuse."""

    def __init__(self, model: TinyCausalLM, config: RadixSchedulerConfig) -> None:
        super().__init__(model, config)
        parameter = next(model.parameters())
        self.cache = RadixPagedKVCache.from_config(
            model.config,
            num_pages=config.num_pages,
            page_size=config.page_size,
            dtype=parameter.dtype,
            device=parameter.device,
        )

    @property
    def prefill_input_tokens(self) -> int:
        return sum(step.input_tokens for step in self.history if step.phase == "prefill")

    def reset_prefix_cache(self) -> None:
        self.cache.reset_prefix_cache()

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
        if self._prefilling is None:
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
        return super()._prefill_step()

    def _release(self, entry: _PagedEntry, *, finished: bool) -> None:
        request = entry.request
        request_id = request.request_id
        if request_id in self.cache.allocator.request_ids:
            if finished:
                cached_token_ids = tuple(request.prompt_token_ids) + tuple(request.output_token_ids)
                self.cache.finish_request(request_id, cached_token_ids)
            else:
                self.cache.release_request(request_id)
        self._entries.pop(request_id, None)
        if finished:
            self._finished_requests += 1
        else:
            self._aborted_requests += 1
