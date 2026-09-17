from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mysglang.core import FinishReason, IncrementalOutput, Request, SamplingParams
from mysglang.modeling.tiny import TinyCausalLM
from mysglang.scheduler import ContinuousBatchScheduler, SchedulerConfig
from mysglang.tokenizer import ByteTokenizer

from .service import GenerationChunk, ServiceStats


@dataclass(frozen=True)
class ContinuousGenerationSession:
    service: ContinuousBatchGenerationService
    request: Request
    prompt_tokens: int

    def __aiter__(self) -> AsyncIterator[GenerationChunk]:
        return self.service._run_session(self)


class ContinuousBatchGenerationService:
    """Adapt the step scheduler to the async session interface used by HTTP."""

    def __init__(
        self,
        model: TinyCausalLM,
        tokenizer: ByteTokenizer,
        scheduler_config: SchedulerConfig,
    ) -> None:
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("model vocab_size must match tokenizer vocab_size")
        self.tokenizer = tokenizer
        self.scheduler = ContinuousBatchScheduler(model, scheduler_config)
        self._request_ids = itertools.count()
        self._queues: dict[str, asyncio.Queue[IncrementalOutput | BaseException]] = {}
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def stats(self) -> ServiceStats:
        stats = self.scheduler.stats
        return ServiceStats(
            active_requests=stats.active_requests,
            max_active_requests=stats.max_active_requests,
            finished_requests=stats.finished_requests,
            aborted_requests=stats.aborted_requests,
        )

    def start(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        ignore_eos: bool = False,
    ) -> ContinuousGenerationSession:
        prompt_token_ids = self.tokenizer.encode(prompt)
        params = SamplingParams(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            ignore_eos=ignore_eos,
        )
        if eos_token_id is not None and eos_token_id >= self.tokenizer.vocab_size:
            raise ValueError("eos_token_id must be smaller than model vocab_size")
        request = Request.from_token_ids(
            f"req-{next(self._request_ids)}",
            prompt_token_ids,
            params,
        )
        #进入调度
        self.scheduler.add(request)
        #每个请求的 Queue 保存的是：这个请求在多次 scheduler step / 模型推理中陆续产生的输出事件，尚未被 session/HTTP 消费的 token 事件
        self._queues[request.request_id] = asyncio.Queue()
        #给这个请求准备一个输出通道，之后把它生成的 token 送回对应 HTTP/session
        return ContinuousGenerationSession(self, request, len(prompt_token_ids))

    async def abort(self, request_id: str) -> bool:
        event = self.scheduler.abort(request_id)
        if event is None:
            return False
        queue = self._queues.get(request_id)
        if queue is not None:
            queue.put_nowait(event)
        return True

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._run_scheduler())

    async def _run_scheduler(self) -> None:
        try:
            while self.scheduler.has_work:

#Scheduler 决定本轮做：Prefill或者Decode
                step = self.scheduler.step()


                if step is not None:
                    for event in step.outputs:
                        queue = self._queues.get(event.request_id)
                        if queue is not None:
                            queue.put_nowait(event)
                await asyncio.sleep(0)
        except Exception as exc:
            for queue in self._queues.values():
                queue.put_nowait(exc)
            raise
        finally:
            self._worker_task = None
##多个请求创建多个生成器
    async def _run_session(
        self, session: ContinuousGenerationSession
    ) -> AsyncIterator[GenerationChunk]:
        queue = self._queues[session.request.request_id]
        decoder = self.tokenizer.new_incremental_decoder()
        self._ensure_worker()#_sse/_collect-----_run_session----_ensure_worker-----_run_scheduler

        ##哦哦真正调度队列在scheduler._watting队列    _ensure_worker-----_run_scheduler全局唯一
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                if item.finish_reason is FinishReason.ABORTED:
                    return
                is_eos = item.finish_reason is FinishReason.EOS
                text = decoder.decode(None if is_eos else item.token_id, final=item.finished)
                yield GenerationChunk(
                    request_id=item.request_id,
                    text=text,
                    token_id=item.token_id,
                    output_index=item.output_index,
                    finished=item.finished,
                    finish_reason=item.finish_reason,
                    prompt_tokens=session.prompt_tokens,
                    completion_tokens=item.output_index + 1,
                )
                if item.finished:
                    return
        finally:
            if not session.request.is_terminal:
                await self.abort(session.request.request_id)
            self._queues.pop(session.request.request_id, None)
