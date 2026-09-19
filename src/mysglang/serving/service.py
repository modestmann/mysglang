from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from mysglang.core import FinishReason, IncrementalOutput, Request, SamplingParams
from mysglang.modeling.qwen3 import Qwen3ForCausalLM
from mysglang.scheduler import Scheduler, SchedulerConfig, SchedulerStats


class IncrementalDecoder(Protocol):
    def decode(self, token_id: int | None, *, final: bool) -> str: ...


class Tokenizer(Protocol):
    vocab_size: int
    eos_token_id: int | None

    def encode(self, text: str) -> Sequence[int]: ...

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = True,
    ) -> Sequence[int]: ...

    def new_incremental_decoder(self) -> IncrementalDecoder: ...


@dataclass(frozen=True)
class GenerationChunk:
    request_id: str
    text: str
    token_id: int
    output_index: int
    finished: bool
    finish_reason: FinishReason | None
    prompt_tokens: int
    completion_tokens: int


@dataclass
class GenerationSession:
    """One request-specific view over the shared generation service."""

    service: GenerationService
    request: Request
    prompt_tokens: int
    _consumed: bool = field(default=False, init=False, repr=False)

    def __aiter__(self) -> AsyncIterator[GenerationChunk]:
        if self._consumed:
            raise RuntimeError("a generation session can only be consumed once")
        self._consumed = True
        return self.service._run_session(self)


class GenerationService:
    """Route async sessions through one global continuous-batching worker."""

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        tokenizer: Tokenizer,
        scheduler_config: SchedulerConfig,
    ) -> None:
        if model.config.vocab_size < tokenizer.vocab_size:
            raise ValueError("model vocab_size is smaller than the tokenizer ID space")
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = Scheduler(model, scheduler_config)
        self._request_ids = itertools.count()
        self._queues: dict[str, asyncio.Queue[IncrementalOutput | BaseException]] = {}
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def stats(self) -> SchedulerStats:
        return self.scheduler.stats

    def start(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        ignore_eos: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> GenerationSession:
        prompt_token_ids = tuple(self.tokenizer.encode(prompt))
        return self._start_token_ids(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            ignore_eos=ignore_eos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

    def start_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        ignore_eos: bool = False,
        enable_thinking: bool = True,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> GenerationSession:
        prompt_token_ids = tuple(
            self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
            )
        )
        return self._start_token_ids(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            ignore_eos=ignore_eos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

    def _start_token_ids(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        eos_token_id: int | None,
        ignore_eos: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
    ) -> GenerationSession:
        request = Request.from_token_ids(
            f"req-{next(self._request_ids)}",
            prompt_token_ids,
            SamplingParams(
                max_new_tokens=max_new_tokens,
                eos_token_id=(
                    self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
                ),
                ignore_eos=ignore_eos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            ),
        )
        self.scheduler.validate(request)
        return GenerationSession(self, request, len(prompt_token_ids))

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
                # 全局唯一 worker 每轮只让 Scheduler 选择一次 Prefill 或 Decode。
                step = self.scheduler.step()
                if step is not None:
                    for event in step.outputs:
                        queue = self._queues.get(event.request_id)
                        if queue is not None:
                            queue.put_nowait(event)
                await asyncio.sleep(0)
        except Exception as exc:
            # 把模型/调度异常广播给所有等待中的 session，由消费者上下文负责清理请求。
            for queue in self._queues.values():
                queue.put_nowait(exc)
        finally:
            self._worker_task = None

    async def _run_session(self, session: GenerationSession) -> AsyncIterator[GenerationChunk]:
        request_id = session.request.request_id
        queue: asyncio.Queue[IncrementalOutput | BaseException] = asyncio.Queue()
        decoder = self.tokenizer.new_incremental_decoder()
        self.scheduler.add(session.request)
        # 每个请求有自己的输出 Queue；真正的等待/运行集合由共享 Scheduler 管理。
        self._queues[request_id] = queue
        self._ensure_worker()

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
                await self.abort(request_id)
            self._queues.pop(request_id, None)
