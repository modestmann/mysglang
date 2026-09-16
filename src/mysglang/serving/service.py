from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator
from dataclasses import dataclass

import torch

from mysglang.cache import ContiguousKVCache
from mysglang.core import FinishReason, Request, RequestState, SamplingParams
from mysglang.modeling.tiny import TinyCausalLM
from mysglang.tokenizer import ByteTokenizer


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


@dataclass(frozen=True)
class ServiceStats:
    active_requests: int
    max_active_requests: int
    finished_requests: int
    aborted_requests: int

#一次生成任务所需的上下文
@dataclass(frozen=True)
class GenerationSession:
    service: GenerationService
    request: Request
    input_ids: torch.Tensor
    cache: ContiguousKVCache
    prompt_tokens: int

    def __aiter__(self) -> AsyncIterator[GenerationChunk]:
        return self.service._run_session(self)

#推理服务，真正掌握 model/tokenizer
class GenerationService:
    """Own request lifetimes while serializing model forwards with an async lock."""

    def __init__(self, model: TinyCausalLM, tokenizer: ByteTokenizer) -> None:
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("model vocab_size must match tokenizer vocab_size")
        self.model = model.eval()
        self.tokenizer = tokenizer

        self._model_lock = asyncio.Lock()
        ##锁
        self._request_ids = itertools.count()
        ##request活跃队列 ，允许多个 active Request并发存在，但是调用 模型和tokenizer要串行
        self._active: dict[str, Request] = {}
        self._max_active_requests = 0
        self._finished_requests = 0
        self._aborted_requests = 0

    @property
    def stats(self) -> ServiceStats:
        return ServiceStats(
            active_requests=len(self._active),
            max_active_requests=self._max_active_requests,
            finished_requests=self._finished_requests,
            aborted_requests=self._aborted_requests,
        )
#模型推理入口准备，返回GenerationSession
    def start(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        ignore_eos: bool = False,
    ) -> GenerationSession:
        prompt_token_ids = self.tokenizer.encode(prompt)
        if eos_token_id is not None and eos_token_id >= self.model.config.vocab_size:
            raise ValueError("eos_token_id must be smaller than model vocab_size")
        if len(prompt_token_ids) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("prompt plus max_new_tokens exceeds max_position_embeddings")
        params = SamplingParams(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            ignore_eos=ignore_eos,
        )
        request_id = f"req-{next(self._request_ids)}"
        request = Request.from_token_ids(request_id, prompt_token_ids, params)
        parameter = next(self.model.parameters())
        input_ids = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=parameter.device,
        )
        cache = ContiguousKVCache.from_config(
            self.model.config,
            batch_size=1,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        return GenerationSession(self, request, input_ids, cache, len(prompt_token_ids))

    async def abort(self, request_id: str) -> bool:
        request = self._active.get(request_id)
        if request is None or request.is_terminal:
            return False
        request.abort()
        return True
#推理入口
    async def _run_session(self, session: GenerationSession) -> AsyncIterator[GenerationChunk]:
        request = session.request
        if request.state is not RequestState.WAITING:
            raise RuntimeError("a generation session can only be consumed once")
        if request.request_id in self._active:
            raise RuntimeError(f"request is already running: {request.request_id}")
        self._active[request.request_id] = request
        self._max_active_requests = max(self._max_active_requests, len(self._active))
        decoder = self.tokenizer.new_incremental_decoder()

        try:
            request.start_prefill()
            ##加锁
            async with self._model_lock:
                with torch.inference_mode():
                    logits = self.model(session.input_ids, kv_cache=session.cache)
            ##去锁
            if request.state is RequestState.ABORTED:
                return
            request.start_decode()

            while request.state is RequestState.DECODING:
                next_token = int(logits[:, -1].argmax(dim=-1).item())
                event = request.record_token(next_token)
                is_eos = event.finish_reason is FinishReason.EOS
                text = decoder.decode(None if is_eos else next_token, final=event.finished)
                yield GenerationChunk(
                    request_id=request.request_id,
                    text=text,
                    token_id=next_token,
                    output_index=event.output_index,
                    finished=event.finished,
                    finish_reason=event.finish_reason,
                    prompt_tokens=session.prompt_tokens,
                    completion_tokens=request.num_output_tokens,
                )
                if event.finished:
                    break
                ##让锁
                await asyncio.sleep(0)
                if request.state is RequestState.ABORTED:
                    break
                token_tensor = torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=session.input_ids.device,
                )
                ##加锁
                async with self._model_lock:
                    with torch.inference_mode():
                        logits = self.model(token_tensor, kv_cache=session.cache)
                ##去锁
        finally:
            if not request.is_terminal:
                request.abort()
            self._active.pop(request.request_id, None)
            if request.state is RequestState.FINISHED:
                self._finished_requests += 1
            else:
                self._aborted_requests += 1
