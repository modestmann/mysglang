from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mysglang.core import FinishReason

from .service import GenerationChunk, GenerationService, GenerationSession


# 每个 HTTP 请求会包装成独立 session；这些 session 共享同一个 GenerationService，
# 再由 request 专属 Queue 接收全局 Scheduler worker 路由回来的 token。
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=16, gt=0)
    stream: bool = False
    eos_token_id: int | None = Field(default=None, ge=0)
    ignore_eos: bool = False
    temperature: float = Field(default=0.0, ge=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, gt=0)
    stream: bool = False
    eos_token_id: int | None = Field(default=None, ge=0)
    ignore_eos: bool = False
    enable_thinking: bool = True
    temperature: float = Field(default=0.0, ge=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)


def _finish_reason(reason: FinishReason | None) -> str | None:
    if reason is FinishReason.LENGTH:
        return "length"
    if reason is FinishReason.EOS:
        return "stop"
    return None


def _usage(chunk: GenerationChunk) -> dict[str, int]:
    return {
        "prompt_tokens": chunk.prompt_tokens,
        "completion_tokens": chunk.completion_tokens,
        "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
    }


# 非流式输出：消费完整 session 后一次返回。
async def _collect(session: GenerationSession) -> tuple[str, list[int], GenerationChunk]:
    text_parts: list[str] = []
    token_ids: list[int] = []
    last: GenerationChunk | None = None
    async for chunk in session:
        text_parts.append(chunk.text)
        token_ids.append(chunk.token_id)
        last = chunk
    if last is None:
        raise RuntimeError("generation produced no output")
    return "".join(text_parts), token_ids, last


# 流式输出：逐 token 编码为 SSE；断连或取消时必须 abort 并释放 KV 页面。
async def _sse(
    session: GenerationSession,
    http_request: HTTPRequest,
    encode: Callable[[GenerationChunk], dict],
) -> AsyncIterator[bytes]:
    completed = False
    try:
        async with aclosing(session.__aiter__()) as stream:
            async for chunk in stream:
                if await http_request.is_disconnected():
                    await session.service.abort(session.request.request_id)
                    return
                yield f"data: {json.dumps(encode(chunk), ensure_ascii=False)}\n\n".encode()
                completed = chunk.finished
        if completed:
            yield b"data: [DONE]\n\n"
    except asyncio.CancelledError:
        await session.service.abort(session.request.request_id)
        raise
    finally:
        if not completed:
            await session.service.abort(session.request.request_id)


def create_app(service: GenerationService) -> FastAPI:
    app = FastAPI(title="MySGLang", version="0.1.0")

    def start_session(
        prompt: str,
        max_tokens: int,
        eos_token_id: int | None,
        ignore_eos: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int | None,
    ) -> GenerationSession:
        try:
            return service.start(
                prompt,
                max_new_tokens=max_tokens,
                eos_token_id=eos_token_id,
                ignore_eos=ignore_eos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # 保留简单 generate 与 OpenAI 风格 chat 两种请求格式。
    @app.post("/generate")
    async def generate(payload: GenerateRequest, request: HTTPRequest):
        session = start_session(
            payload.prompt,
            payload.max_tokens,
            payload.eos_token_id,
            payload.ignore_eos,
            payload.temperature,
            payload.top_k,
            payload.top_p,
            payload.seed,
        )
        if payload.stream:
            return StreamingResponse(
                _sse(
                    session,
                    request,
                    lambda chunk: {
                        "id": chunk.request_id,
                        "text": chunk.text,
                        "token_id": chunk.token_id,
                        "output_index": chunk.output_index,
                        "finished": chunk.finished,
                        "finish_reason": _finish_reason(chunk.finish_reason),
                    },
                ),
                media_type="text/event-stream",
            )

        text, token_ids, last = await _collect(session)
        return {
            "id": last.request_id,
            "text": text,
            "token_ids": token_ids,
            "finish_reason": _finish_reason(last.finish_reason),
            "usage": _usage(last),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: HTTPRequest):
        if not payload.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        try:
            session = service.start_chat(
                [message.model_dump() for message in payload.messages],
                max_new_tokens=payload.max_tokens,
                eos_token_id=payload.eos_token_id,
                ignore_eos=payload.ignore_eos,
                enable_thinking=payload.enable_thinking,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                seed=payload.seed,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload.stream:
            return StreamingResponse(
                _sse(
                    session,
                    request,
                    lambda chunk: {
                        "id": f"chatcmpl-{chunk.request_id}",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.text},
                                "finish_reason": _finish_reason(chunk.finish_reason),
                            }
                        ],
                    },
                ),
                media_type="text/event-stream",
            )

        text, _token_ids, last = await _collect(session)
        return {
            "id": f"chatcmpl-{last.request_id}",
            "object": "chat.completion",
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _finish_reason(last.finish_reason),
                }
            ],
            "usage": _usage(last),
        }

    return app
