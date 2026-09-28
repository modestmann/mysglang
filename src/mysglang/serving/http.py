from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mysglang.core import FinishReason

from .parsers import ReasoningParser, ToolCallStreamBuffer, parse_tool_calls
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
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # 字符串或 OpenAI content parts：text 与 image_url（data URL / http(s) / 本地路径）。
    content: str | list[dict[str, Any]] | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_template(self) -> dict[str, Any]:
        # content parts（文本与 image_url）原样交给服务层拆分图片。
        message = self.model_dump(exclude_none=True)
        if self.content is None:
            message["content"] = ""
        for call in message.get("tool_calls", ()):
            # 历史里的 OpenAI arguments 是 JSON 字符串；模板需要逐参数展开的 dict。
            function = call.get("function", {})
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    pass
        return message


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3"
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    eos_token_id: int | None = Field(default=None, ge=0)
    ignore_eos: bool = False
    # 兼容旧字段；OpenAI 客户端通常用 chat_template_kwargs={"enable_thinking": ...}。
    enable_thinking: bool = True
    chat_template_kwargs: dict[str, Any] | None = None
    # 把 <think>...</think> 拆到 reasoning_content（对应 SGLang/vLLM 的 reasoning parser）。
    separate_reasoning: bool = True
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float = Field(default=0.0, ge=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)

    @property
    def token_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 16

    @property
    def active_tools(self) -> list[dict[str, Any]] | None:
        return None if self.tool_choice == "none" else self.tools


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


def _sse_line(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


# 流式输出：逐 token 编码为 SSE；断连或取消时必须 abort 并释放 KV 页面。
async def _sse(
    session: GenerationSession,
    http_request: HTTPRequest,
    encode: Callable[[GenerationChunk], list[dict]],
) -> AsyncIterator[bytes]:
    completed = False
    try:
        async with aclosing(session.__aiter__()) as stream:
            async for chunk in stream:
                if await http_request.is_disconnected():
                    await session.service.abort(session.request.request_id)
                    return
                for payload in encode(chunk):
                    yield _sse_line(payload)
                completed = chunk.finished
        if completed:
            yield b"data: [DONE]\n\n"
    except asyncio.CancelledError:
        await session.service.abort(session.request.request_id)
        raise
    finally:
        if not completed:
            await session.service.abort(session.request.request_id)


class _ChatOutputParser:
    """Per-request reasoning/tool-call routing shared by streaming and non-streaming."""

    def __init__(
        self,
        service: GenerationService,
        payload: ChatCompletionRequest,
        enable_thinking: bool,
    ) -> None:
        reasoning = (
            payload.separate_reasoning
            and enable_thinking
            and getattr(service.tokenizer, "supports_reasoning", False)
        )
        self.reasoning = ReasoningParser() if reasoning else None
        self.tools = payload.active_tools
        self.tool_buffer = ToolCallStreamBuffer() if self.tools else None

    def feed(self, text: str, *, final: bool) -> tuple[str, str]:
        reasoning, content = ("", text) if self.reasoning is None else self.reasoning.feed(
            text, final=final
        )
        if self.tool_buffer is not None:
            content = self.tool_buffer.feed(content, final=final)
        return reasoning, content

    def tool_calls(self) -> list[dict[str, Any]]:
        if self.tool_buffer is None or not self.tool_buffer.buffering:
            return []
        _content, calls = parse_tool_calls(self.tool_buffer.buffered_text(), self.tools)
        return [call.to_openai() for call in calls]


def create_app(service: GenerationService, *, model_name: str = "mysglang") -> FastAPI:
    app = FastAPI(title="MySGLang", version="0.1.0")

    def start_session(payload: GenerateRequest) -> GenerationSession:
        try:
            return service.start(
                payload.prompt,
                max_new_tokens=payload.max_tokens,
                eos_token_id=payload.eos_token_id,
                ignore_eos=payload.ignore_eos,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                seed=payload.seed,
                presence_penalty=payload.presence_penalty,
                frequency_penalty=payload.frequency_penalty,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "mysglang"}],
        }

    # 保留简单 generate 与 OpenAI 风格 chat 两种请求格式。
    @app.post("/generate")
    async def generate(payload: GenerateRequest, request: HTTPRequest):
        session = start_session(payload)
        if payload.stream:
            return StreamingResponse(
                _sse(
                    session,
                    request,
                    lambda chunk: [
                        {
                            "id": chunk.request_id,
                            "text": chunk.text,
                            "token_id": chunk.token_id,
                            "output_index": chunk.output_index,
                            "finished": chunk.finished,
                            "finish_reason": _finish_reason(chunk.finish_reason),
                        }
                    ],
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
        template_kwargs = dict(payload.chat_template_kwargs or {})
        enable_thinking = bool(template_kwargs.pop("enable_thinking", payload.enable_thinking))
        try:
            session = service.start_chat(
                [message.to_template() for message in payload.messages],
                max_new_tokens=payload.token_limit,
                eos_token_id=payload.eos_token_id,
                ignore_eos=payload.ignore_eos,
                enable_thinking=enable_thinking,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                seed=payload.seed,
                presence_penalty=payload.presence_penalty,
                frequency_penalty=payload.frequency_penalty,
                tools=payload.active_tools,
                chat_template_kwargs=template_kwargs,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        parser = _ChatOutputParser(service, payload, enable_thinking)
        created = int(time.time())

        if payload.stream:
            started = False

            def encode(chunk: GenerationChunk) -> list[dict]:
                nonlocal started
                reasoning, content = parser.feed(chunk.text, final=chunk.finished)
                deltas: list[dict[str, Any]] = []
                if not started:
                    deltas.append({"role": "assistant"})
                    started = True
                if reasoning:
                    deltas.append({"reasoning_content": reasoning})
                if content:
                    deltas.append({"content": content})
                finish_reason = _finish_reason(chunk.finish_reason)
                if chunk.finished:
                    calls = parser.tool_calls()
                    if calls:
                        deltas.append(
                            {
                                "tool_calls": [
                                    {"index": index, **call} for index, call in enumerate(calls)
                                ]
                            }
                        )
                        finish_reason = "tool_calls"
                if not deltas and not chunk.finished:
                    return []
                deltas = deltas or [{}]
                events = []
                for index, delta in enumerate(deltas):
                    last_delta = index == len(deltas) - 1
                    event = {
                        "id": f"chatcmpl-{chunk.request_id}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": payload.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason if last_delta else None,
                            }
                        ],
                    }
                    if chunk.finished and last_delta:
                        event["usage"] = _usage(chunk)
                    events.append(event)
                return events

            return StreamingResponse(_sse(session, request, encode), media_type="text/event-stream")

        text, _token_ids, last = await _collect(session)
        reasoning, content = parser.feed(text, final=True)
        calls = parser.tool_calls()
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if parser.reasoning is not None:
            message["reasoning_content"] = reasoning
        finish_reason = _finish_reason(last.finish_reason)
        if calls:
            message["tool_calls"] = calls
            finish_reason = "tool_calls"
        elif message["content"] is None:
            message["content"] = ""
        return {
            "id": f"chatcmpl-{last.request_id}",
            "object": "chat.completion",
            "created": created,
            "model": payload.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": _usage(last),
        }

    return app
