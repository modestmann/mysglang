"""Split Qwen3.5/3.6 chat output into reasoning, content and tool calls.

Thinking: with ``enable_thinking`` the chat template already ends the prompt with
``<think>\\n``, so generated text is reasoning until ``</think>``. Tool calls use the
``qwen3_coder`` XML layout::

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Beijing
    </parameter>
    </function>
    </tool_call>

Both parsers are incremental: streamed deltas never contain a partially generated tag.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0


@dataclass
class ReasoningParser:
    """Incrementally route text to ``reasoning_content`` until ``</think>``."""

    in_reasoning: bool = True
    _pending: str = ""
    _started: bool = False
    _strip_newlines: bool = False

    def feed(self, delta: str, *, final: bool = False) -> tuple[str, str]:
        """Return ``(reasoning_delta, content_delta)`` for newly decoded text."""
        text = self._pending + delta
        self._pending = ""
        reasoning = ""
        if self.in_reasoning:
            if not self._started:
                # 模板通常已把 <think> 放进 prompt；若模型自己输出了它，也要去掉。
                stripped = text.lstrip()
                if not final and len(stripped) < len(THINK_START) and THINK_START.startswith(
                    stripped
                ):
                    self._pending = text
                    return "", ""
                if stripped.startswith(THINK_START):
                    text = stripped[len(THINK_START) :]
                self._started = True
            end = text.find(THINK_END)
            if end < 0:
                hold = 0 if final else _partial_suffix(text, THINK_END)
                self._pending = text[len(text) - hold :]
                return text[: len(text) - hold], ""
            reasoning = text[:end]
            text = text[end + len(THINK_END) :]
            self.in_reasoning = False
            self._strip_newlines = True
        if self._strip_newlines:
            # 模板在 </think> 后接两个换行；它们可能被拆到后续 delta 中。
            text = text.lstrip("\n")
            if text:
                self._strip_newlines = False
        return reasoning, text


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


def _parameter_types(tools: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for tool in tools or ():
        function = tool.get("function", tool)
        properties = (function.get("parameters") or {}).get("properties") or {}
        result[function.get("name", "")] = {
            name: str(schema.get("type", "string")) for name, schema in properties.items()
        }
    return result


def _convert(value: str, declared_type: str | None) -> Any:
    if declared_type in {None, "string"}:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_tool_calls(
    text: str,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, list[ToolCall]]:
    """Split final assistant content into plain text and qwen3_coder tool calls."""
    start = text.find(TOOL_CALL_START)
    if start < 0:
        return text, []
    types = _parameter_types(tools)
    calls: list[ToolCall] = []
    for block in text[start:].split(TOOL_CALL_START)[1:]:
        body = block.split(TOOL_CALL_END, 1)[0]
        function = _FUNCTION.search(body)
        if function is None:
            continue
        name = function.group(1).strip()
        parameter_types = types.get(name, {})
        arguments = {
            key.strip(): _convert(value, parameter_types.get(key.strip()))
            for key, value in _PARAMETER.findall(function.group(2))
        }
        calls.append(ToolCall(f"call_{uuid.uuid4().hex[:24]}", name, arguments))
    if not calls:
        return text, []
    return text[:start].rstrip(), calls


@dataclass
class ToolCallStreamBuffer:
    """Pass content through until a ``<tool_call>`` starts, then buffer the remainder."""

    _pending: str = ""
    _buffer: list[str] = field(default_factory=list)
    buffering: bool = False

    def feed(self, delta: str, *, final: bool = False) -> str:
        if self.buffering:
            self._buffer.append(delta)
            return ""
        text = self._pending + delta
        self._pending = ""
        start = text.find(TOOL_CALL_START)
        if start >= 0:
            self.buffering = True
            self._buffer.append(text[start:])
            return text[:start]
        hold = 0 if final else _partial_suffix(text, TOOL_CALL_START)
        self._pending = text[len(text) - hold :]
        return text[: len(text) - hold]

    def buffered_text(self) -> str:
        return "".join(self._buffer)
