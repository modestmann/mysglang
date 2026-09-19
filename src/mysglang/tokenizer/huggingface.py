from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _is_cjk(codepoint: int) -> bool:
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x2F800 <= codepoint <= 0x2FA1F
    )


def _printable_prefix(text: str) -> str:
    if text.endswith("\n"):
        return text
    if text and _is_cjk(ord(text[-1])):
        return text
    if len(text) > 1 and _is_cjk(ord(text[-2])):
        return text[:-1]
    return text[: text.rfind(" ") + 1]


class HuggingFaceIncrementalDecoder:
    """Decode complete token context while emitting only its stable new suffix."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._sent_text = ""

    def decode(self, token_id: int | None, *, final: bool) -> str:
        if token_id is not None:
            self._token_ids.append(token_id)
        decoded = self._tokenizer.decode(
            self._token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded.startswith(self._sent_text):
            # Tokenizer byte fallback can temporarily end in U+FFFD. It was never emitted by
            # _printable_prefix, so a non-prefix here indicates a tokenizer contract violation.
            raise RuntimeError("incremental tokenizer output rewrote already emitted text")
        printable = decoded if final else _printable_prefix(decoded)
        # A stable CJK prefix may be followed by an unfinished Latin word (for example
        # ``我是`` -> ``我是AI``). In that case _printable_prefix returns an earlier boundary;
        # already emitted text remains valid and must not be retracted.
        if len(printable) < len(self._sent_text):
            printable = self._sent_text
        delta = printable[len(self._sent_text) :]
        self._sent_text = printable
        return delta


class HuggingFaceTokenizer:
    """Local Hugging Face tokenizer with Qwen chat-template and streaming support."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        # ``tokenizer.vocab_size`` excludes added tokens; len(tokenizer) includes IDs such as
        # Qwen's im_start/im_end and thinking markers.
        self.vocab_size = len(tokenizer)
        self.eos_token_id: int | None = tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> HuggingFaceTokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HuggingFaceTokenizer requires the 'transformers' package") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("prompt must be a non-empty string")
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = True,
    ) -> list[int]:
        if not messages:
            raise ValueError("messages must not be empty")
        result = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if isinstance(result, Mapping):
            result = result["input_ids"]
        if hasattr(result, "tolist"):
            result = result.tolist()
        if result and isinstance(result[0], list):
            if len(result) != 1:
                raise ValueError("chat template unexpectedly returned a batch")
            result = result[0]
        return [int(token_id) for token_id in result]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def new_incremental_decoder(self) -> HuggingFaceIncrementalDecoder:
        return HuggingFaceIncrementalDecoder(self._tokenizer)
