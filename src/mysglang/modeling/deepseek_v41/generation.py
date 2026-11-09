"""单请求生成与多轮会话状态；复用缓存时保持原始 token 边界。"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from .model import ReferenceCache
from .runtime import DeepSeekV41Model

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"


def format_prompt(prompt: str, *, first_turn: bool = True) -> str:
    """官方纯文本、关闭 thinking 的 user/assistant 边界。"""
    prefix = BOS_TOKEN if first_turn else ""
    return prefix + "<｜User｜>" + prompt + "<｜Assistant｜></think>"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.float().flatten(), expected.float().flatten()
    return {
        "max_abs_error": (a - b).abs().max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "greedy_equal": torch.equal(actual.argmax(-1), expected.argmax(-1)),
    }


@torch.inference_mode()
def generate_tokens(
    model: DeepSeekV41Model,
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    cache: ReferenceCache | None = None,
    compare_torch: bool = False,
    on_token: Callable[[int], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """只有新输入进入 prefill；调用方提供的 cache 在每次成功 forward 后推进。"""
    cache = model.new_cache() if cache is None else cache
    reused_tokens = cache.length
    prompt_tokens = reused_tokens + len(token_ids)
    if not token_ids or max_new_tokens <= 0 or prompt_tokens + max_new_tokens > model.max_seq_len:
        raise ValueError("empty prompt or prompt + max_new_tokens exceeds max_seq_len")
    if compare_torch and reused_tokens:
        raise ValueError("Torch comparison requires an empty cache")
    eos = tokenizer.token_to_id(EOS_TOKEN)
    if eos is None:
        raise ValueError("tokenizer is missing the DeepSeek EOS token")
    device, tp = model.embed_weight.device, model.context.tp
    tokens = torch.tensor([list(token_ids)], device=device, dtype=torch.long)
    _synchronize(device)
    start = time.perf_counter()
    logits = model(tokens, cache)
    _synchronize(device)
    prefill_seconds = time.perf_counter() - start
    comparison, oracle_cache = [], None
    backend = model.context.backend
    if compare_torch:
        oracle_cache = model.new_cache()
        model.context.backend = "torch"
        try:
            expected = model(tokens, oracle_cache)
        finally:
            model.context.backend = backend
        comparison.append(_metrics(logits, expected))
        if log is not None and tp.rank == 0:
            log(f"Torch comparison (prefill): {comparison[-1]}")
    generated, decode_seconds = [], []
    for step in range(max_new_tokens):
        next_token = logits.argmax(-1, keepdim=True)
        if tp.enabled:
            source = dist.get_global_rank(tp.process_group, 0) if tp.process_group else 0
            dist.broadcast(next_token, src=source, group=tp.process_group)
        token = next_token.item()
        generated.append(token)
        if tp.rank == 0:
            if on_token is not None:
                on_token(token)
            elif log is not None:
                text = tokenizer.decode([token], skip_special_tokens=False)
                log(f"token {step + 1}: {token} {text!r}")
        if token == eos or step + 1 == max_new_tokens:
            break
        _synchronize(device)
        start = time.perf_counter()
        logits = model(next_token, cache)
        _synchronize(device)
        decode_seconds.append(time.perf_counter() - start)
        if oracle_cache is not None and step < 2:
            model.context.backend = "torch"
            try:
                expected = model(next_token, oracle_cache)
            finally:
                model.context.backend = backend
            comparison.append(_metrics(logits, expected))
            if log is not None and tp.rank == 0:
                log(f"Torch comparison (decode {step + 1}): {comparison[-1]}")
        else:
            oracle_cache = None
    return {
        "prompt_tokens": prompt_tokens,
        "prefill_tokens": len(token_ids),
        "reused_tokens": reused_tokens,
        "token_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "finish_reason": "eos" if generated[-1] == eos else "length",
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "torch_comparison": comparison,
    }


class ChatInputError(ValueError):
    """可恢复的输入错误，必须在模型计算前抛出。"""


@dataclass(frozen=True)
class PreparedTurn:
    owner: object
    revision: int
    prompt: str
    token_ids: tuple[int, ...]


class ChatSession:
    """权重保持常驻；成功完成一轮后才提交历史和新缓存。"""

    def __init__(self, model: DeepSeekV41Model, tokenizer: Any, *, max_new_tokens: int):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model, self.tokenizer, self.max_new_tokens = model, tokenizer, max_new_tokens
        self.eos = tokenizer.token_to_id(EOS_TOKEN)
        if self.eos is None:
            raise ValueError("tokenizer is missing the DeepSeek EOS token")
        self._owner = object()
        self._revision = 0
        self.history_tokens: tuple[int, ...] = ()
        self.cache = model.new_cache()
        self.turn_count = 0

    def clear(self) -> None:
        self.history_tokens = ()
        self.cache = self.model.new_cache()
        self.turn_count = 0
        self._revision += 1

    def prepare(self, prompt: str) -> PreparedTurn:
        if not prompt.strip():
            raise ChatInputError("请输入非空问题。")
        text = format_prompt(prompt, first_turn=not self.history_tokens)
        suffix = tuple(self.tokenizer.encode(text, add_special_tokens=False).ids)
        token_ids = self.history_tokens + suffix
        # 长度停止时还需插入 EOS 关闭这一轮；不截断历史或破坏 Engram/KV 状态。
        required = len(token_ids) + self.max_new_tokens + 1
        if required > self.model.max_seq_len:
            raise ChatInputError(
                f"上下文空间不足：本轮输入后 {len(token_ids)} tokens，"
                f"回答预算 {self.max_new_tokens} + 1 个结束符，上限 {self.model.max_seq_len}。"
                "请缩短输入或用 /clear 清空会话；现有历史未改变。"
            )
        return PreparedTurn(self._owner, self._revision, prompt, token_ids)

    def respond(self, turn: PreparedTurn, *, on_token: Callable[[int], None] | None = None) -> dict:
        if turn.owner is not self._owner or turn.revision != self._revision:
            raise ValueError("prepared turn belongs to another or changed chat session")
        # runtime 的 forward 发布新 layers/tokens，不原地修改旧状态；浅复制即可回滚整轮。
        working = replace(self.cache)
        result = generate_tokens(
            self.model,
            self.tokenizer,
            turn.token_ids[working.length :],
            max_new_tokens=self.max_new_tokens,
            cache=working,
            on_token=on_token,
        )
        history = turn.token_ids + tuple(result["token_ids"])
        if result["finish_reason"] != "eos":
            history += (self.eos,)
        # 末个采样 token 尚未 forward；下一轮连同 EOS/user 边界一起送入，不能跳过。
        self.history_tokens, self.cache = history, working
        self.turn_count += 1
        self._revision += 1
        return {
            "prompt": turn.prompt,
            **result,
            "turn": self.turn_count,
            "history_tokens": len(history),
        }
