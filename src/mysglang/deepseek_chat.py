"""DeepSeek 终端会话：rank 0 读写终端，Gloo 广播输入和控制命令。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TextIO

import torch.distributed as dist

from .modeling.deepseek_v41.generation import ChatInputError, ChatSession


class TextStreamer:
    """使用 tokenizer 的增量解码器，等待完整 UTF-8 字符，结束时补齐剩余文本。"""

    def __init__(self, tokenizer: Any, write: Callable[[str], None]):
        from tokenizers.decoders import DecodeStream

        self.tokenizer, self.write = tokenizer, write
        self.decoder = DecodeStream(skip_special_tokens=True)
        self.tokens: list[int] = []
        self.emitted = ""

    def push(self, token: int) -> None:
        self.tokens.append(token)
        chunk = self.decoder.step(self.tokenizer, token)
        if chunk:
            self.write(chunk)
            self.emitted += chunk

    def finish(self) -> None:
        text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        if not text.startswith(self.emitted):
            raise RuntimeError("streaming decoder changed previously emitted text")
        remaining = text[len(self.emitted) :]
        if remaining:
            self.write(remaining)
        self.emitted = text


class ChatChannel:
    """等待人类输入使用独立 Gloo group，不留下等待中的 NCCL collective。"""

    def __init__(self, *, rank: int, world_size: int, group: dist.ProcessGroup | None = None):
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid chat rank/world size")
        self.rank, self.world_size, self.group = rank, world_size, group
        if world_size > 1 and (group is None or dist.get_backend(group) != "gloo"):
            raise ValueError("distributed interactive input requires a separate Gloo group")

    def receive(self, read: Callable[[], str]) -> str:
        payload = [None]
        if self.rank == 0:
            try:
                payload[0] = read()
            except (EOFError, KeyboardInterrupt):
                payload[0] = "/exit"
        if self.world_size > 1:
            source = dist.get_global_rank(self.group, 0)
            dist.broadcast_object_list(payload, src=source, group=self.group)
        return payload[0]


def run_interactive(
    session: ChatSession,
    channel: ChatChannel,
    *,
    read: Callable[[], str],
    output: TextIO,
    on_turn: Callable[[dict], None] | None = None,
    streamer_factory: Callable[..., TextStreamer] = TextStreamer,
) -> None:
    def write(text: str) -> None:
        if channel.rank == 0:
            output.write(text)
            output.flush()

    help_text = "/clear（或 /reset）清空会话；/exit（或 /quit）退出；/help 查看帮助。\n"
    write("DeepSeek V4.1 已就绪，可以连续对话。\n" + help_text)
    session_number = 1
    while True:
        write("你> ")
        prompt = channel.receive(read)
        command = prompt.strip()
        if command in {"/exit", "/quit"}:
            write("已退出。\n")
            return
        if command in {"/clear", "/reset"}:
            session.clear()
            session_number += 1
            write("会话已清空。\n")
            continue
        if command == "/help":
            write(help_text)
            continue
        if not command:
            continue
        try:
            turn = session.prepare(prompt)
        except ChatInputError as exc:
            write(str(exc) + "\n")
            continue
        write("DeepSeek> ")
        streamer = streamer_factory(session.tokenizer, write) if channel.rank == 0 else None
        result = session.respond(turn, on_token=streamer.push if streamer is not None else None)
        if streamer is not None:
            streamer.finish()
        write("\n")
        if result["finish_reason"] == "length":
            write(f"[已达到本轮 {session.max_new_tokens} token 上限，可继续提问。]\n")
        if on_turn is not None and channel.rank == 0:
            on_turn({"session": session_number, **result})
