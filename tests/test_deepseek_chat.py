"""多轮 token 边界、缓存提交、终端输入同步及 UTF-8 流式解码回归。"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mysglang.deepseek_chat import ChatChannel, TextStreamer, run_interactive
from mysglang.deepseek_cli import generate, main
from mysglang.modeling.deepseek_v41.generation import (
    BOS_TOKEN,
    EOS_TOKEN,
    ChatInputError,
    ChatSession,
    format_prompt,
)
from mysglang.modeling.deepseek_v41.model import ReferenceCache
from mysglang.modeling.parallel import TensorParallelContext

HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class TinyTokenizer:
    markers = {BOS_TOKEN: 0, EOS_TOKEN: 1, "<｜User｜>": 2, "<｜Assistant｜>": 3, "</think>": 4}

    def token_to_id(self, text: str) -> int | None:
        return self.markers.get(text)

    def encode(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        parts = re.split("(" + "|".join(map(re.escape, self.markers)) + ")", text)
        ids = []
        for part in parts:
            if part in self.markers:
                ids.append(self.markers[part])
            else:
                ids.extend(8 + ord(char) % 48 for char in part)
        return SimpleNamespace(ids=ids)

    def decode(self, ids, *, skip_special_tokens=True):
        return "".join(chr(65 + token % 26) for token in ids if token >= 5)


class ScriptedModel:
    def __init__(self, predictions=(10, 1, 11, 1, 12, 1), *, max_seq_len=200, distributed=False):
        self.embed_weight = torch.empty(1)
        tp = TensorParallelContext.from_distributed() if distributed else TensorParallelContext()
        self.context = SimpleNamespace(tp=tp, backend="torch")
        self.max_seq_len = max_seq_len
        self.owner = object()
        self.predictions = predictions
        self.inputs: list[list[int]] = []
        self.fail_on_call: int | None = None

    def new_cache(self):
        return ReferenceCache(self.owner)

    def __call__(self, tokens, cache):
        self.inputs.append(tokens[0].tolist())
        cache.tokens = (
            tokens.clone() if cache.tokens is None else torch.cat((cache.tokens, tokens), 1)
        )
        if self.fail_on_call == len(self.inputs):
            raise RuntimeError("injected generation failure")
        result = torch.full((1, 64), -100.0)
        result[0, self.predictions[(len(self.inputs) - 1) % len(self.predictions)]] = 100
        return result


class SimpleStreamer:
    def __init__(self, tokenizer, write):
        self.tokenizer, self.write = tokenizer, write

    def push(self, token):
        self.write(self.tokenizer.decode([token]))

    def finish(self):
        pass


def _distributed_chat_worker(rank: int, root: str):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(root / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    try:
        predictions = (10, 1, 11, 1, 12, 1) if rank == 0 else (20, 21, 22, 23, 24, 25)
        model = ScriptedModel(predictions, distributed=True)
        session = ChatSession(model, TinyTokenizer(), max_new_tokens=3)
        prompts = iter(["hello", "follow", "/help", "", "x" * 500, "/clear", "again", "/exit"])
        reads, generations = [], []

        def read():
            if rank != 0:
                raise AssertionError("nonzero rank read stdin")
            if not reads:
                # 用户等待超过数据面 group timeout，仍不应有数据面 collective 超时。
                time.sleep(3)
            value = next(prompts)
            reads.append(value)
            return value

        output = io.StringIO()
        run_interactive(
            session,
            ChatChannel(rank=rank, world_size=2, group=control),
            read=read,
            output=output,
            on_turn=generations.append,
            streamer_factory=SimpleStreamer,
        )
        (root / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "inputs": model.inputs,
                    "history": session.history_tokens,
                    "cache_length": session.cache.length,
                    "turns": session.turn_count,
                    "reads": reads,
                    "output": output.getvalue(),
                    "generations": generations,
                }
            )
        )
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


class ChatSessionTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()

    def test_second_turn_prefills_only_uncached_eos_and_new_user_tokens(self):
        model = ScriptedModel()
        chat = ChatSession(model, self.tokenizer, max_new_tokens=3)
        first = chat.respond(chat.prepare("hello"))
        history, previous_length = chat.history_tokens, chat.cache.length
        self.assertEqual(len(history), previous_length + 1)
        self.assertEqual(history[-1], 1)
        second_plan = chat.prepare("follow")
        self.assertEqual(second_plan.token_ids[: len(history)], history)
        second = chat.respond(second_plan)
        self.assertEqual(first["finish_reason"], "eos")
        self.assertEqual(second["reused_tokens"], previous_length)
        self.assertEqual(
            model.inputs[2],
            [1] + self.tokenizer.encode(format_prompt("follow", first_turn=False)).ids,
        )
        self.assertEqual(chat.history_tokens.count(0), 1)
        self.assertEqual(chat.history_tokens.count(1), 2)
        self.assertEqual(chat.turn_count, 2)

    def test_length_stop_closes_assistant_and_preserves_unforwarded_content(self):
        model = ScriptedModel((10, 11, 12, 13))
        chat = ChatSession(model, self.tokenizer, max_new_tokens=2)
        first = chat.respond(chat.prepare("hello"))
        self.assertEqual(first["finish_reason"], "length")
        self.assertEqual(chat.history_tokens[-2:], (11, 1))
        self.assertEqual(len(chat.history_tokens), chat.cache.length + 2)
        chat.respond(chat.prepare("next"))
        self.assertEqual(model.inputs[2][:3], [11, 1, 2])

    def test_clear_resets_cache_history_and_rejects_stale_prepared_turns(self):
        model = ScriptedModel()
        chat = ChatSession(model, self.tokenizer, max_new_tokens=3)
        chat.respond(chat.prepare("first"))
        stale = chat.prepare("stale")
        previous = chat.cache
        chat.clear()
        self.assertIs(chat.model, model)
        self.assertIsNot(chat.cache, previous)
        self.assertEqual((chat.history_tokens, chat.cache.length, chat.turn_count), ((), 0, 0))
        with self.assertRaisesRegex(ValueError, "changed chat session"):
            chat.respond(stale)
        self.assertEqual(chat.prepare("new").token_ids[0], 0)

    def test_failed_decode_does_not_commit_a_partial_conversation(self):
        model = ScriptedModel()
        chat = ChatSession(model, self.tokenizer, max_new_tokens=3)
        chat.respond(chat.prepare("first"))
        cache, history, tokens = chat.cache, chat.history_tokens, chat.cache.tokens.clone()
        model.fail_on_call = len(model.inputs) + 2
        with self.assertRaisesRegex(RuntimeError, "injected"):
            chat.respond(chat.prepare("second"))
        self.assertIs(chat.cache, cache)
        self.assertEqual(chat.history_tokens, history)
        self.assertEqual(chat.turn_count, 1)
        torch.testing.assert_close(chat.cache.tokens, tokens)

    def test_context_limit_rejects_without_forward_or_history_truncation(self):
        model = ScriptedModel(max_seq_len=32)
        chat = ChatSession(model, self.tokenizer, max_new_tokens=3)
        chat.respond(chat.prepare("ok"))
        calls, history, cache = len(model.inputs), chat.history_tokens, chat.cache
        with self.assertRaisesRegex(ChatInputError, "现有历史未改变"):
            chat.prepare("too long" * 10)
        self.assertEqual(len(model.inputs), calls)
        self.assertEqual(chat.history_tokens, history)
        self.assertIs(chat.cache, cache)
        self.assertIsNotNone(chat.prepare("ok"))

    def test_boundary_reserves_end_marker_even_after_length_stop(self):
        prompt_length = len(self.tokenizer.encode(format_prompt("ok")).ids)
        model = ScriptedModel((10, 11), max_seq_len=prompt_length + 3)
        chat = ChatSession(model, self.tokenizer, max_new_tokens=2)
        chat.respond(chat.prepare("ok"))
        self.assertEqual(len(chat.history_tokens), model.max_seq_len)
        with self.assertRaises(ChatInputError):
            chat.prepare("next")

    def test_legacy_one_shot_raw_prompt_and_independent_cache(self):
        model = ScriptedModel()
        one = generate(model, self.tokenizer, "hello", max_new_tokens=3, raw=False)
        two = generate(model, self.tokenizer, "raw", max_new_tokens=3, raw=True)
        self.assertEqual(one["reused_tokens"], 0)
        self.assertEqual(two["reused_tokens"], 0)
        self.assertEqual(model.inputs[0], self.tokenizer.encode(format_prompt("hello")).ids)
        self.assertEqual(model.inputs[2], self.tokenizer.encode("raw").ids)

    @unittest.skipUnless(HAS_NUMPY, "requires numpy for DeepSeek reference")
    def test_incremental_multiturn_matches_full_prefix_fp32_oracle(self):
        from mysglang.modeling.deepseek_v41 import DeepSeekV41Config, DeepSeekV41ReferenceModel

        torch.manual_seed(19)
        torch.set_num_threads(2)
        config = DeepSeekV41Config.tiny()
        reference = DeepSeekV41ReferenceModel(
            config, token_map=list(range(config.vocab_size)), max_seq_len=256
        )

        class Adapter:
            context = SimpleNamespace(tp=TensorParallelContext(), backend="torch")
            embed_weight = reference.embed.weight
            max_seq_len = 256
            new_cache = reference.new_cache

            def __call__(self, tokens, cache):
                return reference(tokens, cache)[:, -1]

        chat = ChatSession(Adapter(), self.tokenizer, max_new_tokens=3)
        for prompt in ("hi", "follow", "last"):
            turn = chat.prepare(prompt)
            prefix, expected = list(turn.token_ids), []
            for _ in range(chat.max_new_tokens):
                token = reference(torch.tensor([prefix]))[:, -1].argmax(-1).item()
                prefix.append(token)
                expected.append(token)
                if token == 1:
                    break
            actual = chat.respond(turn)
            self.assertEqual(actual["token_ids"], expected)


class ChatControlTest(unittest.TestCase):
    def test_eof_and_keyboard_interrupt_become_exit_commands(self):
        channel = ChatChannel(rank=0, world_size=1)
        for error in (EOFError, KeyboardInterrupt):

            def read(error=error):
                raise error()

            self.assertEqual(channel.receive(read), "/exit")

    def test_blank_help_clear_and_eof(self):
        model = ScriptedModel()
        chat = ChatSession(model, TinyTokenizer(), max_new_tokens=3)
        prompts = iter(["", "/help", "question", "/clear"])

        def read():
            try:
                return next(prompts)
            except StopIteration:
                raise EOFError from None

        output = io.StringIO()
        run_interactive(
            chat,
            ChatChannel(rank=0, world_size=1),
            read=read,
            output=output,
            streamer_factory=SimpleStreamer,
        )
        self.assertEqual(len(model.inputs), 2)
        self.assertIn("会话已清空", output.getvalue())
        self.assertTrue(output.getvalue().endswith("已退出。\n"))
        self.assertEqual(chat.history_tokens, ())

    def test_two_ranks_share_history_commands_sampling_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_chat_worker, args=(directory,), nprocs=2, join=True)
            ranks = [json.loads((Path(directory) / f"rank-{r}.json").read_text()) for r in (0, 1)]
        for key in ("inputs", "history", "cache_length", "turns"):
            self.assertEqual(ranks[0][key], ranks[1][key])
        self.assertEqual(ranks[1]["reads"], [])
        self.assertEqual(ranks[1]["output"], "")
        self.assertEqual(len(ranks[0]["inputs"]), 6)
        self.assertEqual([item["session"] for item in ranks[0]["generations"]], [1, 1, 2])
        self.assertEqual(ranks[0]["generations"][2]["reused_tokens"], 0)
        self.assertGreater(ranks[0]["generations"][1]["reused_tokens"], 0)
        self.assertIn("上下文空间不足", ranks[0]["output"])

    def test_invalid_cli_modes_fail_before_model_loading(self):
        cases = [
            [],
            ["--interactive", "--prompt", "x"],
            ["--interactive", "--raw-prompt"],
            ["--interactive", "--compare-torch"],
            ["--interactive", "--max-new-tokens", "0"],
            ["--interactive", "--max-new-tokens", "512", "--max-seq-len", "512"],
        ]
        for flags in cases:
            with (
                self.subTest(flags=flags),
                redirect_stderr(io.StringIO()),
                patch(
                    "mysglang.deepseek_cli.torch.cuda.is_available",
                    side_effect=AssertionError("invalid arguments must fail before CUDA setup"),
                ),
            ):
                with self.assertRaises(SystemExit) as caught:
                    main(["--model", "unused", "--config", "unused", *flags])
                self.assertEqual(caught.exception.code, 2)


@unittest.skipUnless(HAS_TOKENIZERS, "requires tokenizers for UTF-8 streaming")
class StreamingTest(unittest.TestCase):
    def tokenizer(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            ["hello world", "你好 世界"],
            trainers.BpeTrainer(
                vocab_size=270,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=list(TinyTokenizer.markers),
            ),
        )
        return tokenizer

    def test_cjk_emoji_and_latin_stream_matches_complete_decode(self):
        tokenizer = self.tokenizer()
        text = "我是DeepSeek，很高兴认识你！😀\nHello, 世界👩🏽‍💻."
        ids = tokenizer.encode(text, add_special_tokens=False).ids + [
            tokenizer.token_to_id(EOS_TOKEN)
        ]
        pieces = []
        streamer = TextStreamer(tokenizer, pieces.append)
        for token in ids:
            streamer.push(token)
            self.assertNotIn("\ufffd", "".join(pieces))
        streamer.finish()
        self.assertEqual("".join(pieces), tokenizer.decode(ids, skip_special_tokens=True))
        self.assertEqual("".join(pieces), text)
        self.assertGreater(len(pieces), 1)

    def test_truncated_utf8_is_flushed_consistently_at_turn_end(self):
        tokenizer = self.tokenizer()
        ids = tokenizer.encode("😀", add_special_tokens=False).ids
        pieces = []
        streamer = TextStreamer(tokenizer, pieces.append)
        for token in ids[:-1]:
            streamer.push(token)
        self.assertEqual(pieces, [])
        streamer.finish()
        self.assertEqual("".join(pieces), tokenizer.decode(ids[:-1], skip_special_tokens=True))


if __name__ == "__main__":
    unittest.main()
