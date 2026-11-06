from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import torch

from benchmarks.benchmark_generation import _parser as benchmark_parser
from benchmarks.benchmark_generation import _resolve_kv_pool
from mysglang.cli import (
    _LoadedRuntime,
    _make_backend,
    _parser,
    _rank_device,
    _read_distributed_launch,
    _run,
    _scheduler_config,
)
from mysglang.modeling import TorchAttentionBackend
from mysglang.scheduler import SchedulerConfig
from mysglang.server import _server_parser
from mysglang.serving import GenerationService
from mysglang.tokenizer import ByteTokenizer
from tests.helpers import make_model


class CliBackendTest(unittest.TestCase):
    def test_auto_selects_flashinfer_only_when_available_on_cuda(self) -> None:
        for device, available in (("cpu", False), ("cpu", True), ("cuda", False), ("cuda", True)):
            with self.subTest(device=device, available=available):
                with (
                    patch(
                        "mysglang.cli.importlib.util.find_spec",
                        return_value=object() if available else None,
                    ) as find,
                    patch("mysglang.cli.FlashInferBackend") as flashinfer,
                ):
                    backend = _make_backend("auto", torch.device(device))
                    if device == "cuda" and available:
                        self.assertIs(backend, flashinfer.return_value)
                        flashinfer.assert_called_once_with()
                    else:
                        self.assertIsInstance(backend, TorchAttentionBackend)
                        flashinfer.assert_not_called()
                    if device == "cuda":
                        find.assert_called_once_with("flashinfer")
                    else:
                        find.assert_not_called()

    def test_explicit_backend_selection_does_not_silently_fall_back(self) -> None:
        with patch(
            "mysglang.cli.FlashInferBackend", side_effect=RuntimeError("missing flashinfer")
        ):
            with self.assertRaisesRegex(RuntimeError, "missing flashinfer"):
                _make_backend("flashinfer", torch.device("cuda"))
            self.assertIsInstance(
                _make_backend("torch", torch.device("cuda")), TorchAttentionBackend
            )
        with self.assertRaisesRegex(ValueError, "unknown attention backend"):
            _make_backend("flash", torch.device("cuda"))

    def test_entry_points_accept_flashinfer_and_reject_removed_backend(self) -> None:
        for factory, required in (
            (_parser, []),
            (_server_parser, []),
            (benchmark_parser, ["--model", "."]),
        ):
            with self.subTest(entry_point=factory.__module__):
                parser = factory()
                args = parser.parse_args([*required, "--backend", "flashinfer", "--page-size", "1"])
                self.assertEqual((args.backend, args.page_size), ("flashinfer", 1))
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parser.parse_args([*required, "--backend", "flash"])
                self.assertEqual(error.exception.code, 2)

    def test_scheduler_defaults_to_small_pages_and_honors_overrides(self) -> None:
        for flags, expected in (([], 16), (["--page-size", "1"], 1), (["--page-size", "32"], 32)):
            with self.subTest(flags=flags):
                args = _parser().parse_args(flags)
                self.assertEqual(_scheduler_config(args).page_size, expected)
        with self.assertRaises(ValueError):
            _scheduler_config(_parser().parse_args(["--page-size", "0"]))

    def test_benchmark_page_size_preserves_kv_token_budget(self) -> None:
        for flags, page_size in (([], 16), (["--page-size", "1"], 1)):
            with self.subTest(flags=flags):
                args = benchmark_parser().parse_args(["--model", ".", *flags])
                _resolve_kv_pool(args)
                self.assertEqual(args.backend, "flashinfer")
                self.assertEqual(args.page_size, page_size)
                self.assertEqual(args.num_pages * args.page_size, args.kv_tokens)


class CliTensorParallelLaunchTest(unittest.TestCase):
    def test_single_process_is_the_default(self) -> None:
        args = _parser().parse_args([])
        with patch.dict("os.environ", {}, clear=True):
            launch = _read_distributed_launch(args)

        self.assertEqual(launch.world_size, 1)
        self.assertEqual(launch.rank, 0)
        self.assertEqual(launch.local_rank, 0)

    def test_torchrun_environment_defines_the_tp_ranks(self) -> None:
        args = _parser().parse_args(["--tensor-parallel-size", "4"])
        environment = {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "2"}
        with patch.dict("os.environ", environment, clear=True):
            launch = _read_distributed_launch(args)

        self.assertEqual((launch.world_size, launch.rank, launch.local_rank), (4, 2, 2))
        self.assertEqual(_rank_device("cpu", launch), torch.device("cpu"))

    def test_explicit_tp_size_must_match_torchrun(self) -> None:
        args = _parser().parse_args(["--tensor-parallel-size", "4"])
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "torchrun launched WORLD_SIZE=1"):
                _read_distributed_launch(args)

    def test_indexed_cuda_device_is_rejected_for_tp(self) -> None:
        args = _parser().parse_args(["--tensor-parallel-size", "2"])
        environment = {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}
        with patch.dict("os.environ", environment, clear=True):
            launch = _read_distributed_launch(args)
        with self.assertRaisesRegex(ValueError, "LOCAL_RANK selects it"):
            _rank_device("cuda:0", launch)

    def test_ngram_speculative_flags_are_explicit(self) -> None:
        args = _parser().parse_args(
            [
                "--speculative-ngram-max-tokens",
                "4",
                "--speculative-ngram-min-match",
                "3",
                "--speculative-ngram-max-match",
                "12",
            ]
        )

        self.assertEqual(args.speculative_ngram_max_tokens, 4)
        self.assertEqual(args.speculative_ngram_min_match, 3)
        self.assertEqual(args.speculative_ngram_max_match, 12)


class CliShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_exit_after_generation_waits_for_worker(self) -> None:
        service = GenerationService(
            make_model(seed=789, vocab_size=256, max_position_embeddings=128),
            ByteTokenizer(),
            SchedulerConfig(max_running_requests=1, num_pages=48, page_size=4),
        )
        args = _parser().parse_args(["--max-new-tokens", "2"])

        with patch("mysglang.cli._load_runtime", return_value=_LoadedRuntime(service)):
            with patch("builtins.input", side_effect=["hi", "/exit"]):
                await _run(args)

        self.assertEqual(service.stats.finished_requests, 1)
        self.assertIsNone(service._worker_task)


if __name__ == "__main__":
    unittest.main()
