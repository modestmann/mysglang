from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from mysglang.cli import _LoadedRuntime, _parser, _rank_device, _read_distributed_launch, _run
from mysglang.scheduler import SchedulerConfig
from mysglang.serving import GenerationService
from mysglang.tokenizer import ByteTokenizer
from tests.helpers import make_model


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
