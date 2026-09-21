from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from mysglang.cli import _parser, _rank_device, _read_distributed_launch


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


if __name__ == "__main__":
    unittest.main()
