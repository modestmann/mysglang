#各rank同步器
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from mysglang.core import IncrementalOutput, Request
from mysglang.modeling.qwen3 import Qwen3ForCausalLM

from .config import SchedulerConfig
from .coordination import SchedulerBatchPlan
from .scheduler import Scheduler, SchedulerStats, SchedulerStep


@dataclass(frozen=True)
class _Command:
    name: str
    payload: object = None


@dataclass(frozen=True)
class _CommandOutcome:
    error: str | None
    result: object


class _ProcessGroupCoordinator:
    """Small CPU control plane kept separate from model tensor collectives."""

    def __init__(self, process_group: dist.ProcessGroup | None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        backend = str(dist.get_backend(process_group))
        if backend != "gloo":
            raise RuntimeError(
                "TP scheduler control messages require a Gloo process group; "
                "keep model tensors on the NCCL TP group"
            )
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.source_rank = (
            0 if process_group is None else dist.get_global_rank(process_group, 0)
        )

    @property
    def is_driver(self) -> bool:
        return self.rank == 0

    def send_command(self, command: _Command) -> None:
        if not self.is_driver:
            raise RuntimeError("only TP rank 0 may send scheduler commands")
        objects: list[object] = [command]
        self._broadcast_objects(objects)

    def receive_command(self) -> _Command:
        if self.is_driver:
            raise RuntimeError("TP rank 0 does not receive worker commands")
        objects: list[object] = [None]
        self._broadcast_objects(objects)
        command = objects[0]
        if not isinstance(command, _Command):
            raise RuntimeError("received an invalid TP scheduler command")
        return command

    def validate_batch(self, plan: SchedulerBatchPlan) -> None:
        objects: list[object] = [plan if self.is_driver else None]
        self._broadcast_objects(objects)
        authoritative = objects[0]
        if not isinstance(authoritative, SchedulerBatchPlan):
            raise RuntimeError("rank 0 broadcast an invalid SchedulerBatchPlan")

        mismatches: list[object] = [None] * self.world_size
        self._all_gather_objects(mismatches, plan != authoritative)
        bad_ranks = tuple(index for index, mismatch in enumerate(mismatches) if mismatch)
        if bad_ranks:
            raise RuntimeError(
                "TP ranks produced different SchedulerBatchPlan values; "
                f"mismatched ranks: {bad_ranks}"
            )

    def sync_token_ids(self, token_ids: tuple[int, ...]) -> tuple[int, ...]:
        objects: list[object] = [token_ids if self.is_driver else None]
        self._broadcast_objects(objects)
        result = objects[0]
        if not isinstance(result, tuple) or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in result
        ):
            raise RuntimeError("rank 0 broadcast invalid sampled token IDs")
        return result

    def complete_command(
        self,
        result: object,
        error: BaseException | None,
    ) -> None:
        local = _CommandOutcome(
            error=None if error is None else f"{type(error).__name__}: {error}",
            result=result,
        )
        outcomes: list[object] = [None] * self.world_size
        self._all_gather_objects(outcomes, local)
        failures = tuple(
            (rank, outcome.error)
            for rank, outcome in enumerate(outcomes)
            if isinstance(outcome, _CommandOutcome) and outcome.error is not None
        )
        if failures:
            raise RuntimeError(f"TP scheduler command failed: {failures}") from error
        if any(not isinstance(outcome, _CommandOutcome) for outcome in outcomes):
            raise RuntimeError("received an invalid TP scheduler command outcome")
        if any(outcome.result != outcomes[0].result for outcome in outcomes[1:]):
            raise RuntimeError("TP ranks produced different scheduler command results")

    def _broadcast_objects(self, objects: list[object]) -> None:
        # Scheduler.step runs under torch.inference_mode(). Object collectives
        # allocate temporary CPU tensors and then update them in-place, which
        # PyTorch rejects for inference tensors. Keep the model forward in
        # inference mode, but create control-plane temporaries as normal tensors.
        with torch.inference_mode(False):
            dist.broadcast_object_list(
                objects,
                src=self.source_rank,
                group=self.process_group,
            )

    def _all_gather_objects(self, output: list[object], value: object) -> None:
        with torch.inference_mode(False):
            dist.all_gather_object(output, value, group=self.process_group)


class TensorParallelScheduler:
    """Rank-0 facade over one deterministically mirrored Scheduler per TP rank.

    Rank 0 owns the public API. Nonzero ranks enter :meth:`run_worker_loop` and
    apply the commands broadcast by rank 0. Each rank owns an independent KV
    pool containing only its local KV heads, while allocator and radix metadata
    remain identical because every scheduler command is replayed in order.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        config: SchedulerConfig,
        *,
        control_group: dist.ProcessGroup | None = None,
    ) -> None:
        tensor_parallel = model.tensor_parallel
        if not tensor_parallel.enabled:
            raise ValueError("TensorParallelScheduler requires a TP model")
        self._coordinator = _ProcessGroupCoordinator(control_group)
        if self._coordinator.rank != tensor_parallel.rank:
            raise ValueError("control-group rank does not match the model TP rank")
        if self._coordinator.world_size != tensor_parallel.world_size:
            raise ValueError("control-group size does not match the model TP size")
        self._scheduler = Scheduler(model, config, coordinator=self._coordinator)
        self._shutdown = False

    @property
    def rank(self) -> int:
        return self._coordinator.rank

    @property
    def is_driver(self) -> bool:
        return self._coordinator.is_driver

    @property
    def has_work(self) -> bool:
        self._require_driver()
        return self._scheduler.has_work

    @property
    def stats(self) -> SchedulerStats:
        self._require_driver()
        return self._scheduler.stats

    def validate(self, request: Request) -> None:
        self._require_driver()
        self._scheduler.validate(request)

    def add(self, request: Request) -> None:
        self._dispatch(_Command("add", request))

    def abort(self, request_id: str) -> IncrementalOutput | None:
        result = self._dispatch(_Command("abort", request_id))
        if result is not None and not isinstance(result, IncrementalOutput):
            raise RuntimeError("distributed abort returned an invalid result")
        return result

    def step(self) -> SchedulerStep | None:
        result = self._dispatch(_Command("step"))
        if result is not None and not isinstance(result, SchedulerStep):
            raise RuntimeError("distributed step returned an invalid result")
        return result

    def reset_prefix_cache(self) -> None:
        self._dispatch(_Command("reset_prefix_cache"))

    def check_integrity(self) -> None:
        self._scheduler.check_integrity()

    def shutdown(self) -> None:
        self._require_driver()
        if self._shutdown:
            return
        self._dispatch(_Command("shutdown"))
        self._shutdown = True

    def run_worker_loop(self) -> None:
        if self.is_driver:
            raise RuntimeError("TP rank 0 drives commands and must not enter the worker loop")
        if self._shutdown:
            raise RuntimeError("TP worker has already shut down")
        while True:
            command = self._coordinator.receive_command()
            self._execute_and_complete(command)
            if command.name == "shutdown":
                self._shutdown = True
                return

    def _dispatch(self, command: _Command) -> Any:
        self._require_driver()
        if self._shutdown:
            raise RuntimeError("TP scheduler has shut down")
        self._coordinator.send_command(command)
        return self._execute_and_complete(command)

    def _execute_and_complete(self, command: _Command) -> Any:
        result: object = None
        error: BaseException | None = None
        try:
            result = self._apply_command(command)
        except BaseException as exc:
            error = exc
        self._coordinator.complete_command(result, error)
        return result

    def _apply_command(self, command: _Command) -> object:
        if command.name == "add":
            if not isinstance(command.payload, Request):
                raise TypeError("add command requires a Request")
            self._scheduler.add(command.payload)
            return None
        if command.name == "abort":
            if not isinstance(command.payload, str):
                raise TypeError("abort command requires a request ID")
            return self._scheduler.abort(command.payload)
        if command.name == "step":
            return self._scheduler.step()
        if command.name == "reset_prefix_cache":
            self._scheduler.reset_prefix_cache()
            return None
        if command.name == "shutdown":
            # Every rank must drop captured NCCL graphs before the caller destroys
            # the process group. Otherwise graph-owned communicators can keep worker
            # processes and their GPU allocations alive after benchmark completion.
            self._scheduler.close()
            return None
        raise ValueError(f"unknown TP scheduler command: {command.name}")

    def _require_driver(self) -> None:
        if not self.is_driver:
            raise RuntimeError("only TP rank 0 exposes the scheduler API")
