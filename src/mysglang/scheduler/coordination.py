#多rank需要核对的数据类型
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SchedulerBatchPlan:
    """CPU description that must agree before TP ranks enter one forward."""

    phase: str
    execution: str
    request_ids: tuple[str, ...]
    input_token_ids: tuple[int, ...]
    append_lengths: tuple[int, ...]
    logits_indices: tuple[int, ...] | None
    starts: tuple[int, ...]
    page_tables: tuple[tuple[int, ...], ...]


class SchedulerCoordinator(Protocol):
    """Coordination hooks used by a Scheduler mirrored on every TP rank."""

    def validate_batch(self, plan: SchedulerBatchPlan) -> None: ...

    def sync_token_ids(self, token_ids: tuple[int, ...]) -> tuple[int, ...]: ...
