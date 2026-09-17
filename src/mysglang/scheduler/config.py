from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    max_running_requests: int = 8
    prefill_token_budget: int = 64
    max_consecutive_prefill_steps: int = 1#当 Prefill 和 Decode 都有工作时，最多允许连续执行多少个 Prefill step，之后必须执行一次 Decode

    def __post_init__(self) -> None:
        for name in (
            "max_running_requests",
            "prefill_token_budget",
            "max_consecutive_prefill_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PagedSchedulerConfig(SchedulerConfig):
    num_pages: int = 64
    page_size: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("num_pages", "page_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
