from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for the single paged + radix scheduler."""

    max_running_requests: int = 8
    # 仅限制新增 Prefill token；混合 batch 另为每个 Decode 请求保留一个 token。
    prefill_token_budget: int = 64
    num_pages: int = 64
    page_size: int = 4

    def __post_init__(self) -> None:
        for name in (
            "max_running_requests",
            "prefill_token_budget",
            "num_pages",
            "page_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
