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
    # 只捕获这些精确 batch size；不补 dummy request，也不改变调度顺序。
    decode_cuda_graph_batch_sizes: tuple[int, ...] = ()
    # 0 表示关闭。开启后只为 greedy 请求生成无模型 n-gram 草稿。
    speculative_ngram_max_tokens: int = 0
    speculative_ngram_min_match: int = 2
    speculative_ngram_max_match: int = 8

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
        buckets = self.decode_cuda_graph_batch_sizes
        if not isinstance(buckets, tuple):
            raise TypeError("decode_cuda_graph_batch_sizes must be a tuple")
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in buckets
        ):
            raise ValueError("CUDA Graph batch sizes must be positive integers")
        if tuple(sorted(set(buckets))) != buckets:
            raise ValueError("CUDA Graph batch sizes must be sorted and unique")
        if buckets and buckets[-1] > self.max_running_requests:
            raise ValueError("CUDA Graph batch size cannot exceed max_running_requests")
        for name in (
            "speculative_ngram_max_tokens",
            "speculative_ngram_min_match",
            "speculative_ngram_max_match",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.speculative_ngram_max_tokens < 0:
            raise ValueError("speculative_ngram_max_tokens must be non-negative")
        if self.speculative_ngram_min_match <= 0:
            raise ValueError("speculative_ngram_min_match must be positive")
        if self.speculative_ngram_max_match < self.speculative_ngram_min_match:
            raise ValueError(
                "speculative_ngram_max_match must be at least speculative_ngram_min_match"
            )
