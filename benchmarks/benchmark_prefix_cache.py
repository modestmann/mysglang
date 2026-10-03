"""Measure what the radix prefix cache saves on a shared-prefix workload.

Every request is <shared prefix> + <its own suffix>, like many users sharing one
system prompt. Requests run one at a time so TTFT is not mixed with queueing.
Two passes over the same requests:

  cold: reset_prefix_cache() before every request, so each one prefills fully;
  warm: reset once, then the first request populates the cache for the rest.

Greedy outputs of the two passes must be identical; a mismatch means the cached
KV differs from recomputed KV.

Example:
  python benchmarks/benchmark_prefix_cache.py --model ../KuiperQwen3/artifacts/qwen3-0.6b/hf-source
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from benchmark_generation import (
    _consume,
    _resolve_kv_pool,
    _run_text,
)
from benchmark_generation import (
    _parser as _generation_parser,
)

from mysglang.cli import _load_runtime


def _parser():
    parser = _generation_parser()
    parser.description = __doc__
    parser.set_defaults(
        output=Path("benchmarks/results/prefix-cache.jsonl"),
        name="prefix-cache",
        max_new_tokens=16,
        kv_tokens=4096,
    )
    parser.add_argument("--prefix-tokens", type=int, default=1024)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--requests", type=int, default=8)
    return parser


def _repeat_to(ids: tuple[int, ...], target: int) -> tuple[int, ...]:
    return (ids * ((target + len(ids) - 1) // len(ids)))[:target]


def _prompts(service: Any, args: Any) -> list[tuple[int, ...]]:
    encode = service.tokenizer.encode
    prefix = _repeat_to(
        tuple(encode("你是一个推理服务助手。请遵守以下规范并简洁回答用户问题。")), args.prefix_tokens
    )
    return [
        prefix + _repeat_to(tuple(encode(f"用户 {index} 的问题：第 {index} 个请求。")), args.suffix_tokens)
        for index in range(args.requests)
    ]


async def _run_pass(service: Any, prompts: list[tuple[int, ...]], args: Any, cold: bool):
    service.scheduler.reset_prefix_cache()
    results = []
    for prompt in prompts:
        if cold:
            service.scheduler.reset_prefix_cache()
        prefilled_before = service.stats.prefill_input_tokens
        session = service._start_token_ids(
            prompt,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=None,
            ignore_eos=True,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            seed=args.seed,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result = await _consume(session, time.perf_counter())
        result["prompt_tokens"] = len(prompt)
        result["prefilled_tokens"] = service.stats.prefill_input_tokens - prefilled_before
        results.append(result)
    return results


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [r["ttft_s"] * 1000 for r in results]
    prompt = sum(r["prompt_tokens"] for r in results)
    prefilled = sum(r["prefilled_tokens"] for r in results)
    return {
        "ttft_ms_first": ttft[0],
        "ttft_ms_rest_p50": statistics.median(ttft[1:]) if len(ttft) > 1 else None,
        "ttft_ms_all_p50": statistics.median(ttft),
        "prompt_tokens": prompt,
        "prefilled_tokens": prefilled,
        "cache_hit_rate": 1 - prefilled / prompt,
    }


async def _main(args: Any) -> None:
    args.max_running_requests = 1
    args.decode_cuda_graph_batch_sizes = ()
    _resolve_kv_pool(args)
    runtime = _load_runtime(args)
    if runtime.scheduler is not None and not runtime.scheduler.is_driver:
        raise RuntimeError("run this benchmark with a single rank")
    service = runtime.service
    try:
        prompts = _prompts(service, args)
        await _run_pass(service, prompts[:1], args, cold=True)  # warmup: kernels, allocator
        cold = await _run_pass(service, prompts, args, cold=True)
        warm = await _run_pass(service, prompts, args, cold=False)
        identical = [c["token_ids"] == w["token_ids"] for c, w in zip(cold, warm)]
        result = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "name": args.name,
            "config": {
                "model": str(args.model.resolve()),
                "backend": args.backend,
                "dtype": args.dtype,
                "prefix_tokens": args.prefix_tokens,
                "suffix_tokens": args.suffix_tokens,
                "requests": args.requests,
                "max_new_tokens": args.max_new_tokens,
                "num_pages": args.num_pages,
                "page_size": args.page_size,
            },
            "cold": _summary(cold),
            "warm": _summary(warm),
            "greedy_outputs_identical": all(identical),
            "environment": {
                "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        c, w = result["cold"], result["warm"]
        print(
            f"prefix {args.prefix_tokens} + suffix {args.suffix_tokens} tokens, "
            f"{args.requests} requests, one at a time\n"
            f"cold: TTFT p50 {c['ttft_ms_all_p50']:.2f} ms, prefilled {c['prefilled_tokens']}"
            f"/{c['prompt_tokens']} tokens (hit rate {c['cache_hit_rate']:.1%})\n"
            f"warm: TTFT first {w['ttft_ms_first']:.2f} ms, rest p50 {w['ttft_ms_rest_p50']:.2f} ms, "
            f"prefilled {w['prefilled_tokens']}/{w['prompt_tokens']} "
            f"(hit rate {w['cache_hit_rate']:.1%})\n"
            f"TTFT speedup on cache hits: {c['ttft_ms_all_p50'] / w['ttft_ms_rest_p50']:.2f}x; "
            f"greedy outputs identical: {result['greedy_outputs_identical']}\n"
            f"appended to {args.output}"
        )
    finally:
        await service.aclose()


if __name__ == "__main__":
    asyncio.run(_main(_parser().parse_args()))
