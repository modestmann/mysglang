"""Reproducible end-to-end benchmark for one or more Qwen3 TP ranks.

Examples:
  python benchmarks/benchmark_generation.py --model /root/models/Qwen3-0.6B
  torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_generation.py \
      --model /root/models/Qwen3-0.6B --tensor-parallel-size 4

Only rank 0 writes a result. Other ranks remain in the TP scheduler worker loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from mysglang.cli import _load_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/raw.jsonl"))
    parser.add_argument("--name", default="generation")
    parser.add_argument("--backend", choices=("flash", "torch", "auto"), default="flash")
    parser.add_argument(
        "--moe-dispatch",
        choices=("naive", "sorted", "grouped", "triton_grouped", "all_to_all"),
        default="sorted",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-pages", type=int, default=256)
    parser.add_argument("--prefill-token-budget", type=int)
    parser.add_argument("--warmup-new-tokens", type=int, default=4)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system")
    parser.add_argument("--prompt")
    parser.add_argument("--gpu-sample-ms", type=int, default=200)
    return parser


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _summary_ms(values: list[float]) -> dict[str, float | None]:
    milliseconds = [value * 1000 for value in values]
    return {
        "mean": statistics.fmean(milliseconds) if milliseconds else None,
        "p50": _percentile(milliseconds, 0.50),
        "p95": _percentile(milliseconds, 0.95),
        "max": max(milliseconds) if milliseconds else None,
    }


def _run_text(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


async def _gpu_sampler(
    interval_ms: int,
    stop: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    query = (
        "index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw"
    )
    while not stop.is_set():
        output = await asyncio.to_thread(
            _run_text,
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        )
        if output:
            timestamp = time.time()
            for line in output.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 7:
                    samples.append(
                        {
                            "timestamp": timestamp,
                            "index": int(fields[0]),
                            "uuid": fields[1],
                            "name": fields[2],
                            "memory_used_mib": float(fields[3]),
                            "memory_total_mib": float(fields[4]),
                            "utilization_gpu_percent": float(fields[5]),
                            "power_draw_w": None if fields[6] == "[N/A]" else float(fields[6]),
                        }
                    )
        try:
            await asyncio.wait_for(stop.wait(), interval_ms / 1000)
        except TimeoutError:
            pass


def _fixed_prompt_ids(service: Any, target: int) -> tuple[int, ...]:
    if target <= 0:
        raise ValueError("--prompt-tokens must be positive")
    seed = tuple(service.tokenizer.encode("性能测试：请简洁介绍大语言模型推理。"))
    return (seed * ((target + len(seed) - 1) // len(seed)))[:target]


async def _consume(session: Any, started: float) -> dict[str, Any]:
    token_ids: list[int] = []
    token_times: list[float] = []
    async for chunk in session:
        token_ids.append(chunk.token_id)
        token_times.append(time.perf_counter())
    return {
        "request_id": session.request.request_id,
        "token_ids": token_ids,
        "ttft_s": token_times[0] - started,
        "e2e_s": token_times[-1] - started,
        "itl_s": [right - left for left, right in zip(token_times, token_times[1:])],
    }


async def _one_batch(
    service: Any,
    args: argparse.Namespace,
    output_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    prompt_ids = _fixed_prompt_ids(service, args.prompt_tokens)
    sessions = [
        service._start_token_ids(
            prompt_ids,
            max_new_tokens=output_tokens,
            eos_token_id=None,
            ignore_eos=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed + index,
        )
        for index in range(args.concurrency)
    ]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    results = await asyncio.gather(*(_consume(session, started) for session in sessions))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return results, time.perf_counter() - started


def _gpu_summary(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index in sorted({sample["index"] for sample in samples}):
        current = [sample for sample in samples if sample["index"] == index]
        result.append(
            {
                "index": index,
                "uuid": current[0]["uuid"],
                "name": current[0]["name"],
                "sample_count": len(current),
                "peak_memory_used_mib": max(x["memory_used_mib"] for x in current),
                "mean_utilization_gpu_percent": statistics.fmean(
                    x["utilization_gpu_percent"] for x in current
                ),
                "peak_power_draw_w": max(
                    (x["power_draw_w"] for x in current if x["power_draw_w"] is not None),
                    default=None,
                ),
            }
        )
    return result


def _scheduler_delta(before: Any, after: Any) -> dict[str, int]:
    cumulative = {
        "finished_requests",
        "aborted_requests",
        "model_forwards",
        "prefill_input_tokens",
        "cuda_graph_captures",
        "cuda_graph_replays",
    }
    result = asdict(after)
    for name in cumulative:
        result[name] -= getattr(before, name)
    return result


async def _driver(service: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup_new_tokens:
        await _one_batch(service, args, args.warmup_new_tokens)
        service.scheduler.reset_prefix_cache()
    scheduler_before = service.stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    samples: list[dict[str, Any]] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(_gpu_sampler(args.gpu_sample_ms, stop, samples))
    try:
        requests, elapsed = await _one_batch(service, args, args.max_new_tokens)
    finally:
        stop.set()
        await sampler

    ttft = [request["ttft_s"] for request in requests]
    e2e = [request["e2e_s"] for request in requests]
    itl = [value for request in requests for value in request["itl_s"]]
    output_tokens = sum(len(request["token_ids"]) for request in requests)
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "config": {
            "model": str(args.model.resolve()),
            "backend": args.backend,
            "moe_dispatch": args.moe_dispatch,
            "dtype": args.dtype,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "concurrency": args.concurrency,
            "prompt_tokens_per_request": args.prompt_tokens,
            "max_new_tokens_per_request": args.max_new_tokens,
            "num_pages": args.num_pages,
            "prefill_token_budget": args.prefill_token_budget,
            "cuda_graph": args.cuda_graph,
            "sampling": {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed": args.seed,
            },
        },
        "metrics": {
            "elapsed_s": elapsed,
            "requests_per_s": len(requests) / elapsed,
            "output_tokens": output_tokens,
            "output_tokens_per_s": output_tokens / elapsed,
            "ttft_ms": _summary_ms(ttft),
            "itl_ms": _summary_ms(itl),
            "e2e_ms": _summary_ms(e2e),
            "rank0_peak_torch_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
            ),
            "rank0_peak_torch_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None
            ),
        },
        "scheduler": _scheduler_delta(scheduler_before, service.stats),
        "gpu_summary": _gpu_summary(samples),
        "requests": requests,
        "environment": {
            "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "flash_attn": _run_text(
                [os.sys.executable, "-c", "import flash_attn; print(flash_attn.__version__)"]
            ),
            "nvidia_smi": _run_text(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
            "gpu_topology": _run_text(["nvidia-smi", "topo", "-m"]),
            "command": " ".join(os.sys.argv),
        },
    }


async def _main(args: argparse.Namespace) -> None:
    args.max_running_requests = args.concurrency
    args.decode_cuda_graph_batch_sizes = (args.concurrency,) if args.cuda_graph else ()
    runtime = _load_runtime(args)
    try:
        if runtime.scheduler is not None and not runtime.scheduler.is_driver:
            runtime.scheduler.run_worker_loop()
            return
        if runtime.service is None:
            raise RuntimeError("rank 0 did not create a generation service")
        result = await _driver(runtime.service, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps({"output": str(args.output), **result["metrics"]}, indent=2))
    finally:
        try:
            if runtime.scheduler is not None and runtime.scheduler.is_driver:
                runtime.scheduler.shutdown()
        finally:
            if runtime.owns_process_group and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(_main(_parser().parse_args()))
