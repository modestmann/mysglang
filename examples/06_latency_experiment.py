"""Chapter 6 experiment: report online latency metrics for a mixed workload."""

import math
import time

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM
from mysglang.scheduler import ContinuousBatchScheduler, SchedulerConfig
from mysglang.tokenizer import ByteTokenizer


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(22)
    tokenizer = ByteTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCausalLM(
        ModelConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=176,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    ).to(device).eval()
    scheduler = ContinuousBatchScheduler(
        model,
        SchedulerConfig(max_running_requests=8, prefill_token_budget=16),
    )
    workload = [
        ("seed", "seed", 12),
        ("short", "tiny", 2),
        ("long-prompt", "x" * 48, 4),
        ("chinese", "你好世界" * 4, 6),
        ("long-output", "generate", 12),
        ("medium", "m" * 24, 5),
        ("short-2", "ok", 3),
        ("mixed", "调度器" * 5, 7),
    ]
    submitted: dict[str, float] = {}
    token_times: dict[str, list[float]] = {request_id: [] for request_id, _, _ in workload}
    completed: dict[str, float] = {}

    def submit(item: tuple[str, str, int]) -> None:
        request_id, prompt, max_new_tokens = item
        submitted[request_id] = time.perf_counter()
        scheduler.add(
            Request.from_token_ids(
                request_id,
                tokenizer.encode(prompt),
                SamplingParams(max_new_tokens=max_new_tokens),
            )
        )

    experiment_start = time.perf_counter()
    submit(workload[0])
    scheduler.step()  # The seed request is decoding when the remaining requests arrive.
    first_time = time.perf_counter()
    token_times["seed"].append(first_time)

    for item in workload[1:]:
        submit(item)

    while scheduler.has_work:
        step = scheduler.step()
        now = time.perf_counter()
        for event in step.outputs:
            token_times[event.request_id].append(now)
            if event.finished:
                completed[event.request_id] = now
    experiment_end = time.perf_counter()

    ttft_ms = [
        (times[0] - submitted[request_id]) * 1_000
        for request_id, times in token_times.items()
    ]
    tpot_ms = [
        (times[-1] - times[0]) * 1_000 / (len(times) - 1)
        for times in token_times.values()
        if len(times) > 1
    ]
    e2e_ms = [
        (completed[request_id] - submitted[request_id]) * 1_000
        for request_id in submitted
    ]
    total_output_tokens = sum(len(times) for times in token_times.values())
    elapsed = experiment_end - experiment_start

    print(f"device: {device}")
    print(f"requests: {len(workload)}")
    print(f"output tokens: {total_output_tokens}")
    print(f"output throughput: {total_output_tokens / elapsed:.2f} token/s")
    print(f"TTFT p50/p95/p99: {percentile(ttft_ms, 0.50):.3f} / "
          f"{percentile(ttft_ms, 0.95):.3f} / {percentile(ttft_ms, 0.99):.3f} ms")
    print(f"TPOT p50/p95/p99: {percentile(tpot_ms, 0.50):.3f} / "
          f"{percentile(tpot_ms, 0.95):.3f} / {percentile(tpot_ms, 0.99):.3f} ms")
    print(f"E2E  p50/p95/p99: {percentile(e2e_ms, 0.50):.3f} / "
          f"{percentile(e2e_ms, 0.95):.3f} / {percentile(e2e_ms, 0.99):.3f} ms")
    print(f"max decode batch size: {scheduler.stats.max_decode_batch_size}")
    print(f"max wait steps: {scheduler.stats.max_wait_steps}")


if __name__ == "__main__":
    main()
