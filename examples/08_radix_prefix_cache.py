"""Chapter 8 experiment: measure page-aligned prefix reuse."""

import argparse
import json
import time
from pathlib import Path

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM
from mysglang.scheduler import (
    PagedBatchScheduler,
    PagedSchedulerConfig,
    RadixBatchScheduler,
    RadixSchedulerConfig,
)
from mysglang.tokenizer import ByteTokenizer


def run_workload(scheduler, prompts: list[list[int]], max_new_tokens: int) -> dict:
    outputs: list[list[int]] = []
    started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        request_id = f"req-{index}"
        scheduler.add(
            Request.from_token_ids(
                request_id,
                prompt,
                SamplingParams(max_new_tokens=max_new_tokens),
            )
        )
        output: list[int] = []
        while scheduler.has_work:
            step = scheduler.step()
            output.extend(event.token_id for event in step.outputs)
            scheduler.check_integrity()
        outputs.append(output)
    elapsed = time.perf_counter() - started
    prefill_tokens = sum(step.input_tokens for step in scheduler.history if step.phase == "prefill")
    return {
        "elapsed_ms": elapsed * 1_000,
        "prefill_input_tokens": prefill_tokens,
        "model_forwards": scheduler.stats.model_forwards,
        "outputs": outputs,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    torch.manual_seed(808)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()
    model = (
        TinyCausalLM(
            ModelConfig(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=176,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=256,
            )
        )
        .to(device)
        .eval()
    )
    shared = "You are a concise assistant. Context: MySGLang lesson eight. Question: "
    suffixes = (
        "What is radix?",
        "Why align pages?",
        "When do we evict?",
        "How is KV reused?",
    )
    prompts = [tokenizer.encode(shared + suffix) for suffix in suffixes]
    common = dict(
        max_running_requests=4,
        prefill_token_budget=32,
        num_pages=128,
        page_size=4,
    )
    baseline = run_workload(PagedBatchScheduler(model, PagedSchedulerConfig(**common)), prompts, 8)
    radix_scheduler = RadixBatchScheduler(model, RadixSchedulerConfig(**common))
    radix = run_workload(radix_scheduler, prompts, 8)
    prefix_stats = radix_scheduler.cache.prefix_cache.stats
    result = {
        "device": str(device),
        "request_count": len(prompts),
        "page_size": common["page_size"],
        "outputs_match": baseline.pop("outputs") == radix.pop("outputs"),
        "baseline": baseline,
        "radix": radix,
        "prefix_cache": {
            "cached_pages": prefix_stats.cached_pages,
            "matched_tokens": prefix_stats.matched_tokens,
            "evicted_pages": prefix_stats.evicted_pages,
        },
    }
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as output:
            output.write(line + "\n")


if __name__ == "__main__":
    main()
