"""Chapter 7 experiment: inspect physical page use during generation."""

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM
from mysglang.generation import greedy_generate_cached
from mysglang.scheduler import PagedBatchScheduler, PagedSchedulerConfig
from mysglang.tokenizer import ByteTokenizer


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(33)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()
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
    config = PagedSchedulerConfig(
        max_running_requests=4,
        prefill_token_budget=16,
        num_pages=16,
        page_size=4,
    )
    scheduler = PagedBatchScheduler(model, config)
    prompts = ["alpha", "中文", "gamma", "delta"]
    max_new_tokens = 8
    actual = {f"req-{index}": [] for index in range(len(prompts))}
    for index, prompt in enumerate(prompts):
        scheduler.add(
            Request.from_token_ids(
                f"req-{index}",
                tokenizer.encode(prompt),
                SamplingParams(max_new_tokens=max_new_tokens),
            )
        )

    peak_allocated = 0
    peak_reserved = 0
    print("step phase   batch allocated reserved free")
    while scheduler.has_work:
        step = scheduler.step()
        for event in step.outputs:
            actual[event.request_id].append(event.token_id)
        scheduler.check_integrity()
        stats = scheduler.cache.allocator.stats
        peak_allocated = max(peak_allocated, stats.allocated_pages)
        peak_reserved = max(peak_reserved, stats.reserved_pages)
        print(
            f"{step.index:>4} {step.phase:<7} {step.batch_size:>5} "
            f"{stats.allocated_pages:>9} {stats.reserved_pages:>8} "
            f"{stats.free_pages:>4}"
        )

    outputs_match = True
    for index, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt)
        expected = greedy_generate_cached(
            model,
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens,
        )[0, len(prompt_ids) :].tolist()
        outputs_match &= actual[f"req-{index}"] == expected

    element_size = next(model.parameters()).element_size()
    slot_token_capacity = config.max_running_requests * model.config.max_position_embeddings
    slot_bytes = (
        2
        * model.config.num_layers
        * slot_token_capacity
        * model.config.num_key_value_heads
        * model.config.head_dim
        * element_size
    )
    final_stats = scheduler.cache.allocator.stats
    print(f"device: {device}")
    print(f"outputs match isolated cached generation: {outputs_match}")
    print(f"fixed-slot token capacity: {slot_token_capacity}")
    print(f"paged-pool token capacity: {config.num_pages * config.page_size}")
    print(f"fixed-slot KV bytes: {slot_bytes}")
    print(f"paged-pool KV bytes: {scheduler.cache.memory_bytes}")
    print(f"peak allocated/reserved pages: {peak_allocated}/{peak_reserved}")
    print(f"final free pages: {final_stats.free_pages}/{final_stats.total_pages}")


if __name__ == "__main__":
    main()
