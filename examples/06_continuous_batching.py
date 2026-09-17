"""Chapter 6 experiment: compare isolated generation with scheduler batches."""

import time

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM
from mysglang.generation import greedy_generate_cached
from mysglang.scheduler import ContinuousBatchScheduler, SchedulerConfig
from mysglang.tokenizer import ByteTokenizer


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(11)
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
    scheduler = ContinuousBatchScheduler(
        model,
        SchedulerConfig(max_running_requests=4, prefill_token_budget=16),
    )
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

    started = time.perf_counter()
    while scheduler.has_work:
        step = scheduler.step()
        for event in step.outputs:
            actual[event.request_id].append(event.token_id)
    elapsed_ms = (time.perf_counter() - started) * 1_000

    outputs_match = True
    for index, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        expected = greedy_generate_cached(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
        )[0, len(prompt_ids) :].tolist()
        outputs_match &= actual[f"req-{index}"] == expected

    isolated_forward_calls = len(prompts) * max_new_tokens
    print(f"device: {device}")
    print(f"outputs match isolated cached generation: {outputs_match}")
    print(f"isolated model forward calls: {isolated_forward_calls}")
    print(f"scheduled model forward calls: {scheduler.stats.model_forwards}")
    print(f"maximum decode batch size: {scheduler.stats.max_decode_batch_size}")
    print(f"scheduled elapsed time: {elapsed_ms:.3f} ms")
    print("step trace:")
    for step in scheduler.history:
        print(
            f"  {step.index:02d} {step.phase:7s} "
            f"batch={step.batch_size} input_tokens={step.input_tokens} "
            f"requests={','.join(step.request_ids)}"
        )


if __name__ == "__main__":
    main()
