"""Chapter 4: compare uncached and contiguous-KV-cache greedy decoding."""

import argparse
import time
from collections.abc import Callable

import torch

from mysglang import ModelConfig, TinyCausalLM, greedy_generate, greedy_generate_cached


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ms(fn: Callable[[], torch.Tensor], device: torch.device, repeats: int) -> float:
    for _ in range(2):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1_000 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if use_cuda else "cpu")
    max_length = args.prompt_length + args.new_tokens
    if min(args.prompt_length, args.new_tokens, args.repeats) <= 0:
        raise ValueError("prompt length, new tokens and repeats must be positive")

    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=176,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_length,
    )
    model = TinyCausalLM(config).to(device).eval()
    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_length), device=device)

    uncached = lambda: greedy_generate(model, prompt, max_new_tokens=args.new_tokens)
    cached = lambda: greedy_generate_cached(model, prompt, max_new_tokens=args.new_tokens)
    uncached_output = uncached()
    cached_output = cached()
    if not torch.equal(uncached_output, cached_output):
        raise AssertionError("cached and uncached generation produced different tokens")

    uncached_ms = measure_ms(uncached, device, args.repeats)
    cached_ms = measure_ms(cached, device, args.repeats)
    n, p = args.new_tokens, args.prompt_length
    uncached_projected = n * p + n * (n - 1) // 2
    cached_projected = p + n - 1

    print(f"device: {device}")
    print(f"prompt/new tokens: {p}/{n}")
    print(f"projected tokens without cache: {uncached_projected}")
    print(f"projected tokens with cache:    {cached_projected}")
    print(f"uncached latency: {uncached_ms:.3f} ms")
    print(f"cached latency:   {cached_ms:.3f} ms")
    print(f"speedup: {uncached_ms / cached_ms:.2f}x")
    print("tokens match: True")


if __name__ == "__main__":
    main()
