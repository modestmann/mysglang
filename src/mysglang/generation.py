from __future__ import annotations

import torch

from .cache import ContiguousKVCache
from .modeling.tiny import TinyCausalLM


@torch.inference_mode()
def greedy_generate(
    model: TinyCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """The deliberately slow baseline: recompute the whole prefix every step."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("milestone 1 generation supports exactly one sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    output = input_ids
    for _ in range(max_new_tokens):
        if output.size(1) >= model.config.max_position_embeddings:
            break
        next_token = model(output)[:, -1].argmax(dim=-1, keepdim=True)
        output = torch.cat((output, next_token), dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return output


@torch.inference_mode()
def greedy_generate_cached(
    model: TinyCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate with a per-layer cache, forwarding only one token after prefill."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("cached generation supports exactly one sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0 or input_ids.size(1) >= model.config.max_position_embeddings:
        return input_ids

    parameter = next(model.parameters())
    cache = ContiguousKVCache.from_config(
        model.config,
        batch_size=1,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    output = input_ids
    logits = model(input_ids, kv_cache=cache)
    for step in range(max_new_tokens):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        output = torch.cat((output, next_token), dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        if step + 1 == max_new_tokens or output.size(1) >= model.config.max_position_embeddings:
            break
        logits = model(next_token, kv_cache=cache)
    return output
