from __future__ import annotations

import torch

from mysglang.core import SamplingParams


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one row after temperature, top-k and nucleus filtering."""
    if logits.ndim != 1:
        raise ValueError("sampler expects one rank-1 logits row")
    if params.is_greedy:
        return int(logits.argmax().item())
    if generator is None:
        raise ValueError("random sampling requires a per-request generator")

    scores = logits.float() / params.temperature
    if params.top_k and params.top_k < scores.numel():
        threshold = torch.topk(scores, params.top_k).values[-1]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)

    if params.top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        # Keep the first token that crosses top_p, rather than truncating before it.
        remove = cumulative - sorted_probabilities >= params.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sampled_index = torch.multinomial(
            sorted_probabilities,
            1,
            generator=generator,
        )
        return int(sorted_indices[sampled_index].item())
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
