from __future__ import annotations

from collections.abc import Sequence

import torch

from mysglang.core import SamplingParams


def apply_penalties(#重复惩罚
    logits: torch.Tensor,
    params: SamplingParams,
    output_token_ids: Sequence[int],
) -> torch.Tensor:
    """Presence/frequency penalties over the request's generated tokens only."""
    if not params.has_penalties or not output_token_ids:
        return logits
    tokens, counts = torch.unique(
        torch.tensor(output_token_ids, dtype=torch.long, device=logits.device),
        return_counts=True,
    )
    penalty = params.presence_penalty + params.frequency_penalty * counts.to(logits.dtype)
    adjusted = logits.clone()
    adjusted[tokens] -= penalty
    return adjusted


def sampling_probs(
    logits: torch.Tensor,
    params: SamplingParams,
    output_token_ids: Sequence[int] = (),
    *,
    apply_request_penalties: bool = True,
) -> torch.Tensor:
    """Normalized distribution after penalties, temperature, top-k and nucleus filtering."""
    if params.is_greedy:
        raise ValueError("greedy requests have no sampling distribution")
    if apply_request_penalties:
        logits = apply_penalties(logits, params, output_token_ids)
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
        probabilities = probabilities.scatter(
            0, sorted_indices, sorted_probabilities.masked_fill(remove, 0)
        )
        probabilities = probabilities / probabilities.sum()
    return probabilities


def greedy_token(
    logits: torch.Tensor,
    params: SamplingParams,
    output_token_ids: Sequence[int] = (),
) -> int:
    return int(apply_penalties(logits, params, output_token_ids).argmax().item())


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    generator: torch.Generator | None = None,
    output_token_ids: Sequence[int] = (),
) -> int:
    """Sample one row after penalties, temperature, top-k and nucleus filtering."""
    if logits.ndim != 1:
        raise ValueError("sampler expects one rank-1 logits row")
    if params.is_greedy:
        return greedy_token(logits, params, output_token_ids)
    if generator is None:
        raise ValueError("random sampling requires a per-request generator")
    probabilities = sampling_probs(logits, params, output_token_ids)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def speculative_accept(
    draft: Sequence[int],#MTP 的猜测，一串 token
    draft_probs: Sequence[torch.Tensor | None],#每个猜测当初的概率表 q；n-gram 或 greedy 时是 None
    target_logits: torch.Tensor,#	主模型验证时每个位置的 logits，比猜测多 1 行
    params: SamplingParams,
    history: Sequence[int],
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, ...]:
    """Verify a draft chain against ``len(draft) + 1`` target logits rows.

    Greedy requests accept while the draft equals the (penalized) target argmax. For
    random sampling this is standard speculative sampling: accept ``d`` with probability
    ``min(1, p(d) / q(d))``; on rejection sample from ``normalize(max(0, p - q))``. A draft
    without ``q`` (n-gram, or a greedy MTP draft) is deterministic, i.e. ``q = onehot(d)``.
    The emitted tokens then follow exactly the distribution of sequential sampling.
    """
    tokens = list(history)
    output: list[int] = []
    for index, token in enumerate(draft):
        row = target_logits[index]
        if params.is_greedy:
            target = greedy_token(row, params, tokens)
            if target != token:
                return (*output, target)
        else:
            assert generator is not None
            probabilities = sampling_probs(row, params, tokens)
            q = draft_probs[index] if index < len(draft_probs) else None
            q_token = 1.0 if q is None else float(q[token])
            accept = min(1.0, float(probabilities[token]) / max(q_token, 1e-20))
            if float(torch.rand((), generator=generator, device=row.device)) >= accept:
                if q is None:
                    residual = probabilities.clone()
                    residual[token] = 0
                else:
                    residual = (probabilities - q.to(probabilities.dtype)).clamp_min(0)
                total = residual.sum()
                residual = probabilities if float(total) <= 0 else residual / total
                corrected = int(torch.multinomial(residual, 1, generator=generator).item())
                return (*output, corrected)
        output.append(token)
        tokens.append(token)
    bonus = sample_token(
        target_logits[len(draft)],
        params,
        generator=generator,
        output_token_ids=tokens,
    )
    return (*output, bonus)
