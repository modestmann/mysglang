from __future__ import annotations

from collections.abc import Iterable

import torch

from mysglang import ModelConfig, Request, SamplingParams, TinyCausalLM


def make_model(
    *,
    seed: int = 2026,
    vocab_size: int = 64,
    max_position_embeddings: int = 64,
) -> TinyCausalLM:
    torch.manual_seed(seed)
    return TinyCausalLM(
        ModelConfig(
            vocab_size=vocab_size,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=max_position_embeddings,
        )
    ).eval()


def make_request(
    request_id: str,
    prompt_token_ids: Iterable[int],
    max_new_tokens: int = 4,
) -> Request:
    return Request.from_token_ids(
        request_id,
        prompt_token_ids,
        SamplingParams(max_new_tokens=max_new_tokens),
    )


@torch.inference_mode()
def reference_generate(
    model: TinyCausalLM,
    prompt_token_ids: Iterable[int],
    max_new_tokens: int,
) -> list[int]:
    """Small full-recomputation oracle kept outside the production package."""

    parameter = next(model.parameters())
    tokens = torch.tensor(
        [tuple(prompt_token_ids)],
        dtype=torch.long,
        device=parameter.device,
    )
    output: list[int] = []
    for _ in range(max_new_tokens):
        token_id = int(model(tokens)[:, -1].argmax(dim=-1).item())
        output.append(token_id)
        next_token = torch.tensor([[token_id]], dtype=torch.long, device=parameter.device)
        tokens = torch.cat((tokens, next_token), dim=1)
    return output


def drain_scheduler(scheduler) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    while scheduler.has_work:
        step = scheduler.step()
        if step is None:
            raise AssertionError("scheduler reported work but made no progress")
        for event in step.outputs:
            outputs.setdefault(event.request_id, []).append(event.token_id)
        scheduler.check_integrity()
    return outputs
