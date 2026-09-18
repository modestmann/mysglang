"""Shared RadixAttention app builder for chapter 8 examples."""

import torch

from mysglang import ModelConfig, TinyCausalLM
from mysglang.scheduler import RadixBatchScheduler, RadixSchedulerConfig
from mysglang.serving import ContinuousBatchGenerationService
from mysglang.serving.http import create_app
from mysglang.tokenizer import ByteTokenizer


def build_app_and_service(*, device: str | torch.device | None = None):
    torch.manual_seed(8)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    config = RadixSchedulerConfig(
        max_running_requests=8,
        prefill_token_budget=32,
        num_pages=64,
        page_size=4,
    )
    service = ContinuousBatchGenerationService(
        model,
        ByteTokenizer(),
        config,
        scheduler_type=RadixBatchScheduler,
    )
    return create_app(service), service
