"""Chapter 3: compare tiny MySGLang logits with Hugging Face Qwen3."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from mysglang import ModelConfig, TinyCausalLM, greedy_generate


def main() -> None:
    torch.manual_seed(321)
    config = ModelConfig(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    hf_config = Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        attention_bias=False,
        tie_word_embeddings=config.tie_word_embeddings,
        use_cache=False,
    )
    hf_config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(hf_config).eval()
    model = TinyCausalLM(config).eval()
    model.load_state_dict(
        {
            name.removeprefix("model."): value
            for name, value in reference.state_dict().items()
        },
        strict=True,
    )

    prompt = torch.tensor([[1, 5, 9]])
    with torch.inference_mode():
        expected_logits = reference(prompt, use_cache=False).logits.float()
        actual_logits = model(prompt)
        expected_tokens = prompt
        for _ in range(4):
            next_token = reference(expected_tokens, use_cache=False).logits[:, -1].argmax(
                dim=-1, keepdim=True
            )
            expected_tokens = torch.cat((expected_tokens, next_token), dim=1)
        actual_tokens = greedy_generate(model, prompt, max_new_tokens=4)

    error = (actual_logits - expected_logits).abs()
    print(f"max absolute logit error: {error.max().item():.3e}")
    print(f"mean absolute logit error: {error.mean().item():.3e}")
    print(f"Hugging Face tokens: {expected_tokens.tolist()[0]}")
    print(f"MySGLang tokens:     {actual_tokens.tolist()[0]}")
    print(f"tokens match: {torch.equal(actual_tokens, expected_tokens)}")


if __name__ == "__main__":
    main()
