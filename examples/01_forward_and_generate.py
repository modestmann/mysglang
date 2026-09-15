"""Chapter 3 smoke test: inspect logits and run the uncached generation loop."""

import torch

from mysglang import ModelConfig, TinyCausalLM, greedy_generate


def main() -> None:
    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=80,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = TinyCausalLM(config).eval()
    prompt = torch.tensor([[1, 5, 9, 2]], dtype=torch.long)
    logits = model(prompt)
    output = greedy_generate(model, prompt, max_new_tokens=6)

    print(f"logits shape: {tuple(logits.shape)}")
    print(f"prompt tokens: {prompt.tolist()[0]}")
    print(f"output tokens: {output.tolist()[0]}")
    print("Note: random weights produce meaningless tokens; this milestone tests mechanics.")


if __name__ == "__main__":
    main()
