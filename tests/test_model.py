import unittest

import torch

from mysglang import ModelConfig, TinyCausalLM

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    Qwen3Config = None
    Qwen3ForCausalLM = None

from tests.helpers import make_model


class ModelRegressionTest(unittest.TestCase):
    @torch.inference_mode()
    def test_causal_prefix_is_unchanged_by_future_tokens(self) -> None:
        model = make_model()
        prefix = torch.tensor([[1, 2, 3]])
        extended = torch.tensor([[1, 2, 3, 4, 5]])

        torch.testing.assert_close(
            model(prefix),
            model(extended)[:, : prefix.size(1)],
            atol=1e-5,
            rtol=1e-5,
        )

    @unittest.skipUnless(Qwen3ForCausalLM is not None, "requires the model dependency")
    @torch.inference_mode()
    def test_logits_match_hugging_face_qwen3(self) -> None:
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
            {name.removeprefix("model."): value for name, value in reference.state_dict().items()},
            strict=True,
        )
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        expected = reference(input_ids, use_cache=False).logits.float()
        actual = model(input_ids)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
