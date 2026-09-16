import unittest

import torch

from mysglang import ContiguousKVCache, ModelConfig, TinyCausalLM, greedy_generate

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    Qwen3Config = None
    Qwen3ForCausalLM = None


@unittest.skipUnless(Qwen3ForCausalLM is not None, "requires the model optional dependency")
class HuggingFaceQwen3AlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(321)
        self.config = ModelConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        hf_config = Qwen3Config(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=self.config.rms_norm_eps,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": self.config.rope_theta,
            },
            attention_bias=False,
            tie_word_embeddings=self.config.tie_word_embeddings,
            use_cache=False,
        )
        hf_config._attn_implementation = "eager"
        self.reference = Qwen3ForCausalLM(hf_config).eval()
        self.model = TinyCausalLM(self.config).eval()
        mapped_state = {
            name.removeprefix("model."): value
            for name, value in self.reference.state_dict().items()
        }
        self.model.load_state_dict(mapped_state, strict=True)

    @torch.inference_mode()
    def test_logits_match_hugging_face_eager_attention(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected = self.reference(input_ids, use_cache=False).logits.float()
        actual = self.model(input_ids)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @torch.inference_mode()
    def test_uncached_greedy_tokens_match_hugging_face(self) -> None:
        prompt = torch.tensor([[1, 5, 9]])
        expected = prompt
        for _ in range(4):
            next_token = self.reference(expected, use_cache=False).logits[:, -1].argmax(
                dim=-1, keepdim=True
            )
            expected = torch.cat((expected, next_token), dim=1)

        actual = greedy_generate(self.model, prompt, max_new_tokens=4)
        self.assertTrue(torch.equal(actual, expected))

    @torch.inference_mode()
    def test_cached_decode_logits_match_hugging_face_dynamic_cache(self) -> None:
        prompt = torch.tensor([[1, 5, 9]])
        parameter = next(self.model.parameters())
        cache = ContiguousKVCache.from_config(
            self.config,
            batch_size=1,
            dtype=parameter.dtype,
            device=parameter.device,
        )

        expected = self.reference(prompt, use_cache=True)
        actual_logits = self.model(prompt, kv_cache=cache)
        torch.testing.assert_close(actual_logits, expected.logits.float(), atol=1e-5, rtol=1e-5)

        past_key_values = expected.past_key_values
        next_token = expected.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(3):
            expected = self.reference(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            actual_logits = self.model(next_token, kv_cache=cache)
            torch.testing.assert_close(
                actual_logits, expected.logits.float(), atol=1e-5, rtol=1e-5
            )
            past_key_values = expected.past_key_values
            next_token = expected.logits[:, -1].argmax(dim=-1, keepdim=True)
