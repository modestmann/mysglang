import unittest

import torch
import torch.nn.functional as F

from mysglang import ModelConfig, Qwen3ForCausalLM, load_huggingface_state_dict

try:
    from transformers import (
        Qwen3Config,
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )
    from transformers import (
        Qwen3ForCausalLM as HFQwen3ForCausalLM,
    )
except ImportError:
    Qwen3Config = None
    HFQwen3ForCausalLM = None
    Qwen3MoeConfig = None
    Qwen3MoeForCausalLM = None

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

    @unittest.skipUnless(HFQwen3ForCausalLM is not None, "requires the model dependency")
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
        reference = HFQwen3ForCausalLM(hf_config).eval()
        model = Qwen3ForCausalLM(config).eval()
        load_huggingface_state_dict(model, reference.state_dict())
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        expected_layers: list[torch.Tensor] = []
        actual_layers: list[torch.Tensor] = []
        reference_hooks = [
            layer.register_forward_hook(
                lambda _module, _inputs, output: expected_layers.append(output.detach())
            )
            for layer in reference.model.layers
        ]
        model_hooks = [
            layer.register_forward_hook(
                lambda _module, _inputs, output: actual_layers.append(output.detach())
            )
            for layer in model.layers
        ]

        expected = reference(input_ids, use_cache=False).logits.float()
        actual = model(input_ids)
        for hook in reference_hooks + model_hooks:
            hook.remove()

        self.assertEqual(len(actual_layers), config.num_layers)
        for actual_layer, expected_layer in zip(actual_layers, expected_layers):
            torch.testing.assert_close(actual_layer, expected_layer, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(Qwen3MoeForCausalLM is not None, "requires the model dependency")
    @torch.inference_mode()
    def test_logits_match_hugging_face_qwen3_moe(self) -> None:
        torch.manual_seed(654)
        config = ModelConfig(
            model_type="qwen3_moe",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            norm_topk_prob=True,
        )
        hf_config = Qwen3MoeConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            decoder_sparse_step=config.decoder_sparse_step,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            num_experts=config.num_experts,
            norm_topk_prob=config.norm_topk_prob,
            tie_word_embeddings=False,
            use_cache=False,
        )
        hf_config._attn_implementation = "eager"
        reference = Qwen3MoeForCausalLM(hf_config).eval()
        model = Qwen3ForCausalLM(config).eval()
        load_huggingface_state_dict(model, reference.state_dict())
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        expected = reference(input_ids, use_cache=False).logits.float()
        actual = model(input_ids)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        naive_model = Qwen3ForCausalLM(config, moe_dispatch_backend="naive").eval()
        load_huggingface_state_dict(naive_model, reference.state_dict())
        torch.testing.assert_close(
            naive_model(input_ids),
            actual,
            atol=1e-5,
            rtol=1e-5,
        )

        for backend in ("grouped", "triton_grouped", "all_to_all"):
            optimized_model = Qwen3ForCausalLM(
                config,
                moe_dispatch_backend=backend,
            ).eval()
            load_huggingface_state_dict(optimized_model, reference.state_dict())
            torch.testing.assert_close(
                optimized_model(input_ids),
                actual,
                atol=1e-5,
                rtol=1e-5,
            )

        # Exercise the memory-safe skew fallback: padding four active experts to the
        # busiest expert would exceed twice the real assignment count.
        experts = optimized_model.layers[0].mlp.experts
        expert_inputs = torch.randn(13, config.hidden_size)
        local_experts = torch.tensor([0] * 10 + [1, 2, 3])
        expected_expert_outputs = torch.empty_like(expert_inputs)
        for expert_idx in range(config.num_experts):
            indices = torch.where(local_experts == expert_idx)[0]
            gate, up = F.linear(
                expert_inputs[indices],
                experts.gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)
            expected_expert_outputs[indices] = F.linear(
                F.silu(gate) * up,
                experts.down_proj[expert_idx],
            )
        torch.testing.assert_close(
            experts._grouped_expert_gemm(expert_inputs, local_experts),
            expected_expert_outputs,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            experts._triton_grouped_expert_gemm(expert_inputs, local_experts),
            expected_expert_outputs,
            atol=1e-5,
            rtol=1e-5,
        )

        # Older Qwen3MoE checkpoints store one gate/up/down tensor per expert instead of
        # Transformers 5's packed expert tensors. Both layouts feed the same runtime model.
        legacy_state: dict[str, torch.Tensor] = {}
        for name, weight in reference.state_dict().items():
            if name.endswith(".experts.gate_up_proj"):
                intermediate = config.moe_intermediate_size
                for expert_idx in range(config.num_experts):
                    prefix = name.removesuffix(".gate_up_proj")
                    legacy_state[f"{prefix}.{expert_idx}.gate_proj.weight"] = weight[
                        expert_idx, :intermediate
                    ]
                    legacy_state[f"{prefix}.{expert_idx}.up_proj.weight"] = weight[
                        expert_idx, intermediate:
                    ]
            elif name.endswith(".experts.down_proj"):
                prefix = name.removesuffix(".down_proj")
                for expert_idx in range(config.num_experts):
                    legacy_state[f"{prefix}.{expert_idx}.down_proj.weight"] = weight[expert_idx]
            else:
                legacy_state[name] = weight
        legacy_model = Qwen3ForCausalLM(config).eval()
        load_huggingface_state_dict(legacy_model, legacy_state)
        torch.testing.assert_close(
            legacy_model(input_ids),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
