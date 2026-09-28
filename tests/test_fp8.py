"""Block-FP8 quantization, kernels and checkpoint loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from mysglang import ModelConfig, Qwen3ForCausalLM
from mysglang.config import Fp8QuantConfig
from mysglang.modeling.fp8 import (
    FP8_DTYPE,
    dequantize_block_fp8,
    fp8_linear,
    quantize_block_fp8,
)
from tests.helpers import export_hf_state_dict
from tests.test_hybrid import make_hybrid

CUDA_AVAILABLE = torch.cuda.is_available()


class BlockQuantizationTest(unittest.TestCase):
    def test_round_trip_error_is_bounded_per_block_including_ragged_edges(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(3, 70, 45) * torch.logspace(-2, 1, 45)
        quantized, scale = quantize_block_fp8(weight, (16, 32))
        self.assertEqual(quantized.dtype, FP8_DTYPE)
        self.assertEqual(tuple(scale.shape), (3, 5, 2))
        restored = dequantize_block_fp8(quantized, scale, (16, 32), torch.float32)
        # e4m3 keeps 3 mantissa bits; the worst relative error of a block is 1/16 of its max.
        for row in range(0, 70, 16):
            for column in range(0, 45, 32):
                block = (slice(None), slice(row, row + 16), slice(column, column + 32))
                limit = weight[block].abs().amax(dim=(-2, -1), keepdim=True) / 16
                self.assertTrue(bool(((restored[block] - weight[block]).abs() <= limit).all()))

    def test_cpu_linear_uses_the_dequantized_weight(self) -> None:
        weight, scale = quantize_block_fp8(torch.randn(24, 40), (8, 8))
        x = torch.randn(5, 40)
        expected = F.linear(x, dequantize_block_fp8(weight, scale, (8, 8), torch.float32))
        torch.testing.assert_close(fp8_linear(x, weight, scale, (8, 8)), expected)

    def test_modules_to_not_convert_matches_path_fragments(self) -> None:
        quant = Fp8QuantConfig(
            modules_to_not_convert=(
                "lm_head",
                "model.language_model.layers.0.linear_attn.in_proj_a",
                "mtp.fc",
            )
        )
        self.assertFalse(quant.is_quantized("lm_head"))
        self.assertFalse(quant.is_quantized("layers.0.linear_attn.in_proj_a"))
        self.assertTrue(quant.is_quantized("layers.10.linear_attn.in_proj_a"))
        self.assertFalse(quant.is_quantized("mtp.fc"))
        self.assertTrue(quant.is_quantized("mtp.layers.0.self_attn.q_proj"))


@unittest.skipUnless(CUDA_AVAILABLE, "requires CUDA")
class Fp8CudaKernelTest(unittest.TestCase):
    def test_triton_linear_matches_dequantized_reference(self) -> None:
        torch.manual_seed(1)
        for rows, columns, depth, block in (
            (1, 512, 256, (128, 128)),
            (37, 300, 640, (128, 128)),
            (5, 96, 48, (16, 16)),
        ):
            with self.subTest(rows=rows, block=block):
                weight, scale = quantize_block_fp8(
                    torch.randn(columns, depth, device="cuda") * 0.05, block
                )
                x = torch.randn(rows, depth, device="cuda", dtype=torch.bfloat16)
                expected = F.linear(
                    x.float(), dequantize_block_fp8(weight, scale, block, torch.float32)
                )
                actual = fp8_linear(x, weight, scale, block).float()
                torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)

    def test_triton_grouped_moe_matches_dequantized_experts(self) -> None:
        from mysglang.modeling.triton_moe import triton_grouped_mlp

        torch.manual_seed(2)
        experts, hidden, intermediate = 6, 256, 128
        gate_up, gate_up_scale = quantize_block_fp8(
            torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.05, (128, 128)
        )
        down, down_scale = quantize_block_fp8(
            torch.randn(experts, hidden, intermediate, device="cuda") * 0.05, (128, 128)
        )
        x = torch.randn(29, hidden, device="cuda", dtype=torch.bfloat16)
        # -1 and 99 are other ranks' experts and must contribute zeros.
        ids = torch.tensor([*torch.randint(0, experts, (27,)).tolist(), -1, 99], device="cuda")
        actual = triton_grouped_mlp(
            x,
            ids,
            gate_up,
            down,
            gate_up_scale=gate_up_scale,
            down_scale=down_scale,
            block_size=(128, 128),
        )
        expected = triton_grouped_mlp(
            x,
            ids,
            dequantize_block_fp8(gate_up, gate_up_scale, (128, 128), torch.bfloat16),
            dequantize_block_fp8(down, down_scale, (128, 128), torch.bfloat16),
        )
        torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
        self.assertEqual(float(actual[-2:].abs().sum()), 0.0)


class Fp8CheckpointTest(unittest.TestCase):
    def _write_checkpoint(self, directory: Path, model: Qwen3ForCausalLM) -> None:
        from safetensors.torch import save_file

        config = model.config
        state_dict = export_hf_state_dict(model)
        # Qwen3.5 stores language-model weights under model.language_model.*.
        state_dict = {
            name.replace("model.", "model.language_model.", 1): tensor.contiguous()
            for name, tensor in state_dict.items()
        }
        # Vision tensors must be ignored by the text-only loader.
        state_dict["model.visual.patch_embed.proj.weight"] = torch.zeros(2, 2)
        save_file(state_dict, str(directory / "model.safetensors"))
        text_config = {
            "model_type": f"{config.model_type}_text",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "attn_output_gate": True,
            "layer_types": list(config.layer_types),
            "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
            "linear_key_head_dim": config.linear_key_head_dim,
            "linear_value_head_dim": config.linear_value_head_dim,
            "linear_num_key_heads": config.linear_num_key_heads,
            "linear_num_value_heads": config.linear_num_value_heads,
            "mtp_num_hidden_layers": config.mtp_num_hidden_layers,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
            "eos_token_id": 3,
            "dtype": "float32",
        }
        outer = {
            "model_type": config.model_type,
            "text_config": text_config,
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": list(config.quantization.weight_block_size),
                "modules_to_not_convert": list(config.quantization.modules_to_not_convert),
            },
        }
        (directory / "config.json").write_text(json.dumps(outer))
        (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [5, 3]}))

    @torch.inference_mode()
    def test_fp8_safetensors_load_in_fp8_and_dequantized_modes(self) -> None:
        quant = Fp8QuantConfig(
            weight_block_size=(4, 4),
            modules_to_not_convert=("lm_head", "mtp.fc", "in_proj_a", "in_proj_b"),
        )
        reference = make_hybrid(seed=5, moe=True, mtp=True, head_dim=8, quantization=quant)
        tokens = torch.randint(0, 48, (1, 20))
        expected = reference(tokens)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._write_checkpoint(path, reference)
            loaded_config = ModelConfig.from_pretrained(path)
            self.assertEqual(loaded_config.stop_token_ids, (3, 5))
            fp8 = Qwen3ForCausalLM.from_pretrained(path, enable_mtp=True)
            self.assertEqual(fp8.layers[0].linear_attn.in_proj_qkvz.weight.dtype, FP8_DTYPE)
            self.assertEqual(fp8.layers[0].linear_attn.in_proj_ba.weight.dtype, torch.float32)
            self.assertEqual(fp8.layers[2].mlp.experts.gate_up_proj.dtype, FP8_DTYPE)
            self.assertEqual(fp8.mtp.fc.weight.dtype, torch.float32)
            ignored = fp8.checkpoint_load_report.ignored_tensors
            self.assertIn("model.visual.patch_embed.proj.weight", ignored)
            torch.testing.assert_close(fp8(tokens), expected)

            dequantized = Qwen3ForCausalLM.from_pretrained(path, fp8_weights=False)
            self.assertEqual(
                dequantized.layers[0].linear_attn.in_proj_qkvz.weight.dtype, torch.float32
            )
            torch.testing.assert_close(dequantized(tokens), expected)


class RealConfigTest(unittest.TestCase):
    def test_parses_the_qwen3_6_35b_a3b_fp8_config(self) -> None:
        path = Path(__file__).parent / "fixtures" / "qwen3_6_35b_a3b_fp8_config.json"
        config = ModelConfig.from_pretrained(path)
        self.assertEqual(config.model_type, "qwen3_5_moe")
        self.assertEqual(config.num_layers, 40)
        self.assertEqual(len(config.full_attention_layers), 10)
        self.assertEqual(config.rotary_dim, 64)
        self.assertEqual(config.num_experts, 256)
        self.assertEqual(config.shared_expert_intermediate_size, 512)
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.quantization.weight_block_size, (128, 128))
        self.assertFalse(config.quantization.is_quantized("layers.0.linear_attn.in_proj_b"))
        self.assertTrue(config.quantization.is_quantized("layers.0.linear_attn.in_proj_qkv"))
        self.assertFalse(config.quantization.is_quantized("mtp.fc"))


if __name__ == "__main__":
    unittest.main()
