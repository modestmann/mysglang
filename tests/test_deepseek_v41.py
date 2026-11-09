"""DeepSeek V4.1 参考路径的因果性、增量状态与模型专属数学约束。"""

import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from mysglang import ModelConfig, Qwen3ForCausalLM
from mysglang.modeling.deepseek_v41 import DeepSeekV41Config, DeepSeekV41ReferenceModel
from mysglang.modeling.deepseek_v41.engram import EngramHasher
from mysglang.modeling.deepseek_v41.model import Block, Gate
from mysglang.modeling.deepseek_v41.ops import candidate_mask, hc_split, rotary, sparse_attention
from mysglang.modeling.registry import build_reference_model, load_model_config

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


def make_deepseek(seed=3, **changes):
    torch.manual_seed(seed)
    config = replace(DeepSeekV41Config.tiny(), **changes)
    return DeepSeekV41ReferenceModel(config, token_map=list(range(config.vocab_size)))


class DeepSeekConfigTest(unittest.TestCase):
    def test_native_and_hf_nested_config_agree(self):
        native = asdict(DeepSeekV41Config.tiny())
        aliases = {
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "moe_inter_dim": "moe_intermediate_size",
            "rope_head_dim": "qk_rope_head_dim",
            "norm_eps": "rms_norm_eps",
            "n_activated_experts": "num_experts_per_tok",
            "score_func": "scoring_func",
            "route_scale": "routed_scaling_factor",
            "window_size": "sliding_window",
            "kv_source_layers": "kv_source_layer_ids",
            "index_source_layers": "index_source_layer_ids",
            "candidate_source_layer": "candidate_source_layer_id",
            "engram_pad_id": "engram_pad_token_id",
            "n_mtp_layers": "num_nextn_predict_layers",
            "dspark_n_activated_experts": "dspark_num_experts_per_tok",
        }
        text = {aliases.get(k, k): v for k, v in native.items()}
        text["model_type"] = "deepseek_v41_text"
        text["rope_scaling"] = {
            "factor": text.pop("rope_factor"),
            "original_max_position_embeddings": text.pop("original_seq_len"),
            "beta_fast": text.pop("beta_fast"),
            "beta_slow": text.pop("beta_slow"),
        }
        hf = {
            "model_type": "deepseek_v41",
            "text_config": text,
            "vision_config": {"num_hidden_layers": 0},
        }
        self.assertEqual(DeepSeekV41Config.from_dict(hf), DeepSeekV41Config.from_dict(native))

    def test_rejects_invalid_shared_state_topology(self):
        for changes in (
            {"kv_source_layers": (3,)},
            {"index_source_layers": (1, 5)},
            {"compress_ratios": (0, 2, 1, 1, 1, 1)},
            {"candidate_source_layer": 2},
            {"kv_source_layers": (3, 1)},
            {"rope_head_dim": 3},
            {"n_activated_experts": 5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(DeepSeekV41Config.tiny(), **changes)
        with self.assertRaisesRegex(ValueError, "missing"):
            DeepSeekV41Config.from_dict({"model_type": "deepseek_v41"})

    def test_rejects_qwen_quantization_format(self):
        config = asdict(DeepSeekV41Config.tiny())
        config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        with self.assertRaisesRegex(ValueError, "MXFP8/MXFP4"):
            DeepSeekV41Config.from_dict(config)

    def test_registry_preserves_qwen_and_recognizes_deepseek(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = replace(
                DeepSeekV41Config.tiny(), engram_layer_ids=(), engram_num_embeddings=()
            )
            path.write_text(json.dumps(asdict(config)))
            config = load_model_config(directory)
            model = build_reference_model(config, token_map=list(range(config.vocab_size)))
            self.assertIsInstance(model, DeepSeekV41ReferenceModel)
            path.write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "vocab_size": 64,
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 2,
                        "max_position_embeddings": 128,
                    }
                )
            )
            config = load_model_config(path)
            self.assertIsInstance(config, ModelConfig)
            self.assertIsInstance(build_reference_model(config), Qwen3ForCausalLM)

    def test_release_dimensions_rejected_before_allocating_parameters(self):
        with patch("torch.nn.Embedding") as embedding:
            with self.assertRaisesRegex(ValueError, "small models"):
                DeepSeekV41ReferenceModel(DeepSeekV41Config())
            embedding.assert_not_called()


class DeepSeekMathTest(unittest.TestCase):
    def test_hc_residual_matrix_rows_are_sources_columns_are_destinations(self):
        # 非对称置换可区分 comb @ residual 和 comb.T @ residual。
        residual = torch.tensor([[[[1.0], [3.0], [7.0]]]])
        comb = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]])
        actual = Block.post(torch.zeros(1, 1, 1), residual, torch.zeros(1, 1, 3), comb)
        torch.testing.assert_close(actual, torch.tensor([[[[7.0], [1.0], [3.0]]]]))

    def test_sink_adds_denominator_mass_without_a_value(self):
        q = torch.zeros(1, 1, 1, 2)
        kv = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
        sink = torch.tensor([math.log(2)])
        actual = sparse_attention(q, kv, torch.tensor([[[0, 1, -1]]]), sink)
        torch.testing.assert_close(actual, torch.tensor([[[[2.0, 3.0]]]]))
        empty = sparse_attention(q, kv, torch.full((1, 1, 3), -1), sink)
        torch.testing.assert_close(empty, torch.zeros_like(q))

    def test_routing_bias_selects_experts_but_does_not_weight_them(self):
        gate = Gate(DeepSeekV41Config.tiny())
        with torch.no_grad():
            gate.weight.zero_()
            gate.weight[:, 0] = torch.tensor([4.0, 1.0, 2.0, -2.0])
            gate.bias.copy_(torch.tensor([-20.0, 20.0, 10.0, 0.0]))
        x = torch.zeros(1, 32)
        x[0, 0] = 1
        weights, indices = gate(x)
        self.assertEqual(indices.tolist(), [[1, 2]])
        values = [math.sqrt(math.log1p(math.exp(v))) for v in (1.0, 2.0)]
        expected = torch.tensor([[1.5 * value / sum(values) for value in values]])
        torch.testing.assert_close(weights, expected)

    def test_candidate_selection_keeps_newest_partial_block_and_is_stable(self):
        # 最新块分数极低，仍必须占据一个名额；同分优先选择更早的块。
        scores = torch.tensor([[[10.0, 10.0, 10.0, 10.0, -100.0, -torch.inf]]])
        mask = candidate_mask(scores, torch.tensor([[[5]]]), 2, 2)
        self.assertEqual(mask.tolist(), [[[True, True, False, False, True, True]]])
        none = candidate_mask(torch.full_like(scores, -torch.inf), torch.tensor([[[0]]]), 2, 2)
        self.assertFalse(bool(none.any()))

    def test_hc_sinkhorn_keeps_positive_mixes_and_doubly_stochastic_residual(self):
        config = DeepSeekV41Config.tiny()
        torch.manual_seed(1)
        pre, post, comb = hc_split(torch.randn(2, 3, 24), torch.ones(3), torch.zeros(24), config)
        self.assertTrue(bool((pre > 0).all() and (post > 0).all()))
        torch.testing.assert_close(comb.sum(-1), torch.ones(2, 3, 4), atol=1e-5, rtol=0)
        torch.testing.assert_close(comb.sum(-2), torch.ones(2, 3, 4), atol=2e-6, rtol=0)

    def test_rotary_inverse_and_unrotated_prefix(self):
        config = DeepSeekV41Config.tiny()
        x = torch.randn(2, 5, 4, 16)
        positions = torch.arange(31, 36)
        for compressed in (False, True):
            y = rotary(x, positions, config, compressed)
            torch.testing.assert_close(y[..., :-8], x[..., :-8], atol=0, rtol=0)
            torch.testing.assert_close(rotary(y, positions, config, compressed, inverse=True), x)

    @unittest.skipUnless(NUMPY_AVAILABLE, "requires numpy for Engram")
    def test_engram_dead_token_cuts_history_and_requires_exact_vocabulary(self):
        config = DeepSeekV41Config.tiny()
        hasher = EngramHasher(config, list(range(64)))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            EngramHasher(config, [0] * 64)
        with self.assertRaisesRegex(ValueError, "prime buckets"):
            EngramHasher(replace(config, engram_num_embeddings=(153, 220)), list(range(64)))
        tokens = torch.tensor([[10, 11, 12, 13, 14]])
        mask = torch.tensor([[True, True, False, True, True]])
        actual = hasher(tokens, token_mask=mask)
        fresh = hasher(tokens[:, 3:])
        torch.testing.assert_close(actual[:, 3:], fresh, rtol=0, atol=0)


@unittest.skipUnless(NUMPY_AVAILABLE, "requires numpy for Engram")
class DeepSeekStateTest(unittest.TestCase):
    def test_equal_index_scores_do_not_depend_on_chunk_width(self):
        model = make_deepseek()
        with torch.no_grad():
            for layer in model.layers:
                if layer.attn.indexer is not None:
                    layer.attn.indexer.weights_proj.weight.zero_()
        tokens = torch.randint(0, 64, (2, 17))
        cache = model.new_cache()
        pieces = [model(tokens[:, :3], cache)]
        pieces.extend(model(tokens[:, i : i + 1], cache) for i in range(3, 17))
        torch.testing.assert_close(torch.cat(pieces, 1), model(tokens), atol=4e-6, rtol=4e-6)

    def test_prefill_chunks_and_decode_match_full_causal_forward(self):
        for seed in (0, 3, 17):
            model = make_deepseek(seed)
            tokens = torch.randint(0, 64, (2, 19))
            expected = model(tokens)
            for ends in ((3, 4, 9, 19), tuple(range(1, 20))):
                with self.subTest(seed=seed, ends=ends):
                    cache = model.new_cache()
                    pieces, start = [], 0
                    for end in ends:
                        pieces.append(model(tokens[:, start:end], cache))
                        start = end
                    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=4e-6, rtol=4e-6)
                    self.assertEqual(cache.length, tokens.shape[1])
                    for i, state in enumerate(cache.layers):
                        if i in model.config.kv_source_layers:
                            ratio = model.config.compress_ratios[i]
                            self.assertEqual(state.compressed.shape[1], 19 // ratio)
                        else:
                            self.assertIsNone(state.compressed)

    def test_window_128_boundary_and_incomplete_compression_group(self):
        model = make_deepseek(window_size=128)
        tokens = torch.randint(0, 64, (1, 133))
        cache = model.new_cache()
        first = model(tokens[:, :127], cache)
        self.assertEqual(cache.layers[1].pending_kv.shape[1], 1)
        last = model(tokens[:, 127:], cache)
        torch.testing.assert_close(torch.cat((first, last), 1), model(tokens), atol=4e-6, rtol=4e-6)

    def test_future_tokens_do_not_change_prefix_logits(self):
        model = make_deepseek()
        tokens = torch.randint(0, 64, (2, 17))
        original = model(tokens)
        tokens[:, 7:] = (tokens[:, 7:] + 3) % 64
        changed = model(tokens)
        torch.testing.assert_close(original[:, :7], changed[:, :7], atol=3e-6, rtol=3e-6)

    def test_interleaved_request_caches_and_new_cache_have_no_shared_state(self):
        model = make_deepseek()
        a, b = torch.randint(0, 64, (2, 11)), torch.randint(0, 64, (2, 9))
        ca, cb = model.new_cache(), model.new_cache()
        model(a[:, :3], ca)
        model(b[:, :4], cb)
        torch.testing.assert_close(model(a[:, 3:], ca), model(a)[:, 3:], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(model(b[:, 4:], cb), model(b)[:, 4:], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(model(a, model.new_cache()), model(a))

    def test_invalid_input_does_not_advance_cache(self):
        model = make_deepseek()
        cache = model.new_cache()
        model(torch.tensor([[1, 2, 3]]), cache)
        before = cache.tokens.clone()
        for invalid in (
            torch.tensor([[64]]),
            torch.ones(2, 1, dtype=torch.long),
            torch.ones(1, 260, dtype=torch.long),
        ):
            with self.assertRaises(ValueError):
                model(invalid, cache)
            torch.testing.assert_close(cache.tokens, before)
        with self.assertRaisesRegex(ValueError, "another model"):
            make_deepseek(4)(torch.tensor([[1]]), cache)

    def test_layer_failure_does_not_publish_partially_updated_state(self):
        model = make_deepseek()
        cache = model.new_cache()
        model(torch.tensor([[1, 2, 3]]), cache)
        layers, tokens = cache.layers, cache.tokens
        with patch.object(model.layers[3], "forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                model(torch.tensor([[4, 5]]), cache)
        self.assertIs(cache.layers, layers)
        self.assertIs(cache.tokens, tokens)
        torch.testing.assert_close(
            model(torch.tensor([[4, 5]]), cache),
            model(torch.tensor([[1, 2, 3, 4, 5]]))[:, 3:],
            atol=3e-6,
            rtol=3e-6,
        )


if __name__ == "__main__":
    unittest.main()
