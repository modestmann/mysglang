"""量化 checkpoint 的真实切片、分片生成和 H20 算子数值回归。"""

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mysglang.modeling.deepseek_v41.config import DeepSeekV41Config
from mysglang.modeling.deepseek_v41.loader import load_rank_weights
from mysglang.modeling.deepseek_v41.model import DeepSeekV41ReferenceModel
from mysglang.modeling.deepseek_v41.quantization import (
    decode_ue8m0,
    dequantize_weight,
    fake_quantize,
    fp4_codes,
    fp4_values,
    quantize_weight,
    quantized_linear,
    sparse_attention_torch,
)
from mysglang.modeling.deepseek_v41.runtime import DeepSeekV41Model
from mysglang.modeling.parallel import TensorParallelContext

HAS_DEPS = (
    importlib.util.find_spec("safetensors") is not None
    and importlib.util.find_spec("numpy") is not None
    and hasattr(torch, "float8_e8m0fnu")
)
HAS_TRITON = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def write_quantized_fixture(root: Path):
    from safetensors.torch import save_file

    c = replace(
        DeepSeekV41Config.tiny(),
        dim=64,
        moe_inter_dim=64,
        head_dim=64,
        rope_head_dim=32,
        q_lora_rank=64,
        o_groups=4,
        o_lora_rank=32,
        index_n_heads=4,
        index_head_dim=32,
        engram_head_dim=32,
        dtype="fp8",
        expert_dtype="fp4",
    )
    torch.manual_seed(17)
    source = DeepSeekV41ReferenceModel(c, token_map=list(range(c.vocab_size)))
    tensors = {}
    for name, value in source.state_dict().items():
        kind = None
        if ".engram.embed.weight" in name:
            kind = "engram"
        elif ".ffn.experts." in name:
            kind = "fp4"
        elif name.endswith(".weight") and any(
            f".{p}." in name for p in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "w1", "w2", "w3")
        ):
            if ".compressor." not in name:
                kind = "fp8"
        if kind:
            tensors[name], tensors[name[:-7] + ".scale"] = quantize_weight(value, kind)
        else:
            fp32 = "hc_" in name or name.endswith(("attn_sink", ".gate.bias"))
            tensors[name] = value.to(torch.float32 if fp32 else torch.bfloat16)
    # 两个 shard，确保 weight/scale 可以跨文件，不依赖遍历先后。
    mapping = {}
    for part in range(2):
        subset = {name: t for i, (name, t) in enumerate(tensors.items()) if i % 2 == part}
        shard = f"model-{part}.safetensors"
        save_file(subset, root / shard)
        mapping.update({name: shard for name in subset})
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": mapping,
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return c, tensors


def distributed_fixture_worker(rank, root, rendezvous, config, device_type):
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = f"cuda:{rank}" if device_type == "cuda" else "cpu"
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
    )
    try:
        weights, _ = load_rank_weights(root, config, rank=rank, world_size=2, device=device)
        model = DeepSeekV41Model(
            config,
            weights,
            token_map=list(range(config.vocab_size)),
            tensor_parallel=TensorParallelContext.from_distributed(),
            backend="triton" if device_type == "cuda" else "torch",
        )
        tokens = torch.tensor([[3, 8, 12, 7, 19]], device=device)
        cache = model.new_cache()
        outputs = [model(tokens, cache)]
        for _ in range(3):
            outputs.append(model(outputs[-1].argmax(-1, keepdim=True), cache))
        if rank == 0:
            torch.save(torch.stack(outputs).cpu(), Path(root) / "distributed.pt")
    finally:
        dist.destroy_process_group()


class QuantizationTest(unittest.TestCase):
    def test_ue8_exponents_and_special_values(self):
        values = decode_ue8m0(torch.tensor([0, 1, 126, 127, 128, 254, 255], dtype=torch.uint8))
        torch.testing.assert_close(
            values[:6], torch.tensor([2**-127, 2**-126, 0.5, 1, 2, 2.0**127])
        )
        self.assertTrue(torch.isnan(values[-1]))

    def test_fp4_all_codes_and_ties_to_even(self):
        expected = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
        codes = torch.arange(16, dtype=torch.uint8)
        torch.testing.assert_close(fp4_values(codes), expected)
        self.assertEqual(fp4_codes(expected).tolist(), codes.tolist())
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        self.assertEqual(fp4_codes(ties).tolist(), [0, 2, 2, 4, 4, 6, 6])

    def test_fp4_packed_order_and_per_row_scale(self):
        w = torch.full((2, 16), 0x21, dtype=torch.uint8)
        s = torch.tensor([[127], [128]], dtype=torch.uint8)
        got = dequantize_weight(w, s, "fp4").float()
        torch.testing.assert_close(got[0], torch.tensor([0.5, 1.0] * 16))
        torch.testing.assert_close(got[1], got[0] * 2)

    @unittest.skipUnless(HAS_TRITON, "requires CUDA and Triton")
    def test_cuda_rounding_and_mixed_precision_linear(self):
        torch.manual_seed(13)
        for kind in ("fp8", "index4", "kv4"):
            x = torch.randn(13, 512, device="cuda", dtype=torch.bfloat16)
            x[0].zero_()
            torch.testing.assert_close(
                fake_quantize(x, kind, backend="triton"), fake_quantize(x, kind), atol=0, rtol=0
            )
        for kind in ("fp4", "fp8"):
            w, s = quantize_weight(torch.randn(96, 256) * 0.1, kind)
            w, s = w.cuda(), s.view(torch.uint8).cuda()
            for rows in (1, 5, 33):
                x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
                torch.testing.assert_close(
                    quantized_linear(x, w, s, kind),
                    quantized_linear(x, w, s, kind, backend="torch"),
                    atol=0.032,
                    rtol=0.01,
                )

    @unittest.skipUnless(HAS_TRITON, "requires CUDA and Triton")
    def test_sparse_attention_mask_sink_and_multiple_tiles(self):
        from mysglang.modeling.deepseek_v41.triton_quant import sparse_attention_cuda

        torch.manual_seed(3)
        q = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(2, 70, 64, dtype=torch.bfloat16, device="cuda")
        ids = torch.arange(70, device="cuda").expand(2, 3, -1).clone()
        ids[:, 0] = -1
        ids[:, 1, 30:] = -1
        sink = torch.randn(4, device="cuda")
        actual = sparse_attention_cuda(q, kv, ids, sink)
        expected = sparse_attention_torch(q, kv, ids, sink)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        self.assertEqual(actual[:, 0].count_nonzero().item(), 0)


@unittest.skipUnless(HAS_DEPS, "requires numpy, safetensors and UE8M0-capable PyTorch")
class QuantizedRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.source = write_quantized_fixture(self.root)
        torch.set_num_threads(2)

    def model(self, device="cpu", backend="torch"):
        weights, report = load_rank_weights(self.root, self.config, device=device, chunk_bytes=257)
        return DeepSeekV41Model(
            self.config, weights, token_map=list(range(self.config.vocab_size)), backend=backend
        ), report

    def test_loader_actual_shards_scales_padding_and_wo_a(self):
        ranks = [
            load_rank_weights(self.root, self.config, rank=r, world_size=4, chunk_bytes=511)[0]
            for r in range(4)
        ]
        for name in ("layers.0.attn.wq_b.weight", "layers.0.attn.wq_b.scale", "embed.weight"):
            got = torch.cat([rank[name].view(torch.uint8) for rank in ranks], 0)
            torch.testing.assert_close(got, self.source[name].view(torch.uint8))
        for name in ("layers.0.attn.wo_b.weight", "layers.0.attn.wo_b.scale"):
            got = torch.cat([rank[name].view(torch.uint8) for rank in ranks], 1)
            torch.testing.assert_close(got, self.source[name].view(torch.uint8))
        table = "layers.1.engram.embed.weight"
        scale = table[:-7] + ".scale"
        self.assertEqual(ranks[-1][table][-2:].view(torch.uint8).count_nonzero().item(), 0)
        self.assertTrue((ranks[-1][scale][-2:] == 127).all())
        for rank, state in enumerate(ranks):
            owned = {int(n.split(".")[4]) for n in state if ".ffn.experts." in n}
            self.assertEqual(owned, {rank})
        name = "layers.0.attn.wo_a.weight"
        got = torch.cat([rank[name] for rank in ranks], 0)
        expected = dequantize_weight(self.source[name], self.source[name[:-7] + ".scale"], "fp8")
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        self.assertNotIn(name[:-7] + ".scale", ranks[0])

    def test_bf16_compressor_weights_compute_and_cache_in_fp32(self):
        from mysglang.modeling.deepseek_v41.model import LayerState

        model, _ = self.model()
        compressor = model.layers[1].attn.compressor
        self.assertEqual(compressor.wkv.weight.dtype, torch.bfloat16)
        x = torch.randn(1, 1, self.config.dim).bfloat16()
        new = LayerState()
        compressor(x, LayerState(), new)
        self.assertEqual(new.pending_kv.dtype, torch.float32)
        self.assertEqual(new.pending_scores.dtype, torch.float32)
        expected = torch.nn.functional.linear(x.float(), compressor.wkv.weight.float())
        torch.testing.assert_close(new.pending_kv, expected, atol=0, rtol=0)

    def test_incremental_cache_matches_full_prefix_and_rolls_back_failure(self):
        model, report = self.model()
        self.assertGreater(report["loaded_bytes"], 0)
        tokens = torch.tensor([[3, 8, 12, 7, 19, 21, 5, 4]])
        expected = model(tokens, full_logits=True)
        cache = model.new_cache()
        actual = torch.cat(
            [
                model(tokens[:, :1], cache, full_logits=True),
                model(tokens[:, 1:3], cache, full_logits=True),
                model(tokens[:, 3:5], cache, full_logits=True),
                model(tokens[:, 5:], cache, full_logits=True),
            ],
            1,
        )
        torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.08)
        self.assertEqual(cache.length, 8)
        before = cache.layers
        with patch.object(model.layers[3].ffn, "forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                model(tokens[:, :1], cache)
        self.assertEqual(cache.length, 8)
        self.assertIs(cache.layers, before)

    def test_two_rank_gloo_generation_matches_single_rank(self):
        # 真实短提示的压缩位置数少于 index_topk；避免未训练随机权重在
        # 稀疏检索同分附近的离散切换掩盖 TP/EP 通信数值检查。
        self.config = replace(self.config, index_topk=32, candidate_topk_blocks=32)
        model, _ = self.model()
        # 官方 RowParallelLinear 在 collective 前将各 partial GEMM 舍入为
        # BF16；单进程 oracle 显式重现该舍入，其余计算保留完整未分片权重。
        # 直接拿一次完整 GEMM 当作多卡 oracle 会在后续 FP8/FP4 边界处分叉。
        for layer in model.layers:
            projection = layer.attn.wo_b

            def split_projection(x, projection=projection):
                width = x.shape[-1] // 2
                partials = []
                for rank in range(2):
                    start, stop = rank * width, (rank + 1) * width
                    partials.append(
                        quantized_linear(
                            x[..., start:stop],
                            projection.weight[:, start:stop],
                            projection.scale[:, start // 32 : stop // 32],
                            "fp8",
                            backend="torch",
                        ).float()
                    )
                return (partials[0] + partials[1]).bfloat16()

            projection.forward = split_projection
        cache = model.new_cache()
        outputs = [model(torch.tensor([[3, 8, 12, 7, 19]]), cache)]
        for _ in range(3):
            outputs.append(model(outputs[-1].argmax(-1, keepdim=True), cache))
        mp.spawn(
            distributed_fixture_worker,
            args=(str(self.root), str(self.root / "rendezvous"), self.config, "cpu"),
            nprocs=2,
            join=True,
        )
        got = torch.load(self.root / "distributed.pt", weights_only=True)
        expected = torch.stack(outputs)
        torch.testing.assert_close(got, expected, atol=0.06, rtol=0.08)
        torch.testing.assert_close(got.argmax(-1), expected.argmax(-1))

    @unittest.skipUnless(HAS_TRITON, "requires CUDA and Triton")
    def test_cuda_quantized_runtime_matches_torch(self):
        model, _ = self.model("cuda", "triton")
        tokens = torch.tensor([[3, 8, 12, 7, 19]], device="cuda")
        got = model(tokens)
        model.context.backend = "torch"
        expected = model(tokens)
        torch.testing.assert_close(got, expected, atol=0.08, rtol=0.10)
        torch.testing.assert_close(got.argmax(-1), expected.argmax(-1))

    @unittest.skipUnless(
        HAS_TRITON and torch.cuda.device_count() >= 2, "requires two CUDA devices and Triton"
    )
    def test_two_rank_nccl_generation_matches_gloo(self):
        config = replace(self.config, index_topk=32, candidate_topk_blocks=32)
        mp.spawn(
            distributed_fixture_worker,
            args=(str(self.root), str(self.root / "cpu-rendezvous"), config, "cpu"),
            nprocs=2,
            join=True,
        )
        expected = torch.load(self.root / "distributed.pt", weights_only=True)
        mp.spawn(
            distributed_fixture_worker,
            args=(str(self.root), str(self.root / "cuda-rendezvous"), config, "cuda"),
            nprocs=2,
            join=True,
        )
        got = torch.load(self.root / "distributed.pt", weights_only=True)
        torch.testing.assert_close(got, expected, atol=0.08, rtol=0.10)
        torch.testing.assert_close(got.argmax(-1), expected.argmax(-1))


if __name__ == "__main__":
    unittest.main()
