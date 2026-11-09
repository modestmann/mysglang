"""小模型 BF16/量化语义与官方 Python 模型计算图对照，参考目录只读。

官方 GEMM 替换为反量化权重 + Torch GEMM，激活量化使用已测原语。
这验证计算图、量化位置和增量状态，不宣称与官方融合 CUDA kernel 逐位相同。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmarks.validate_deepseek_v41 import load_definitions, reference_model
from mysglang.modeling.deepseek_v41.loader import load_rank_weights
from mysglang.modeling.deepseek_v41.quantization import (
    dequantize_weight,
    fake_quantize,
    fp4_codes,
    sparse_attention_torch,
)
from mysglang.modeling.deepseek_v41.runtime import DeepSeekV41Model
from tests.test_deepseek_v41_runtime import write_quantized_fixture


def quantized_reference(config, source: Path, sglang: Path, weights):
    state, quantized_names = {}, set()
    for name, weight in weights.items():
        if name.endswith(".scale"):
            continue
        if weight.dtype in {torch.float8_e4m3fn, torch.uint8}:
            kind = (
                "fp4"
                if weight.dtype == torch.uint8
                else ("engram" if ".engram.embed." in name else "fp8")
            )
            weight = dequantize_weight(weight, weights[name[:-7] + ".scale"], kind)
            if kind != "engram":
                quantized_names.add(name)
        state[name] = weight
    model = reference_model(config, source, sglang, state).bfloat16()
    # ratio-2 的跨 token 投影/门控状态必须保持 FP32，不能随激活转 BF16。
    for layer in model.layers:
        compressor = getattr(layer.attn, "compressor", None)
        if compressor is not None and hasattr(compressor, "kv_state"):
            compressor.kv_state = compressor.kv_state.float()
            compressor.score_state = compressor.score_state.float()
    for name, parameter in model.named_parameters():
        value = state[name]
        if ".compressor." in name and ("wkv" in name or "wgate" in name):
            if config.compress_ratios[int(name.split(".")[1])] > 1:
                value = value.float()
        parameter.data = value.clone()
    module = sys.modules[type(model).__module__]
    quantized_ids = {
        id(parameter) for name, parameter in model.named_parameters() if name in quantized_names
    }

    def linear(x, weight, bias=None):
        if id(weight) in quantized_ids:
            x = fake_quantize(x, "fp8")
        return F.linear(x.to(weight.dtype), weight, bias)

    def act_quant(x, *args, **kwargs):
        x.copy_(fake_quantize(x, "fp8"))

    def fp4_act_quant(x, block_size=32, *args, **kwargs):
        x.copy_(fake_quantize(x, "kv4" if block_size == 16 else "index4"))

    module.linear = linear
    module.act_quant, module.fp4_act_quant = act_quant, fp4_act_quant
    module.sparse_attn = lambda q, kv, sink, indices, scale: sparse_attention_torch(
        q, kv, indices, sink
    )
    model.head.forward = lambda x, full_logits=False: F.linear(x.float(), model.head.weight.float())
    return model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = args.reference / "inference"
    path = args.sglang / "python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py"
    external = load_definitions(
        path,
        {"quantize_to_e2m1_codes"},
        {"torch": torch, "_E2M1_MAGNITUDES": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)},
    )
    values = torch.linspace(-8, 8, 4097)
    torch.testing.assert_close(fp4_codes(values), external.quantize_to_e2m1_codes(values))
    with tempfile.TemporaryDirectory() as temp:
        config, _ = write_quantized_fixture(Path(temp))
        # 与真实短提示一样，让所有可见压缩位置参与 attention。
        config = replace(config, index_topk=32, candidate_topk_blocks=32)
        weights, _ = load_rank_weights(temp, config)
        ours = DeepSeekV41Model(
            config, weights, token_map=list(range(config.vocab_size)), backend="torch"
        )
        reference = quantized_reference(config, source, args.sglang, weights)
        tokens = torch.tensor([[3, 8, 12, 7, 19, 21, 5, 4]])
        cache, actual, expected = ours.new_cache(), [], []
        for start, end in ((0, 5), (5, 6), (6, 7), (7, 8)):
            actual.append(ours(tokens[:, start:end], cache, full_logits=True))
            expected.append(reference(tokens[:, start:end], start)[1])
        actual, expected = torch.cat(actual, 1), torch.cat(expected, 1)
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
        result = {
            "kind": "small quantized weights / official graph / Torch primitive substitution",
            "model_source_sha256": hashlib.sha256((source / "model.py").read_bytes()).hexdigest(),
            "sglang_quant_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fp4_codes_checked": values.numel(),
            "max_abs_error": (actual - expected).abs().max().item(),
            "cosine_similarity": F.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0
            ).item(),
            "greedy_equal": torch.equal(actual.argmax(-1), expected.argmax(-1)),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
