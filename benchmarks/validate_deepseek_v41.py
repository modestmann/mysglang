"""对照官方 Python 模型及本地 vLLM/SGLang 数学实现，验证小模型 FP32 路径。

只提取显式列出的函数/类；不导入框架或 CUDA 扩展，不改动参考目录。
官方模型的存储/GEMM/激活量化替换为 FP32，稀疏同分统一按位置排序。
这项验证不代表真实 MXFP4/MXFP8 数值对齐或整模型推理验收。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal

import numpy as np
import torch
from sympy import isprime
from torch import nn
from torch.nn import functional as F

from mysglang.modeling.deepseek_v41 import DeepSeekV41Config, DeepSeekV41ReferenceModel
from mysglang.modeling.deepseek_v41.engram import EngramHasher, compressed_token_map


def load_definitions(path: Path, names: set[str], env: dict, *, stable_index: bool = False):
    source = path.read_text()
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in nodes} != names:
        raise ValueError(f"reference definitions changed: {path}")

    class StableIndexTopk(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "topk":
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_stable_topk", ctx=ast.Load()),
                        args=[node.func.value, *node.args],
                        keywords=node.keywords,
                    ),
                    node,
                )
            return node

    if stable_index:
        nodes = [
            StableIndexTopk().visit(n) if n.name in {"Indexer", "select_candidate_blocks"} else n
            for n in nodes
        ]
    module = ModuleType("_deepseek_oracle_" + hashlib.sha256(str(path).encode()).hexdigest()[:12])
    module.__dict__.update(env)
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module.__name__] = module
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def _stable_topk(tensor, k, dim=-1, largest=True, sorted=True):
    ids = torch.argsort(tensor, dim=dim, descending=largest, stable=True).narrow(dim, 0, k)
    return SimpleNamespace(indices=ids, values=tensor.gather(dim, ids))


def _sparse_oracle(q, kv, sink, indices, scale):
    # 按 query 显式取值，与模型内批量 gather 的实现独立。
    result = torch.zeros_like(q)
    for batch in range(q.shape[0]):
        for pos in range(q.shape[1]):
            picked = indices[batch, pos]
            values = kv[batch, picked[picked >= 0].long()]
            logits = q[batch, pos] @ values.T * scale
            probabilities = torch.cat((logits, sink[:, None]), -1).softmax(-1)
            result[batch, pos] = probabilities[:, :-1] @ values
    return result


def reference_model(config, source: Path, sglang: Path, state_dict):
    env = dict(
        torch=torch,
        dist=torch.distributed,
        nn=nn,
        F=F,
        np=np,
        math=math,
        dataclass=dataclass,
        Literal=Literal,
        contextmanager=contextmanager,
        lru_cache=lru_cache,
        isprime=isprime,
    )
    mhc = load_definitions(
        sglang / "python/sglang/test/kernels/mhc.py", {"hc_split_sinkhorn_torch"}, env
    )
    engram = load_definitions(
        source / "engram.py",
        {
            "find_next_prime",
            "build_compressed_token_map",
            "compute_hash_multipliers",
            "EngramLayout",
            "NgramHashState",
        },
        env,
    )
    engram.build_compressed_token_map = lambda _: (
        list(range(config.vocab_size)),
        config.vocab_size,
    )
    env.update(
        EngramLayout=engram.EngramLayout,
        NgramHashState=engram.NgramHashState,
        hc_split_sinkhorn=mhc.hc_split_sinkhorn_torch,
        sparse_attn=_sparse_oracle,
        _stable_topk=_stable_topk,
        world_size=1,
        rank=0,
        fp8_block_size=32,
        fp4_block_size=32,
        scale_fmt="ue8m0",
        scale_dtype=torch.float32,
        act_quant=lambda *a, **kw: None,
        fp4_act_quant=lambda *a, **kw: None,
        linear=F.linear,
        ParallelEngramEmbedding=nn.Embedding,
    )
    ref = load_definitions(
        source / "model.py",
        {
            "ModelArgs",
            "set_dtype",
            "ParallelEmbedding",
            "Linear",
            "ColumnParallelLinear",
            "RowParallelLinear",
            "RMSNorm",
            "Engram",
            "precompute_freqs_cis",
            "apply_rotary_emb",
            "get_window_topk_idxs",
            "Compressor",
            "Indexer",
            "select_candidate_blocks",
            "Attention",
            "Gate",
            "Expert",
            "MoE",
            "Block",
            "ParallelHead",
            "make_identity_pre_mix",
            "SharedAttentionRuntime",
            "Transformer",
            "sample",
        },
        env,
        stable_index=True,
    )

    def fp32_linear_init(self, in_features, out_features, bias=False, dtype=None):
        nn.Module.__init__(self)
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = None

    ref.Linear.__init__ = fp32_linear_init
    ref.shared_attn = ref.SharedAttentionRuntime()
    accepted = {f.name for f in fields(ref.ModelArgs)}
    args = ref.ModelArgs(**{k: v for k, v in asdict(config).items() if k in accepted})
    args.max_batch_size, args.max_seq_len, args.temperature = 2, 256, 0.0
    model = ref.Transformer(args)
    model.load_state_dict(state_dict, strict=True)
    # 返回所有位置，以便检查每个 prefill token，而不只比较最后一个。
    model.head.forward = lambda x, full_logits=False: F.linear(x.float(), model.head.weight)
    return model


def framework_engram_checks(config, ours, vllm: Path, sglang: Path, tokens):
    env = dict(torch=torch, np=np)
    vref = load_definitions(
        vllm / "vllm/models/deepseek_v41/common/engram.py",
        {
            "_is_prime",
            "find_next_prime",
            "compute_hash_multipliers",
            "EngramLayout",
        },
        env,
    )
    args = SimpleNamespace(**asdict(config), engram_pad_token_id=config.engram_pad_id)
    layout = vref.EngramLayout(args)
    hashes = ours.engram_hash
    torch.testing.assert_close(torch.tensor(layout.primes).flatten(1), hashes.primes)
    torch.testing.assert_close(layout.offsets, hashes.offsets)
    multipliers = vref.compute_hash_multipliers(
        config.engram_layer_ids, config.engram_max_ngram_size, config.engram_compressed_vocab_size
    )
    torch.testing.assert_close(multipliers, hashes.multipliers)
    sref = load_definitions(
        sglang / "python/sglang/srt/layers/engram.py", {"compute_engram_hash_ids"}, env
    )
    length = tokens.shape[1]
    shifts = torch.arange(config.engram_max_ngram_size)
    positions = torch.arange(length)[:, None] - shifts
    windows = tokens[:, positions.clamp_min(0)].flatten(0, 1)
    blocked = (positions < 0).expand(tokens.shape[0], -1, -1).flatten(0, 1)
    expected = sref.compute_engram_hash_ids(
        windows,
        blocked,
        hashes.pad_id,
        hashes.token_map,
        hashes.multipliers,
        torch.tensor(layout.primes),
        hashes.offsets,
    )
    torch.testing.assert_close(expected, hashes(tokens).flatten(0, 1), rtol=0, atol=0)


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def validate_tokenizer(path: Path, source: Path, vllm: Path, sglang: Path) -> dict:
    from tokenizers import Tokenizer

    class TokenizerView:
        def __init__(self):
            self.backend_tokenizer = Tokenizer.from_file(str(path))

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    tokenizer = TokenizerView()
    actual = compressed_token_map(tokenizer)
    config = DeepSeekV41Config.from_json(source / "config.json")
    hasher = EngramHasher(config, actual)
    checked = []
    for name, file in (
        ("official", source / "engram.py"),
        ("vllm", vllm / "vllm/models/deepseek_v41/common/engram.py"),
        ("sglang", sglang / "python/sglang/srt/layers/engram.py"),
    ):
        module = load_definitions(file, {"build_compressed_token_map"}, {})
        expected, size = module.build_compressed_token_map(tokenizer)
        if expected != actual or size != config.engram_compressed_vocab_size:
            raise AssertionError(f"tokenizer map differs from {name}")
        checked.append(name)
    # 使用发布版参数核对哈希空间，无需分配 203 GB 的 Engram 查找表。
    holder = SimpleNamespace(engram_hash=hasher)
    torch.manual_seed(1017)
    tokens = torch.randint(0, config.vocab_size, (2, 17))
    framework_engram_checks(config, holder, vllm, sglang, tokens)
    return {
        "vocab_size": len(tokenizer),
        "compressed_vocab_size": len(set(actual)),
        "token_map_sha256": hashlib.sha256(np.asarray(actual, dtype="<i8").tobytes()).hexdigest(),
        "tokenizer_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "matches": checked,
        "release_config_hashes_match_frameworks": True,
    }


def validate(source: Path, vllm: Path, sglang: Path) -> dict:
    config = DeepSeekV41Config.tiny()
    results = []
    for seed in (0, 3, 17):
        torch.manual_seed(seed)
        ours = DeepSeekV41ReferenceModel(config, token_map=list(range(config.vocab_size)))
        expected = reference_model(config, source, sglang, ours.state_dict())
        tokens = torch.randint(0, config.vocab_size, (2, 19))
        framework_engram_checks(config, ours, vllm, sglang, tokens)
        reference_hidden, reference_routes = {}, {}
        for index, block in enumerate(expected.layers):
            block.register_forward_hook(
                lambda module, inputs, output, i=index: reference_hidden.update(
                    {i: output[0].clone()}
                )
            )
            block.ffn.gate.register_forward_hook(
                lambda module, inputs, output, i=index: reference_routes.update(
                    {i: output[1].clone()}
                )
            )
        cache = ours.new_cache()
        worst_logits = worst_hidden = 0.0
        # 官方参考只接受一次完整 prefill，随后逐 token decode。
        for start, end in [(0, 7), *((i, i + 1) for i in range(7, tokens.shape[1]))]:
            trace = {}
            actual_logits = ours(tokens[:, start:end], cache, trace=trace)
            _, ref_logits, _ = expected(tokens[:, start:end], start_pos=start)
            torch.testing.assert_close(actual_logits, ref_logits, atol=8e-6, rtol=8e-6)
            worst_logits = max(worst_logits, (actual_logits - ref_logits).abs().max().item())
            for index in range(config.n_layers):
                actual = trace[f"layers.{index}.hidden"]
                torch.testing.assert_close(actual, reference_hidden[index], atol=8e-6, rtol=8e-6)
                worst_hidden = max(
                    worst_hidden, (actual - reference_hidden[index]).abs().max().item()
                )
                routes = trace[f"layers.{index}.routes"].flatten(0, 1)
                torch.testing.assert_close(routes, reference_routes[index], atol=0, rtol=0)
        results.append(
            {
                "seed": seed,
                "max_logits_abs_error": worst_logits,
                "max_hidden_abs_error": worst_hidden,
                "expert_indices_equal": True,
            }
        )
    return {
        "scope": "tiny text FP32; unquantized weights/activations; identity compressed token map",
        "reference_adjustments": [
            "FP32 storage/GEMM and sparse-attention oracle",
            "SGLang pure-Torch Sinkhorn oracle",
            "stable position order for equal sparse-index scores",
            "return logits at every position",
        ],
        "not_validated": [
            "MXFP4/MXFP8 rounding",
            "real checkpoint outputs",
            "tokenizer normalization",
            "TP execution",
            "vision",
            "DSpark",
            "paged scheduler",
        ],
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
        "config": asdict(config),
        "vllm_revision": revision(vllm),
        "sglang_revision": revision(sglang),
        "reference_source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("model.py", "engram.py")
        },
        "vllm_engram_layout_and_multipliers_equal": True,
        "sglang_engram_hashes_equal": True,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="官方 inference/ 目录")
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="可选：同时核对真实 tokenizer.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = validate(args.reference, args.vllm, args.sglang)
    project = Path(__file__).resolve().parents[1]
    result["mysglang_revision"] = revision(project)
    result["mysglang_worktree_dirty"] = bool(
        subprocess.check_output(
            ["git", "-C", str(project), "status", "--porcelain"], text=True
        ).strip()
    )
    implementation_files = sorted((project / "src/mysglang/modeling/deepseek_v41").glob("*.py"))
    implementation_files.append(Path(__file__).resolve())
    result["implementation_sha256"] = {
        str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in implementation_files
    }
    if args.tokenizer:
        result["tokenizer_validation"] = validate_tokenizer(
            args.tokenizer, args.reference, args.vllm, args.sglang
        )
        result["not_validated"].remove("tokenizer normalization")
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
