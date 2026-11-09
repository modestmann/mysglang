"""原生量化纯文本推理：attention TP + routed expert EP + Engram 行分片。

该路径使用有界连续缓存和等长 batch，尚未连接分页 Scheduler/HTTP 服务。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ..parallel import TensorParallelContext
from .config import DeepSeekV41Config
from .engram import EngramHasher
from .loader import WeightStore, validate_parallel
from .model import Block, LayerState, ReferenceCache, SharedAttention, append
from .ops import candidate_mask, hc_split, rotary
from .quantization import dequantize_weight, fake_quantize, quantized_linear, sparse_attention_torch


@dataclass
class RuntimeContext:
    tp: TensorParallelContext
    backend: str = "triton"

    def quantize(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        return fake_quantize(x, kind, backend=self.backend)


class NativeLinear(nn.Module):
    def __init__(
        self,
        store: WeightStore,
        name: str,
        context: RuntimeContext,
        *,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context = context
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", store.pop(name + ".weight"))
        self.kind = {torch.uint8: "fp4", torch.float8_e4m3fn: "fp8"}.get(self.weight.dtype)
        if self.kind:
            self.register_buffer("scale", store.pop(name + ".scale"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.numel():
            dtype = torch.bfloat16 if self.kind else (self.compute_dtype or self.weight.dtype)
            return x.new_empty((*x.shape[:-1], self.weight.shape[0]), dtype=dtype)
        if self.kind:
            return quantized_linear(
                x, self.weight, self.scale, self.kind, backend=self.context.backend
            )
        dtype = self.compute_dtype or self.weight.dtype
        return F.linear(x.to(dtype), self.weight.to(dtype))


class NativeNorm(nn.Module):
    def __init__(self, store: WeightStore, name: str, eps: float):
        super().__init__()
        self.register_buffer("weight", store.pop(name + ".weight"))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        return (
            h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()
        ).to(x.dtype)


class NativeExpert(nn.Module):
    def __init__(self, store, name, context, config):
        super().__init__()
        for proj in ("w1", "w2", "w3"):
            setattr(self, proj, NativeLinear(store, name + "." + proj, context))
        self.limit = config.swiglu_limit

    def forward(self, x, weights=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            gate, up = gate.clamp_max(self.limit), up.clamp(-self.limit, self.limit)
        hidden = F.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights
        return self.w2(hidden.to(x.dtype))


class NativeMoE(nn.Module):
    def __init__(self, store, name, context, config):
        super().__init__()
        self.context, self.config = context, config
        self.register_buffer("gate_weight", store.pop(name + ".gate.weight"))
        self.register_buffer("gate_bias", store.pop(name + ".gate.bias"))
        if config.vision_n_layers:
            self.register_buffer("gate_bias_vl", store.pop(name + ".gate.bias_vl"))
        local = config.n_routed_experts // context.tp.world_size
        self.begin, self.end = context.tp.rank * local, (context.tp.rank + 1) * local
        self.experts = nn.ModuleDict(
            {
                str(i): NativeExpert(store, f"{name}.experts.{i}", context, config)
                for i in range(self.begin, self.end)
            }
        )
        self.shared_experts = NativeExpert(store, name + ".shared_experts", context, config)

    def route(self, x):
        c = self.config
        logits = F.linear(x.float(), self.gate_weight.float()) / c.gate_temp
        if c.score_func == "sqrtsoftplus":
            scores = F.softplus(logits).sqrt()
        elif c.score_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(-1)
        indices = (scores + self.gate_bias.float()).topk(c.n_activated_experts, -1).indices
        weights = scores.gather(-1, indices)
        if c.norm_topk_prob and c.n_activated_experts > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * c.route_scale, indices

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        weights, indices = self.route(x)
        result = torch.zeros_like(x, dtype=torch.float32)
        # 每层只同步一次活跃 expert 列表；所有 rank 均执行最后的 collective。
        counts = torch.bincount(indices.flatten(), minlength=self.config.n_routed_experts).tolist()
        for expert in range(self.begin, self.end):
            if counts[expert]:
                rows, slots = torch.where(indices == expert)
                values = self.experts[str(expert)](x[rows], weights[rows, slots, None])
                result.index_add_(0, rows, values.float())
        self.context.tp.all_reduce(result)
        result = result + self.shared_experts(x).float()
        return result.to(x.dtype).reshape(shape), indices.reshape(*shape[:-1], -1)


class NativeEngram(nn.Module):
    def __init__(self, store, name, context, config, layer):
        super().__init__()
        self.context, self.config = context, config
        self.hash_index = config.engram_layer_ids.index(layer)
        self.register_buffer("weight", store.pop(name + ".embed.weight"))
        self.register_buffer("scale", store.pop(name + ".embed.scale"))
        self.register_buffer("q_weight", store.pop(name + ".q_weight"))
        self.register_buffer("k_weight", store.pop(name + ".k_weight"))
        self.wkv = NativeLinear(store, name + ".wkv", context)

    def lookup(self, hashes):
        rows = self.weight.shape[0]
        local = hashes - self.context.tp.rank * rows
        valid = (local >= 0) & (local < rows)
        local = local.clamp(0, rows - 1).flatten()
        # CUDA 不支持所有 float8 gather；按原始 byte 读取后恢复 dtype。
        weight = self.weight.view(torch.uint8)[local].view(torch.float8_e4m3fn)
        scale = self.scale[local]
        values = dequantize_weight(weight, scale, "engram").reshape(*hashes.shape, -1)
        values.masked_fill_(~valid.unsqueeze(-1), 0)
        return self.context.tp.all_reduce(values)

    def forward(self, x, hashes):
        c = self.config
        projected = self.wkv(self.lookup(hashes).flatten(-2))
        key, value = projected.float().split((c.hc_mult * c.dim, c.dim), -1)
        key = key.unflatten(-1, (c.hc_mult, c.dim))
        h = x.float()
        dot = (h * key * self.q_weight.float() * self.k_weight.float()).sum(-1)
        dot = dot * torch.rsqrt(h.square().mean(-1) + c.norm_eps)
        dot = dot * torch.rsqrt(key.square().mean(-1) + c.norm_eps) / math.sqrt(c.dim)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(x.dtype)


class NativeCompressor(nn.Module):
    def __init__(self, store, name, context, config, ratio):
        super().__init__()
        self.ratio = ratio
        # 发布版权重以 BF16 存储，但 ratio-2 的投影/softmax pooling 必须用 FP32。
        # 保持存储布局不变，计算时提升精度，不能只把 BF16 GEMM 输出转 FP32。
        self.wkv = NativeLinear(
            store, name + ".wkv", context, compute_dtype=torch.float32 if ratio > 1 else None
        )
        self.norm = NativeNorm(store, name + ".norm", config.norm_eps)
        if ratio > 1:
            self.wgate = NativeLinear(store, name + ".wgate", context, compute_dtype=torch.float32)

    def forward(self, x, old, new):
        values = self.wkv(x)
        if self.ratio == 1:
            return self.norm(values)
        values = append(old.pending_kv, values)
        scores = append(old.pending_scores, self.wgate(x))
        complete = values.shape[1] // self.ratio * self.ratio
        new.pending_kv, new.pending_scores = values[:, complete:], scores[:, complete:]
        values = values[:, :complete].unflatten(1, (-1, self.ratio))
        scores = scores[:, :complete].unflatten(1, (-1, self.ratio))
        return self.norm((values * scores.softmax(2)).sum(2).to(x.dtype))


class NativeIndexer(nn.Module):
    def __init__(self, store, name, context, config, owns_k):
        super().__init__()
        self.context, self.config = context, config
        self.wq_b = NativeLinear(store, name + ".wq_b", context)
        self.weights_proj = NativeLinear(store, name + ".weights_proj", context)
        if owns_k:
            self.wk = NativeLinear(store, name + ".wk", context)
            self.k_norm = NativeNorm(store, name + ".k_norm", config.norm_eps)

    def scores(self, x, qr, keys, positions):
        c = self.config
        heads = c.index_n_heads // self.context.tp.world_size
        q = self.wq_b(qr).unflatten(-1, (heads, c.index_head_dim))
        q = rotary(q, positions, c, True).to(x.dtype)
        q = self.context.quantize(q, "index4")
        scores = torch.einsum("bshd,btd->bsht", q.float(), keys.float()).relu()
        weights = self.weights_proj(x).float() / math.sqrt(c.index_head_dim * c.index_n_heads)
        return self.context.tp.all_reduce((scores * weights.unsqueeze(-1)).sum(-2))


class NativeAttention(nn.Module):
    def __init__(self, store, name, context, config, layer):
        super().__init__()
        self.context, self.config, self.layer = context, config, layer
        self.ratio = config.compress_ratios[layer]
        self.register_buffer("attn_sink", store.pop(name + ".attn_sink"))
        for proj in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            setattr(self, proj, NativeLinear(store, name + "." + proj, context))
        self.q_norm = NativeNorm(store, name + ".q_norm", config.norm_eps)
        self.kv_norm = NativeNorm(store, name + ".kv_norm", config.norm_eps)
        self.compressor = (
            NativeCompressor(store, name + ".compressor", context, config, self.ratio)
            if layer in config.kv_source_layers
            else None
        )
        self.indexer = (
            NativeIndexer(store, name + ".indexer", context, config, self.compressor is not None)
            if layer in config.index_source_layers
            else None
        )

    def forward(self, x, positions, old, shared, trace):
        c, ctx = self.config, self.context
        batch, length, _ = x.shape
        start, end = int(positions[0]), int(positions[-1]) + 1
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (c.n_heads // ctx.tp.world_size, c.head_dim))
        q = rotary(q, positions, c, bool(self.ratio)).to(x.dtype)
        kv = rotary(self.kv_norm(self.wkv(x)), positions, c, bool(self.ratio)).to(x.dtype)
        new = LayerState(window=append(old.window, ctx.quantize(kv, "fp8")))
        kv = new.window
        windows = (
            positions[:, None] - c.window_size + 1 + torch.arange(c.window_size, device=x.device)
        )
        indices = windows.masked_fill(windows < 0, -1).expand(batch, -1, -1)
        if self.ratio:
            count = end // self.ratio
            if self.compressor is not None:
                latent = self.compressor(x, old, new)
                compressed_positions = torch.arange(start // self.ratio, count, device=x.device)
                compressed_positions = compressed_positions * self.ratio
                keys = self.indexer.k_norm(self.indexer.wk(latent))
                keys = rotary(keys, compressed_positions, c, True).to(x.dtype)
                new.index_keys = append(old.index_keys, ctx.quantize(keys, "index4"))
                rotated = rotary(latent, compressed_positions, c, True).to(x.dtype)
                new.compressed = append(old.compressed, ctx.quantize(rotated, "kv4"))
                shared.compressed, shared.index_keys = new.compressed, new.index_keys
            if self.indexer is not None:
                lengths = ((positions + 1) // self.ratio).view(1, length, 1)
                scores = self.indexer.scores(x, qr, shared.index_keys, positions)
                scores = scores.masked_fill(
                    torch.arange(count, device=x.device) >= lengths, -torch.inf
                )
                if self.layer == c.candidate_source_layer:
                    shared.candidates = candidate_mask(
                        scores, lengths, c.candidate_block_size, c.candidate_topk_blocks
                    )
                elif 0 <= c.candidate_source_layer < self.layer:
                    scores = scores.masked_fill(~shared.candidates, -torch.inf)
                selected = scores.argsort(dim=-1, descending=True, stable=True)[..., : c.index_topk]
                selected = selected.sort(-1).values
                valid = torch.isfinite(scores.gather(-1, selected))
                shared.indices = selected.masked_fill(~valid, -1)
                trace[f"layers.{self.layer}.index"] = shared.indices
            selected = torch.where(shared.indices >= 0, shared.indices + end, -1)
            indices = torch.cat((indices, selected), -1)
            kv = torch.cat((kv, shared.compressed), 1)
        if ctx.backend == "triton":
            from .triton_quant import sparse_attention_cuda

            output = sparse_attention_cuda(q, kv, indices, self.attn_sink)
        else:
            output = sparse_attention_torch(q, kv, indices, self.attn_sink)
        output = rotary(output, positions, c, bool(self.ratio), inverse=True).to(x.dtype)
        groups = c.o_groups // ctx.tp.world_size
        output = ctx.quantize(output.reshape(batch, length, groups, -1), "fp8")
        weight = self.wo_a.weight.reshape(groups, c.o_lora_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", output, weight).flatten(-2)
        return ctx.tp.all_reduce(self.wo_b(output).float()).to(x.dtype), new


class NativeBlock(Block):
    def __init__(self, store, context, config, layer):
        nn.Module.__init__(self)
        self.config, self.layer = config, layer
        root = f"layers.{layer}"
        self.attn = NativeAttention(store, root + ".attn", context, config, layer)
        self.ffn = NativeMoE(store, root + ".ffn", context, config)
        self.engram = (
            NativeEngram(store, root + ".engram", context, config, layer)
            if layer in config.engram_layer_ids
            else None
        )
        for part in ("attn", "ffn"):
            setattr(
                self, part + "_norm", NativeNorm(store, root + f".{part}_norm", config.norm_eps)
            )
            for suffix in ("fn", "base", "scale"):
                name = f"hc_{part}_{suffix}"
                self.register_buffer(name, store.pop(root + "." + name))

    def mixes(self, x, name):
        flat = x.float().flatten(-2)
        projection = F.linear(flat, getattr(self, f"hc_{name}_fn").float())
        projection = projection * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.config.norm_eps
        )
        return hc_split(
            projection,
            getattr(self, f"hc_{name}_scale").float(),
            getattr(self, f"hc_{name}_base").float(),
            self.config,
        )

    @staticmethod
    def pre(x, mix):
        return Block.pre(x.float(), mix.float()).to(x.dtype)

    @staticmethod
    def post(output, residual, post, comb):
        return Block.post(output.float(), residual.float(), post, comb).to(residual.dtype)


class _DiscardTrace(dict):
    def __setitem__(self, key, value):
        pass


class DeepSeekV41Model(nn.Module):
    """已加载的 rank 本地模型；只读权重，缓存归调用方所有。"""

    def __init__(
        self,
        config: DeepSeekV41Config,
        weights: dict[str, torch.Tensor],
        *,
        token_map: Sequence[int] | None,
        tensor_parallel: TensorParallelContext | None = None,
        backend: str = "triton",
        max_seq_len: int = 4096,
    ):
        super().__init__()
        tp = tensor_parallel or TensorParallelContext()
        validate_parallel(config, tp.world_size)
        if backend not in {"torch", "triton"} or max_seq_len <= 0:
            raise ValueError("invalid backend or context length")
        if config.engram_layer_ids and token_map is None:
            raise ValueError("Engram requires a compressed tokenizer map")
        self.config, self.max_seq_len = config, max_seq_len
        self.context = RuntimeContext(tp, backend)
        self._cache_owner = object()
        store = WeightStore(weights)
        self.register_buffer("embed_weight", store.pop("embed.weight"))
        self.register_buffer("head_weight", store.pop("head.weight"))
        device = self.embed_weight.device
        if backend == "triton" and device.type != "cuda":
            raise ValueError("Triton runtime requires a CUDA device")
        self.engram_hash = (
            EngramHasher(config, token_map).to(device) if config.engram_layer_ids else None
        )
        self.layers = nn.ModuleList(
            NativeBlock(store, self.context, config, layer) for layer in range(config.n_layers)
        )
        self.norm = NativeNorm(store, "norm", config.norm_eps)
        store.finish()

    def new_cache(self) -> ReferenceCache:
        return ReferenceCache(self._cache_owner, layers=[LayerState() for _ in self.layers])

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ReferenceCache | None = None,
        *,
        trace: dict[str, torch.Tensor] | None = None,
        full_logits: bool = False,
    ):
        if input_ids.ndim != 2 or min(input_ids.shape) <= 0 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a nonempty int64 [batch, sequence] tensor")
        if input_ids.device != self.embed_weight.device:
            raise ValueError("input and model devices differ")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("token id outside vocabulary")
        target = cache
        cache = self.new_cache() if cache is None else cache
        if cache.owner is not self._cache_owner:
            raise ValueError("cache belongs to another model")
        if cache.tokens is not None and (
            cache.tokens.shape[0] != input_ids.shape[0] or cache.tokens.device != input_ids.device
        ):
            raise ValueError("cache batch size/device cannot change")
        start = cache.length
        if start + input_ids.shape[1] > self.max_seq_len:
            raise ValueError("runtime context limit exceeded")
        tokens = append(cache.tokens, input_ids)
        hashes = self.engram_hash(tokens, start=start) if self.engram_hash is not None else None
        positions = torch.arange(start, tokens.shape[1], device=input_ids.device)
        local = input_ids - self.context.tp.rank * self.embed_weight.shape[0]
        valid = (local >= 0) & (local < self.embed_weight.shape[0])
        h = F.embedding(local.clamp(0, self.embed_weight.shape[0] - 1), self.embed_weight)
        h.masked_fill_(~valid.unsqueeze(-1), 0)
        self.context.tp.all_reduce(h)
        h = h.unsqueeze(-2).expand(-1, -1, self.config.hc_mult, -1)
        pre_mix = h.new_zeros(*h.shape[:-1], dtype=torch.float32)
        pre_mix[..., 0] = 1
        shared, new_states = SharedAttention(), []
        captured = _DiscardTrace() if trace is None else trace
        if hashes is not None:
            captured["hashes"] = hashes
        for layer, state in zip(self.layers, cache.layers, strict=True):
            h, pre_mix, new = layer(h, pre_mix, positions, state, shared, hashes, captured)
            new_states.append(new)
        h = self.norm(NativeBlock.pre(h, pre_mix))
        if not full_logits:
            h = h[:, -1]
        logits = F.linear(h.float(), self.head_weight.float())
        tp = self.context.tp
        if tp.enabled:
            gathered = [torch.empty_like(logits) for _ in range(tp.world_size)]
            dist.all_gather(gathered, logits, group=tp.process_group)
            logits = torch.cat(gathered, -1)
        if target is not None:
            target.tokens, target.layers = tokens.clone(), new_states
        return logits
