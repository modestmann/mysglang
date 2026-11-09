"""纯文本、小规模 FP32 oracle。缓存属于调用方，无全局运行态。

这里保留完整历史以便检查因果关系；尚未实现分页、量化、TP 或生产服务接口。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from .config import DeepSeekV41Config
from .engram import Engram, EngramHasher
from .ops import RMSNorm, candidate_mask, hc_split, rotary, sparse_attention


@dataclass
class LayerState:
    window: torch.Tensor | None = None
    pending_kv: torch.Tensor | None = None
    pending_scores: torch.Tensor | None = None
    compressed: torch.Tensor | None = None
    index_keys: torch.Tensor | None = None


@dataclass
class ReferenceCache:
    """等长 batch 的参考缓存。一次 forward 成功后才发布新状态。"""

    owner: object
    tokens: torch.Tensor | None = None
    layers: list[LayerState] = field(default_factory=list)

    @property
    def length(self) -> int:
        return 0 if self.tokens is None else self.tokens.shape[1]


@dataclass
class SharedAttention:
    compressed: torch.Tensor | None = None
    index_keys: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def append(previous: torch.Tensor | None, current: torch.Tensor) -> torch.Tensor:
    return current if previous is None else torch.cat((previous, current), dim=1)


class Compressor(nn.Module):
    def __init__(self, config: DeepSeekV41Config, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.wkv = nn.Linear(config.dim, config.head_dim, bias=False)
        self.norm = RMSNorm(config.head_dim, config.norm_eps)
        if ratio > 1:
            self.wgate = nn.Linear(config.dim, config.head_dim, bias=False)

    def forward(self, x: torch.Tensor, old: LayerState, new: LayerState) -> torch.Tensor:
        values = self.wkv(x)
        if self.ratio == 1:
            return self.norm(values)
        values = append(old.pending_kv, values)
        scores = append(old.pending_scores, self.wgate(x))
        complete = values.shape[1] // self.ratio * self.ratio
        new.pending_kv, new.pending_scores = values[:, complete:], scores[:, complete:]
        values = values[:, :complete].unflatten(1, (-1, self.ratio))
        scores = scores[:, :complete].unflatten(1, (-1, self.ratio))
        return self.norm((values * scores.softmax(2)).sum(2))


class Indexer(nn.Module):
    def __init__(self, config: DeepSeekV41Config, owns_k: bool):
        super().__init__()
        self.config = config
        self.wq_b = nn.Linear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.dim, config.index_n_heads, bias=False)
        if owns_k:
            self.wk = nn.Linear(config.head_dim, config.index_head_dim, bias=False)
            self.k_norm = RMSNorm(config.index_head_dim, config.norm_eps)

    def scores(
        self, x: torch.Tensor, qr: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        c = self.config
        q = self.wq_b(qr).unflatten(-1, (c.index_n_heads, c.index_head_dim))
        q = rotary(q, positions, c, True)
        scores = torch.einsum("bshd,btd->bsht", q, keys).relu()
        weights = self.weights_proj(x) / math.sqrt(c.index_head_dim * c.index_n_heads)
        return (scores * weights.unsqueeze(-1)).sum(-2)


class Attention(nn.Module):
    def __init__(self, config: DeepSeekV41Config, layer: int):
        super().__init__()
        c = self.config = config
        self.layer = layer
        self.ratio = c.compress_ratios[layer]
        self.attn_sink = nn.Parameter(torch.zeros(c.n_heads))
        self.wq_a = nn.Linear(c.dim, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.norm_eps)
        self.wq_b = nn.Linear(c.q_lora_rank, c.n_heads * c.head_dim, bias=False)
        self.wkv = nn.Linear(c.dim, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.norm_eps)
        self.wo_a = nn.Linear(
            c.n_heads * c.head_dim // c.o_groups, c.o_groups * c.o_lora_rank, bias=False
        )
        self.wo_b = nn.Linear(c.o_groups * c.o_lora_rank, c.dim, bias=False)
        self.compressor = Compressor(c, self.ratio) if layer in c.kv_source_layers else None
        self.indexer = (
            Indexer(c, self.compressor is not None) if layer in c.index_source_layers else None
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        old: LayerState,
        shared: SharedAttention,
        trace: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, LayerState]:
        c = self.config
        batch, length, _ = x.shape
        start, end = int(positions[0]), int(positions[-1]) + 1
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (c.n_heads, c.head_dim))
        q = rotary(q, positions, c, bool(self.ratio))
        kv = rotary(self.kv_norm(self.wkv(x)), positions, c, bool(self.ratio))
        new = LayerState(window=append(old.window, kv))
        kv = new.window
        offsets = torch.arange(c.window_size, device=x.device)
        windows = positions[:, None] - c.window_size + 1 + offsets
        windows = windows.masked_fill(windows < 0, -1).expand(batch, -1, -1)
        indices = windows
        if self.ratio:
            count = end // self.ratio
            if self.compressor is not None:
                latent = self.compressor(x, old, new)
                first_group = start // self.ratio
                compressed_positions = (
                    torch.arange(first_group, count, device=x.device) * self.ratio
                )
                keys = self.indexer.k_norm(self.indexer.wk(latent))
                keys = rotary(keys, compressed_positions, c, True)
                new.index_keys = append(old.index_keys, keys)
                rotated = rotary(latent, compressed_positions, c, True)
                new.compressed = append(old.compressed, rotated)
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
                # 同时过滤候选外的 -inf，而不只过滤因果范围外的位置。
                valid = torch.isfinite(scores.gather(-1, selected))
                shared.indices = selected.masked_fill(~valid, -1)
                trace[f"layers.{self.layer}.index"] = shared.indices
            compressed_indices = torch.where(shared.indices >= 0, shared.indices + end, -1)
            indices = torch.cat((indices, compressed_indices), -1)
            kv = torch.cat((kv, shared.compressed), 1)
        output = sparse_attention(q, kv, indices, self.attn_sink)
        output = rotary(output, positions, c, bool(self.ratio), inverse=True)
        output = output.reshape(batch, length, c.o_groups, -1)
        weight = self.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", output, weight).flatten(-2)
        return self.wo_b(output), new


class Gate(nn.Module):
    def __init__(self, config: DeepSeekV41Config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.dim))
        self.bias = nn.Parameter(torch.zeros(config.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        logits = F.linear(x, self.weight) / c.gate_temp
        if c.score_func == "sqrtsoftplus":
            scores = F.softplus(logits).sqrt()
        elif c.score_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(-1)
        indices = (scores + self.bias).topk(c.n_activated_experts, -1).indices
        weights = scores.gather(-1, indices)
        if c.norm_topk_prob and c.n_activated_experts > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * c.route_scale, indices


class Expert(nn.Module):
    def __init__(self, config: DeepSeekV41Config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.moe_inter_dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.moe_inter_dim, bias=False)
        self.w2 = nn.Linear(config.moe_inter_dim, config.dim, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        gate, up = self.w1(x), self.w3(x)
        if self.limit > 0:
            gate, up = gate.clamp_max(self.limit), up.clamp(-self.limit, self.limit)
        hidden = F.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights
        return self.w2(hidden)


class MoE(nn.Module):
    def __init__(self, config: DeepSeekV41Config):
        super().__init__()
        self.gate = Gate(config)
        self.experts = nn.ModuleList(Expert(config) for _ in range(config.n_routed_experts))
        self.shared_experts = Expert(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        weights, indices = self.gate(x)
        result = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == i)
            if rows.numel():
                result.index_add_(0, rows, expert(x[rows], weights[rows, slots, None]))
        result = result + self.shared_experts(x)
        return result.reshape(shape), indices.reshape(*shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, config: DeepSeekV41Config, layer: int):
        super().__init__()
        c = self.config = config
        self.layer = layer
        self.attn = Attention(c, layer)
        self.ffn = MoE(c)
        self.engram = Engram(c, layer) if layer in c.engram_layer_ids else None
        self.attn_norm = RMSNorm(c.dim, c.norm_eps)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        count = c.hc_mult * (2 + c.hc_mult)
        for name in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{name}_fn", nn.Parameter(torch.empty(count, c.hc_mult * c.dim))
            )
            self.register_parameter(f"hc_{name}_base", nn.Parameter(torch.zeros(count)))
            self.register_parameter(f"hc_{name}_scale", nn.Parameter(torch.full((3,), 0.1)))

    def mixes(self, x: torch.Tensor, name: str):
        flat = x.flatten(-2)
        projection = F.linear(flat, getattr(self, f"hc_{name}_fn"))
        projection = projection * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.config.norm_eps
        )
        return hc_split(
            projection,
            getattr(self, f"hc_{name}_scale"),
            getattr(self, f"hc_{name}_base"),
            self.config,
        )

    @staticmethod
    def pre(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        return (x * mix.unsqueeze(-1)).sum(-2)

    @staticmethod
    def post(
        output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        # comb 的行是来源流、列是目标流；将来源维度求和。
        return output.unsqueeze(-2) * post.unsqueeze(-1) + torch.einsum(
            "bsij,bsid->bsjd", comb, residual
        )

    def forward(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        positions: torch.Tensor,
        old: LayerState,
        shared: SharedAttention,
        hashes: torch.Tensor | None,
        trace: dict[str, torch.Tensor],
    ):
        if self.engram is not None:
            x = self.engram(x, hashes[:, :, self.engram.hash_index])
        pre, post, comb = self.mixes(x, "attn")
        attn, state = self.attn(self.attn_norm(self.pre(x, pre_mix)), positions, old, shared, trace)
        x = self.post(attn, x, post, comb)
        next_pre, post, comb = self.mixes(x, "ffn")
        ffn, routes = self.ffn(self.ffn_norm(self.pre(x, pre)))
        x = self.post(ffn, x, post, comb)
        trace[f"layers.{self.layer}.routes"] = routes
        trace[f"layers.{self.layer}.hidden"] = x
        return x, next_pre, state


class DeepSeekV41ReferenceModel(nn.Module):
    """提供 FP32 logits 和增量状态的数值基准，不接受真实量化 checkpoint。"""

    def __init__(
        self,
        config: DeepSeekV41Config,
        *,
        token_map: Sequence[int] | None = None,
        max_seq_len: int = 256,
    ):
        super().__init__()
        # 在任何参数分配前拦住发布版尺寸；真实权重必须走后续量化分片加载器。
        from .checkpoint import expected_backbone_shapes

        estimate = sum(math.prod(shape) for shape in expected_backbone_shapes(config).values())
        if estimate > 10_000_000:
            raise ValueError(
                "FP32 reference is limited to small models; use DeepSeekV41Config.tiny()"
            )
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if config.engram_layer_ids and token_map is None:
            raise ValueError("Engram requires an explicit compressed token map")
        self.config, self.max_seq_len = config, max_seq_len
        self._cache_owner = object()
        self.engram_hash = EngramHasher(config, token_map) if config.engram_layer_ids else None
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.float()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith((".bias", ".attn_sink", "_base")):
                    parameter.zero_()
                elif name.endswith("_scale"):
                    parameter.fill_(0.1)
                elif "norm.weight" in name or name.endswith(("q_weight", "k_weight")):
                    parameter.fill_(1)
                else:
                    parameter.normal_(std=0.1)

    def new_cache(self) -> ReferenceCache:
        return ReferenceCache(self._cache_owner, layers=[LayerState() for _ in self.layers])

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ReferenceCache | None = None,
        *,
        trace: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or min(input_ids.shape) <= 0 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a nonempty int64 [batch, sequence] tensor")
        if input_ids.device != self.embed.weight.device or self.embed.weight.dtype != torch.float32:
            raise ValueError(
                "reference model requires FP32 parameters and inputs on the same device"
            )
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
            raise ValueError("reference context limit exceeded")
        tokens = append(cache.tokens, input_ids)
        hashes = self.engram_hash(tokens, start=start) if self.engram_hash is not None else None
        positions = torch.arange(start, tokens.shape[1], device=input_ids.device)
        h = self.embed(input_ids).unsqueeze(-2).expand(-1, -1, self.config.hc_mult, -1)
        pre_mix = h.new_zeros(*h.shape[:-1])
        pre_mix[..., 0] = 1
        shared, new_states = SharedAttention(), []
        captured = {} if trace is None else trace
        if hashes is not None:
            captured["hashes"] = hashes
        for layer, state in zip(self.layers, cache.layers, strict=True):
            h, pre_mix, new = layer(h, pre_mix, positions, state, shared, hashes, captured)
            new_states.append(new)
        logits = self.head(self.norm(Block.pre(h, pre_mix)))
        if target is not None:
            target.tokens = tokens.clone()
            target.layers = new_states
        return logits
