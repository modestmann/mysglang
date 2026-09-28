"""Qwen3.5 Gated DeltaNet: linear attention with a fixed-size recurrent state.

For each value head the layer keeps a matrix state ``S`` of shape ``[head_k, head_v]``
and, per token ``t``, applies the gated delta rule::

    S = exp(g_t) * S                          # data-dependent decay
    S = S + k_t ⊗ beta_t * (v_t - S^T k_t)    # delta-rule correction
    o_t = S^T q_t

A causal depthwise convolution first mixes the last ``kernel`` projected inputs. The
state therefore consists of the last ``kernel - 1`` conv inputs plus ``S``; both live in
``RecurrentStatePool`` slots between forwards.

Kernels: the PyTorch reference uses the chunked algorithm for multi-token segments and
the per-token recurrence for Decode, mirroring Hugging Face. ``fla`` (flash-linear-
attention) provides the corresponding varlen Triton kernels on CUDA.
"""

from __future__ import annotations

import importlib.util

import torch
import torch.nn.functional as F
from torch import nn

from mysglang.cache import LinearAttentionLogEntry, RecurrentStateBatch
from mysglang.config import ModelConfig

from .parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelContext

FLA_AVAILABLE = importlib.util.find_spec("fla") is not None
LINEAR_ATTENTION_BACKENDS = ("auto", "torch", "fla")


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence of the chunked algorithm (a port of the HF reference).

    Inputs are float32 ``[T, H, D]`` / ``[T, H]`` with L2-normalized Q/K; the state is
    ``[H, Dk, Dv]``. Returns ``[T, H, Dv]`` outputs and the final state.
    """
    length = query.size(0)
    query, key, value = (x.transpose(0, 1) for x in (query, key, value))
    decay, beta = decay.transpose(0, 1), beta.transpose(0, 1)
    pad = (chunk_size - length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    decay = F.pad(decay, (0, pad))
    beta = F.pad(beta, (0, pad))
    query = query * query.size(-1) ** -0.5

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    heads = query.size(0)

    def chunks(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(heads, -1, chunk_size, x.size(-1))

    query, key, value, key_beta, value_beta = map(
        chunks, (query, key, value, key_beta, value_beta)
    )
    decay = decay.reshape(heads, -1, chunk_size).cumsum(dim=-1)
    diagonal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device)
    )
    decay_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    attention = -((key_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(
        diagonal_mask, 0
    )
    # Forward substitution: (I + A)^-1 for the strictly lower-triangular chunk matrix.
    for row_index in range(1, chunk_size):
        row = attention[..., row_index, :row_index].clone()
        sub = attention[..., :row_index, :row_index].clone()
        attention[..., row_index, :row_index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attention = attention + torch.eye(chunk_size, dtype=attention.dtype, device=query.device)
    value = attention @ value_beta
    key_cumdecay = attention @ (key_beta * decay.exp().unsqueeze(-1))

    state = initial_state.to(torch.float32)
    output = torch.zeros_like(value)
    for chunk_index in range(query.size(1)):
        chunk_query = query[:, chunk_index]
        chunk_key = key[:, chunk_index]
        chunk_decay = decay[:, chunk_index]
        intra = chunk_query @ chunk_key.transpose(-1, -2) * decay_mask[:, chunk_index]
        corrected = value[:, chunk_index] - key_cumdecay[:, chunk_index] @ state
        inter = (chunk_query * chunk_decay.unsqueeze(-1).exp()) @ state
        output[:, chunk_index] = inter + intra @ corrected
        state = state * chunk_decay[:, -1, None, None].exp() + (
            chunk_key * (chunk_decay[:, -1, None] - chunk_decay).exp().unsqueeze(-1)
        ).transpose(-1, -2) @ corrected

    output = output.reshape(heads, -1, output.size(-1))[:, :length]
    return output.transpose(0, 1), state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched per-token recurrence over ``[B, S, H, D]`` padded segments.

    ``lengths`` masks padded steps, so every request stops updating its state at its own
    length. This is the Decode path and the verification path whose prefix is replayed.
    """
    query = query * query.size(-1) ** -0.5
    state = state.to(torch.float32)
    output = value.new_zeros((*value.shape[:3], value.size(-1)))
    for step in range(query.size(1)):
        updated = state * decay[:, step].exp()[..., None, None]
        memory = (updated * key[:, step, :, :, None]).sum(dim=-2)
        delta = (value[:, step] - memory) * beta[:, step, :, None]
        updated = updated + key[:, step, :, :, None] * delta[:, :, None, :]
        output[:, step] = (updated * query[:, step, :, :, None]).sum(dim=-2)
        if lengths is None:
            state = updated
        else:
            active = (lengths > step)[:, None, None, None]
            state = torch.where(active, updated, state)
    return output, state


def _pad_segments(packed: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
    """``[T, ...]`` packed rows -> ``[B, max_len, ...]`` zero-padded segments."""
    if len(set(lengths)) == 1:
        return packed.reshape(len(lengths), lengths[0], *packed.shape[1:])
    padded = packed.new_zeros((len(lengths), max(lengths), *packed.shape[1:]))
    offset = 0
    for index, length in enumerate(lengths):
        padded[index, :length] = packed[offset : offset + length]
        offset += length
    return padded


def _unpad_segments(padded: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
    if len(set(lengths)) == 1:
        return padded.reshape(-1, *padded.shape[2:])
    return torch.cat([padded[index, :length] for index, length in enumerate(lengths)])


def causal_conv1d_segments(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    initial_states: torch.Tensor,
    lengths: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv + SiLU for packed segments continuing their conv states.

    ``inputs`` is ``[T, C]``, ``weight`` ``[C, kernel]`` and ``initial_states``
    ``[B, C, kernel - 1]`` (the previous inputs of each request). Returns packed outputs
    and every request's new conv state.
    """
    kernel = weight.size(-1)
    padded = _pad_segments(inputs, lengths).transpose(1, 2)
    window = torch.cat((initial_states.to(inputs.dtype), padded), dim=-1)
    output = F.conv1d(window, weight.unsqueeze(1), groups=weight.size(0))
    output = F.silu(output).transpose(1, 2)
    if len(set(lengths)) == 1:
        final_states = window[..., -(kernel - 1) :]
    else:
        final_states = torch.stack(
            [window[index, :, length : length + kernel - 1] for index, length in enumerate(lengths)]
        )
    return _unpad_segments(output, lengths), final_states.contiguous()


class GatedRMSNorm(nn.Module):
    """RMSNorm on each value head, then multiplied by SiLU(z) (the output gate)."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        hidden = x.float()
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.eps)
        hidden = self.weight * hidden.to(dtype)
        return (hidden * F.silu(gate.float())).to(dtype)


class _ConvWeight(nn.Module):
    """Holds the depthwise conv weight under HF's ``conv1d.weight`` name and shape."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)


class Qwen3_5GatedDeltaNet(nn.Module):
    """One linear-attention token mixer, sharded by heads under tensor parallelism."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        state_layer_idx: int,
        tensor_parallel: TensorParallelContext,
        *,
        backend: str = "auto",
        quantize=None,
    ) -> None:
        super().__init__()
        if backend not in LINEAR_ATTENTION_BACKENDS:
            raise ValueError(f"linear attention backend must be one of {LINEAR_ATTENTION_BACKENDS}")
        if backend == "fla" and not FLA_AVAILABLE:
            raise RuntimeError("the fla linear-attention backend requires flash-linear-attention")
        self.layer_idx = layer_idx
        self.state_layer_idx = state_layer_idx
        self.backend = backend
        self.tensor_parallel = tensor_parallel
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_k_heads = tensor_parallel.local_size(
            config.linear_num_key_heads, "linear_num_key_heads"
        )
        self.num_v_heads = tensor_parallel.local_size(
            config.linear_num_value_heads, "linear_num_value_heads"
        )
        self.global_key_dim = config.linear_num_key_heads * self.head_k_dim
        self.global_value_dim = config.linear_num_value_heads * self.head_v_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        prefix = f"layers.{layer_idx}.linear_attn"
        # HF 分开存 in_proj_qkv / in_proj_z；各逻辑段独立切分后在本 rank 融合为一次 GEMM。
        self.in_proj_qkvz = ColumnParallelLinear(
            config.hidden_size,
            (
                self.global_key_dim,
                self.global_key_dim,
                self.global_value_dim,
                self.global_value_dim,
            ),
            tensor_parallel,
            quant=quantize(f"{prefix}.in_proj_qkv", f"{prefix}.in_proj_z") if quantize else None,
        )
        # b/a 只有每头一个标量，官方 FP8 checkpoint 也保留 BF16。
        self.in_proj_ba = ColumnParallelLinear(
            config.hidden_size,
            (config.linear_num_value_heads, config.linear_num_value_heads),
            tensor_parallel,
            quant=quantize(f"{prefix}.in_proj_b", f"{prefix}.in_proj_a") if quantize else None,
        )
        self.conv1d = _ConvWeight(self.conv_dim, self.conv_kernel_size)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(1, 16)))
        self.norm = GatedRMSNorm(self.head_v_dim, config.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            self.global_value_dim,
            config.hidden_size,
            tensor_parallel,
            quant=quantize(f"{prefix}.out_proj") if quantize else None,
        )

    # 以下三个方法告诉 loader 如何从 HF 的全局张量中取本 rank 的分片。
    def qkv_source_rows(self, part: int) -> slice:
        """Rows of HF ``in_proj_qkv`` (and ``conv1d``) for q/k/v ``part`` on this rank."""
        offsets = (0, self.global_key_dim, 2 * self.global_key_dim)
        local = self.in_proj_qkvz.source_segment(part)
        return slice(offsets[part] + local.start, offsets[part] + local.stop)

    def conv_local_rows(self, part: int) -> slice:
        return self.in_proj_qkvz.local_segment(part)

    def head_shard(self) -> slice:
        return self.in_proj_ba.source_segment(0)

    def _use_fla(self, x: torch.Tensor) -> bool:
        if self.backend == "torch":
            return False
        usable = FLA_AVAILABLE and x.is_cuda and x.dtype in {torch.float16, torch.bfloat16}
        if self.backend == "fla" and not usable:
            raise RuntimeError("the fla backend requires CUDA float16/bfloat16 tensors")
        return usable

    def forward(
        self,
        x: torch.Tensor,
        state_batch: RecurrentStateBatch | None,
    ) -> torch.Tensor:
        original_shape = x.shape
        tokens = x.reshape(-1, x.size(-1))
        if state_batch is None:
            # 无缓存的整段前向（oracle 路径）：每行 batch 都从零状态开始。
            batch_rows = original_shape[0] if x.ndim == 3 else 1
            lengths = (tokens.size(0) // batch_rows,) * batch_rows
            conv_states = tokens.new_zeros((batch_rows, self.conv_dim, self.conv_kernel_size - 1))
            ssm_states = tokens.new_zeros(
                (batch_rows, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                dtype=torch.float32,
            )
            cu_seqlens = None
        else:
            if state_batch.total_tokens != tokens.size(0):
                raise ValueError("recurrent batch metadata disagrees with the token count")
            lengths = state_batch.append_lengths
            pool = state_batch.pool
            conv_states = pool.conv_states[self.state_layer_idx].index_select(0, state_batch.slots)
            ssm_states = pool.ssm_states[self.state_layer_idx].index_select(0, state_batch.slots)
            cu_seqlens = state_batch.cu_seqlens

        qkvz = self.in_proj_qkvz(tokens)
        mixed = qkvz[:, : self.conv_dim]
        gate_z = qkvz[:, self.conv_dim :]
        b, a = self.in_proj_ba(tokens).split((self.num_v_heads, self.num_v_heads), dim=-1)
        conv_out, new_conv_states = causal_conv1d_segments(
            mixed,
            self.conv1d.weight.squeeze(1),
            conv_states,
            lengths,
        )
        query, key, value = conv_out.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        query = query.reshape(-1, self.num_k_heads, self.head_k_dim)
        key = key.reshape(-1, self.num_k_heads, self.head_k_dim)
        value = value.reshape(-1, self.num_v_heads, self.head_v_dim)
        ratio = self.num_v_heads // self.num_k_heads
        if ratio > 1:
            query = query.repeat_interleave(ratio, dim=1)
            key = key.repeat_interleave(ratio, dim=1)
        beta = b.sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        speculative = state_batch is not None and state_batch.speculative
        if speculative:
            assert state_batch is not None and state_batch.log is not None
            state_batch.log.entries[self.state_layer_idx] = LinearAttentionLogEntry(
                conv_inputs=mixed,
                query=query,
                key=key,
                value=value,
                decay=decay,
                beta=beta,
            )
        core, new_ssm_states = self._delta_rule(
            query, key, value, decay, beta, ssm_states, lengths, cu_seqlens, speculative
        )
        if state_batch is not None and not speculative:
            # 普通前向直接推进状态；verify 前向保持 slot 不变，等接受长度确定后再 replay。
            pool = state_batch.pool
            pool.conv_states[self.state_layer_idx].index_copy_(
                0, state_batch.slots, new_conv_states.to(pool.conv_states.dtype)
            )
            pool.ssm_states[self.state_layer_idx].index_copy_(
                0, state_batch.slots, new_ssm_states.to(pool.ssm_states.dtype)
            )

        core = self.norm(
            core.to(x.dtype).reshape(-1, self.head_v_dim),
            gate_z.reshape(-1, self.head_v_dim),
        )
        output = self.out_proj(core.reshape(tokens.size(0), self.value_dim))
        return output.reshape(*original_shape[:-1], output.size(-1))

    def _delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        states: torch.Tensor,
        lengths: tuple[int, ...],
        cu_seqlens: torch.Tensor | None,
        speculative: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_fla(query):
            return self._fla_delta_rule(
                query, key, value, decay, beta, states, lengths, cu_seqlens, speculative
            )
        query = l2norm(query.float())
        key = l2norm(key.float())
        value = value.float()
        beta = beta.float()
        if speculative or max(lengths) == 1:
            # Decode 与 verify 都走逐 token 递归；verify 的前缀之后会被原样 replay。
            # 等长 batch 不创建任何 host 张量，因此可以进入 CUDA Graph。
            uniform = len(set(lengths)) == 1
            output, final = torch_recurrent_gated_delta_rule(
                _pad_segments(query, lengths),
                _pad_segments(key, lengths),
                _pad_segments(value, lengths),
                _pad_segments(decay, lengths),
                _pad_segments(beta, lengths),
                states,
                None if uniform else torch.tensor(lengths, device=query.device),
            )
            return _unpad_segments(output, lengths), final
        outputs = []
        finals = []
        offset = 0
        for index, length in enumerate(lengths):
            segment = slice(offset, offset + length)
            offset += length
            if length == 1:
                output, final = torch_recurrent_gated_delta_rule(
                    query[segment][None],
                    key[segment][None],
                    value[segment][None],
                    decay[segment][None],
                    beta[segment][None],
                    states[index : index + 1],
                )
                outputs.append(output[0])
                finals.append(final[0])
                continue
            output, final = torch_chunk_gated_delta_rule(
                query[segment],
                key[segment],
                value[segment],
                decay[segment],
                beta[segment],
                states[index],
            )
            outputs.append(output)
            finals.append(final)
        return torch.cat(outputs), torch.stack(finals)

    def _fla_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        states: torch.Tensor,
        lengths: tuple[int, ...],
        cu_seqlens: torch.Tensor | None,
        speculative: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        recurrent = speculative or max(lengths) == 1
        if recurrent and len(set(lengths)) == 1:
            # 等长的 Decode/verify 用 [batch, length] 布局，不需要 cu_seqlens，可进入 CUDA Graph。
            def dense(x: torch.Tensor) -> torch.Tensor:
                return _pad_segments(x, lengths).contiguous()

            output, final = fused_recurrent_gated_delta_rule(
                q=dense(query),
                k=dense(key),
                v=dense(value),
                g=dense(decay),
                beta=dense(beta),
                initial_state=states.contiguous(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            return _unpad_segments(output, lengths), final
        if cu_seqlens is None:
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            cu_seqlens = torch.tensor(offsets, dtype=torch.long, device=query.device)
        arguments = dict(
            q=query.unsqueeze(0).contiguous(),
            k=key.unsqueeze(0).contiguous(),
            v=value.unsqueeze(0).contiguous(),
            g=decay.unsqueeze(0).contiguous(),
            beta=beta.unsqueeze(0).contiguous(),
            initial_state=states.contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        if recurrent:
            output, final = fused_recurrent_gated_delta_rule(**arguments)
        else:
            output, final = chunk_gated_delta_rule(**arguments)
        return output.squeeze(0), final

    @torch.no_grad()
    def commit_speculative_masked(
        self,
        state_batch: RecurrentStateBatch,
        accepted_lengths: torch.Tensor,
    ) -> None:
        """Fixed-width replay for equal-length verify segments (CUDA Graph friendly).

        Tokens past a request's accepted length get ``g = 0`` and ``beta = 0``: the decay
        ``exp(0) = 1`` keeps the state and the delta-rule correction is zero, so they are
        identity updates. The conv state is the window ending at the accepted length.
        """
        if state_batch.log is None:
            raise ValueError("commit_speculative requires a verify-mode recurrent batch")
        lengths = state_batch.append_lengths
        if len(set(lengths)) != 1:
            raise ValueError("masked commit requires equal verify lengths")
        entry = state_batch.log.entries[self.state_layer_idx]
        pool = state_batch.pool
        batch_size, width = len(lengths), lengths[0]
        steps = torch.arange(width, device=accepted_lengths.device)
        active = steps[None, :] < accepted_lengths[:, None]
        decay = torch.where(active[..., None], entry.decay.view(batch_size, width, -1), 0.0)
        beta = torch.where(active[..., None], entry.beta.view(batch_size, width, -1), 0.0)
        ssm_states = pool.ssm_states[self.state_layer_idx].index_select(0, state_batch.slots)
        _, new_ssm_states = self._delta_rule(
            entry.query,
            entry.key,
            entry.value,
            decay.reshape(-1, decay.size(-1)),
            beta.reshape(-1, beta.size(-1)).to(entry.beta.dtype),
            ssm_states,
            lengths,
            None,
            True,
        )
        conv_states = pool.conv_states[self.state_layer_idx].index_select(0, state_batch.slots)
        window = torch.cat(
            (
                conv_states.to(entry.conv_inputs.dtype),
                entry.conv_inputs.view(batch_size, width, -1).transpose(1, 2),
            ),
            dim=-1,
        )
        kernel = self.conv_kernel_size - 1
        index = accepted_lengths[:, None, None] + torch.arange(kernel, device=window.device)
        new_conv_states = window.gather(2, index.expand(-1, window.size(1), -1))
        pool.conv_states[self.state_layer_idx].index_copy_(
            0, state_batch.slots, new_conv_states.to(pool.conv_states.dtype)
        )
        pool.ssm_states[self.state_layer_idx].index_copy_(
            0, state_batch.slots, new_ssm_states.to(pool.ssm_states.dtype)
        )

    @torch.no_grad()
    def commit_speculative(
        self,
        state_batch: RecurrentStateBatch,
        accepted_lengths: tuple[int, ...],
    ) -> None:
        """Advance each slot by its accepted verify prefix, replayed from the log."""
        if state_batch.log is None:
            raise ValueError("commit_speculative requires a verify-mode recurrent batch")
        entry = state_batch.log.entries[self.state_layer_idx]
        pool = state_batch.pool
        keep = [(index, length) for index, length in enumerate(accepted_lengths) if length > 0]
        if not keep:
            return
        rows = []
        for index, length in keep:
            start = sum(state_batch.append_lengths[:index])
            if length > state_batch.append_lengths[index]:
                raise ValueError("accepted length exceeds the verified segment")
            rows.extend(range(start, start + length))
        row_index = torch.tensor(rows, dtype=torch.long, device=entry.query.device)
        batch_index = torch.tensor([index for index, _ in keep], device=state_batch.slots.device)
        slots = state_batch.slots.index_select(0, batch_index)
        lengths = tuple(length for _, length in keep)

        conv_states = pool.conv_states[self.state_layer_idx].index_select(0, slots)
        _, new_conv_states = causal_conv1d_segments(
            entry.conv_inputs.index_select(0, row_index),
            self.conv1d.weight.squeeze(1),
            conv_states,
            lengths,
        )
        ssm_states = pool.ssm_states[self.state_layer_idx].index_select(0, slots)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        _, new_ssm_states = self._delta_rule(
            entry.query.index_select(0, row_index),
            entry.key.index_select(0, row_index),
            entry.value.index_select(0, row_index),
            entry.decay.index_select(0, row_index),
            entry.beta.index_select(0, row_index),
            ssm_states,
            lengths,
            torch.tensor(offsets, dtype=torch.long, device=slots.device),
            True,
        )
        pool.conv_states[self.state_layer_idx].index_copy_(
            0, slots, new_conv_states.to(pool.conv_states.dtype)
        )
        pool.ssm_states[self.state_layer_idx].index_copy_(
            0, slots, new_ssm_states.to(pool.ssm_states.dtype)
        )
