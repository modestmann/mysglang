"""Per-request recurrent state for Gated DeltaNet (linear attention) layers.

Full-attention layers keep one K/V row per token in the paged pool. A DeltaNet layer
instead folds the whole history into a fixed-size state, so each active request owns
one *slot* holding, for every linear-attention layer:

- ``conv_states``: the last ``kernel - 1`` pre-convolution inputs of the causal conv;
- ``ssm_states``: the ``[heads, head_k_dim, head_v_dim]`` delta-rule state in float32.

Unlike paged K/V, a state cannot be truncated after the fact: it already absorbed every
token it saw. Speculative verification therefore runs in *log* mode: layers read their
slot state but do not write it back, and record the per-token inputs they used. Once
the scheduler knows how many candidates were accepted, the model replays only the
accepted prefix from the untouched slot state (see ``SpeculativeStateLog``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass
class LinearAttentionLogEntry:
    """Inputs one DeltaNet layer used during a verify forward, in packed token order."""

    conv_inputs: torch.Tensor  # [tokens, conv_dim] before the causal convolution
    query: torch.Tensor  # [tokens, heads, head_k_dim]
    key: torch.Tensor  # [tokens, heads, head_k_dim]
    value: torch.Tensor  # [tokens, heads, head_v_dim]
    decay: torch.Tensor  # [tokens, heads], log-space gate g
    beta: torch.Tensor  # [tokens, heads]


@dataclass
class SpeculativeStateLog:
    entries: dict[int, LinearAttentionLogEntry] = field(default_factory=dict)


@dataclass(frozen=True)
class RecurrentStateBatch:
    """Slot selection shared by every DeltaNet layer in one cached forward."""

    pool: RecurrentStatePool
    request_ids: tuple[str, ...]
    slots: torch.Tensor
    append_lengths: tuple[int, ...]
    cu_seqlens: torch.Tensor
    log: SpeculativeStateLog | None = None

    @property
    def speculative(self) -> bool:
        return self.log is not None

    @property
    def total_tokens(self) -> int:
        return sum(self.append_lengths)


@dataclass(frozen=True)
class RecurrentDecodeBuffer:
    """Fixed-address slot metadata reused by one exact Decode (or verify) batch shape."""

    slots: torch.Tensor
    cu_seqlens: torch.Tensor
    append_length: int
    log: SpeculativeStateLog | None = None

    @property
    def batch_size(self) -> int:
        return self.slots.numel()


@dataclass(frozen=True)
class RecurrentStateStats:
    total_slots: int
    free_slots: int
    memory_bytes: int


class RecurrentStatePool:
    """Fixed-address state slots for every linear-attention layer of one TP rank."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        conv_dim: int,
        conv_kernel_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dtype: torch.dtype,
        device: torch.device | str,
        ssm_dtype: torch.dtype = torch.float32,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "num_slots": num_slots,
            "conv_dim": conv_dim,
            "conv_kernel_size": conv_kernel_size - 1,
            "num_heads": num_heads,
            "head_k_dim": head_k_dim,
            "head_v_dim": head_v_dim,
        }
        if invalid := [name for name, value in dimensions.items() if value <= 0]:
            raise ValueError(f"recurrent state dimensions must be positive: {invalid}")
        self.conv_states = torch.zeros(
            (num_layers, num_slots, conv_dim, conv_kernel_size - 1),
            dtype=conv_dtype,
            device=device,
        )
        self.ssm_states = torch.zeros(
            (num_layers, num_slots, num_heads, head_k_dim, head_v_dim),
            dtype=ssm_dtype,
            device=device,
        )
        self._free_slots = list(range(num_slots))
        self._slots: dict[str, int] = {}

    @property
    def num_slots(self) -> int:
        return self.ssm_states.size(1)

    @property
    def num_layers(self) -> int:
        return self.ssm_states.size(0)

    @property
    def device(self) -> torch.device:
        return self.ssm_states.device

    @property
    def request_ids(self) -> frozenset[str]:
        return frozenset(self._slots)

    @property
    def memory_bytes(self) -> int:
        return (
            self.conv_states.numel() * self.conv_states.element_size()
            + self.ssm_states.numel() * self.ssm_states.element_size()
        )

    @property
    def stats(self) -> RecurrentStateStats:
        return RecurrentStateStats(
            total_slots=self.num_slots,
            free_slots=len(self._free_slots),
            memory_bytes=self.memory_bytes,
        )

    def can_reserve(self) -> bool:
        return bool(self._free_slots)

    def reserve(self, request_id: str) -> int:
        if request_id in self._slots:
            raise ValueError(f"request already owns a recurrent state slot: {request_id}")
        if not self._free_slots:
            raise RuntimeError("no free recurrent state slot")
        # 最小编号优先，使 TP 各 rank 的镜像调度得到相同 slot。
        slot = min(self._free_slots)
        self._free_slots.remove(slot)
        # 新请求从零状态开始；slot 上一任主人的状态必须清除。
        self.conv_states[:, slot].zero_()
        self.ssm_states[:, slot].zero_()
        self._slots[request_id] = slot
        return slot

    def release(self, request_id: str) -> int:
        slot = self._slots.pop(request_id)
        self._free_slots.append(slot)
        return slot

    def slot(self, request_id: str) -> int:
        try:
            return self._slots[request_id]
        except KeyError as exc:
            raise KeyError(f"request has no recurrent state slot: {request_id}") from exc

    def prepare_batch(
        self,
        request_ids: Sequence[str],
        append_lengths: Sequence[int],
        *,
        speculative: bool = False,
    ) -> RecurrentStateBatch:
        requests = tuple(request_ids)
        lengths = tuple(append_lengths)
        if len(requests) != len(lengths):
            raise ValueError("recurrent batch lengths must match request IDs")
        if len(set(requests)) != len(requests):
            raise ValueError("request_ids must be unique within a recurrent batch")
        if any(length <= 0 for length in lengths):
            raise ValueError("recurrent append lengths must be positive")
        slots = [self.slot(request_id) for request_id in requests]
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        return RecurrentStateBatch(
            pool=self,
            request_ids=requests,
            slots=torch.tensor(slots, dtype=torch.long, device=self.device),
            append_lengths=lengths,
            cu_seqlens=torch.tensor(cumulative, dtype=torch.long, device=self.device),
            log=SpeculativeStateLog() if speculative else None,
        )

    def allocate_decode_buffer(
        self,
        batch_size: int,
        *,
        append_length: int = 1,
        speculative: bool = False,
    ) -> RecurrentDecodeBuffer:
        """Stable slot/offset tensors so a CUDA Graph can read this batch's states."""
        return RecurrentDecodeBuffer(
            slots=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            cu_seqlens=torch.arange(
                0,
                (batch_size + 1) * append_length,
                append_length,
                dtype=torch.long,
                device=self.device,
            ),
            append_length=append_length,
            # Graph replay does not re-run Python, so the verify log object (whose
            # tensors live in the graph pool) is owned by the buffer and reused.
            log=SpeculativeStateLog() if speculative else None,
        )

    def prepare_decode_batch(
        self,
        request_ids: Sequence[str],
        buffer: RecurrentDecodeBuffer,
    ) -> RecurrentStateBatch:
        requests = tuple(request_ids)
        if len(requests) != buffer.batch_size:
            raise ValueError("request batch size does not match the recurrent decode buffer")
        slots = [self.slot(request_id) for request_id in requests]
        buffer.slots.copy_(torch.tensor(slots, dtype=torch.long))
        return RecurrentStateBatch(
            pool=self,
            request_ids=requests,
            slots=buffer.slots,
            append_lengths=(buffer.append_length,) * len(requests),
            cu_seqlens=buffer.cu_seqlens,
            log=buffer.log,
        )

    def snapshot(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy the selected slots' states (used around CUDA Graph warmup)."""
        return (
            self.conv_states.index_select(1, slots).clone(),
            self.ssm_states.index_select(1, slots).clone(),
        )

    def restore(self, slots: torch.Tensor, snapshot: tuple[torch.Tensor, torch.Tensor]) -> None:
        conv, ssm = snapshot
        self.conv_states.index_copy_(1, slots, conv)
        self.ssm_states.index_copy_(1, slots, ssm)

    def check_integrity(self) -> None:
        owned = list(self._slots.values())
        if len(owned) != len(set(owned)):
            raise RuntimeError("a recurrent state slot has multiple owners")
        if set(owned) & set(self._free_slots):
            raise RuntimeError("a recurrent state slot is both owned and free")
        if set(owned) | set(self._free_slots) != set(range(self.num_slots)):
            raise RuntimeError("recurrent state slot accounting does not cover the pool")
