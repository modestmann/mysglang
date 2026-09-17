from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mysglang.config import ModelConfig


class SlotKVCache:
    """Fixed contiguous cache slots for a continuously changing decode batch.

    Each request owns one slot while it is active. Selected slots may have
    different sequence lengths; append gathers their valid prefixes into a
    temporary padded batch and returns the corresponding attention mask.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_kv_heads: int,
        max_length: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "num_slots": num_slots,
            "num_kv_heads": num_kv_heads,
            "max_length": max_length,
            "head_dim": head_dim,
        }
        if invalid := [name for name, value in dimensions.items() if value <= 0]:
            raise ValueError(f"cache dimensions must be positive: {invalid}")

        shape = (num_layers, num_slots, num_kv_heads, max_length, head_dim)
        self._keys = torch.empty(shape, dtype=dtype, device=device)
        self._values = torch.empty_like(self._keys)
        #这里对kvcche进行了精细化管理  [层，batch的request数目]
        self._lengths = [[0] * num_slots for _ in range(num_layers)]

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        *,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> SlotKVCache:
        return cls(
            num_layers=config.num_layers,
            num_slots=num_slots,
            num_kv_heads=config.num_key_value_heads,
            max_length=config.max_position_embeddings,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )

    @property
    def num_layers(self) -> int:
        return self._keys.size(0)

    @property
    def num_slots(self) -> int:
        return self._keys.size(1)

    @property
    def max_length(self) -> int:
        return self._keys.size(3)

    def lengths(self, slot_ids: Sequence[int]) -> tuple[int, ...]:
        slots = self._validate_slots(slot_ids)
        result = tuple(self._lengths[0][slot] for slot in slots)
        for layer_idx in range(1, self.num_layers):
            current = tuple(self._lengths[layer_idx][slot] for slot in slots)
            if current != result:
                raise RuntimeError("KV cache layers have inconsistent slot lengths")
        return result

    def layer_lengths(self, layer_idx: int, slot_ids: Sequence[int]) -> tuple[int, ...]:
        self._validate_layer(layer_idx)
        slots = self._validate_slots(slot_ids)
        return tuple(self._lengths[layer_idx][slot] for slot in slots)

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append equal-size chunks and return padded K/V plus a causal mask."""
        self._validate_layer(layer_idx)
        slots = self._validate_slots(slot_ids)

        if len(set(slots)) != len(slots):
            raise ValueError("slot_ids must be unique within a batch")

        expected = (len(slots), self._keys.size(2), self._keys.size(4))
        actual = (key.size(0), key.size(1), key.size(3)) if key.ndim == 4 else None

        if actual != expected:
            raise ValueError(
                "key must have shape "
                f"[{len(slots)}, {self._keys.size(2)}, sequence, {self._keys.size(4)}]"
            )
        if value.shape != key.shape:
            raise ValueError("key and value must have the same shape")
        if key.device != self._keys.device or value.device != self._values.device:
            raise ValueError("key/value device must match the cache device")
        if key.dtype != self._keys.dtype or value.dtype != self._values.dtype:
            raise ValueError("key/value dtype must match the cache dtype")

        starts = self.layer_lengths(layer_idx, slots)
        chunk_length = key.size(2)
        ends = tuple(start + chunk_length for start in starts)

        if max(ends) > self.max_length:
            raise ValueError(f"KV cache capacity exceeded: {max(ends)} > {self.max_length}")

        with torch.no_grad():
            for batch_index, (slot, start, end) in enumerate(zip(slots, starts, ends)):
                self._keys[layer_idx, slot, :, start:end].copy_(key[batch_index])
                self._values[layer_idx, slot, :, start:end].copy_(value[batch_index])
                self._lengths[layer_idx][slot] = end
        #每层用每层的KV
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=self._keys.device)
        max_end = max(ends)
        keys = self._keys[layer_idx].index_select(0, slot_tensor)[:, :, :max_end]
        values = self._values[layer_idx].index_select(0, slot_tensor)[:, :, :max_end]

        query_positions = torch.tensor(starts, device=key.device)[:, None]#增加维度
        query_positions = query_positions + torch.arange(chunk_length, device=key.device)[None, :]
        key_positions = torch.arange(max_end, device=key.device)
        attention_mask = key_positions[None, None, :] <= query_positions[:, :, None]
        return keys, values, attention_mask[:, None, :, :]

    def reset_slot(self, slot_id: int) -> None:
        slot = self._validate_slots((slot_id,))[0]
        for layer_lengths in self._lengths:
            layer_lengths[slot] = 0

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")

    def _validate_slots(self, slot_ids: Sequence[int]) -> tuple[int, ...]:
        slots = tuple(slot_ids)
        if not slots:
            raise ValueError("slot_ids must not be empty")
        if any(not isinstance(slot, int) or isinstance(slot, bool) for slot in slots):
            raise TypeError("slot_ids must contain integers")
        if any(not 0 <= slot < self.num_slots for slot in slots):
            raise IndexError("slot_id out of range")
        return slots
