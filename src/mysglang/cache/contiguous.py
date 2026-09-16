from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mysglang.config import ModelConfig


class ContiguousKVCache:
    """A fixed-capacity, per-request KV cache used to teach incremental decode.

    Storage layout is [layers, batch, kv_heads, max_length, head_dim]. This
    class is inference-only: appending copies detached K/V tensors into the
    preallocated buffers.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_length: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "batch_size": batch_size,
            "num_kv_heads": num_kv_heads,
            "max_length": max_length,
            "head_dim": head_dim,
        }
        if invalid := [name for name, value in dimensions.items() if value <= 0]:
            raise ValueError(f"cache dimensions must be positive: {invalid}")

        shape = (num_layers, batch_size, num_kv_heads, max_length, head_dim)
        self._keys = torch.empty(shape, dtype=dtype, device=device)
        self._values = torch.empty_like(self._keys)
        self._lengths = [0] * num_layers

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        max_length: int | None = None,
    ) -> ContiguousKVCache:
        capacity = config.max_position_embeddings if max_length is None else max_length
        if capacity > config.max_position_embeddings:
            raise ValueError("cache max_length exceeds max_position_embeddings")
        return cls(
            num_layers=config.num_layers,
            batch_size=batch_size,
            num_kv_heads=config.num_key_value_heads,
            max_length=capacity,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )

    @property
    def num_layers(self) -> int:
        return self._keys.size(0)

    @property
    def batch_size(self) -> int:
        return self._keys.size(1)

    @property
    def max_length(self) -> int:
        return self._keys.size(3)

    @property
    def length(self) -> int:
        first = self._lengths[0]
        if any(length != first for length in self._lengths[1:]):
            raise RuntimeError("KV cache layers have inconsistent lengths")
        return first

    def layer_length(self, layer_idx: int) -> int:
        self._validate_layer(layer_idx)
        return self._lengths[layer_idx]

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return this layer's complete valid prefix."""
        self._validate_layer(layer_idx)
        expected = (self.batch_size, self._keys.size(2), self._keys.size(4))
        actual = (key.size(0), key.size(1), key.size(3)) if key.ndim == 4 else None
        if actual != expected:
            raise ValueError(
                "key must have shape "
                f"[{self.batch_size}, {self._keys.size(2)}, sequence, {self._keys.size(4)}]"
            )
        if value.shape != key.shape:
            raise ValueError("key and value must have the same shape")
        if key.device != self._keys.device or value.device != self._values.device:
            raise ValueError("key/value device must match the cache device")
        if key.dtype != self._keys.dtype or value.dtype != self._values.dtype:
            raise ValueError("key/value dtype must match the cache dtype")

        start = self._lengths[layer_idx]
        end = start + key.size(2)
        if end > self.max_length:
            raise ValueError(f"KV cache capacity exceeded: {end} > {self.max_length}")

        with torch.no_grad():
            self._keys[layer_idx, :, :, start:end].copy_(key)
            self._values[layer_idx, :, :, start:end].copy_(value)
        self._lengths[layer_idx] = end
        return (
            self._keys[layer_idx, :, :, :end],
            self._values[layer_idx, :, :, :end],
        )

    def reset(self) -> None:
        self._lengths = [0] * self.num_layers

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")
