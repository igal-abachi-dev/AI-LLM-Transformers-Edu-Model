"""Preallocated or projection-dtype-lazy key/value caches for autoregressive inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minifrontier.config import ModelConfig


@dataclass(slots=True)
class LayerKVCache:
    keys: torch.Tensor | None
    values: torch.Tensor | None
    _batch_size: int
    n_kv_heads: int
    _capacity: int
    head_dim: int
    requested_device: torch.device
    ring: bool = False
    length: int = 0
    visible_start: int = 0
    _rollback_indices: torch.Tensor | None = None
    _rollback_keys: torch.Tensor | None = None
    _rollback_values: torch.Tensor | None = None
    _rollback_length: int | None = None

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        n_kv_heads: int,
        capacity: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype | None,
        ring: bool = False,
    ) -> LayerKVCache:
        if min(batch_size, n_kv_heads, capacity, head_dim) <= 0:
            raise ValueError("all cache dimensions must be positive")
        shape = (batch_size, n_kv_heads, capacity, head_dim)
        keys = torch.zeros(shape, device=device, dtype=dtype) if dtype is not None else None
        values = torch.zeros(shape, device=device, dtype=dtype) if dtype is not None else None
        return cls(keys, values, batch_size, n_kv_heads, capacity, head_dim, device, ring)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dtype(self) -> torch.dtype | None:
        return self.keys.dtype if self.keys is not None else None

    @property
    def device(self) -> torch.device:
        return self.keys.device if self.keys is not None else self.requested_device

    def _allocate_from(self, update: torch.Tensor) -> None:
        shape = (self.batch_size, self.n_kv_heads, self.capacity, self.head_dim)
        self.keys = update.new_zeros(shape)
        self.values = update.new_zeros(shape)

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape != values.shape:
            raise ValueError("cache key/value shapes must match")
        if keys.ndim != 4:
            raise ValueError("cache updates must be [batch, kv_heads, sequence, head_dim]")
        if start_pos != self.length:
            raise ValueError(f"cache append expected start_pos {self.length}, got {start_pos}")
        if keys.shape[:2] != (self.batch_size, self.n_kv_heads) or keys.shape[3] != self.head_dim:
            raise ValueError("cache update shape is incompatible with allocated cache")
        if keys.device != self.requested_device or values.device != self.requested_device:
            raise ValueError("cache update device does not match allocated cache")
        update_length = keys.shape[2]
        end_pos = start_pos + update_length
        if not self.ring and end_pos > self.capacity:
            raise ValueError(f"cache capacity {self.capacity} exceeded by position {end_pos}")
        if self.keys is None or self.values is None:
            self._allocate_from(keys)
        assert self.keys is not None and self.values is not None
        if keys.dtype != self.keys.dtype or values.dtype != self.values.dtype:
            raise ValueError("cache update dtype does not match allocated cache")
        if not self.ring:
            self.keys[:, :, start_pos:end_pos].copy_(keys.detach())
            self.values[:, :, start_pos:end_pos].copy_(values.detach())
            self.visible_start = 0
            self.length = end_pos
            return self.keys[:, :, :end_pos], self.values[:, :, :end_pos]

        prior_keys, prior_values = self._chronological()
        # The incoming chunk contains the current key for its first query, so a W-token
        # local window needs at most W-1 historical keys in the temporary attention view.
        # Keeping all W old keys makes unmasked single-token SDPA attend W+1 positions.
        prior_limit = self.capacity - 1
        if prior_limit == 0:
            prior_keys = prior_keys[:, :, :0]
            prior_values = prior_values[:, :, :0]
        else:
            prior_keys = prior_keys[:, :, -prior_limit:]
            prior_values = prior_values[:, :, -prior_limit:]
        # Ring writes may overwrite the storage backing these chronological views.
        prior_keys = prior_keys.clone()
        prior_values = prior_values.clone()
        stored_keys = keys[:, :, -self.capacity :]
        stored_values = values[:, :, -self.capacity :]
        stored_start = end_pos - stored_keys.shape[2]
        write_positions = torch.arange(stored_start, end_pos, device=keys.device) % self.capacity
        rollback_indices = torch.unique(write_positions)
        self._rollback_indices = rollback_indices
        self._rollback_keys = self.keys.index_select(2, rollback_indices).clone()
        self._rollback_values = self.values.index_select(2, rollback_indices).clone()
        self._rollback_length = self.length
        self.keys.index_copy_(2, write_positions, stored_keys.detach())
        self.values.index_copy_(2, write_positions, stored_values.detach())
        self.length = end_pos
        self.visible_start = max(0, end_pos - self.capacity)
        if prior_keys.shape[2] == 0:
            return keys, values
        return torch.cat((prior_keys, keys), dim=2), torch.cat((prior_values, values), dim=2)

    def _chronological(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None or self.values is None or self.length == 0:
            if self.keys is None or self.values is None:
                raise RuntimeError("ring cache must be allocated before it can be read")
            return self.keys[:, :, :0], self.values[:, :, :0]
        stored = min(self.length, self.capacity)
        first = (self.length - stored) % self.capacity
        if first + stored <= self.capacity:
            return (
                self.keys[:, :, first : first + stored],
                self.values[:, :, first : first + stored],
            )
        split = self.capacity - first
        return (
            torch.cat((self.keys[:, :, first:], self.keys[:, :, : stored - split]), dim=2),
            torch.cat((self.values[:, :, first:], self.values[:, :, : stored - split]), dim=2),
        )

    def commit(self) -> None:
        self._rollback_indices = None
        self._rollback_keys = None
        self._rollback_values = None
        self._rollback_length = None

    def reset(self) -> None:
        self.length = 0
        self.visible_start = 0
        self.commit()

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError("invalid cache truncation length")
        if self.ring and self._rollback_length == length:
            assert self.keys is not None and self.values is not None
            assert self._rollback_indices is not None
            assert self._rollback_keys is not None and self._rollback_values is not None
            self.keys.index_copy_(2, self._rollback_indices, self._rollback_keys)
            self.values.index_copy_(2, self._rollback_indices, self._rollback_values)
        elif self.ring and length != self.length and self.length > self.capacity:
            raise ValueError("cannot truncate committed history after ring-cache wrap")
        self.length = length
        self.visible_start = max(0, length - self.capacity) if self.ring else 0
        self.commit()

    def allocated_bytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()

    def logical_bytes(self) -> int:
        if self.keys is None:
            return 0
        stored_length = min(self.length, self.capacity) if self.ring else self.length
        elements = 2 * self.batch_size * self.n_kv_heads * stored_length * self.head_dim
        return elements * self.keys.element_size()


class KVCache:
    """A consistent collection of layer caches for one model invocation stream."""

    def __init__(self, layers: list[LayerKVCache], config: ModelConfig) -> None:
        if len(layers) != config.n_layers:
            raise ValueError("cache layer count does not match model configuration")
        self.layers = layers
        self.config = config

    @classmethod
    def allocate(
        cls,
        config: ModelConfig,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        capacity: int | None = None,
        bounded_local: bool = False,
    ) -> KVCache:
        resolved_capacity = capacity or config.max_seq_len
        if resolved_capacity > config.max_seq_len:
            raise ValueError("cache capacity cannot exceed model max_seq_len")
        torch_device = torch.device(device)
        layers = [
            LayerKVCache.allocate(
                batch_size=batch_size,
                n_kv_heads=config.n_kv_heads,
                capacity=(
                    min(config.local_window, resolved_capacity)
                    if bounded_local and config.is_local_layer(layer_index)
                    else resolved_capacity
                ),
                head_dim=config.head_dim,
                device=torch_device,
                dtype=dtype,
                ring=bounded_local and config.is_local_layer(layer_index),
            )
            for layer_index in range(config.n_layers)
        ]
        return cls(layers, config)

    @property
    def length(self) -> int:
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise RuntimeError(f"cache layers have inconsistent lengths: {sorted(lengths)}")
        return next(iter(lengths))

    @property
    def capacity(self) -> int:
        return max(layer.capacity for layer in self.layers)

    def validate(self, *, batch_size: int, device: torch.device) -> None:
        for layer in self.layers:
            if layer.batch_size != batch_size:
                raise ValueError("cache batch size does not match tokens")
            if layer.device != device:
                raise ValueError("cache device does not match model inputs")

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def truncate(self, length: int) -> None:
        for layer in self.layers:
            layer.truncate(length)

    def commit(self) -> None:
        for layer in self.layers:
            layer.commit()

    def allocated_bytes(self) -> int:
        return sum(layer.allocated_bytes() for layer in self.layers)

    def logical_bytes(self) -> int:
        return sum(layer.logical_bytes() for layer in self.layers)
