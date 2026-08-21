"""Preallocated or projection-dtype-lazy key/value caches for autoregressive inference.

Beginner's map of this file
---------------------------
Generating a 500-token answer naively means re-running the whole conversation
through every layer for each new word. But notice: an old token's Key and Value
never change. They depend only on that token and the ones before it, which are
already fixed. So compute them once and write them down.

That notebook is the KV cache, and it is the difference between "re-read the
whole book for every word" and "glance at your notes". Only K and V are stored --
the Query is always for the brand-new token.

Two storage policies live here:

* **Linear** (``ring=False``) -- one slot per position, filled left to right.
  Used by global/full-attention layers, which genuinely need all of history.
  Memory grows with every token generated.
* **Ring** (``ring=True``) -- a fixed number of slots, where writing past the end
  wraps around and overwrites the oldest entry. Used by Modern's local layers: if
  a layer can only look back ``local_window`` tokens, storing more is pointless.
  Memory for those layers stops growing entirely, however long the chat runs.

The price of a cache is memory, and it is often the real limit on how many users
a served model can handle at once -- which is exactly what GQA and the hybrid
schedule are attacking.

A cache is inference-only. Training needs gradients through every position, so
``MiniFrontier.forward`` refuses to accept a cache and ``labels`` together.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minifrontier.config import ModelConfig


@dataclass(slots=True)
class LayerKVCache:
    """The stored keys and values for ONE layer.

    ``keys``/``values`` are both ``[batch, n_kv_heads, capacity, head_dim]``, or
    ``None`` until the first append decides their dtype. ``length`` counts tokens
    seen so far, which for a ring cache can exceed ``capacity``.
    """

    keys: torch.Tensor | None
    values: torch.Tensor | None
    _batch_size: int
    n_kv_heads: int
    _capacity: int
    head_dim: int
    requested_device: torch.device
    ring: bool = False
    # Tokens appended so far, in total. For a ring cache this keeps counting past
    # `capacity`; `visible_start` then marks the oldest position still stored.
    length: int = 0
    visible_start: int = 0
    # A forward pass can fail after a ring append has already overwritten history,
    # and the overwritten data is gone for good. These four fields keep just
    # enough of the previous contents for `truncate` to undo the last append.
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
        """Reserve storage for one layer, optionally deferring the dtype choice.

        Passing ``dtype=None`` leaves the tensors unallocated until the first
        append, so the cache adopts whatever dtype the projections actually produce
        (BF16 under autocast, FP32 otherwise) instead of guessing and then failing
        a dtype check mid-generation.
        """

        if min(batch_size, n_kv_heads, capacity, head_dim) <= 0:
            raise ValueError("all cache dimensions must be positive")
        # A bare `torch.device("cuda")` has `index=None`, but every real CUDA tensor
        # reports an explicit index (`cuda:0`) once allocated -- so the two compare
        # UNEQUAL despite naming the same physical device. Resolve the index now,
        # while the cache is still unallocated, so `validate()` later compares two
        # fully-qualified devices instead of failing on this mismatch.
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
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
        """Store this call's K/V and return everything the layer may now attend to.

        The returned tensors are the *whole visible history*, not just what was
        passed in -- which is why ``attention.py`` overwrites its local ``key`` and
        ``value`` with the result. Appends must be contiguous: this is a stream,
        not random access, and ``start_pos`` is checked against ``length`` to catch
        a desynchronized caller immediately.
        """

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
            # Linear path: drop the new K/V into slots [start_pos, end_pos) and hand
            # back a VIEW of everything written so far. No copy, no concatenation --
            # this is why preallocating the full capacity up front pays off.
            # `.detach()` because a cache must never keep a gradient graph alive.
            self.keys[:, :, start_pos:end_pos].copy_(keys.detach())
            self.values[:, :, start_pos:end_pos].copy_(values.detach())
            self.visible_start = 0
            self.length = end_pos
            return self.keys[:, :, :end_pos], self.values[:, :, :end_pos]

        # Ring path. Slot order no longer matches time order once the buffer has
        # wrapped, so read the surviving history out chronologically BEFORE
        # overwriting anything.
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
        # If this single chunk is longer than the whole ring, only its tail can
        # survive -- the rest would be overwritten by its own successors anyway.
        stored_keys = keys[:, :, -self.capacity :]
        stored_values = values[:, :, -self.capacity :]
        stored_start = end_pos - stored_keys.shape[2]
        # Map absolute positions onto ring slots; `% capacity` is the wrap-around.
        # index_copy_ then writes them all in one shot, wrap included.
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
        # Hand attention a temporary chronological view: surviving history followed
        # by the new tokens. Unlike the linear path this is a real concatenation,
        # because the stored slots are rotated relative to time.
        if prior_keys.shape[2] == 0:
            return keys, values
        return torch.cat((prior_keys, keys), dim=2), torch.cat((prior_values, values), dim=2)

    def _chronological(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a ring buffer back in time order.

        Slot ``i`` holds positions ``i``, ``i + capacity``, ``i + 2*capacity``, and
        so on, so once the ring has wrapped the oldest surviving token is no longer
        in slot 0. This works out where "oldest" currently lives and, when the live
        span straddles the end of the tensor, stitches the two pieces back together.
        """

        if self.keys is None or self.values is None or self.length == 0:
            if self.keys is None or self.values is None:
                raise RuntimeError("ring cache must be allocated before it can be read")
            return self.keys[:, :, :0], self.values[:, :, :0]
        stored = min(self.length, self.capacity)
        # Slot holding the oldest token that is still inside the window.
        first = (self.length - stored) % self.capacity
        # Contiguous case: the live span does not run off the end of the tensor.
        if first + stored <= self.capacity:
            return (
                self.keys[:, :, first : first + stored],
                self.values[:, :, first : first + stored],
            )
        # Wrapped case: the live span is in two pieces, tail first then head.
        split = self.capacity - first
        return (
            torch.cat((self.keys[:, :, first:], self.keys[:, :, : stored - split]), dim=2),
            torch.cat((self.values[:, :, first:], self.values[:, :, : stored - split]), dim=2),
        )

    def commit(self) -> None:
        """Accept the last append as permanent history and free the undo copies."""

        self._rollback_indices = None
        self._rollback_keys = None
        self._rollback_values = None
        self._rollback_length = None

    def reset(self) -> None:
        """Forget everything -- start a new conversation reusing the same storage."""

        self.length = 0
        self.visible_start = 0
        self.commit()

    def truncate(self, length: int) -> None:
        """Rewind to ``length`` tokens, undoing the last append if it is still undoable.

        A linear cache can always rewind, because old slots are never overwritten.
        A ring cache can only undo the append it still holds rollback data for:
        once a write is committed and has wrapped, the overwritten tokens are
        genuinely gone, so asking for them back is an error rather than a silent
        wrong answer.
        """

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
        """Memory actually reserved -- what the allocator is holding for this layer."""

        if self.keys is None or self.values is None:
            return 0
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()

    def logical_bytes(self) -> int:
        """Memory currently carrying real tokens.

        This is the number that grows as a conversation gets longer -- and for a
        ring cache, the number that stops growing once the window is full. Quoting
        both figures is more honest than either alone.
        """

        if self.keys is None:
            return 0
        stored_length = min(self.length, self.capacity) if self.ring else self.length
        elements = 2 * self.batch_size * self.n_kv_heads * stored_length * self.head_dim
        return elements * self.keys.element_size()


class KVCache:
    """A consistent collection of layer caches for one model invocation stream.

    One ``LayerKVCache`` per layer, kept in lockstep: they must all hold the same
    number of tokens or generation has silently desynchronized. Reading ``.length``
    is where that invariant gets checked.
    """

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
        """Allocate one cache per layer for a single generation stream.

        ``bounded_local=True`` is the memory win of the Modern preset: local layers
        get a ring buffer capped at ``local_window`` instead of a linear cache the
        length of the whole context. Global layers always keep full history.
        """

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
