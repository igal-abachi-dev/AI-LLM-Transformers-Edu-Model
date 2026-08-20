"""Readable and optimized causal MHA/GQA for Edu and Modern presets.

Beginner's map of this file
---------------------------
Attention is a weighted average, and nothing more exotic than that. Each token
produces three vectors from its own residual-stream card:

* **Query** -- "here is what I am looking for"
* **Key**   -- "here is what I am"
* **Value** -- "here is what you get if you pick me"

Compare one token's Query against every allowed token's Key (a dot product),
turn those scores into percentages with softmax, and take that weighted blend of
the Values. The blend is what attention adds back onto the residual stream.

The same maths is written here more than once, on purpose:

* ``manual_scaled_dot_product_attention`` -- the teaching version. Slow, builds
  the entire score matrix, spells out every step in FP32. Read this one first.
* ``"sdpa"`` -- ``F.scaled_dot_product_attention``, PyTorch's fused kernel. Same
  answer, far faster, and it never materializes the score matrix in one piece.
* ``"flex"`` -- FlexAttention, used for local layers because it can skip whole
  tiles of the score grid that the sliding window guarantees are masked out.
* ``"auto"`` -- Flex for local layers, SDPA for global ones.

The tests assert these agree within a documented tolerance. Learn from the manual
one; run the fast ones.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from minifrontier.cache import LayerKVCache
from minifrontier.config import AttentionImplementation, ModelConfig
from minifrontier.layers import RMSNorm
from minifrontier.rope import apply_rotary

# Building a FlexAttention BlockMask costs real time, and it depends only on
# shapes -- never on the data flowing through. So identical (length, window,
# device) requests reuse one object across layers, decode steps, and calls.
_BLOCK_MASK_CACHE: dict[tuple[Any, ...], Any] = {}


def clear_block_mask_cache() -> None:
    """Clear cached FlexAttention masks; primarily useful for bounded tests."""

    _BLOCK_MASK_CACHE.clear()


def block_mask_cache_size() -> int:
    return len(_BLOCK_MASK_CACHE)


def _flex_block_mask(
    *,
    query_length: int,
    key_length: int,
    query_start: int,
    key_start: int,
    window_size: int | None,
    device: torch.device,
) -> Any:
    """Build, or reuse, the banded/causal BlockMask that FlexAttention needs.

    FlexAttention does not take a boolean grid. It takes ``mask_mod``, a small
    function answering "is this (query, key) pair allowed?", compiles it, and uses
    it to work out which *tiles* of the score matrix are entirely masked so it can
    skip them completely. That is where the speedup on local layers comes from.

    The two predicates below are deliberately identical to the ones in
    ``masking.build_attention_mask``; the tests compare the results.
    """

    key = (
        device.type,
        device.index,
        query_length,
        key_length,
        query_start,
        key_start,
        window_size,
    )
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    def mask_mod(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        # Same two rules as masking.py, expressed one pair at a time. Indexes are
        # relative to this call, so shift them into absolute token positions first.
        query_position = query_index + query_start
        key_position = key_index + key_start
        allowed = key_position <= query_position  # causal: no peeking ahead
        if window_size is not None:
            allowed &= key_position >= query_position - window_size + 1  # window
        return allowed

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
    )
    _BLOCK_MASK_CACHE[key] = block_mask
    return block_mask


def _expand_kv_for_reference(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Repeat compact K/V only for the explicit teaching implementation.

    Under GQA there are fewer K/V heads than query heads, so this physically copies
    each K/V head ``queries_per_kv`` times to make the sharing obvious: query heads
    0 and 1 both receive a copy of KV group 0, and so on.

    The fast paths never do this. They pass ``enable_gqa=True`` and let the kernel
    reuse one K/V head across several query heads with no copy at all -- copying
    here would waste memory in precisely the place GQA exists to save it.
    """

    kv_heads = tensor.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if query_heads == kv_heads:
        return tensor
    return tensor.repeat_interleave(query_heads // kv_heads, dim=1)


def manual_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Readable attention reference over ``[batch, heads, sequence, head_dim]``.

    ``softmax(Q @ K^T / sqrt(head_dim) + mask) @ V``, spelled out step by step and
    computed entirely in FP32. This is the correctness baseline the optimized
    kernels are measured against, so it favours clarity over speed everywhere.

    It is also memory-hungry in a way the fused kernels are not: the ``scores``
    tensor below is ``[batch, heads, queries, keys]``, which at a 2,048-token
    context is millions of numbers per head. That is exactly the quadratic wall
    that SDPA and FlexAttention were built to get around.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be four-dimensional")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key batch/head dimensions are incompatible")
    if query.shape[1] != key.shape[1]:
        raise ValueError("manual attention requires equal expanded query and KV head counts")
    if mask.shape != (query.shape[-2], key.shape[-2]):
        raise ValueError("mask must have shape [query_sequence, key_sequence]")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    # A query with no visible keys would softmax a row of all -inf into NaN, and
    # the NaN would then spread silently. Catch it here, where the cause is clear.
    if not mask.any(dim=-1).all():
        raise ValueError("every query must be allowed to attend at least one key")

    # Shrink the scores BEFORE softmax. A dot product of d random numbers grows
    # like sqrt(d), so without this the scores get large, softmax saturates into
    # "100% one token, 0% everything else", and the gradients stop flowing.
    scale = 1.0 / math.sqrt(query.shape[-1])
    # [B, H, Sq, D] @ [B, H, D, Sk] -> [B, H, Sq, Sk]: one score per (query, key).
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    # Disallowed pairs become -inf, which softmax turns into exactly zero weight.
    # The mask is [Sq, Sk] and broadcasts over batch and heads.
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    # Each query's row now sums to 1: "60% of that token, 30% of this one, ...".
    probabilities = F.softmax(scores, dim=-1)
    probabilities = F.dropout(probabilities, p=dropout_p, training=training)
    # The weighted blend of value vectors -> [B, H, Sq, D]. This is the answer.
    output = torch.matmul(probabilities, value.float())
    return output.to(dtype=query.dtype)


class CausalSelfAttention(nn.Module):
    """Full or sliding causal MHA/GQA with explicit and first-party optimized paths.

    One of these lives in every ``TransformerBlock``. It receives an already
    normalized ``[batch, sequence, d_model]`` tensor, gathers information across
    tokens, and returns the same shape for the block to add onto the residual
    stream.

    ``layer_index`` is stored because in the Modern preset a layer's behaviour
    depends on where it sits in the stack: local or global, RoPE or NoPE, Flex or
    SDPA. In the Edu preset every layer behaves identically.
    """

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.head_dim = config.head_dim
        self.is_local = config.is_local_layer(layer_index)
        self.position_encoding = config.position_encoding_for_layer(layer_index)
        # The three roles, one matrix each, all reading the same input. Q gets one
        # head per query head; K and V get n_kv_heads. Under MHA those are equal;
        # under GQA k_proj/v_proj are narrower -- and since the KV cache stores
        # exactly what those two produce, that is where the memory saving comes
        # from. bias=False everywhere: RMSNorm in front makes biases pointless.
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        # Mixes the separate heads' answers back into one vector per token.
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        # QK-Norm (Modern only): an RMSNorm over ONE head's head_dim numbers,
        # applied to Q and K before they are compared. It caps how large the dot
        # products can grow, which is what prevents the sudden loss spikes that
        # kill big training runs. Both are None on Edu.
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None

    def resolved_implementation(
        self,
        override: AttentionImplementation | None = None,
        *,
        cache: LayerKVCache | None = None,
        sequence_length: int | None = None,
    ) -> AttentionImplementation:
        """Pick the kernel for this specific call, honouring any per-call override.

        One surprise lives here, and it looks like a bug until you know why: while
        generating a single token, a local layer drops from Flex back to SDPA.
        FlexAttention earns its keep by skipping whole tiles of a large score grid,
        and with exactly one query row there is no grid to skip -- only compile and
        launch overhead left over.
        """

        selected = self.config.attention_impl_for_layer(self.layer_index, override)
        if selected == "flex" and self.is_local and cache is not None and sequence_length == 1:
            return "sdpa"
        return selected

    def forward(
        self,
        inputs: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        *,
        implementation: AttentionImplementation | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: LayerKVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch, sequence, _ = inputs.shape
        # One matmul per role, then split the flat output width into heads.
        # [B, S, d_model] -> [B, S, H, D]
        query = self.q_proj(inputs).view(batch, sequence, self.config.n_heads, self.head_dim)
        key = self.k_proj(inputs).view(batch, sequence, self.config.n_kv_heads, self.head_dim)
        value = self.v_proj(inputs).view(batch, sequence, self.config.n_kv_heads, self.head_dim)
        # Move heads next to batch so each head becomes an independent [S, D]
        # problem that the matmuls below can batch over.
        query = query.transpose(1, 2)  # [B, Hq, S, D]
        key = key.transpose(1, 2)  # [B, Hkv, S, D]
        value = value.transpose(1, 2)
        # Order matters here and is easy to get backwards on a whiteboard: this
        # codebase normalizes FIRST and rotates SECOND. Some papers do the
        # reverse. The tests pin this order; follow the code, not the diagram.
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        # Position stamps go on Q and K only, never on V. A global layer running
        # the NoPE experiment skips this entirely and sees an unordered pile.
        if self.position_encoding == "rope":
            query = apply_rotary(query, cosine, sine)
            key = apply_rotary(key, cosine, sine)
        key_start = 0
        if cache is not None:
            # Returns the whole visible history, not just this chunk: the new K/V
            # stitched onto everything the cache already holds.
            key, value = cache.append(key, value, start_pos=start_pos)
            if cache.ring:
                # A ring cache has already dropped anything older than `capacity`,
                # so column 0 of what came back is no longer absolute position 0.
                # The masks need to know where the key axis actually begins.
                key_start = max(0, start_pos - cache.capacity + 1)

        dropout_p = self.config.dropout if self.training else 0.0
        selected = self.resolved_implementation(
            implementation,
            cache=cache,
            sequence_length=sequence,
        )
        if selected == "manual":
            # Teaching path: expand the compact K/V so the sharing is visible, then
            # run the fully spelled-out reference above.
            if attention_mask is None:
                raise ValueError("manual attention requires an explicit shared mask")
            attended = manual_scaled_dot_product_attention(
                query,
                _expand_kv_for_reference(key, self.config.n_heads),
                _expand_kv_for_reference(value, self.config.n_heads),
                mask=attention_mask,
                dropout_p=dropout_p,
                training=self.training,
            )
        elif selected == "sdpa":
            # SDPA is fastest when it is told the SHAPE of the mask rather than
            # handed one, because a described mask needs no memory and unlocks the
            # fused kernel. This ladder picks the cheapest correct description --
            # every branch computes the same answer, just at a different price.
            if self.is_local:
                if cache is not None and cache.ring and sequence == 1:
                    # Decoding one token against a ring cache that already holds at
                    # most `window` keys: everything stored is both in the past and
                    # inside the window, so no mask is needed at all.
                    sdpa_mask = None
                    is_causal = False
                elif attention_mask is None:
                    raise ValueError("local SDPA requires an explicit shared window mask")
                else:
                    # Banded window mask; add batch and head axes to broadcast:
                    # [Sq, Sk] -> [1, 1, Sq, Sk].
                    sdpa_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                    is_causal = False
            elif start_pos == 0:
                # Ordinary training/prefill pass on a global layer. `is_causal=True`
                # lets the kernel generate the triangle itself and take its fastest
                # path -- this is the branch that runs during training.
                sdpa_mask = None
                is_causal = True
            elif sequence == 1:
                # One new token against the full cached history: every stored key is
                # in its past, so again nothing needs masking.
                sdpa_mask = None
                is_causal = False
            else:
                # Several tokens at once, part way through a cached stream. The
                # triangle is offset by `start_pos`, which `is_causal` (which always
                # assumes the diagonal starts at 0) cannot express -- so we must
                # hand over a real mask here.
                if attention_mask is None:
                    raise ValueError("chunked cached SDPA requires an offset mask")
                sdpa_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                is_causal = False
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                # Under GQA this tells the kernel to broadcast each K/V head across
                # its group of query heads internally: no repeat, no extra memory.
                enable_gqa=self.config.n_heads != self.config.n_kv_heads,
            )
        elif selected == "flex":
            # Local layers: the banded mask leaves whole tiles of the score grid
            # provably empty, and Flex is the path that can skip them outright.
            if dropout_p:
                raise ValueError("FlexAttention path requires dropout=0")
            attended = flex_attention(
                query,
                key,
                value,
                block_mask=_flex_block_mask(
                    query_length=sequence,
                    key_length=key.shape[-2],
                    query_start=start_pos,
                    key_start=key_start,
                    window_size=self.config.local_window if self.is_local else None,
                    device=query.device,
                ),
                enable_gqa=self.config.n_heads != self.config.n_kv_heads,
            )
        else:
            raise ValueError(f"unknown attention implementation: {selected}")

        # Glue the heads' separate answers back into one vector per token.
        # [B, H, S, D] -> [B, S, H, D] -> [B, S, d_model]. `contiguous()` is needed
        # because `view` cannot reinterpret a transposed (non-contiguous) tensor.
        merged = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        # out_proj's output is what gets added onto the residual stream, so its
        # weights are initialized extra small (see MiniFrontier.__init__) to keep
        # the stream from growing as layers stack up.
        return self.out_proj(merged)
