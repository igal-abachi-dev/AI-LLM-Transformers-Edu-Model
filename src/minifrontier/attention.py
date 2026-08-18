"""Readable and optimized causal MHA/GQA for Edu and Modern presets."""

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
        query_position = query_index + query_start
        key_position = key_index + key_start
        allowed = key_position <= query_position
        if window_size is not None:
            allowed &= key_position >= query_position - window_size + 1
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
    """Repeat compact K/V only for the explicit teaching implementation."""

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
    """Readable attention reference over ``[batch, heads, sequence, head_dim]``."""

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
    if not mask.any(dim=-1).all():
        raise ValueError("every query must be allowed to attend at least one key")

    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probabilities = F.softmax(scores, dim=-1)
    probabilities = F.dropout(probabilities, p=dropout_p, training=training)
    output = torch.matmul(probabilities, value.float())
    return output.to(dtype=query.dtype)


class CausalSelfAttention(nn.Module):
    """Full or sliding causal MHA/GQA with explicit and first-party optimized paths."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.head_dim = config.head_dim
        self.is_local = config.is_local_layer(layer_index)
        self.position_encoding = config.position_encoding_for_layer(layer_index)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None

    def resolved_implementation(
        self,
        override: AttentionImplementation | None = None,
        *,
        cache: LayerKVCache | None = None,
        sequence_length: int | None = None,
    ) -> AttentionImplementation:
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
        query = self.q_proj(inputs).view(batch, sequence, self.config.n_heads, self.head_dim)
        key = self.k_proj(inputs).view(batch, sequence, self.config.n_kv_heads, self.head_dim)
        value = self.v_proj(inputs).view(batch, sequence, self.config.n_kv_heads, self.head_dim)
        query = query.transpose(1, 2)  # [B, Hq, S, D]
        key = key.transpose(1, 2)  # [B, Hkv, S, D]
        value = value.transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.position_encoding == "rope":
            query = apply_rotary(query, cosine, sine)
            key = apply_rotary(key, cosine, sine)
        key_start = 0
        if cache is not None:
            key, value = cache.append(key, value, start_pos=start_pos)
            if cache.ring:
                key_start = max(0, start_pos - cache.capacity + 1)

        dropout_p = self.config.dropout if self.training else 0.0
        selected = self.resolved_implementation(
            implementation,
            cache=cache,
            sequence_length=sequence,
        )
        if selected == "manual":
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
            if self.is_local:
                if cache is not None and cache.ring and sequence == 1:
                    sdpa_mask = None
                    is_causal = False
                elif attention_mask is None:
                    raise ValueError("local SDPA requires an explicit shared window mask")
                else:
                    sdpa_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                    is_causal = False
            elif start_pos == 0:
                sdpa_mask = None
                is_causal = True
            elif sequence == 1:
                sdpa_mask = None
                is_causal = False
            else:
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
                enable_gqa=self.config.n_heads != self.config.n_kv_heads,
            )
        elif selected == "flex":
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

        merged = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.out_proj(merged)
