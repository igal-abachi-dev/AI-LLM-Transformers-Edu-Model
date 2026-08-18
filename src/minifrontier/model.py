"""The complete MiniFrontier Edu decoder-only language model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from minifrontier.attention import CausalSelfAttention
from minifrontier.cache import KVCache, LayerKVCache
from minifrontier.config import AttentionImplementation, ModelConfig
from minifrontier.layers import RMSNorm, SwiGLU
from minifrontier.loss import next_token_loss
from minifrontier.masking import build_attention_mask
from minifrontier.rope import RoPE


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TransformerBlock(nn.Module):
    """Pre-norm attention and SwiGLU residual block."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config, layer_index)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)

    def forward(
        self,
        inputs: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        *,
        attention_impl: AttentionImplementation | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: LayerKVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        inputs = inputs + self.attention(
            self.attention_norm(inputs),
            cosine,
            sine,
            implementation=attention_impl,
            attention_mask=attention_mask,
            cache=cache,
            start_pos=start_pos,
        )
        return inputs + self.feed_forward(self.ffn_norm(inputs))


class MiniFrontier(nn.Module):
    """A compact causal LM sharing one readable Edu/Modern implementation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_index) for layer_index in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        residual_std = config.resolved_init_std / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down_proj.weight, mean=0.0, std=residual_std)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.resolved_init_std)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        attention_impl: AttentionImplementation | None = None,
        cache: KVCache | None = None,
        position_start: int | None = None,
        logits_to_keep: int | None = None,
        activation_checkpointing: bool = False,
    ) -> ModelOutput:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must use an integer dtype")
        if tokens.shape[1] == 0 or tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.config.max_seq_len}]")
        if labels is not None and labels.shape != tokens.shape:
            raise ValueError("labels must have the same shape as tokens")
        if cache is not None and self.training:
            raise ValueError("KV cache is inference-only; call model.eval() first")
        if cache is not None and labels is not None:
            raise ValueError("cached inference does not compute training loss")
        if logits_to_keep is not None and logits_to_keep <= 0:
            raise ValueError("logits_to_keep must be positive when provided")
        if logits_to_keep is not None and labels is not None:
            raise ValueError("selective logits cannot be used while computing loss")
        if activation_checkpointing and (not self.training or cache is not None):
            raise ValueError("activation checkpointing requires uncached training mode")

        start_pos = cache.length if cache is not None else 0
        if position_start is not None and position_start != start_pos:
            raise ValueError(f"expected position_start {start_pos}, got {position_start}")
        end_pos = start_pos + tokens.shape[1]
        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"positions [{start_pos}, {end_pos}) exceed max_seq_len {self.config.max_seq_len}"
            )

        hidden = self.token_embedding(tokens)
        if cache is not None:
            cache.validate(batch_size=tokens.shape[0], device=hidden.device)
        positions = torch.arange(start_pos, end_pos, device=tokens.device)
        cosine, sine = self.rope(positions, dtype=hidden.dtype, device=hidden.device)
        full_mask = None
        local_mask = None
        try:
            for layer_index, block in enumerate(self.blocks):
                layer_cache = cache.layers[layer_index] if cache is not None else None
                resolved_impl = block.attention.resolved_implementation(
                    attention_impl,
                    cache=layer_cache,
                    sequence_length=tokens.shape[1],
                )
                needs_full_mask = resolved_impl == "manual" or (
                    resolved_impl == "sdpa"
                    and not block.attention.is_local
                    and cache is not None
                    and tokens.shape[1] > 1
                    and start_pos > 0
                )
                ring_single_decode = (
                    layer_cache is not None and layer_cache.ring and tokens.shape[1] == 1
                )
                needs_local_mask = block.attention.is_local and (
                    resolved_impl == "manual"
                    or (resolved_impl == "sdpa" and not ring_single_decode)
                )
                if needs_full_mask and full_mask is None:
                    full_mask = build_attention_mask(
                        tokens.shape[1],
                        end_pos,
                        query_start=start_pos,
                        device=tokens.device,
                    )
                if needs_local_mask and local_mask is None:
                    key_start = (
                        max(0, start_pos - layer_cache.capacity + 1)
                        if layer_cache is not None and layer_cache.ring
                        else 0
                    )
                    local_mask = build_attention_mask(
                        tokens.shape[1],
                        end_pos - key_start,
                        query_start=start_pos,
                        key_start=key_start,
                        window_size=self.config.local_window,
                        device=tokens.device,
                    )
                selected_mask = local_mask if block.attention.is_local else full_mask
                if activation_checkpointing:

                    def block_forward(
                        value: torch.Tensor,
                        current_block: TransformerBlock = block,
                        current_mask: torch.Tensor | None = selected_mask,
                    ) -> torch.Tensor:
                        return current_block(
                            value,
                            cosine,
                            sine,
                            attention_impl=attention_impl,
                            attention_mask=current_mask,
                            start_pos=start_pos,
                        )

                    hidden = checkpoint(block_forward, hidden, use_reentrant=False)
                else:
                    hidden = block(
                        hidden,
                        cosine,
                        sine,
                        attention_impl=attention_impl,
                        attention_mask=selected_mask,
                        cache=layer_cache,
                        start_pos=start_pos,
                    )
            normalized = self.final_norm(hidden)
            if logits_to_keep is not None:
                normalized = normalized[:, -logits_to_keep:]
            logits = self.lm_head(normalized)
            loss = None
            if labels is not None:
                loss = next_token_loss(logits, labels, loss_mask=loss_mask)
            if cache is not None:
                cache.commit()
            return ModelOutput(logits=logits, loss=loss)
        except Exception:
            if cache is not None:
                cache.truncate(start_pos)
            raise

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
        top_k: int | None = None,
        top_p: float = 1.0,
        validate_logits: bool = False,
    ) -> torch.Tensor:
        """Generate with the M3 preallocated KV-cache implementation."""

        from minifrontier.generation import generate

        return generate(
            self,
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            generator=generator,
            validate_logits=validate_logits,
        )

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
