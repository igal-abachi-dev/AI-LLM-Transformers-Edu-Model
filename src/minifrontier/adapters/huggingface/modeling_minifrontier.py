"""Standalone Transformers model matching the native MiniFrontier tensor graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Cache, DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_minifrontier import MiniFrontierConfig


class MiniFrontierRMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(inputs.dtype)


def rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class MiniFrontierRoPE(nn.Module):
    def __init__(self, config: MiniFrontierConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(
        self, position_ids: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse_frequency = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device)
                / self.head_dim
            )
        )
        frequencies = position_ids.float().unsqueeze(-1) * inverse_frequency
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary(inputs: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    cosine = cosine.unsqueeze(1).to(inputs.dtype)
    sine = sine.unsqueeze(1).to(inputs.dtype)
    return inputs * cosine + rotate_half(inputs) * sine


def eager_attention_forward(
    module: MiniFrontierAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    probabilities = F.softmax(scores, dim=-1).to(query.dtype)
    probabilities = F.dropout(probabilities, p=dropout, training=module.training)
    output = torch.matmul(probabilities, value)
    return output.transpose(1, 2).contiguous(), probabilities


def _attention_mask(
    *,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    key_length: int,
    local_window: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions.view(1, 1, 1, key_length) <= cache_position.view(1, 1, -1, 1)
    if local_window is not None:
        allowed &= key_positions.view(1, 1, 1, key_length) >= (
            cache_position.view(1, 1, -1, 1) - local_window + 1
        )
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[-1] != key_length:
            raise ValueError("attention_mask must have shape [batch, key_length]")
        allowed &= attention_mask[:, None, None, :].to(torch.bool)
    additive = torch.zeros((), dtype=dtype, device=device).expand(allowed.shape).clone()
    return additive.masked_fill(~allowed, torch.finfo(dtype).min)


class MiniFrontierAttention(nn.Module):
    is_causal = True

    def __init__(self, config: MiniFrontierConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_index
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.num_key_value_groups = config.n_heads // config.n_kv_heads
        self.is_local = config.is_local_layer(layer_index)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_norm = (
            MiniFrontierRMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        )
        self.k_norm = (
            MiniFrontierRMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.Tensor,
        output_attentions: bool,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, sequence, self.config.n_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch, sequence, self.config.n_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, sequence, self.config.n_kv_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.config.uses_rope(self.layer_idx):
            query = apply_rotary(query, cosine, sine)
            key = apply_rotary(key, cosine, sine)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        if self.config._attn_implementation == "vllm":
            selected_mask = attention_mask
        else:
            selected_mask = _attention_mask(
                attention_mask=attention_mask,
                cache_position=cache_position,
                key_length=key.shape[-2],
                local_window=self.config.local_window if self.is_local else None,
                dtype=query.dtype,
                device=query.device,
            )
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attended, weights = attention_interface(
            self,
            query,
            key,
            value,
            selected_mask,
            scaling=self.scaling,
            dropout=self.config.dropout if self.training else 0.0,
            **kwargs,
        )
        attended = attended.reshape(batch, sequence, -1)
        return self.out_proj(attended), weights if output_attentions else None


class MiniFrontierSwiGLU(nn.Module):
    def __init__(self, config: MiniFrontierConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiniFrontierDecoderLayer(nn.Module):
    def __init__(self, config: MiniFrontierConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = MiniFrontierRMSNorm(config.d_model, config.norm_eps)
        self.attention = MiniFrontierAttention(config, layer_index)
        self.ffn_norm = MiniFrontierRMSNorm(config.d_model, config.norm_eps)
        self.feed_forward = MiniFrontierSwiGLU(config)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        attended, weights = self.attention(self.attention_norm(hidden_states), *args, **kwargs)
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, weights


class MiniFrontierPreTrainedModel(PreTrainedModel):
    config_class = MiniFrontierConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules: ClassVar[list[str]] = ["MiniFrontierDecoderLayer"]
    _supports_sdpa = True
    _supports_attention_backend = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.config.init_std or self.config.d_model**-0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)


class MiniFrontierModel(MiniFrontierPreTrainedModel):
    def __init__(self, config: MiniFrontierConfig) -> None:
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = MiniFrontierRoPE(config)
        self.blocks = nn.ModuleList(
            MiniFrontierDecoderLayer(config, index) for index in range(config.n_layers)
        )
        self.final_norm = MiniFrontierRMSNorm(config.d_model, config.norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        hidden_states = self.token_embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                past_length + hidden_states.shape[1],
                device=hidden_states.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1)
        cosine, sine = self.rope(position_ids, dtype=hidden_states.dtype)
        all_hidden = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        for block in self.blocks:
            if all_hidden is not None:
                all_hidden.append(hidden_states)
            hidden_states, weights = block(
                hidden_states,
                cosine,
                sine,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                output_attentions=output_attentions,
                **kwargs,
            )
            if all_attentions is not None:
                all_attentions.append(weights)
        hidden_states = self.final_norm(hidden_states)
        if all_hidden is not None:
            all_hidden.append(hidden_states)
        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, past_key_values, all_hidden, all_attentions)
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden) if all_hidden is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )


class MiniFrontierForCausalLM(MiniFrontierPreTrainedModel, GenerationMixin):
    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "model.token_embedding.weight"
    }

    def __init__(self, config: MiniFrontierConfig) -> None:
        super().__init__(config)
        self.model = MiniFrontierModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.token_embedding = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if logits_to_keep is not None:
            hidden_states = hidden_states[:, -logits_to_keep:]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            values = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return ((loss,) if loss is not None else ()) + tuple(
                value for value in values if value is not None
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
