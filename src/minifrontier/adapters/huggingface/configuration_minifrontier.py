"""Transformers configuration preserving every frozen MiniFrontier architecture flag."""

from __future__ import annotations

from transformers import PretrainedConfig


class MiniFrontierConfig(PretrainedConfig):
    model_type = "minifrontier"

    def __init__(
        self,
        vocab_size: int = 16_384,
        max_seq_len: int = 2_048,
        n_layers: int = 20,
        d_model: int = 768,
        n_heads: int = 12,
        n_kv_heads: int = 12,
        d_ff: int = 2_048,
        norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        qk_norm: bool = False,
        attention_pattern: str = "full",
        local_window: int = 512,
        global_position_encoding: str = "rope",
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        init_std: float | None = None,
        preset: str = "edu",
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs: object,
    ) -> None:
        tie_embeddings = bool(kwargs.pop("tie_word_embeddings", tie_embeddings))
        kwargs.pop("head_dim", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("intermediate_size", None)
        kwargs.pop("max_position_embeddings", None)
        kwargs.pop("num_attention_heads", None)
        kwargs.pop("num_hidden_layers", None)
        kwargs.pop("num_key_value_heads", None)
        use_cache = bool(kwargs.pop("use_cache", True))
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.d_model = d_model
        self.hidden_size = d_model
        self.n_heads = n_heads
        self.num_attention_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_key_value_heads = n_kv_heads
        self.d_ff = d_ff
        self.intermediate_size = d_ff
        self.norm_eps = norm_eps
        self.rms_norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.attention_pattern = attention_pattern
        self.local_window = local_window
        self.global_position_encoding = global_position_encoding
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.init_std = init_std
        self.preset = preset
        self.head_dim = d_model // n_heads
        self.use_cache = use_cache
        self.sliding_window = local_window
        self.layer_types = [
            "sliding_attention" if self.is_local_layer(index) else "full_attention"
            for index in range(n_layers)
        ]
        self._validate_minifrontier()

    def _validate_minifrontier(self) -> None:
        positive = (
            self.vocab_size,
            self.max_seq_len,
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.d_ff,
            self.local_window,
        )
        if min(positive) <= 0:
            raise ValueError("MiniFrontier dimensions must be positive")
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("invalid MiniFrontier query/KV head divisibility")
        if self.head_dim % 2:
            raise ValueError("MiniFrontier RoPE head_dim must be even")
        if self.local_window > self.max_seq_len:
            raise ValueError("local_window cannot exceed max_seq_len")
        if self.preset not in ("edu", "modern"):
            raise ValueError("preset must be edu or modern")
        if self.attention_pattern not in ("full", "hybrid"):
            raise ValueError("attention_pattern must be full or hybrid")
        if self.global_position_encoding not in ("rope", "none"):
            raise ValueError("global_position_encoding must be rope or none")
        if self.preset == "edu" and (
            self.n_heads != self.n_kv_heads
            or self.qk_norm
            or self.attention_pattern != "full"
            or self.global_position_encoding != "rope"
        ):
            raise ValueError("Edu configuration violates the frozen architecture")
        if self.preset == "modern" and self.n_kv_heads >= self.n_heads:
            raise ValueError("Modern requires fewer KV heads than query heads")

    def is_local_layer(self, layer_index: int) -> bool:
        return self.attention_pattern == "hybrid" and (layer_index + 1) % 4 != 0

    def uses_rope(self, layer_index: int) -> bool:
        return self.is_local_layer(layer_index) or self.global_position_encoding == "rope"
