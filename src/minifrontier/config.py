"""Small, validated configuration objects for MiniFrontier."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Preset = Literal["edu", "modern"]
AttentionPattern = Literal["full", "hybrid"]
PositionMode = Literal["rope", "none"]
AttentionImplementation = Literal["auto", "manual", "sdpa", "flex"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """All architecture choices needed by the compact neural core."""

    vocab_size: int = 16_384
    max_seq_len: int = 2_048
    n_layers: int = 20
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 12
    d_ff: int = 2_048
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    qk_norm: bool = False
    attention_pattern: AttentionPattern = "full"
    local_window: int = 512
    global_position_encoding: PositionMode = "rope"
    dropout: float = 0.0
    tie_embeddings: bool = True
    attention_impl: AttentionImplementation = "sdpa"
    init_std: float | None = None
    preset: Preset = "edu"

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
            "local_window": self.local_window,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.init_std is not None and self.init_std <= 0:
            raise ValueError("init_std must be positive when provided")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.local_window > self.max_seq_len:
            raise ValueError("local_window cannot exceed max_seq_len")
        if self.preset == "edu":
            if self.n_kv_heads != self.n_heads:
                raise ValueError("Edu requires MHA: n_kv_heads must equal n_heads")
            if self.qk_norm:
                raise ValueError("Edu keeps qk_norm disabled")
            if self.attention_pattern != "full":
                raise ValueError("Edu uses full attention in every layer")
            if self.global_position_encoding != "rope":
                raise ValueError("Edu uses RoPE in every layer")
        if self.preset == "modern" and self.n_kv_heads >= self.n_heads:
            raise ValueError("Modern must use fewer KV heads than query heads")
        if self.global_position_encoding == "none" and self.attention_pattern != "hybrid":
            raise ValueError("global NoPE only applies to hybrid attention")
        if self.preset not in ("edu", "modern"):
            raise ValueError(f"unknown preset: {self.preset}")
        if self.attention_pattern not in ("full", "hybrid"):
            raise ValueError(f"unknown attention_pattern: {self.attention_pattern}")
        if self.global_position_encoding not in ("rope", "none"):
            raise ValueError(f"unknown global_position_encoding: {self.global_position_encoding}")
        if self.attention_impl not in ("auto", "manual", "sdpa", "flex"):
            raise ValueError(f"unknown attention_impl: {self.attention_impl}")
        if (
            self.dropout > 0
            and self.attention_pattern == "hybrid"
            and self.attention_impl in ("auto", "flex")
        ):
            raise ValueError("hybrid FlexAttention requires dropout=0")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def queries_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def resolved_init_std(self) -> float:
        return self.init_std if self.init_std is not None else self.d_model**-0.5

    def is_local_layer(self, layer_index: int) -> bool:
        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer_index {layer_index} is outside [0, {self.n_layers})")
        return self.attention_pattern == "hybrid" and (layer_index + 1) % 4 != 0

    def position_encoding_for_layer(self, layer_index: int) -> PositionMode:
        """Resolve the positional policy without coupling it to an attention backend."""

        if self.is_local_layer(layer_index):
            return "rope"
        return self.global_position_encoding

    def attention_impl_for_layer(
        self,
        layer_index: int,
        override: AttentionImplementation | None = None,
    ) -> AttentionImplementation:
        """Resolve ``auto`` to Flex locally and fused-eligible SDPA globally."""

        selected = override or self.attention_impl
        if selected != "auto":
            return selected
        return "flex" if self.is_local_layer(layer_index) else "sdpa"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_toml(cls, path: str | Path) -> ModelConfig:
        with Path(path).open("rb") as file:
            values = tomllib.load(file)
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid model configuration in {path}: {error}") from error

    @classmethod
    def tiny_edu(
        cls,
        *,
        vocab_size: int = 64,
        max_seq_len: int = 32,
        n_layers: int = 2,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 96,
        attention_impl: AttentionImplementation = "sdpa",
    ) -> ModelConfig:
        """Return a CPU-friendly Edu config that exercises the exact production code."""

        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            d_ff=d_ff,
            local_window=min(16, max_seq_len),
            attention_impl=attention_impl,
            preset="edu",
        )

    @classmethod
    def tiny_modern(
        cls,
        *,
        vocab_size: int = 64,
        max_seq_len: int = 32,
        n_layers: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        d_ff: int = 96,
        local_window: int = 8,
        qk_norm: bool = True,
        global_position_encoding: PositionMode = "rope",
        attention_impl: AttentionImplementation = "auto",
    ) -> ModelConfig:
        """Return a CPU-friendly Modern config using the production implementation."""

        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            qk_norm=qk_norm,
            attention_pattern="hybrid",
            local_window=local_window,
            global_position_encoding=global_position_encoding,
            attention_impl=attention_impl,
            preset="modern",
        )
