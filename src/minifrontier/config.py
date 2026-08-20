"""Small, validated configuration objects for MiniFrontier.

Beginner's map of this file
---------------------------
Every architectural decision in the whole model lives in one frozen dataclass.
The 29k-parameter toy and the 150M release run *identical code*; they differ only
in the numbers below. That is the central design idea of this repository, and it
is why this is the first file to read.

The two presets are:

* **Edu** -- the classic modern baseline: pre-RMSNorm, RoPE, full causal
  multi-head attention in every layer, SwiGLU, tied embeddings.
* **Modern** -- Edu plus the three upgrades the field actually adopted between
  2019 and 2025: GQA (fewer key/value heads than query heads), QK-Norm, and a
  3-local-then-1-global attention schedule. Plus one live experiment, NoPE.

``__post_init__`` is deliberately strict. A wrong combination here would either
crash twenty layers later with a confusing shape error, or -- much worse -- train
quietly into something that is not the preset you asked for.
"""

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

    # How many distinct tokens exist. One 16,384-entry byte-level BPE vocabulary
    # is shared by every model size in this project.
    vocab_size: int = 16_384
    # The longest sequence the model may ever see, in tokens. Fixes the RoPE table
    # size and the maximum KV-cache length.
    max_seq_len: int = 2_048
    # How many TransformerBlocks are stacked. Depth is "how many rounds of
    # thinking" the model gets before it has to commit to a guess.
    n_layers: int = 20
    # Width of the residual stream: how many numbers describe one token.
    d_model: int = 768
    # Attention query heads -- independent "listeners", each with its own job.
    n_heads: int = 12
    # Attention key/value heads. Equal to n_heads means MHA; fewer means GQA,
    # where several query heads share one set of keys/values to shrink the cache.
    n_kv_heads: int = 12
    # Inner width of the SwiGLU feed-forward, i.e. how big the "thinking room" is.
    d_ff: int = 2_048
    # Numerical floor inside RMSNorm so an all-zero token cannot divide by zero.
    norm_eps: float = 1e-6
    # RoPE base frequency. Larger = slower rotations = longer positional reach.
    rope_theta: float = 10_000.0
    # RMSNorm on Q and K before they are compared. Modern only; see attention.py.
    qk_norm: bool = False
    # "full" = every layer sees all history. "hybrid" = 3 short-sighted layers
    # then 1 far-sighted one, repeating.
    attention_pattern: AttentionPattern = "full"
    # How far back a local (short-sighted) layer may look, in tokens.
    local_window: int = 512
    # Position policy on GLOBAL layers only. "none" is the NoPE experiment.
    global_position_encoding: PositionMode = "rope"
    # Off by default: these models are limited by how much data they see, not by
    # overfitting, so dropout would only slow learning down.
    dropout: float = 0.0
    # Reuse the embedding table as the output scoreboard. Saves vocab * d_model
    # parameters and generally helps small models.
    tie_embeddings: bool = True
    # Which attention kernel to run. See attention_impl_for_layer below.
    attention_impl: AttentionImplementation = "sdpa"
    # Spread of the initial random weights. None means 1/sqrt(d_model).
    init_std: float | None = None
    # Which frozen recipe this config must obey. Enforced in __post_init__.
    preset: Preset = "edu"

    def __post_init__(self) -> None:
        # Fail here, loudly, with a message that names the offending field --
        # rather than at some tensor operation twenty layers deep.
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
        # The heads split d_model evenly between them; head_dim is the quotient.
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        # Each KV head must serve the same number of query heads, so the GQA
        # grouping is uniform: heads 0-1 share KV group 0, heads 2-3 group 1, ...
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        # RoPE pairs features up into 2-D arrows, so it needs an even count.
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
        # The preset guards below are what make "Edu" and "Modern" mean something
        # specific instead of being labels anyone can attach to any config.
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
        # NoPE means "this layer gets no position stamp at all". That is only a
        # coherent idea when local layers underneath have already baked ordering
        # into the residual stream, so it is illegal outside a hybrid schedule.
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
        # FlexAttention has no dropout hook, so allowing both would hand you a
        # model that quietly trains without the dropout you configured.
        if (
            self.dropout > 0
            and self.attention_pattern == "hybrid"
            and self.attention_impl in ("auto", "flex")
        ):
            raise ValueError("hybrid FlexAttention requires dropout=0")

    @property
    def head_dim(self) -> int:
        """Numbers per head. 768 wide with 12 heads gives each head 64 to work with."""

        return self.d_model // self.n_heads

    @property
    def queries_per_kv(self) -> int:
        """How many query heads share one key/value head. 1 means plain MHA."""

        return self.n_heads // self.n_kv_heads

    @property
    def resolved_init_std(self) -> float:
        """Default initial weight spread of 1/sqrt(d_model).

        Wider layers get smaller starting weights, which keeps the size of a
        layer's output roughly equal to the size of its input no matter how big
        the model is. Without that scaling, activations either vanish or explode
        as soon as you add width or depth.
        """

        return self.init_std if self.init_std is not None else self.d_model**-0.5

    def is_local_layer(self, layer_index: int) -> bool:
        """Return True for a short-sighted (sliding-window) layer.

        The 3:1 schedule in one expression: layers 3, 7, 11, 15, 19 are global and
        every other layer is local. Because the count is made 1-based inside the
        modulo, the last layer of any stack whose depth is a multiple of four is
        always global -- deliberately, so the model's final word can see
        everything. Full-attention (Edu) models answer False for every layer.
        """

        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer_index {layer_index} is outside [0, {self.n_layers})")
        return self.attention_pattern == "hybrid" and (layer_index + 1) % 4 != 0

    def position_encoding_for_layer(self, layer_index: int) -> PositionMode:
        """Resolve the positional policy without coupling it to an attention backend.

        Local layers always use RoPE. Only global layers may opt out of position
        information, and only when ``global_position_encoding="none"``.
        """

        if self.is_local_layer(layer_index):
            return "rope"
        return self.global_position_encoding

    def attention_impl_for_layer(
        self,
        layer_index: int,
        override: AttentionImplementation | None = None,
    ) -> AttentionImplementation:
        """Resolve ``auto`` to Flex locally and fused-eligible SDPA globally.

        Different mask shapes suit different kernels. A local layer's banded mask
        leaves whole tiles of the score grid guaranteed empty, which is exactly
        what FlexAttention exploits; a global layer's plain causal triangle is what
        SDPA's fused fast path is already tuned for. ``"auto"`` decides per layer,
        so one hybrid forward pass runs two different kernels.

        The other values are explicit escapes: ``"manual"`` for the readable
        teaching path used by the tests, and pinning ``"sdpa"``/``"flex"``
        everywhere when comparing implementations against each other.
        """

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
        """Return a CPU-friendly Edu config that exercises the exact production code.

        This is ``tiny_edu`` from ``introduction.md``: 64 tokens of vocabulary, 32
        positions, 2 layers, 28,832 parameters. Small enough to run inside a unit
        test in about a second, and it is the same classes the 150M model uses.
        """

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
        """Return a CPU-friendly Modern config using the production implementation.

        ``tiny_modern``: the same toy scale as ``tiny_edu`` with the Modern
        switches flipped -- 2 KV heads instead of 4 (GQA), QK-Norm on, a hybrid
        3-local/1-global schedule with an 8-token window, and the RoPE-versus-NoPE
        choice exposed. Compare the two in ``labs/02_mha_vs_gqa.py``.
        """

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
