"""The complete MiniFrontier Edu decoder-only language model.

Beginner's map of this file
---------------------------
The whole model, bottom to top::

    tokens [B, S]            integer IDs, nothing else
      -> token_embedding     each ID becomes a d_model-number "meaning card"
      -> TransformerBlock    x n_layers: gather from other tokens, then think
      -> final_norm          one last volume adjustment
      -> lm_head             score every token in the vocabulary
    logits [B, S, vocab]

The single most useful mental image is the **residual stream**: picture one
conveyor belt per token carrying its card upward. Each block reads the cards and
*adds* a sticky note; nothing is ever erased. At the top the final card is turned
into a guess about the next token.

``B`` is the batch (how many sequences at once) and ``S`` is the sequence length.

Note that ``forward`` returns scores for *every* position at once, not just the
last one. That is the trick that makes pretraining affordable: a 1,024-token
sequence yields 1,023 predictions from one pass.
"""

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
    """What ``MiniFrontier.forward`` hands back.

    ``logits`` are raw, unnormalized scores over the vocabulary -- one score per
    token per position. They are not probabilities until a softmax is applied.
    ``loss`` is only filled in when ``labels`` were supplied, i.e. during training.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TransformerBlock(nn.Module):
    """Pre-norm attention and SwiGLU residual block.

    The entire Transformer, in two lines (see ``forward`` below)::

        x = x + attention(norm(x))
        x = x + feed_forward(norm(x))

    Read each one as "normalize, do some work, add the result back onto the
    conveyor belt". Three ideas are packed in there:

    * The ``+`` is the **residual connection**. A block never replaces a token's
      vector, it only adds to it. A layer with nothing useful to contribute can
      add roughly zero and everything from below still arrives intact -- which is
      the reason a 20-layer stack trains at all.
    * The norm sits **before** each sublayer ("pre-norm"). The 2017 paper put it
      after; every modern model moved it in front, because post-norm needs careful
      warmup tricks to survive depth and still tends to diverge.
    * Attention and the feed-forward do different jobs. Attention **gathers**:
      "which other tokens should I be listening to?". SwiGLU **thinks**: "given
      what I just gathered, what do I conclude?" -- one token at a time, alone.
    """

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
        # Sublayer 1 -- gather across tokens. Note that `inputs` on the right of
        # the `+` is the untouched original: the norm feeds attention only, it
        # never edits the residual stream itself.
        inputs = inputs + self.attention(
            self.attention_norm(inputs),
            cosine,
            sine,
            implementation=attention_impl,
            attention_mask=attention_mask,
            cache=cache,
            start_pos=start_pos,
        )
        # Sublayer 2 -- think about each token on its own, same add-back shape.
        return inputs + self.feed_forward(self.ffn_norm(inputs))


class MiniFrontier(nn.Module):
    """A compact causal LM sharing one readable Edu/Modern implementation.

    The same class builds the 29k-parameter toy and the 150M release. Which one
    you get depends entirely on the ``ModelConfig`` handed in.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # A lookup table with one row per token ID. That row is everything the
        # model knows about the token before any context is considered.
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Position tables live on the model rather than inside each layer: they
        # depend only on position, so all n_layers layers can share one copy.
        self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)
        # `layer_index` is passed down because in the Modern preset a layer's
        # behaviour (local or global, RoPE or NoPE, Flex or SDPA) depends on it.
        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_index) for layer_index in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        # The scoreboard: turns a finished d_model-wide card into one score per
        # possible next token.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Every Linear and Embedding starts as small random noise around zero.
        self.apply(self._initialize)
        # Each layer adds twice onto the residual stream, so across L layers its
        # variance would grow by a factor of ~2L. Shrinking the two projections
        # that actually write into the stream by 1/sqrt(2L) keeps the stream about
        # the same size at the top as at the bottom. This is the GPT-2 trick, and
        # it matters more the deeper the model gets.
        residual_std = config.resolved_init_std / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down_proj.weight, mean=0.0, std=residual_std)
        # Tied embeddings: the input table and the output scoreboard become the
        # exact same tensor -- "ID -> meaning" on the way in, "meaning -> ID" on
        # the way out. This is an assignment, not a copy, so one gradient update
        # moves both. It saves vocab * d_model weights and usually helps small
        # models; it is also why saving needs shared-tensor-aware safetensors.
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _initialize(self, module: nn.Module) -> None:
        """Fill every weight with small Gaussian noise; see ``resolved_init_std``."""

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
        """Run the stack once and return per-position scores over the vocabulary.

        Most arguments are optional machinery; the beginner's version is
        ``model(tokens)`` for scores, or ``model(tokens, labels=tokens)`` for
        scores plus a training loss. The rest:

        * ``labels`` -- targets for the next-token loss. Usually the same tensor
          as ``tokens``; the shift is applied inside ``loss.py``, not here.
        * ``loss_mask`` -- which positions to score. SFT uses this to learn only
          from the assistant's words.
        * ``cache`` -- a ``KVCache`` for generation, so previous tokens do not have
          to be recomputed. Inference only.
        * ``logits_to_keep`` -- only build the scoreboard for the last N positions.
          During generation only the last one is ever used, and the ``lm_head``
          matmul over a 16k vocabulary is expensive enough to be worth skipping.
        * ``activation_checkpointing`` -- trade compute for memory during training
          by recomputing each block's internals in the backward pass.

        Everything below the docstring up to ``start_pos`` is argument validation:
        catching a mistake here produces a sentence, not a stack trace from inside
        a fused kernel.
        """

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

        # Where this call sits in the sequence. Without a cache every call starts
        # at position 0; with one, it continues after whatever is already stored.
        start_pos = cache.length if cache is not None else 0
        if position_start is not None and position_start != start_pos:
            raise ValueError(f"expected position_start {start_pos}, got {position_start}")
        end_pos = start_pos + tokens.shape[1]
        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"positions [{start_pos}, {end_pos}) exceed max_seq_len {self.config.max_seq_len}"
            )

        if cache is not None:
            cache.validate(
                config=self.config,
                batch_size=tokens.shape[0],
                device=tokens.device,
            )
        # IDs become vectors: [B, S] -> [B, S, d_model]. From here to the top of
        # the stack, `hidden` IS the residual stream.
        hidden = self.token_embedding(tokens)
        # Absolute positions for this call, e.g. [700] when decoding token 700.
        # The rotation tables are computed ONCE and reused by every layer.
        positions = torch.arange(start_pos, end_pos, device=tokens.device)
        cosine, sine = self.rope(positions, dtype=hidden.dtype, device=hidden.device)
        # Masks are built lazily and at most once each. Many layers want the same
        # grid, and several kernel paths need no explicit mask at all -- so this
        # avoids allocating a [Sq, Sk] tensor nobody ends up reading.
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
                # Which mask, if any, this layer's chosen kernel actually needs.
                # "manual" always needs one; SDPA only needs one when it cannot
                # describe the shape itself (see the ladder in attention.py).
                needs_full_mask = resolved_impl == "manual" or (
                    resolved_impl == "sdpa"
                    and not block.attention.is_local
                    and cache is not None
                    and tokens.shape[1] > 1
                    and start_pos > 0
                )
                # Decoding one token against a ring cache: everything stored is
                # already inside the window, so no mask is required.
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
                    # A ring cache has dropped the oldest keys, so the key axis of
                    # the mask no longer starts at absolute position 0.
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
                # Local layers get the banded mask, global layers the triangle.
                # In an Edu model `is_local` is False everywhere, so this is always
                # `full_mask` -- and usually `None`, because plain causal SDPA
                # builds its own.
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

                    # Run the block without storing its intermediate activations,
                    # then recompute them during backward. Roughly 30% more compute
                    # for a large memory saving; note it is incompatible with a KV
                    # cache, which is why forward rejects that combination above.
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
            # Out of the last block; one final volume adjustment before scoring.
            normalized = self.final_norm(hidden)
            if logits_to_keep is not None:
                # Generation only ever reads the final position, so skip the
                # scoreboard matmul for the ones that would be discarded.
                normalized = normalized[:, -logits_to_keep:]
            # [B, S, d_model] -> [B, S, vocab_size]: how good is every token as the
            # continuation of each position?
            logits = self.lm_head(normalized)
            loss = None
            if labels is not None:
                loss = next_token_loss(logits, labels, loss_mask=loss_mask)
            if cache is not None:
                # The pass succeeded, so these tokens are permanent history now and
                # the rollback copies can be released.
                cache.commit()
            return ModelOutput(logits=logits, loss=loss)
        except Exception:
            # A half-written cache is worse than no cache: it would silently
            # corrupt every later token. Roll back to where this call began so the
            # caller sees a clean failure instead of quiet nonsense.
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
        """Generate with the M3 preallocated KV-cache implementation.

        A convenience wrapper: the real loop lives in ``generation.py``, and the
        import is deferred to keep this module free of a circular dependency.
        ``temperature=0`` (the default) means greedy, deterministic decoding.
        """

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
        """Count the model's weights -- the number quoted as "a 150M model".

        With tied embeddings the shared table is counted once, because
        ``parameters()`` deduplicates shared tensors.
        """

        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
