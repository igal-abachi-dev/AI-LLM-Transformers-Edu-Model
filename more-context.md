original prompt for project (to plan it): 
i want to build llm ai that compete with latests chatgpt/grok/claude/gemini/muse
but i want it simple and easy and small/minimal for great, to understand and explain when teaching other about llms/transformers/attention... and good for general chat use and coding
look at latest oss models:grok-1/2,muse glimmer,kimi k3,deepseek v4 pro,glm 5.3, qwen 3.8,gpt 120b oss , minmax m3,llama4
see methods , algos, architecture and design
what python stack should i use? optimal best frontier? PyTorch ? jax? tensorflow? keras(sequential? functional?) , Pre-trained Model Standard  safetensor/jinja? scikit-learn,XGBoost / LightGBM  .... others...
see their llm models source codes for reference,
plan me how to build it correctly , or top 3 plans/designs/architectures...


----------------------


This is the version to build. Freeze it and open an editor.

**They were right about Muon and I was out of date.** `torch.optim.Muon` is real and first-party: `torch.optim.Muon(params, lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, ns_steps=5, adjust_lr_fn=None)`, with docs stating it only supports 2D params and that you should use AdamW for biases, embeddings, and other non-2D parameters. Scrap any plan to write a production Muon.

One trap in Experiment 6, though: the docs' own example pairs Muon at lr=0.02 with AdamW at 3e-4 — a ~60× difference. If you run the A/B with your 4e-4 from §30 on both arms, Muon will lose badly and you'll teach the wrong conclusion. That's exactly what `adjust_lr_fn` is for: the options are "original", "match_rms_adamw" (Moonshot's, which matches AdamW's RMS so Muon can directly reuse the AdamW learning rate), and "spectral_unclamped" (Bernstein's). Use `match_rms_adamw` for the fair single-LR comparison, then a small LR sweep per arm if you want the honest version.

**Their parameter arithmetic checks out.** I recomputed the 50M config: 16384×512 tied embedding = 8.4M, plus 14 × (4×512² + 3×512×1408) = 45M → ~53M Classic; dropping to 2 KV heads saves ~5.5M → ~48M Modern. 150M: 12.6M + 20 × (4×768² + 3×768×2048) = 154M. Both match their table. Someone did the multiplication.

## Where I'll partly concede — §37

You're right that "load Qwen weights into MiniFrontier" is the wrong *gate*. You need a working model before you can port anything, and forcing checkpoint compatibility would drag your dims and conventions toward someone else's.

But your §36 list has a hole it can't see: those tests verify **self-consistency, not correctness**. "manual ≈ SDPA" passes if both paths share the same wrong RoPE convention — rotating adjacent pairs (x₀,x₁),(x₂,x₃) instead of split halves (x₀,x_{d/2}),(x₁,x_{d/2+1}). Norm is preserved, position 0 is correct, cached and uncached agree, the model trains and converges, and you have taught juniors a rotation scheme that no released model uses. Nothing in §36 catches it.

The cheap fix that sidesteps your whole objection: don't port a checkpoint, compare **primitives**. Import `Qwen3RotaryEmbedding` / `apply_rotary_pos_emb` and `Qwen3RMSNorm` from `transformers`, feed random tensors, assert your outputs match. No architecture compatibility, no tokenizer, no dims constraint — just "does my rotation agree with the convention the field actually uses." Twenty lines in `tests/test_rope.py`, and it belongs before M9, not in a someday lab. That's the only external ground truth in the project.

## The real bug in Experiment 4

Your Modern preset pairs a 512-token window with 2048 context — a 4:1 ratio. Muse Glimmer pairs a 2048 window with 131k context: 39 of 52 layers sliding over 2,048 tokens against a 131,072 ceiling, a 64:1 ratio. Different regime entirely.

At 2048 with a 512 window you will measure **no speedup and no memory saving** — masked SDPA at that length costs the same or slightly more than full attention, and the KV cache difference is a rounding error. Meanwhile you're strictly discarding information in 15 of 20 layers, so validation loss goes *up*. Experiment 4 as configured will produce "hybrid attention is worse," which is the opposite of the lesson.

Two options. Either run Experiment 4 at 8K–16K context (the 50M model can do this cheaply) where the quadratic term and the cache actually bite, or keep 2048 and reframe the experiment honestly as *"here is the cost side of the trade, measured; the benefit only appears at 30×+ window-to-context ratios, which is why Glimmer uses 64:1."* Either is teachable. Reporting a null result as if it settles the question is not. And log KV-cache bytes and decode tokens/sec at long sequence, not just val loss — that's where the effect lives.

## §32 is too timid

300M–1B tokens for the canonical 150M is 2–7 tokens per parameter. Chinchilla-optimal is ~20:1, and modern small models are deliberately trained far past that. At 1B tokens your 150M will look worse than the architecture deserves, and — more importantly — undertrained runs are exactly where QK-Norm and attention-schedule effects are smallest and noisiest, so your A/Bs will be measuring seed variance.

The good news is the number: 150M × 3B tokens = 2.7e18 FLOPs, which at a realistic 40–55 TFLOP/s effective on a 4090 is roughly 15–20 hours. One overnight plus a morning, for a properly trained canonical model. Do 3B tokens minimum for M9 and keep the small budgets for smoke tests.

---

That's it. Muon LR handling, one RoPE parity test, fix the window ratio, raise the token budget. Nothing else in the document needs another round — and round five would be planning as procrastination. `uv init`.



--------------

## First, the uncomfortable part

"Minimal enough to teach from" and "competes with Claude/Gemini on chat and coding" are two different artifacts, and no single codebase is both. Frontier capability today comes from ~10²⁵ FLOP pretraining runs, multi-trillion-token curated corpora, and RL infrastructure with 60+ people behind it — Ai2 needed 224 GPUs for 21 extra days just to finish one RL run on a 32B model. What you *can* own end-to-end is the full pipeline shape, at GPT-2-to-3B scale, in ~2–8k lines of readable PyTorch. Then get real chat/coding ability by post-training someone else's base model. Treat those as two projects that share one repo.

## Where the open field actually is (Aug 2026)

Your list needs a few corrections:

- **Muse Glimmer** — Meta's return to open weights: a dense 30B with a 2B ViT-style Perception Encoder, Apache 2.0, distilled from Muse Spark, built for local agentic work, plus an optional DFlash speculative-decoding drafter that proposes token blocks for the main model to verify in parallel. **This is the single most relevant reference for you** — it's the "small model, big teacher" playbook.
- **Kimi K3** — released July 17, 2026, 2.8T params, billed as the largest open-source model. Not K3-the-small-thing.
- **DeepSeek V4 Pro** — 1.6T MoE, MIT, ~80.6% SWE-bench Verified. **GLM-5.2** (not 5.3) — 744B MoE / 40B active, MIT, 1M context. **MiniMax M3** ~428B, community license. **Qwen 3.8 doesn't exist**; the interesting one is **Qwen3-Coder-Next**: 80B MoE with ~3B active, 70.6% SWE-bench Verified, Apache 2.0, runs in ~46GB unified memory.
- **Grok** — grok-1 is Apache 2.0 but a 314B relic; grok-2's Community License allows research/non-commercial use with conditions and explicitly forbids using it to train other models. Useless as a distillation teacher. Skip both.
- **gpt-oss-120b** — the best *readable* frontier-ish code: OpenAI ships an intentionally inefficient reference PyTorch implementation in `gpt_oss/torch/model.py` using basic operators to show the exact architecture. 36 layers, 116.8B total / 5.1B active, o200k_harmony tokenizer, MoE + SwiGLU + GQA + sliding-window + RoPE/YaRN + attention sinks, MXFP4 on MoE weights.
- **Llama 4** — dead end. Behemoth never shipped; Meta's live line is Muse.

## The 2026 consensus recipe (what to actually copy)

Convergent across all of them: RMSNorm pre-norm, SwiGLU, RoPE, **QK-Norm** (quietly universal — cheap stability win), and either GQA or MLA. Ablations consistently order MLA > GQA > MHA, and MLA's KV-cache compression is the reason long context is affordable. Above ~30B everything is sparse MoE with fine-grained experts + a shared expert. Newer efficiency layers: **DeepSeek Sparse Attention** and **Gated DeltaNet** — the pattern is a 3:1 hybrid, most layers using a recurrent state update with every fourth falling back to full attention.

Optimizer is the one place where you get frontier-grade gains for free. **Muon/NorMuon over AdamW**: in modded-nanogpt the AdamW→Muon/NorMuon transition was the single largest efficiency jump — ~1.5× better sample efficiency, lower memory than Adam, <2% wall-clock overhead. It's now used in Kimi K2 and GLM-4.5, and the speedrun record fell from 45 minutes to ~2.2 minutes over a year of refinements.

## Stack

**PyTorch. Not a close call.** Hugging Face Transformers is PyTorch-first, most open LLMs ship PyTorch checkpoints, and FSDP/DeepSpeed are PyTorch-native. JAX only if you're renting TPUs. TensorFlow/Keras for LLMs in 2026 is a career-limiting choice — no reference implementation you'd want to read is written in it, so drop the Sequential-vs-Functional question entirely.

- **Core**: `torch` 2.x (+`torch.compile`, FlexAttention), Triton for custom kernels, `uv` for deps (what nanochat uses).
- **Scale-out**: FSDP2 for one node; **torchtitan** if you go multi-node (PyTorch-native composable 4D parallelism with Float8 training); **torchao** for FP8/INT4/QAT.
- **Tokenizer**: HF `tokenizers` (Rust BPE) or `tiktoken`. Don't hand-roll past the teaching demo.
- **Packaging** (your safetensors/jinja question): `model.safetensors` + `config.json` + `tokenizer.json` + a separate `chat_template.jinja`. That combo is the de facto standard and is what makes vLLM/SGLang/Ollama load your model without custom code. Export GGUF for llama.cpp/LM Studio.
- **Post-training**: `trl` (SFT/DPO/GRPO), `torchtune` or Axolotl, Unsloth for cheap LoRA.
- **Serving**: vLLM or SGLang for GPUs, llama.cpp for laptops.
- **Eval**: `lm-eval-harness` + your own held-out set. Never trust a vendor number.
- **scikit-learn / XGBoost / LightGBM**: not part of an LLM at all — different problem class (tabular). One real use: training a small quality classifier for data filtering, the FineWeb-Edu trick. Keep them out of the model path.

## Top 3 plans

**Plan A — Fork nanochat (teaching-first).** Trains a GPT-2-grade model in ~3 hours on an 8×H100 node for ~$73, with tokenization → pretraining → mid-training → SFT → RL → eval → chat UI in one dependency-light repo; roughly 8,000 lines of readable PyTorch, no LangChain/Megatron abstraction. You get a model you can talk to and a codebase you can step through line by line in front of juniors. Ceiling: it writes simple code and hallucinates confidently. Cost: <$500 including mistakes. **This is your teaching artifact.**

**Plan B — 1–3B modern dense model.** Same harness, upgraded to the 2026 recipe: MLA or GQA+QK-Norm, SwiGLU, 3:1 Gated DeltaNet hybrid, MTP head for speculative decoding, NorMuon, bf16/FP8, 100–300B tokens. Crucially, add **logit distillation from an open teacher** — Muse Glimmer's own recipe was logit distillation from a much larger teacher, then mid-training on longer-context agent-heavy data with richer reasoning traces. Realistic budget $5k–50k. Output: genuinely useful at narrow tasks, still not a daily coding assistant.

**Plan C — Post-train an open base (usable-first).** Start from **Olmo 3 7B/32B** — data recipes, training pipeline, and checkpoints all open, Apache 2.0 across weights, data, code and checkpoints — or Qwen3-Coder-Next / Muse Glimmer. Continue-pretrain on your domain (C#/.NET idioms, internal patterns, Hebrew), then SFT + GRPO/RLVR with verifiable rewards (tests pass / compiles). This is the only route to something you'd actually use, and it's air-gappable, which matters for your work context.

## What I'd do

A + C, in that order, sharing one repo. Week 1–2: rent an 8×H100 for an afternoon, run the nanochat speedrun, read every file, swap AdamW→Muon yourself and watch the loss curve move — that single experiment teaches more about optimizers than any paper. Week 3–4: reimplement MHA→GQA→MLA as three interchangeable modules against `rasbt/LLMs-from-scratch` (it has MLA in `ch04/05_mla` and Gated DeltaNet in `ch04/08_deltanet`) — that's your attention lecture, with runnable code. Then Plan C for the thing you actually put in front of people.

Drop Plan B unless someone else is paying. It's the expensive middle where you learn nothing new and still lose to a 30B download.

----------


To build a language model that is conceptually on par with modern frontier
architectures (like Llama, DeepSeek, Qwen, and Grok) while remaining simple,
clean, and pedagogical for teaching, you must separate architectural principles
from raw compute scale.

Frontier models do not use complicated legacy Transformer blocks (like classic
Post-LN or standard multi-head attention). Modern architectures have converged
on a streamlined set of mathematical primitives.

1. Architectural Anatomy of Modern Open-Weights Models

Across models like Llama, Qwen, DeepSeek, and Grok, the modern LLM architecture
has standardized around five key components:

Input Tokens
     │
     ▼
[ Embedding Layer ]
     │
     ▼
┌───────────────────────────────┐  ◄─── Repeated N Layers
│  RMSNorm                      │
│     │                         │
│  RoPE / Positional Rotation   │
│     │                         │
│  Grouped Query Attention(GQA) │  (or MLA in DeepSeek)
│     │ (Residual Add)          │
│     ▼                         │
│  RMSNorm                      │
│     │                         │
│  SwiGLU Feed-Forward / MoE    │
│     │ (Residual Add)          │
└───────────────────────────────┘
     │
     ▼
[ Final RMSNorm ] ──► [ Un-embedding (LM Head) ] ──► Logits

1.  RMSNorm (Root Mean Square Normalization): Replaces standard LayerNorm. It
    strips out mean-centering and only scales by variance, cutting compute
    overhead by ~20% without losing stability.
2.  RoPE (Rotary Position Embeddings): Encodes position by rotating query and
    key vectors in complex 2D planes. Enables length extrapolation and relative
    position awareness without learned positional tables.
3.  GQA (Grouped Query Attention): Rather than having separate Key/Value heads
    for every Query head (MHA) or 1 single KV head (MQA), H_Q queries share
    H_{KV} key/value heads (typically 4:1 or 8:1 ratio). This reduces KV-cache
    memory during inference by 75–87%.
4.  SwiGLU (Swish Gated Linear Unit): Replaces standard ReLU/GELU feed-forward
    networks:
    \text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}
5.  Pre-Norm + QK-Norm: Normalization happens before sub-layers (Pre-LN) with
    residual additions, and queries/keys are normalized (QK-Norm) before
    dot-product attention to eliminate training spikes (loss divergence at
    scale).

2. The Frontier Python Stack

| Category                  | Recommended Frontier Standard            | Why / When to Use                                                                                                                                                              |
| :------------------------ | :--------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Framework**        | **PyTorch 2.x** (`torch`)                | **Industry Standard.** 95%+ of OSS model codebases (Llama, DeepSeek, Qwen) are native PyTorch. Use `torch.compile` and `torch.nn.functional.scaled_dot_product_attention`.     |
| **Weight Format**         | **`safetensors`**                        | Replaces unsafe Python `.bin`/`pickle`. Memory-mapped (mmap), zero-copy, extremely fast loading.                                                                               |
| **Prompt/Chat Templates** | **`Jinja2`**                             | Standard used by Hugging Face & ChatML. Allows dynamic formatting of system prompts, user turns, tool calling, and thought blocks.                                             |
| **Tokenization**          | **`tiktoken`** or **`tokenizers` (BPE)** | Fast byte-level Byte-Pair Encoding. Byte-level fallback prevents "out-of-vocabulary" errors for code and foreign languages.                                                    |
| **Efficient Inference**   | **`vLLM` / `SGLang` / `llama.cpp`**      | PagedAttention, continuous batching, and tensor parallelism.                                                                                                                   |
| **Tooling to Avoid**      | *Keras / TensorFlow / Scikit-Learn*      | TensorFlow/Keras are rarely used for frontier LLM pretraining. Scikit-learn/XGBoost are for tabular data or external reward classifiers, not generative autoregressive models. |

3. Complete, Minimal Frontier LLM Implementation

Below is a complete, pedagogical implementation in pure PyTorch containing RoPE,
RMSNorm, GQA, and SwiGLU. It is designed to be easily read, taught, and
executed.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 1024               # Hidden dimension
    n_layers: int = 12            # Number of transformer blocks
    n_heads: int = 16             # Query heads
    n_kv_heads: int = 4           # Key/Value heads (GQA ratio = 16/4 = 4)
    vocab_size: int = 32000       # Vocabulary size
    multiple_of: int = 256        # Ensure SwiGLU hidden dim is multiple of this
    max_seq_len: int = 4096       # Maximum sequence length
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0


# 1. RMSNorm (Faster, minimal alternative to LayerNorm)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt( mean(x^2) + eps )
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


# 2. Rotary Position Embeddings (RoPE)
def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # Complex numbers: cos + i*sin

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # Reshape x to complex numbers: (B, H, S, D) -> (B, H, S, D/2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: x.shape[2], :].unsqueeze(0).unsqueeze(1) # Broadcast (1, 1, S, D/2)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_rotated.type_as(x)


# 3. Grouped Query Attention (GQA) with QK-Norm
class GroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.num_queries_per_kv = args.n_heads // args.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # QK-Norm for stability at high learning rates
        self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm & RoPE
        q = apply_rotary_emb(self.q_norm(q), freqs_cis)
        k = apply_rotary_emb(self.k_norm(k), freqs_cis)

        # Expand Key/Value heads to match Query heads if GQA
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled Dot-Product Attention with Flash Attention kernel
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


# 4. SwiGLU Feed-Forward Network
class SwiGLUFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w_gate = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# 5. Full Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attn = GroupedQueryAttention(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = SwiGLUFeedForward(args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# 6. Complete Generative Model
class FrontierLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Weight tying (optional, standard for small models)
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies
        head_dim = args.dim // args.n_heads
        freqs_cis = precompute_rope_freqs(head_dim, args.max_seq_len, args.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seq_len]

        for layer in self.layers:
            h = layer(h, freqs_cis)

        logits = self.output(self.norm(h))
        return logits

4. Top 3 Architecture & Project Blueprints

Depending on whether your priority is education, efficiency, or rapid
deployment, choose one of the following paths:

  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
  │         PLAN 1          │     │         PLAN 2          │     │         PLAN 3          │
  │     Nano-Frontier       │     │     Sparse MoE + MLA    │     │   Distill / Continual   │
  │     (Dense Base)        │     │  (DeepSeek/Grok Style)  │     │       Fine-Tune         │
  ├─────────────────────────┤     ├─────────────────────────┤     ├─────────────────────────┤
  │ • 100M - 500M params    │     │ • Top-2 Router MoE      │     │ • Base: Qwen/Llama (1-3B)│
  │ • Pure PyTorch GQA+RoPE │     │ • Compressed KV (MLA)   │     │ • Custom SFT + DPO/GRPO │
  │ • Best for Teaching     │     │ • Low compute per token │     │ • Best for Real Chat/Code│
  └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘

Plan 1: "The Nano-Frontier" (Best for Teaching & Clear Demos)

  - Goal: A clear, pure-Python model (~120M to 350M parameters) implementing
    modern dense mechanics (Llama 3/Qwen 2.5 style).
  - Specs: 12 layers, 768 hidden dimension, 12 Query heads, 3 KV heads
    (GQA 4:1), 4K context length.
  - Why it works: Trains comfortably on a single consumer GPU (RTX 3090/4090 or
    Apple Silicon with MPS/MLX) in 12–24 hours on a curated dataset (e.g.,
    FineWeb-Edu + StarCoder-sample).
  - Teaching advantage: Fits in <200 lines of clean code; students can inspect
    every matrix multiplication, RoPE transformation, and attention mask.

Plan 2: "The Sparse MoE + MLA Architecture" (DeepSeek & Grok Style)

  - Goal: High parameter capacity with low active FLOPs per token.
  - Core Mechanisms:
      - Multi-Head Latent Attention (MLA): Compresses Keys and Values into a
        shared low-rank latent vector \mathbf{c}^{KV}, cutting KV-cache size
        during decoding down to ~15% of standard GQA.
      - Top-2 Router MoE: Replace the SwiGLU MLP with N parallel experts
        (e.g., 8 experts) plus 1 shared expert. A learned linear router selects
        the top 2 experts per token.
  - Why it works: You get the parameter capacity of a 2B model while only
    spending the compute of a 500M model per forward pass.

Plan 3: "Frontier Small-Scale Distillation" (Best for Real Chat & Coding Utility)

  - Goal: Achieve genuine coding and conversational assistance immediately
    without a million-dollar pre-training budget.
  - Method:
    1.  Take an open-weights foundation model (e.g., Qwen2.5-1.5B or
        Llama-3.2-1B/3B).
    2.  Implement a custom Jinja2 Chat Template with system, reasoning
        (<thought>...</thought>), and tool blocks.
    3.  Perform Direct Preference Optimization (DPO) or GRPO (Group Relative
        Policy Optimization) on synthetic reasoning/coding trajectories
        generated by larger models.

5. ChatML Formatting with Jinja2 and Safetensors

To standardize chat and code interactions across different frontends (like Web
UIs and IDE plugins), use the Jinja2 ChatML standard and save weights in
safetensors:

Chat Template (chat_template.jinja)

{{ bos_token }}
{% for message in messages %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n' }}
{% endif %}

Saving & Loading with Safetensors

from safetensors.torch import save_model, load_model

# Save model weights safely (memory-mapped, secure against arbitrary code execution)
save_model(model, "frontier_llm.safetensors")

# Load model weights back
load_model(model, "frontier_llm.safetensors")

6. Recommended Execution Roadmap

1.  Step 1 (Architecture Validation): Implement the provided FrontierLLM module
    and write unit tests for shapes, RoPE rotations, and causal masking.
2.  Step 2 (Tokenizer Setup): Train a Byte-Level BPE tokenizer using Hugging
    Face tokenizers on a mix of code (Python/JS/Rust) and Markdown text with
    a 32,000 vocab size.
3.  Step 3 (Pretraining): Train on a subset of FineWeb-Edu (5–10B tokens) using
    AdamW with cosine learning rate decay and warmup, using torch.bfloat16.
4.  Step 4 (Instruction & Reasoning Tuning): Fine-tune on multi-turn instruction
    sets (e.g., UltraChat, Magpie) and code-generation sets (e.g., CodeFeedback)
    formatted with your Jinja template.
5.  Step 5 (Inference & Export): Export weights to safetensors and load them
    into vLLM or compile via torch.compile(model, mode="reduce-overhead") for
    fast serving.



------------

Component,Recommendation,Why
Core framework,PyTorch (2.x+),"Best for teaching, debugging, research, most OSS models, Torch.compile, FlashAttention-2/SDPA, FSDP2, TorchTitan. JAX is great for pure research scale (Grok-1 used it) but steeper for teaching. TensorFlow/Keras is outdated for frontier LLMs."
High-level,Hugging Face Transformers + PEFT + TRL + Accelerate / DeepSpeed,Load/train/fine-tune any open model easily. Safetensors is the standard weight format.
Model definition,Pure nn.Module (or torch.nn.Sequential for tiny demos only),Functional style via modules is clearest for teaching. Avoid pure Sequential for real Transformers.
Tokenization,Hugging Face Tokenizers or SentencePiece / tiktoken,BPE/BBPE.
Training efficiency,"FlashAttention-2, Torch.compile, bfloat16/FP8, gradient checkpointing, FSDP2",Essential even for small models.
Serving / inference,"vLLM, SGLang, llama.cpp / GGUF, Hugging Face TGI",For chat/coding demos.
Data / eval,"datasets, lm-eval-harness, EleutherAI, or simple custom loops",
Optional classical ML,"scikit-learn, XGBoost/LightGBM",Only for non-LLM baselines or feature engineering side projects. Not core.


**You cannot realistically build a frontier competitor (matching latest ChatGPT / Grok / Claude / Gemini / Muse quality) as a solo or small-team project with “simple + minimal” constraints.** Those models require massive compute (tens to hundreds of thousands of H100/H200-class GPUs for months), proprietary data pipelines, advanced RL post-training, and specialized infrastructure. Even the open-weight leaders (DeepSeek-V4-Pro ~1.6T total/49B active MoE, Kimi K3 ~2.8T/104B active, GLM-5.x ~750B/40B active, Qwen3.8 ~2.4T/95B active, Llama 4 Maverick ~400B/17B, etc.) are the result of large labs with enormous resources.

What *is* realistic and excellent for your goals (teaching transformers/attention/LLMs clearly + usable general chat + coding) is a **modern, clean, well-documented decoder-only Transformer** in the 100M–3B (or small MoE) range that you fully understand and can explain line-by-line. You can then fine-tune strong open models or distill from them for better performance.

### Current Open-Source Landscape (2026 snapshot)
Almost every strong open model is now **sparse Mixture-of-Experts (MoE)**:
- DeepSeek-V4 series: hybrid Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA), Manifold-Constrained Hyper-Connections (mHC), DeepSeekMoE (shared + fine-grained routed experts), Multi-Token Prediction (MTP), Muon optimizer, 1M context.
- Kimi K3: hybrid Kimi Delta Attention + Gated MLA, LatentMoE, very high sparsity.
- Qwen3.x / Qwen3.8: GQA + QK-Norm + SwiGLU + RoPE (dense) or fine-grained MoE; newer ones mix Gated DeltaNet / hybrid linear attention.
- Llama 4: GQA + MoE (fewer/larger experts), alternating dense/MoE layers in some variants.
- Grok-1 (open): classic MoE (8 experts, top-2), GQA-style, RoPE, JAX/Haiku implementation. Grok-2 similar but closed.
- GLM-5.x, MiniMax-M3, gpt-oss-120B: all MoE-dominant with efficiency tricks (MLA variants, sparse attention, etc.).
- Muse Glimmer: denser multimodal (Meta’s open side).

Common modern building blocks you should implement or study:
- Pre-norm + RMSNorm
- RoPE (or YaRN for long context)
- GQA / MLA (or hybrid sparse attention)
- SwiGLU (or gated variants)
- MoE with load-balancing (auxiliary-loss-free or Sinkhorn-style)
- Optional: Multi-Token Prediction head, QK-Norm, attention sinks

### Recommended Python Stack (Optimal for Understanding + Practical Use)
**Primary recommendation: PyTorch + Hugging Face ecosystem.**

| Component              | Recommendation                          | Why |
|------------------------|-----------------------------------------|-----|
| Core framework        | **PyTorch** (2.x+)                     | Best for teaching, debugging, research, most OSS models, Torch.compile, FlashAttention-2/SDPA, FSDP2, TorchTitan. JAX is great for pure research scale (Grok-1 used it) but steeper for teaching. TensorFlow/Keras is outdated for frontier LLMs. |
| High-level            | Hugging Face Transformers + PEFT + TRL + Accelerate / DeepSpeed | Load/train/fine-tune any open model easily. Safetensors is the standard weight format. |
| Model definition      | Pure `nn.Module` (or torch.nn.Sequential for tiny demos only) | Functional style via modules is clearest for teaching. Avoid pure Sequential for real Transformers. |
| Tokenization          | Hugging Face Tokenizers or SentencePiece / tiktoken | BPE/BBPE. |
| Training efficiency   | FlashAttention-2, Torch.compile, bfloat16/FP8, gradient checkpointing, FSDP2 | Essential even for small models. |
| Serving / inference   | vLLM, SGLang, llama.cpp / GGUF, Hugging Face TGI | For chat/coding demos. |
| Data / eval           | datasets, lm-eval-harness, EleutherAI, or simple custom loops | |
| Optional classical ML | scikit-learn, XGBoost/LightGBM         | Only for non-LLM baselines or feature engineering side projects. Not core. |

Avoid building everything in pure NumPy (too slow) or starting with Keras Sequential (hides too much).

Great reference codebases for teaching:
- Andrej Karpathy’s nanoGPT / nanochat (minimal, clean GPT-style).
- Sebastian Raschka’s LLMs-from-scratch (modern components + recent attention variants).
- xAI Grok-1 (JAX MoE reference).
- Hugging Face implementations of DeepSeek / Qwen / Llama (study the config + modeling files).
- tiny-LLM / mini-modern-llm style single-file or small repos.

### Top 3 Practical Plans / Architectures

**Plan 1: Pure Educational Minimal Transformer (Recommended starting point)**  
Goal: Fully understand and teach every piece (embeddings → RoPE → multi-head / GQA attention → RMSNorm → SwiGLU → residual → LM head).  
- Size: 50–300M parameters (fits on 1× consumer GPU or even CPU for tiny versions).  
- Architecture: Classic decoder-only GPT-style with modern upgrades (pre-norm RMSNorm, RoPE, GQA or full MHA, SwiGLU, optional tied embeddings). No MoE at first.  
- Context: 1k–4k tokens initially.  
- Stack: Pure PyTorch `nn.Module`. Train on TinyStories / FineWeb-Edu / code subsets with simple next-token prediction.  
- Then: Add instruction tuning (SFT) + light preference optimization (DPO/ORPO via TRL).  
- Teaching value: Highest. You can draw every matrix multiply and attention map.  
- Usability: Decent chat + basic coding after good data + SFT. Not frontier, but you own every line.  
- Path to better performance: Distill / continue-pretrain from a strong 1–3B open model (Qwen3-1.7B/4B, Gemma, etc.).

**Plan 2: Modern Small-to-Mid Dense or Light MoE (Best balance for chat + coding)**  
Goal: Something that feels useful while remaining understandable.  
- Size: 1–3B dense **or** small MoE (e.g. 8–16 experts, top-2, ~1–2B active).  
- Architecture: Take Plan 1 and add GQA + QK-Norm + optional simple MoE (or hybrid linear attention like Gated DeltaNet if you want to study 2026 trends). Support 8k–32k context with YaRN/RoPE scaling.  
- Training: Start from an open checkpoint (Qwen3 / Llama-3.2 / Gemma style) via HF, or train from scratch on high-quality data if you have GPU hours. Use LoRA/QLoRA for cheap adaptation, then full fine-tune if possible. Post-train with SFT on chat + coding data + light RL.  
- Stack: PyTorch + Transformers + PEFT + TRL + vLLM for serving. Safetensors throughout.  
- Teaching value: Still high — you implement or deeply inspect the attention/MoE blocks.  
- Usability: Strong general chat and solid coding assistant (especially after domain SFT). Competitive with mid-2024 open models; can approach recent small models with good data.  
- Scaling path: Grow experts or distill larger MoEs.

**Plan 3: Efficiency-Focused Hybrid / Distilled “Frontier-lite” (Most ambitious practical)**  
Goal: Maximize capability per FLOP while keeping the design teachable.  
- Size: 3–8B dense or modest MoE (e.g. DeepSeek-style fine-grained experts with 1 shared + routed, or Llama-4-style fewer larger experts).  
- Architecture: Incorporate 1–2 modern tricks you can implement cleanly:  
  – MLA or simple sparse/compressed attention for long context.  
  – Light MoE with auxiliary-loss-free balancing.  
  – Optional MTP head.  
  – Strong RoPE + YaRN for 128k+.  
- Training strategy: Heavy reliance on distillation / continued pre-training / SFT from strong open teachers (DeepSeek-V4-Flash, Qwen3.8 smaller variants, GLM, Kimi smaller, etc.) + high-quality synthetic data. Full from-scratch pre-training is unrealistic without serious compute.  
- Stack: Same as Plan 2 + TorchTitan / FSDP2 for multi-GPU if available, + quantization (GPTQ/AWQ/GGUF) for deployment.  
- Teaching value: Excellent for advanced topics (MoE routing, sparse attention, residual variants).  
- Usability: Best of the three for real chat + coding. Can feel surprisingly close to larger models on many tasks after good post-training.  
- Reality check: Still far from true frontier, but the highest “usefulness / understandability” ratio.

### Correct Build Process (Any Plan)
1. **Master the basics** — Implement a tiny Transformer from scratch (nanoGPT style). Visualize attention, understand why RoPE > absolute PE, why pre-norm helps, etc.
2. **Choose & study a reference** — Read the modeling code of one strong open model (DeepSeek or Qwen recommended).
3. **Data** — High-quality mixed corpus (web, code, math, books). Clean + dedup. For chat: ShareGPT-style or UltraChat + coding datasets. Synthetic data from stronger models is powerful.
4. **Pre-train / continue-pretrain** — Next-token prediction with modern optimizers (AdamW or Muon-style if you experiment), cosine schedule, large batch if possible.
5. **Post-train** — SFT → preference optimization (DPO/KTO/ORPO) → optional light RL (GRPO/PPO variants). Tool-use / agent data helps coding.
6. **Evaluate & iterate** — lm-eval + human chat + coding benchmarks (HumanEval, SWE-bench lite, etc.).
7. **Deploy** — Quantize, serve with vLLM or llama.cpp, add RAG / tools for practical use.
8. **Document everything** — This is your teaching gold. Keep the code minimal and heavily commented.

**Start with Plan 1 this week.** Once the forward pass, training loop, and generation are crystal-clear, move to Plan 2. Use open models as teachers rather than trying to match their scale from scratch. That combination gives the best educational clarity *and* practical chat/coding performance.

If you tell me your available compute (single GPU? multi-GPU? cloud budget?), preferred language focus, or whether you want pure from-scratch vs. fine-tune-first, I can give a more concrete week-by-week roadmap, config, or even starter code structure.


-----------------


## First, a correction I owe you

I was wrong about Qwen3.8. It exists. Qwen/Qwen3.8-27B shipped on Hugging Face August 14, 2026, Apache 2.0, 55.6 GB of BF16 safetensors in 18 shards — after my knowledge cutoff and after the sources my earlier searches returned. It's a dense, multimodal Qwen3.5-family model with hybrid linear + full attention, a vision tower, and a multi-token-prediction head, 262,144 native context. Qwen3.8-Max (2.4T-A95B) landed alongside it. Doc 1's citation was right and my correction was stale. That's exactly the failure mode of trusting a June leaderboard article in August.

Everything else in my earlier correction holds, and Doc 1 has one of its own: grok-2's weights *were* released — 500 GB across 42 files on HF, TP=8, served via SGLang — just under a license that allows research/non-commercial use with conditions and explicitly prohibits using it to train other large AI models. "Product model, not an open release" undersells it in one direction and oversells the usefulness in another. Either way: skip it.

## The Keras pivot: no

Doc 3's factual claims check out. Keras 3 does ship a first-party Muon optimizer. `keras.ops.dot_product_attention` does GQA when N is a multiple of K, MQA when K==1, causal masking, and flash when conditions are met. The progressive-disclosure argument is coherent. And I'd still reject it, for four reasons that are about your actual workflow, not about API taste:

1. **Every line you want to read is raw PyTorch.** gpt-oss's `gpt_oss/torch/model.py`, DeepSeek V3, nanochat, modded-nanogpt, `rasbt/LLMs-from-scratch`, the HF `muse_glimmer` and `qwen3_5` modeling files. Authoring in Keras means every idea you port — MLA, DeltaNet, MTP, attention sinks, FIM — carries a translation step, and translation is where subtle bugs live. You'd pay that tax on every experiment, forever, to save boilerplate you only write once.
2. **The Functional API is the wrong shape for autoregressive decoding.** A symbolic `keras.Input` graph plus incremental KV-cache decode plus variable-length sliding windows is a fight. In PyTorch, `generate()` with a cache is 40 readable lines — and it's one of your best lessons.
3. **Teaching cost, not saving.** Your juniors need to learn one API. Keras-on-torch means they learn `build`/`call`/symbolic-graph semantics *plus* enough torch to debug when it leaks. Meanwhile `pdb` inside eager torch is the single best pedagogical tool in the stack.
4. **Everything downstream is torch-native.** `trl`, FSDP2, vLLM/SGLang export, FlexAttention for the sliding-window masks you're planning, GGUF conversion. Note also that `keras.layers.GroupQueryAttention`'s documented signature is head_dim / num_query_heads / num_key_value_heads / dropout / use_bias / flash_attention / initializers — `sliding_window` is documented on `MultiHeadAttention`, not obviously on the GQA layer. Doc 3 built its whole hybrid-attention plan on that one kwarg. Verify before you design around it.

If you want a reference to unit-test your attention against, use HF `transformers`' implementation of the same model you're mimicking. Same benefit, no framework commitment.

## What Doc 3 got right — take these

The content changes are better than the framework change, and one is very good:

- **3 local : 1 global as the real V1 config, full attention as the Edu baseline.** Now well-verified: Glimmer's text side is 52 layers, hidden 6656, GQA 32:2, SwiGLU with intermediate 19,968, and attention alternates 3 local then 1 global — 39 of 52 layers sliding-window over 2,048 tokens, 13 global. Gemma 4 uses a 5:1 ratio for comparison, and Glimmer is otherwise closest to Gemma 3 27B including the pre/post RMSNorm placement, with SwiGLU instead of GeGLU and gated attention.
- **Two presets per size (Edu / Modern).** Best idea in either document. `--config 150m-edu` vs `--config 150m` on the same `model.py`, benchmarked side by side, is a better lecture than any slide.
- **AdamW as the correctness baseline; Muon as a measured experiment.** I pushed Muon harder than this; Doc 3 is right on sequencing. Don't touch the optimizer until the model can overfit 100 samples to near-zero loss.
- **FIM.** Cheap and genuinely useful for coding. Critical detail: the FIM special tokens must exist *before* you train the tokenizer, so decide this at day 0 or retrain the tokenizer later.
- **Dropping torchao / torchtitan / Triton from V1.** Agreed.
- **A 50M size for minute-scale cycles.** Agreed — CPU/MPS-runnable matters when you're demoing.

One free win both docs missed: Glimmer applies RoPE with theta 500,000 only to the local attention layers. The global layers are NoPE. That's what makes a 3:1 schedule extrapolate to long context instead of degrading — the local layers carry position, the global layers carry unbiased retrieval. One boolean in your config, and it's a great "why does this work" discussion.

## What both docs get wrong

**No cost numbers anywhere.** Doc 1 lists `MiniFrontier-7B — useful chat/coding model` as a config, like it's a JSON edit. Training a 7B from scratch to actual usefulness is a six-figure compute bill and months of iteration. Doc 3 repeats the same size ladder. This is the single most misleading thing in either plan, and it's the reason my Plan C exists: the only 7B you'll realistically ship is one you continue-pretrain from Olmo 3 or Qwen3.8-27B's smaller siblings.

**The vocab/embedding trap at small scale.** Doc 1 recommends ~50K vocab and a 50M "mechanics" model. At d_model=512, a 50K vocab embedding is 25.6M params — with tied weights, half your model is a lookup table, and your loss curve mostly measures how well you memorized token frequencies. Use 8K–16K vocab for the 50M/150M teaching configs and switch to the real 50–64K vocab only from ~500M up. Same code, different config, and the discrepancy is itself a lesson in parameter budgeting.

**No data plan and no eval plan.** Both docs spend ~80% of their length on architecture, which is the part that matters least for whether your model is any good. "Use mixed data" is not a plan. You need: named corpora (FineWeb-Edu shards, SmolTalk, a code mix), a dedup pass, a contamination check, and — before any of it — a loss target and a harness (`lm-eval-harness` plus a held-out set of your own prompts) wired up on day one. Without evals you cannot tell whether Muon helped or whether you broke the mask.

**Grok-1 as study reference #3.** A 314B int8 JAX/Haiku checkpoint with no training code teaches you nothing gpt-oss doesn't teach better in PyTorch. Cut it.

## Converged spec

Raw PyTorch. Single `model.py`, config-driven, no `nn.Transformer`. RMSNorm pre/post, RoPE (local layers only), GQA with QK-Norm as a flag, SwiGLU, SDPA/FlexAttention, `attention_pattern` in `{full, 3:1}`. Tied embeddings below 500M. Sizes 50M / 150M / 500M / 1.5B, each with Edu and Modern presets. FIM tokens in the tokenizer from day 0. AdamW → Muon as a measured A/B. safetensors + `config.json` + `chat_template.jinja` for release. `trl` for SFT/DPO/GRPO. And Plan C — post-training an open base — for anything you actually intend to use.

The Keras version of this project would work. It would just make you a Keras developer instead of someone who can read DeepSeek's source.


-----------------

The final strategy should be **A + C**, not “train a 7B frontier model from scratch.”
## 1. PyTorch wins for this project
I would reverse my Keras recommendation for the **main implementation**.
Keras 3 itself is absolutely not obsolete or a bad choice. In fact, the current API is more capable than the feedback claims: keras.ops.dot_product_attention directly supports GQA/MQA, and keras.layers.GroupQueryAttention currently *does* have sliding_window, Flash Attention support, and gating. Keras also has a first-party Muon optimizer. ([Keras][1])
But that's not the deciding issue. The deciding issue is that the code you want to **learn from, copy ideas from, diff against, debug, and teach from** is overwhelmingly PyTorch. OpenAI explicitly ships gpt_oss/torch/model.py as an intentionally simple, inefficient implementation using basic PyTorch operators to expose the exact architecture. nanochat likewise exposes tokenization → pretraining → SFT → RL → eval → inference in a small PyTorch project. ([GitHub][2])
PyTorch also now gives us FlexAttention, where causal/sliding/custom attention masks can be described in Python while PyTorch generates the efficient implementation underneath. That eliminates one of the reasons I liked Keras's high-level attention API. ([PyTorch][3])
So:
textMain MiniFrontier implementation:
    Python
    +
    raw PyTorch
Optional teaching notebook:
    Keras 3 / PyTorch backend
## The Keras notebook could still be useful for one lesson showing how the same network looks in a higher-level API, but it should **not** be the canonical model.
# 2. The biggest improvement: stop pretending one trained model serves both goals
I strongly agree with this part of the feedback.
We really have:
textMiniFrontier repo
                           │
             ┌─────────────┴──────────────┐
             │                            │
             ▼                            ▼
        LEARN TRACK                    USE TRACK
   train from scratch              take strong open base
   50M–500M                       7B–80B/MoE
   maybe 1.5B                           │
        │                         continue-pretrain
   understand every line                │
        │                               SFT
   experiments                          │
        │                             RLVR
   lectures                             │
        │                           tools/agents
   cheap runs                           │
                                 actually useful AI
## This is much more intellectually clean.
And it can absolutely be **one repository**. I disagree only with the statement that they need to be “two codebases.” They should be two artifacts/workflows sharing evaluation, datasets, chat protocol, tool protocol and experiments—but we should not force both through one artificial Model abstraction.
# 3. Plan B is now optional, not part of the main roadmap
Previously I proposed:
text50M
150M
500M
1.5B
7B
as though these were merely progressively larger config files.
That's misleading.
The from-scratch track should really be:





































ModelPurpose**50M**debugger / CPU or small-GPU teaching**150M**main educational model**500M**architecture experiments**1–1.5B**optional serious experiment**7B+****not from scratch in normal development**For comparison, OLMo 3 7B was trained on 5.93 trillion tokens, and Ai2 exposes the code, training scripts and intermediate checkpoints precisely because reproducing these systems is a large training project rather than “change hidden_size in JSON.” ([Hugging Face][4])So 7B belongs to the **Use Track**, where we start from a checkpoint.

# 4. Use Track: OLMo 3 is now my first choice
For the practical model, I like the feedback's OLMo suggestion a lot.
Olmo-3-1025-7B is Apache 2.0 and Ai2 publishes the code, training recipes and intermediate checkpoints; OLMo-core contains official 7B and 32B training scripts. ([Hugging Face][4])
That makes it unusually valuable for *your project*, because you can ask:
“How would a real lab implement this?”
and actually inspect the answer.
My Use Track candidates would therefore be:
textGeneral experimentation:
    OLMo 3 7B Base
Coding-heavy:
    Qwen3-Coder-Next-Base
Large local/agent reference:
    Muse Glimmer
## Qwen3-Coder-Next-Base is particularly interesting: 80B total but only 3B activated, 48 layers, with a literal 3 × Gated DeltaNet : 1 × Gated Attention hybrid repeated twelve times, plus sparse MoE. ([Hugging Face][5])
That's a very useful reference for our advanced lessons.
# 5. One correction to the feedback: don't call “3:1” one architecture
There's a subtle but important distinction.
Muse Glimmer's uploaded architecture is:
textSliding attention
Sliding attention
Sliding attention
Full attention
repeated across its decoder, with GQA, local RoPE and full-attention NoPE.
Qwen3-Coder-Next is:
textGated DeltaNet
Gated DeltaNet
Gated DeltaNet
Full gated attention
([Hugging Face][6])
Kimi K3 is:
textKDA
KDA
KDA
Gated MLA
DeepSeek V4 instead interleaves CSA and HCA, with compression and sparse selection.
So there **is** a fascinating 3-cheap/efficient : 1-powerful/global convergence, but the efficient operator differs substantially.
That becomes one of the main lessons of MiniFrontier:
**Frontier models increasingly avoid paying full quadratic attention cost in every layer.**
That's more valuable than blindly copying DeltaNet or KDA.

# 6. Therefore our Modern V1 should stay simpler than Qwen/Kimi
I would freeze two presets.
### Edu
textDecoder-only
RMSNorm
RoPE
MHA
full causal attention
SwiGLU
tied embeddings
This teaches the Transformer.
### Modern
textDecoder-only
RMSNorm
RoPE
GQA
QK-Norm
attention schedule:
    Local
    Local
    Local
    Global
SwiGLU
tied embeddings
And optionally:
textlocal_position = RoPE
global_position = NoPE
because that's an elegant experiment inspired directly by Glimmer rather than a mandatory truth.
**Do not put Gated DeltaNet into V1.**
Make it the first serious architectural extension.
That gives us:
textMHA
 ↓
GQA
 ↓
local/full hybrid
 ↓
Gated DeltaNet hybrid
 ↓
MLA
 ↓
MoE
## as the educational progression.
Much better.
# 7. I also agree on QK-Norm—but not “quietly universal”
Use it in Modern.
Don't claim it's universal.
There are enough current systems and speedrun architectures using QK normalization that it is a worthwhile cheap stability feature; modded-nanogpt explicitly lists QK-Norm among its modernized architecture changes. ([GitHub][7])
But:
Pythonqk_norm: bool
## should remain a config flag so we can A/B it.
That matters because this project is supposed to **teach why choices exist**, not just accumulate fashionable defaults.
# 8. Muon: definitely add it, but exactly as an experiment
This feedback is right.
The modded-nanogpt project reports for its specific benchmark that Muon gives lower optimizer-state memory, about 1.5× better sample efficiency, and under 2% wall-clock overhead compared with its Adam-based reference. ([GitHub][8])
But speedrun results are not proof that Muon will universally give us 1.5× on every scale/data/model.
So our training system should expose:
Pythonoptimizer = "adamw"
# or
optimizer = "muon"
and the curriculum is:
textFirst:
    make AdamW correct
Then:
    train identical seed/config with Muon
Compare:
    validation loss
    tokens to target
    wall-clock to target
    memory
## That's much more scientific.
nanochat itself now contains AdamW + Muon infrastructure, making it another useful implementation reference. ([GitHub][9])
# 9. The vocabulary criticism is good, but I would solve it differently
I agree that 50K vocabulary on a 50M model is silly.
But I **wouldn't change tokenizer vocabulary at every model size**, because then our scaling comparisons become contaminated by a changing tokenizer.
Instead:
textLearn Track tokenizer:
byte-aware BPE
vocab = 16K
used by:
    50M
    150M
    500M
    1.5B
So all four models see exactly the same tokenization.
Then:
textUse Track:
never replace the base model tokenizer
inherit OLMo/Qwen/etc.
That gives us clean experiments.
And yes, reserve chat/tool/FIM tokens from the beginning.
One correction, though: special tokens don't technically **have to exist before tokenizer training**. Hugging Face Tokenizers can add new special tokens later and assign them new IDs. ([Hugging Face][10])
The stronger rule is:
Define them before **model pretraining** so you never have to resize/reinitialize embeddings later.

# 10. FIM stays
Definitely.
The tokenizer should reserve something equivalent to:
text<|fim_prefix|>
<|fim_suffix|>
<|fim_middle|>
<|system|>
<|user|>
<|assistant|>
<|tool_call|>
<|tool_result|>
Then our coding mixture can contain both ordinary:
textprefix → continuation
and:
textprefix + suffix → middle
## examples.
That costs essentially zero architectural complexity.
# 11. Data and evaluation now become first-class modules
This is probably the most important correction to the entire earlier plan.
We shouldn't spend four weeks implementing exotic attention before we have a scoreboard.
nanochat already does this correctly: its training loop periodically measures validation bits-per-byte and its CORE metric rather than deciding from generated samples whether a model “looks better.” ([GitHub][11])
And nanochat's current speedrun has actually moved from FineWeb-Edu to NVIDIA ClimbMix after controlled experiments showed better results for that particular benchmark. ([GitHub][12])
Our minimum evaluation stack should therefore be:
texttraining:
    train loss
    validation loss
    bits/token
    bits/byte
base model:
    CORE-style tasks
    lm-eval-harness subset
chat:
    instruction following
    own held-out prompts
coding:
    HumanEval-like tests
    compile rate
    unit-test pass rate
    FIM tests
agent:
    tool-call correctness
    terminal task success
## lm-eval-harness currently supports dozens of standard evaluations plus custom tasks, Hugging Face models and vLLM. ([GitHub][13])
This should be installed **before architecture experiments start**.
# 12. For code data: be careful with licenses
One place where I'd be more conservative than the pasted feedback is code datasets.
For example, The Stack v2 is huge and useful, but its dataset card explicitly says that it contains repositories under many licenses and users must follow the original license obligations and dataset terms; it also maintains provenance and opt-out mechanisms. ([Hugging Face][14])
So our data layer should track:
textsource
license
document_id
hash
split
language
## from day one.
No anonymous “download 300B GitHub tokens” pipeline.
# 13. Grok drops out of the implementation-study list
Agreed.
Grok-2 weights **do exist**: xAI's repository says about 500 GB across 42 files, intended for TP=8 serving. ([Hugging Face][15])
But the Grok 2 Community License explicitly prohibits using the model, derivatives, **or its generated outputs** to train/create/improve another foundational, large-language or general-purpose AI model, except permitted Grok-2 modifications. ([Hugging Face][16])
So:
textGrok-2:
    benchmark/reference only
not:
    teacher
    distillation source
    synthetic-data generator
## And Grok-1's JAX source isn't important enough to make the core reading list anymore.
# 14. One Qwen correction is still unresolved
I can verify that **qwen3.8-max-preview is real** because Qwen's own qwen-code repository added support for it in July 2026. ([GitHub][17])
I **cannot currently verify from an official Qwen model card** the stronger claim in the feedback that an open-weight Qwen/Qwen3.8-27B was released on August 14. My current official-source search surfaces Qwen3.8-Max Preview activity but the published 27B model card it reliably finds is Qwen3.5-27B. ([GitHub][17])
So I would treat:
textQwen3.8-Max Preview = verified
Qwen3.8-27B open weights = currently unverified
## rather than build the roadmap around it.
# 15. Final repo architecture
I now want this:
textminifrontier/
│
├── minifrontier/
│   ├── model.py
│   ├── attention.py
│   ├── rope.py
│   ├── cache.py
│   ├── generation.py
│   ├── tokenizer.py
│   └── config.py
│
├── learn/
│   ├── pretrain.py
│   ├── midtrain.py
│   ├── sft.py
│   └── experiments/
│       ├── mha_vs_gqa.py
│       ├── qknorm.py
│       ├── full_vs_hybrid.py
│       └── adamw_vs_muon.py
│
├── use/
│   ├── continue_pretrain.py
│   ├── sft.py
│   ├── dpo.py
│   ├── rlvr.py
│   └── tools/
│
├── data/
│   ├── prepare.py
│   ├── dedup.py
│   ├── fim.py
│   └── contamination.py
│
├── eval/
│   ├── base.py
│   ├── chat.py
│   ├── code.py
│   └── agent.py
│
├── configs/
│   ├── 50m-edu.toml
│   ├── 50m-modern.toml
│   ├── 150m-edu.toml
│   ├── 150m-modern.toml
│   ├── 500m-modern.toml
│   └── 1.5b-modern.toml
│
└── tests/
Crucially:
textmodel.py + attention.py + rope.py + cache.py
## should remain small enough that an experienced programmer can genuinely read the complete neural model in an afternoon.
# 16. Final V1 stack
textPython 3.12
Core
    torch
    numpy
Dependency management
    uv
Tokenizer/data
    tokenizers
    datasets
Compatibility
    transformers
    safetensors
    jinja2
Post-training
    trl
Evaluation
    lm-eval
Development
    pytest
Not V1:
textKeras
Triton
TorchTitan
TorchAO
DeepSpeed
Ray
Megatron
XGBoost
LightGBM
scikit-learn
LangChain
And:
texttorch.compile
FlexAttention
## are **features of our PyTorch stack**, not extra architecture frameworks.
# 17. The revised roadmap

**Start from nanochat, don't fork it blindly.** Run its current speedrun first and understand gpt.py, tokenizer, optimizer, base training, SFT, eval and generation. nanochat currently reports GPT-2-grade capability in roughly 1.5–2 hours of 8×H100 pretraining under its current recipe, with the whole pipeline still remaining intentionally small. ([GitHub][18])
**Write MiniFrontier-Edu ourselves.** 50M then 150M: MHA + RMSNorm + RoPE + SwiGLU + KV cache. No cleverness.
**Create MiniFrontier-Modern by changing components, not rewriting the model.** MHA→GQA, QK-Norm, 3-local/1-global, then AdamW→Muon A/B.
**Add FIM and code evaluation before adding exotic architecture.**
**Implement DeltaNet as an optional attention module.** Compare it against local GQA at identical compute. Qwen3-Coder-Next gives us a real modern implementation target. ([Hugging Face][6])
**Only then study MLA and MoE.** They become advanced chapters, not V1 dependencies.
**Start the Use Track from OLMo 3 7B and/or Qwen3-Coder-Next-Base.** Continue-pretrain → SFT → verifiable coding RL → tools. OLMo provides unusually complete checkpoints/recipes; Qwen gives us a very strong coding-oriented sparse model. ([Hugging Face][4])

## Final decision
So the feedback **does materially improve our plan**.
The project I would now build is not “MiniFrontier-7B from scratch.” It is:
**MiniFrontier Learn:** a ~50M–500M, optionally 1.5B, completely understandable modern Transformer implementation in raw PyTorch.**MiniFrontier Use:** post-train a strong open base such as OLMo 3 7B / Qwen3-Coder-Next-Base into the actual general-chat + coding + agent model.
The most important design principle also changes slightly:
**Don't make the tiny model imitate the parameter count of frontier systems. Make it imitate their ideas, experimental discipline, data pipeline, post-training pipeline and tool loop.**
That gives you something you can teach **line-by-line** and something you can actually use—without pretending that a few thousand lines of clean Transformer code erase the compute/data gap to frontier labs.
[1]: https://keras.io/api/ops/nn/?utm_source=chatgpt.com "NN ops"
[2]: https://github.com/openai/gpt-oss?utm_source=chatgpt.com "GitHub - openai/gpt-oss: gpt-oss-120b and gpt-oss-20b are two open-weight language models by OpenAI · GitHub"
[3]: https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/?utm_source=chatgpt.com "FlexAttention + FlashAttention-4: Fast and Flexible – PyTorch"
[4]: https://huggingface.co/allenai/Olmo-3-1025-7B?utm_source=chatgpt.com "allenai/Olmo-3-1025-7B · Hugging Face"
[5]: https://huggingface.co/Qwen/Qwen3-Coder-Next-Base?utm_source=chatgpt.com "Qwen/Qwen3-Coder-Next-Base · Hugging Face"
[6]: https://huggingface.co/Qwen/Qwen3-Coder-Next?utm_source=chatgpt.com "Qwen/Qwen3-Coder-Next · Hugging Face"
[7]: https://github.com/KellerJordan/modded-nanogpt?utm_source=chatgpt.com "GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 90 seconds · GitHub"
[8]: https://github.com/kellerjordan/modded-nanogpt?utm_source=chatgpt.com "GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 90 seconds · GitHub"
[9]: https://github.com/karpathy/nanochat/blob/master/README.md?utm_source=chatgpt.com "nanochat/README.md at master · karpathy/nanochat · GitHub"
[10]: https://huggingface.co/docs/tokenizers/main/api/tokenizer?utm_source=chatgpt.com "Tokenizer · Hugging Face"
[11]: https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py?utm_source=chatgpt.com "nanochat/scripts/base_train.py at master · karpathy/nanochat · GitHub"
[12]: https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md?utm_source=chatgpt.com "nanochat/dev/LEADERBOARD.md at master · karpathy/nanochat · GitHub"
[13]: https://github.com/EleutherAI/lm-evaluation-harness/?utm_source=chatgpt.com "GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models. · GitHub"
[14]: https://huggingface.co/datasets/bigcode/the-stack-v2?utm_source=chatgpt.com "bigcode/the-stack-v2 · Datasets at Hugging Face"
[15]: https://huggingface.co/xai-org/grok-2?utm_source=chatgpt.com "xai-org/grok-2 · Hugging Face"
[16]: https://huggingface.co/xai-org/grok-2/blob/f03a63f535c28dbdbfa8f6f06c6979e82494cff7/LICENSE?utm_source=chatgpt.com "LICENSE · xai-org/grok-2 at f03a63f535c28dbdbfa8f6f06c6979e82494cff7"
[17]: https://github.com/QwenLM/qwen-code/actions/runs/29796610723?utm_source=chatgpt.com "fix(core): support qwen3.8 side queries on DashScope · QwenLM/qwen-code@566760a · GitHub"
[18]: https://github.com/karpathy/nanochat?utm_source=chatgpt.com "GitHub - karpathy/nanochat: The best ChatGPT that $100 can buy. · GitHub"
Yes. Focusing on Qwen3-Coder-Next-Base changes the project substantially: we are no longer designing the Transformer architecture ourselves. We inherit a very strong 80B-total/3B-active pretrained backbone and put our engineering effort into data, post-training, verifiable RL, tool use, agent scaffolding, evaluation, and serving.
There is one important correction to the “Modern preset” above:
Do not modify Qwen3-Coder-Next-Base into Local/Local/Local/Global + tied embeddings.
That was appropriate for our educational MiniFrontier model. The actual Qwen3-Coder-Next-Base should be preserved exactly. It has 48 layers arranged as 12 × [3 × (Gated DeltaNet → MoE), 1 × (Gated Attention → MoE)], 80B total parameters / 3B activated, 16 Q heads and 2 KV heads in the full-attention layers, 512 routed experts with 10 selected plus one shared expert, 256K native context, and untied input/output embeddings. (Hugging Face)
So this is the final project I would build.

Final goal
Call it something like:
FrontierCoder
or keep:
MiniFrontier.Use
The objective is:
Start from Qwen/Qwen3-Coder-Next-Base and create our own general-purpose chat + coding + software-engineering agent through continued training, SFT, preference optimization, verifiable RL, and a minimal tool/agent runtime.
The target system is not merely:
LLM → text
It is:
                    User / IDE / API
                           │
                           ▼
                    Agent Orchestrator
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
              Qwen LLM           Context Manager
                 │
                 ▼
             Tool Router
                 │
     ┌───────────┼───────────┬────────────┐
     ▼           ▼           ▼            ▼
   Files       Search      Terminal       Git
     │                       │            │
     │                    Compiler        Diff
     │                       │
     │                     Tests
     │                       │
     └───────────────┬───────┘
                     ▼
                  Sandbox
                     │
                     ▼
                Observation
                     │
                     └──────→ LLM
This is where much of the frontier-like coding capability will come from. Qwen's own Coder-Next report emphasizes executable environments, tool use, long-horizon interaction, verification, error recovery and reinforcement learning rather than parameter scaling alone. (arXiv)
Backbone architecture — freeze the specification
Our model is:
Qwen3-Coder-Next-Base
and the backbone remains:
Vocabulary
   │
   ▼
Token Embeddings
   │
   ▼

┌─────────────────────────────────────────────┐
│ Block group × 12                            │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ Gated DeltaNet                     │    │
│  │        ↓                            │    │
│  │ Sparse MoE                         │    │
│  └─────────────────────────────────────┘    │
│                   × 3                       │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ Gated full attention               │    │
│  │ GQA: 16 Q / 2 KV                   │    │
│  │ QK normalization                   │    │
│  │ partial RoPE                       │    │
│  │        ↓                            │    │
│  │ Sparse MoE                         │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                   × 12 groups
                   = 48 layers
                         │
                         ▼
                       Norm
                         │
                         ▼
                      LM Head
Qwen documents the exact repeating structure as three Gated DeltaNet layers followed by one Gated Attention layer, each paired with MoE. The full attention has 16 Q heads, two KV heads and 256-dimensional heads. The DeltaNet branch has 16 Q/K heads and 32 value heads with head dimension 128. (Hugging Face)
The model uses partial RoPE: only 25% of the 256-dimensional attention head is rotary-position encoded. Qwen's architecture work also uses zero-centered RMSNorm and output gating for stability; the current Transformers implementation visibly performs Q/K normalization before RoPE and gates the attention output. (Qwen)
The MoE is:
512 routed experts
      │
Router
      │
select 10
      │

1 shared expert
      │
combine
with 512-dimensional expert FFNs. (Hugging Face)
And importantly:
tie_word_embeddings = false
according to the released checkpoint config. We do not change that. (Hugging Face)


What happens to our old Modern preset?
It becomes educational/reference material only.
Our old design:
RMSNorm
RoPE
GQA
QK-Norm

Local
Local
Local
Global
SwiGLU
was actually a simplified approximation of the trend.
Qwen's real implementation is more advanced:
DeltaNet
DeltaNet
DeltaNet
Full Attention
repeat
That is even better for the Use Track because we inherit the difficult part rather than reimplementing it.
So:
MiniFrontier Learn
    Local
    Local
    Local
    Global
MiniFrontier Use
    DeltaNet
    DeltaNet
    DeltaNet
    Full Attention
We should not touch modeling_qwen3_next.py unless we eventually become architecture researchers.
4. The technology stack
I would keep the actual stack surprisingly small.
Language
    Python
Environment/dependencies
    uv
Model
    PyTorch
    Transformers
Training
    Accelerate
    TRL
Efficient adaptation
    PEFT
    LoRA / QLoRA
Data
    datasets
    pyarrow
Storage
    safetensors
Evaluation
    lm-eval
    custom executable evals
Serving
    vLLM
Alternative serving
    SGLang
Agent runtime
    our own small Python loop
Isolation
    Docker
Testing
    pytest
Qwen officially supports the released Base checkpoint through Transformers as AutoModelForCausalLM, and its model card provides both vLLM and SGLang serving paths. (Hugging Face)
TRL currently gives us SFT, DPO, GRPO and other post-training methods, while its GRPO implementation can expose Python tools and stateful per-rollout environments directly to the model. (Hugging Face)
5. Two less-visible dependencies we do want
For Qwen3-Next itself, the Transformers implementation has optimized paths using:
causal-conv1d
flash-linear-attention / FLA
for Gated DeltaNet. If these aren't installed, Transformers has pure-PyTorch fallback implementations. That's useful for correctness/debugging but not what I'd use for serious long-context training. (GitHub)
This still does not mean:
write Triton kernels ourselves
We aren't doing that.
We consume the optimized upstream implementation.
6. Things deliberately excluded
V1 does not need:
Keras
TensorFlow
JAX
TorchTitan
TorchAO
our own Triton
our own CUDA
Megatron
Ray
LangChain
LlamaIndex
scikit-learn
XGBoost
LightGBM
FSDP2 becomes necessary only when we genuinely move into full-model distributed training.
For normal initial adaptation:
Transformers + PEFT + TRL + Accelerate
is enough.
7. Repository structure
I would use this:
frontiercoder/
│
├── pyproject.toml
├── uv.lock
├── README.md
│
├── configs/
│   ├── model/
│   │   └── qwen3-coder-next-base.yaml
│   │
│   ├── train/
│   │   ├── midtrain-lora.yaml
│   │   ├── sft-lora.yaml
│   │   ├── dpo-lora.yaml
│   │   └── grpo-lora.yaml
│   │
│   ├── eval/
│   │   ├── base.yaml
│   │   ├── chat.yaml
│   │   ├── coding.yaml
│   │   └── agent.yaml
│   │
│   └── serve/
│       ├── vllm.yaml
│       └── sglang.yaml
│
├── src/
│   └── frontiercoder/
│       │
│       ├── model/
│       │   ├── load.py
│       │   ├── adapters.py
│       │   └── tokenizer.py
│       │
│       ├── data/
│       │   ├── schemas.py
│       │   ├── registry.py
│       │   ├── prepare.py
│       │   ├── filter.py
│       │   ├── dedup.py
│       │   ├── decontaminate.py
│       │   ├── packing.py
│       │   └── fim.py
│       │
│       ├── train/
│       │   ├── midtrain.py
│       │   ├── sft.py
│       │   ├── dpo.py
│       │   ├── grpo.py
│       │   └── checkpoint.py
│       │
│       ├── rewards/
│       │   ├── base.py
│       │   ├── compile.py
│       │   ├── tests.py
│       │   ├── tool_format.py
│       │   ├── patch.py
│       │   ├── security.py
│       │   └── efficiency.py
│       │
│       ├── agent/
│       │   ├── loop.py
│       │   ├── context.py
│       │   ├── protocol.py
│       │   └── tools/
│       │       ├── files.py
│       │       ├── search.py
│       │       ├── shell.py
│       │       ├── tests.py
│       │       └── git.py
│       │
│       ├── sandbox/
│       │   ├── environment.py
│       │   ├── docker.py
│       │   └── limits.py
│       │
│       ├── eval/
│       │   ├── runner.py
│       │   ├── chat.py
│       │   ├── coding.py
│       │   ├── tool_use.py
│       │   ├── long_context.py
│       │   └── agent.py
│       │
│       └── serving/
│           ├── client.py
│           └── health.py
│
├── scripts/
│   ├── download_model.py
│   ├── prepare_data.py
│   ├── train_mid.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── evaluate.py
│   └── serve.py
│
├── docker/
│   ├── training.Dockerfile
│   ├── inference.Dockerfile
│   └── sandbox.Dockerfile
│
├── tests/
│
└── artifacts/
    └── .gitignore
Notice what's missing:
attention.py
rope.py
moe.py
transformer.py
We don't need them.
Those belong to the Learn Track.
8. Don't modify the tokenizer
This is another important rule.
Use:
AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Coder-Next-Base"
)
and retain Qwen's vocabulary, token IDs and tool-related special tokens.
Qwen specifically warns that Qwen3-Coder updated its special tokens/token IDs and that its function-calling behavior relies on the corresponding parsers in vLLM and SGLang. (GitHub)
So:
NO:
    train our own tokenizer
    change vocabulary
    resize embeddings
    invent new FIM tokens
YES:
    use existing tokenizer
    existing FIM/tool protocol
9. Canonical internal dataset format
Don't store training data as pre-rendered strings.
Store structured conversations:
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "...",
      "tool_calls": []
    }
  ]
}
For agent trajectories:
Task
 │
user request
 │
assistant reasoning/action
 │
tool call
 │
observation
 │
assistant action
 │
tool call
 │
observation
 │
...
 │
final answer
 │
verification
Then rendering into the Qwen chat template happens only during preprocessing/training.
10. Four core dataset schemas
I would formalize four formats.
Language-model sample
id
text
source
license
language
document_hash
split
metadata
Used by continued/mid-training.
Conversation sample
id
messages[]
source
domain
quality_score
verifier_result
Used by SFT.
Preference sample
prompt
chosen
rejected
judge_scores
verification
Used by DPO.
Agent task
task
environment
initial_state
tools[]
max_steps
timeout
verifier
reference_patch?
metadata
Used for GRPO/RLVR.
This separation will save us a lot of pain later.
11. Phase 0 — establish the scoreboard before training
Before changing one weight, evaluate:
Qwen3-Coder-Next-Base
and:
Qwen3-Coder-Next
The instruction-tuned sibling becomes our first reference target. The latter has the exact same 80B/3B-active architecture and is Qwen's own post-trained coding-agent model. (Hugging Face)
Record:
general QA
instruction following
code generation
code repair
FIM/editing
tool calling
repository tasks
terminal tasks
long context
latency
tokens/sec
memory
Do this before SFT.
Otherwise we won't know whether our training helped.
12. Evaluation suite
I would divide evaluation into five buckets:
General
    knowledge
    reasoning
    instruction following
    multilingual
    conversation
Coding
    generation
    debugging
    repair
    review
    refactoring
    FIM
Software engineering
    repo-level issue resolution
    patch correctness
Agentic
    tool use
    terminal
    multi-turn recovery
    tool format correctness
System
    latency
    throughput
    context length
    memory
Qwen itself evaluates Coder-Next on SWE-Bench variants, Terminal-Bench and Aider-style coding workloads, because agentic coding needs repository interaction rather than merely HumanEval-style function generation. (Qwen)
Use lm-eval-harness for standard text evaluations, but our most important coding scores should come from actual executable tasks.
13. Phase 1 — light mid-training
Qwen already did enormous coding-focused mid-training. Their report describes repository-level code, PRs, code review, text/code grounding, hundreds of programming languages, FIM and agent trajectories; repo-level data alone reached roughly 600B tokens in their recipe. (arXiv)
So we should not try to redo Qwen pretraining.
Our mid-training exists only to shift the distribution toward whatever we care about.
Think:
Qwen base knowledge
       +
our desired domains
       +
our preferred coding practices
       +
our documentation
       +
our desired agent workflows
Keep this stage conservative.
14. Mid-training data mixture
I would begin with roughly:
80–90% natural/grounded
10–20% synthetic
as an experimental starting point, not a sacred ratio.
Natural:
high-quality source code
complete repositories
PRs
issues
code reviews
technical documentation
API documentation
software architecture documents
high-quality general text
Synthetic:
grounded technical QA
bug-fix tasks
repo comprehension tasks
code transformation
agent trajectories
This follows the same principle Qwen reports: mainly natural data, with a smaller deliberately constructed synthetic component to move the model toward realistic user workflows without excessive specialization. (arXiv)
15. FIM stays
Qwen3-Coder-Next already supports FIM-oriented training.
Their report specifically includes both chat-style FIM and search-and-replace FIM, and reports search-and-replace performing better in their experiments at equivalent scale. (arXiv)
So our coding data should include:
ordinary continuation



fill middle of code



search/replace patch



git diff
Especially:
before
↓
requested change
↓
minimal patch
rather than teaching the model to rewrite entire files unnecessarily.
16. Phase 2 — SFT is where it becomes an assistant
This is the most important first real training stage.
The Base model knows a lot.
But we have to teach:
what good answers look like
how to follow instructions
when to use tools
how to produce patches
how to explain code
how to ask for missing information
how to recover from errors
how to finish a task
TRL's current SFTTrainer supports conversational datasets, packing and assistant-only/completion-only loss, which fits this stage well. (Hugging Face)
17. Initial SFT mixture
For our first run I'd start approximately:
30% general assistant/chat
30% coding + technical QA
30% multi-turn coding/agent trajectories
10% tool/format/schema exercises
Then change those ratios only from evaluation results.
The general 30% is important.
Otherwise we risk producing:
excellent coding specialist
terrible ChatGPT replacement
Our goal is broader.
18. General-chat SFT should contain
Not just QA.
Train:
direct factual answers
explanations
summaries
rewriting
planning
comparison
reasoning
technical explanation
professional communication
multilingual conversation
instruction following
clarification when actually needed
And vary response lengths.
Otherwise you'll accidentally create a model with one stylistic voice.
19. Coding SFT should contain
A broad spectrum:
implement
complete
debug
review
refactor
explain
optimize
test
document
migrate
design architecture
read unfamiliar repository
find bug
change multiple files
resolve compiler error
resolve failing test
interpret stack trace
Across many languages and ecosystems.
The original Qwen training strategy deliberately includes library use, APIs, I/O, multiple languages and language-specific tooling rather than only competitive programming. (arXiv)
That is the right philosophy.
20. Execution verification is mandatory
An SFT coding answer should ideally go through:
model answer
     │
     ▼
compile?
     │
     ▼
tests?
     │
     ▼
runtime?
     │
     ▼
expected behavior?
     │
     ▼
training sample
Qwen reports using an execution-based user simulator to reject hallucinated/non-functional responses using compiler, runtime and environment feedback before those responses enter high-quality SFT data. (arXiv)
This is a huge lesson:
Don't ask a judge LLM whether code looks correct when you can execute it.
21. Phase 3 — preference training
After SFT:
Prompt
 │
 ├── answer A
 ├── answer B
 ├── answer C
 └── answer D
execute / judge
      │
      ▼
 ranking
Then train chosen versus rejected examples.
DPO is a reasonable first implementation because TRL directly supports it and accepts conversational preference datasets. (Hugging Face)
Use this mainly for:
helpfulness
clarity
instruction adherence
verbosity
professional style
answer organization
user preference
Don't use DPO as a substitute for executable correctness.
22. Phase 4 — RLVR/GRPO is where coding capability should really move
For coding, we have unusually good rewards.
Instead of:
"an AI judge thinks this is an 8/10"
we have:
does it compile?
do the tests pass?
did the bug disappear?
did the patch apply?
does the CLI produce the correct result?
did the tool call parse?
Qwen's report explicitly argues that coding is unusually well suited to execution-driven RL because correctness can often be verified directly. (arXiv)
TRL currently provides GRPO and supports both reward functions and stateful agent environments. (Hugging Face)
23. Keep the reward simple
Start with hard outcomes.
For example:
task failed
    0
task passed
    1
Then perhaps:

valid tool format
all tests passed
security checks passed


timeout
malformed tool call
unnecessary destructive action
Don't start with twelve weighted rewards.
Reward engineering can easily produce models that optimize our scoring function rather than solve the task.


Phase 5 — real multi-turn agent RL
This is the expensive but valuable stage.
Each rollout is:
Task
 │
 ▼
Model
 │
 ▼
read_file
 │
 ▼
Model
 │
 ▼
search_code
 │
 ▼
Model
 │
 ▼
edit_file
 │
 ▼
Model
 │
 ▼
run_tests
 │
 ├── failure
 │     │
 │     └───────────────┐
 │                     ▼
 │                   Model
 │                     │
 │                  fix again
 │
 └── success
       │
       ▼
     reward
This is essentially the direction Qwen took: large collections of executable tasks, environment interaction, high-quality agent trajectories, domain specialization and reinforcement learning. (Qwen)
Agent tools: keep V1 tiny
Only implement:
read_file
write_file
list_files
search_files
grep

run_command
run_tests
git_diff
git_status
Maybe:
web_search
afterward.
That is enough for an astonishing amount of software-engineering work.
No generic agent framework needed.
26. Agent loop itself can be tiny
Conceptually:
while steps < max_steps:
    response = model(messages, tools)
    if response.is_final:
        return response
    tool_result = execute(
        response.tool_call
    )
    messages.append(tool_result)
The intelligence belongs in:
model
+
training
+
environment
not 20,000 lines of agent-framework abstractions.
27. Train multiple tool protocols
This is a really useful result from Qwen's technical report.
They found that training with several tool-call representations improves transfer to different IDE/CLI scaffolds. Their data includes JSON, Python-style calls, XML-style formats such as qwen3_coder, TypeScript interfaces and natural-language definitions. (arXiv)
So don't train only:
{"name":"read_file","arguments":{...}}
Train:
OpenAI JSON
Qwen XML
Python function style
TypeScript schema
MCP-like definitions
while keeping one canonical internal representation.
This should make the model much more robust.
28. Sandboxing architecture
Never let training rollouts operate against the host machine directly.
Use:
Trainer
   │
   ▼
EnvironmentFactory
   │
   ▼
Docker container
   │
   ├── repo
   ├── compiler
   ├── dependencies
   ├── tests
   └── verifier
Every task gets:
clean filesystem
CPU/memory limits
timeout
network policy
temporary credentials only
automatic destruction
This also makes rewards reproducible.
Qwen's own large-scale task synthesis relies on runnable, reproducible environments and stores software-engineering environments as reusable container images before rollouts/evaluation. (arXiv)
29. First training strategy: LoRA, not full 80B fine-tuning
This is important.
“3B active” does not mean we only store a 3B model.
The checkpoint still contains 80B parameters. (Hugging Face)
So start:
Base
  ↓
LoRA SFT
  ↓
LoRA DPO
  ↓
LoRA GRPO
and only pursue full-model tuning if experiments demonstrate a real adapter ceiling.
TRL integrates directly with PEFT for LoRA/QLoRA across its trainers. (Hugging Face)
30. Qwen's MoE needs one PEFT consideration
Modern MoE implementations frequently hold expert matrices as fused multidimensional parameters rather than ordinary nn.Linear modules.
Current PEFT supports precisely this using:
target_parameters
for expert gate_up_proj / down_proj tensors as well as normal target_modules for conventional projections. PEFT warns that unmerged LoRA on expert parameters can add inference overhead and recommends merging the adapter for final deployment. (Hugging Face)
So I would have two adapter presets.
lora-lite
Target only ordinary projections:
attention Q/K/V/O
DeltaNet projections
Cheap experiments.
lora-full
Also adapt selected MoE expert weights.
More capacity.
Then merge the winning adapter into a complete checkpoint for production.
31. Don't immediately do QLoRA
QLoRA is useful and TRL/Transformers support 4-bit PEFT training. (Hugging Face)
But Qwen3-Coder-Next is a relatively unusual combination of:
512-expert MoE
+
Gated DeltaNet
+
full attention
So I would validate:
BF16 LoRA
first on a small run.
Then:
QLoRA
only after verifying numerical equivalence and throughput.
Correctness before savings.
32. Distributed training when we need it
For larger full/adapter runs:
Accelerate
    ↓
FSDP2
is my preferred next step.
FSDP2 shards parameters, gradients and optimizer states across GPUs, supports CPU-efficient loading of very large checkpoints, activation checkpointing and sharded checkpoints. (Hugging Face)
So our stack grows naturally:
1 GPU
    PyTorch
multiple GPUs
    Accelerate
model no longer fits
    FSDP2
No Mega-framework first.
33. Hardware reality
BF16 requires roughly:
[
80B \times 2\ bytes \approx 160GB
]
for weights alone.
So “3B active” gives you excellent compute efficiency, but doesn't magically make an 80B checkpoint occupy 6 GB of memory.
For serious BF16 serving I'd think in terms of a multi-GPU machine rather than one consumer GPU. Quantization can reduce the static weight footprint dramatically, but maximum context length and concurrent requests add additional memory requirements. Qwen explicitly recommends reducing context length when memory is insufficient. (Qwen)
And full 80B training is a completely different hardware class from LoRA.
Therefore the sensible progression is:
adapter experimentation
        ↓
prove gains
        ↓
larger adapter training
        ↓
only then consider full tuning
34. Don't train at 256K initially
The model supports 262,144 tokens natively. (Hugging Face)
That doesn't mean every batch should use them.
Use a context curriculum:
Stage 1
    4K–8K
Stage 2
    16K
Stage 3
    32K
Stage 4
    64K
special long-context runs
    128K–256K
Most user turns don't need 256K.
Reserve expensive long sequences for:
whole repository understanding
long agent histories
very large diffs
documentation sets
35. Data quality pipeline
Every input should pass something like:
Raw source
   ↓
license/provenance
   ↓
parse
   ↓
quality filter
   ↓
secret/PII filter
   ↓
dedup
   ↓
benchmark contamination scan
   ↓
language/domain classification
   ↓
training format
   ↓
train/validation split
For code specifically retain:
repo identity
commit
license
language
file path
hash
That gives us traceability later.
36. Promotion gates
No checkpoint becomes main because “the sample answers looked better.”
Every stage must beat the previous one.
For example:
Base
 ↓
Mid
 ↓
SFT
 ↓
DPO
 ↓
RLVR
 ↓
Agent-RL
and each arrow requires:
general score not materially worse
coding improved
agent improved
tool syntax not worse
long-context not worse
latency acceptable
Keep every benchmark result alongside the checkpoint hash.
37. Serving architecture
Production:
                        App / IDE
                           │
                  OpenAI-compatible API
                           │
                           ▼
                     Agent Service
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
             vLLM                Sandbox pool
                │                     │
                ▼                     ▼
        FrontierCoder model       tools/tests
vLLM exposes an OpenAI-compatible endpoint and the Base model card gives direct vLLM serving instructions. SGLang is also officially supported. (Hugging Face)
I would choose:
vLLM = default
SGLang = benchmark alternative
and switch only if measurements show SGLang is better for our workload.
38. Artifact/checkpoint lifecycle
Keep:
Qwen3-Coder-Next-Base
          │
          ▼
      mid-v001
          │
          ▼
      sft-v001
          │
          ▼
      dpo-v001
          │
          ▼
     rlvr-v001
          │
          ▼
    agent-v001
          │
          ▼
      release
Every checkpoint gets:
config
training manifest
dataset manifests
git commit
dependency lock
metrics
eval report
seed
training duration
hardware
parent checkpoint
Reproducibility matters much more than clever naming.
39. Release format
Final release:
FrontierCoder/
│
├── config.json
├── generation_config.json
│
├── model-00001-of-N.safetensors
├── ...
├── model.safetensors.index.json
│
├── tokenizer.json
├── tokenizer_config.json
│
├── chat_template.jinja
│
├── README.md
└── training_manifest.json
Preserve the Qwen-compatible format.
Then:
Transformers
vLLM
SGLang
can load it without our application being required.
40. Milestones I would actually implement
MilestoneResultM0 — BaselineBase + official Instruct running, complete eval suiteM1 — Dataschemas, provenance, filtering, dedup, code executionM2 — SFTfirst useful general chat + coding assistantM3 — Tool usefiles/shell/git/tests agentM4 — DPOimproved instruction/style/general-chat behaviorM5 — RLVRsingle-turn code improvement via executable rewardsM6 — Agent RLmulti-turn repo/terminal trainingM7 — Domain specializationoptional specialist experts/adaptersM8 — Distillationconsolidate strongest behaviors if usefulM9 — ProductionvLLM API, sandbox workers, monitoringM10 — Iteratedata → train → evaluate → deploy loop
The order matters.
I would not start with continued pretraining.
I'd start with:
baseline
    ↓
SFT
    ↓
agent tools
    ↓
RLVR
and only add continued mid-training once we have evidence that the base model lacks knowledge/representations that SFT cannot provide.
Qwen itself used continued pretraining → SFT → expert specialization → distillation, but we're beginning from the already code-specialized Coder-Next Base, not the more general Qwen3-Next Base that their Coder project started from. That makes a large second mid-training run less urgent for us. (Qwen)
41. The final architecture of the project
The neural architecture is Qwen's.
Our intellectual contribution is everything surrounding it:
                     Qwen3-Coder-Next-Base
                              │
                              ▼
                     High-quality SFT
                              │
                              ▼
                         Preference
                              │
                              ▼
                      Verifiable RL
                              │
                              ▼
                       Agentic RL
                              │
                              ▼
                       FrontierCoder
                              │
            ┌─────────────────┼────────────────┐
            ▼                 ▼                ▼
           Chat             Coding           Agent
            │                 │                │
            └─────────────────┼────────────────┘
                              │
                     Minimal tool runtime
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
               files        terminal        git
                              │
                           compiler
                              │
                            tests
                              │
                         verification
The core decision
For the Use Track, I would now freeze these decisions:
Base:
    Qwen/Qwen3-Coder-Next-Base
Model architecture:
    unchanged
Tokenizer:
    unchanged
Language:
    Python
Framework:
    PyTorch + Transformers
Training:
    TRL + PEFT + Accelerate
Initial adaptation:
    LoRA
Later:
    selective/full fine-tuning only if justified
Alignment:
    SFT → DPO → GRPO/RLVR
Agent:
    our own minimal loop
Tools:
    files + search + shell + tests + git
Environment:
    Docker sandbox
Evaluation:
    executable-first
Serving:
    vLLM
Alternative:
    SGLang
Packaging:
    safetensors + HF-compatible config/tokenizer/template
Main optimization target:
    data quality + executable feedback + agent trajectories
NOT:
    redesigning attention/Transformer architecture
That is the plan I would now consider the production architecture, not a PoC. The model itself is already sophisticated enough; the highest-return work is reproducing the part of Qwen's recipe that made Coder-Next strong: verified data → SFT → tools → executable environments → RL → evaluation → repeat. (arXiv)



------------------------

1. Learn Track (MiniFrontier / educational core)
Raw PyTorch, fully from-scratch, readable in an afternoon.

Sizes: 50M (debugger/CPU), 150M (main teaching model), 500M (architecture experiments), optionally 1–1.5B.
V1 architecture (Edu preset):
Decoder-only · RMSNorm (pre-norm) · RoPE · full MHA · causal attention · SwiGLU · tied embeddings · KV cache.
Modern preset (built by changing components, not rewriting):
Add GQA → QK-Norm (as a flag) → 3-local / 1-global hybrid (local = sliding or local attention, global = full) → optional NoPE on global layers.
Progression of experiments (do these after basic training + evaluation work):
MHA → GQA → local/global hybrid → Gated DeltaNet hybrid → MLA → light MoE.
Tokenizer: fixed byte-aware BPE with ~16k vocab for all Learn sizes (clean scaling comparisons). Reserve special tokens (chat, FIM, tools) from the start.
Optimizers: AdamW first, then A/B with Muon as an experiment.
Data/eval are first-class: train/val loss + bits-per-byte + CORE-style + HumanEval-like + FIM + simple tool-format checks before exotic architecture work.
Start by studying and running nanochat (Karpathy) end-to-end, then write your own clean version.

2. Use Track (FrontierCoder / practical model)
Do not redesign the architecture. Inherit a strong open base and put effort into data, post-training, verification, tools, and agents.
Best current bases (verified as of now):

Primary coding-focused: Qwen/Qwen3-Coder-Next-Base (80B total / 3B active, hybrid 3× Gated DeltaNet + 1× Gated Attention repeated, 512 experts top-10 + 1 shared, 256k context, Apache-2.0). Do not change its architecture, tokenizer, or embeddings.
Strong fully-open alternative: OLMo-3-7B (or 32B) from Ai2 — complete training recipes, intermediate checkpoints, and code make it excellent for learning real-lab practices.
Others (Muse Glimmer, smaller Qwen3 variants, etc.) as secondary references.

Pipeline (LoRA-first):

Establish full baseline evaluation on Base + official Instruct sibling.
Light mid-training only if needed (mostly natural high-quality code/docs + small synthetic).
SFT (mix of general chat + coding + multi-turn agent trajectories + tool formats).
Preference (DPO) for style/helpfulness.
Verifiable RL (GRPO / RLVR) with executable rewards (compile, tests, patch apply, tool correctness).
Multi-turn agent RL in Docker sandboxes.
Minimal agent loop + tools (files, search, shell, tests, git).
Serve with vLLM (primary) or SGLang.

Keep the neural net frozen; your value is in data quality, executable verification, agent trajectories, and evaluation discipline.
Stack (shared, minimal)

Python 3.12 + uv
Core: PyTorch, Transformers, safetensors, tokenizers, datasets
Training: TRL + PEFT (LoRA/QLoRA) + Accelerate (FSDP2 only when needed)
Eval: lm-eval-harness + custom executable coding/agent suites
Serving: vLLM (default)
Agent: your own tiny loop + Docker sandboxes
Explicitly not in V1: Keras, Triton kernels you write, TorchTitan, DeepSpeed, Ray, LangChain, classical ML libraries, etc.


----------

**Focus on the Edu model. Shelve the Use Track.**
That's the answer to your actual question, and it's not a compromise — it's what your original goal was. You said you wanted something small enough to understand and explain when teaching others. The Learn Track *is* that. The Use Track is a different project that happens to also involve LLMs.

------------


The Verdict: Yes, this two-track approach is the definitive, industry-correct way to build this project.

Trying to build a single model that is simultaneously a simple, line-by-line
educational toy and a frontier-grade coding and chat assistant is an impossible
contradiction.

  - To teach and understand deeply: The architecture must be minimal (<300 lines
    of PyTorch), inspectable, transparent, and trainable on a single GPU
    (50M–500M parameters).
  - To compete with modern frontier models for coding/chat: You need trillions
    of tokens of pretraining, extensive software-engineering distributions,
    multi-turn tool calling, verifiable execution feedback, and parameter scale
    (3B to 80B MoE).

The split into Track A (MiniFrontier / Learn) and Track B (FrontierCoder / Use)
resolves this tension without compromising either goal.

                            ┌────────────────────────────────────────┐
                            │          MiniFrontier Project          │
                            └───────────────────┬────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │   Track A: Learn Track      │                   │    Track B: Use Track       │
        │      (MiniFrontier)         │                   │      (FrontierCoder)        │
        ├─────────────────────────────┤                   ├─────────────────────────────┤
        │ • 50M – 500M pure PyTorch   │                   │ • Base: Qwen3-Coder / OLMo  │
        │ • Written line-by-line      │                   │ • Freeze model architecture │
        │ • Teach RoPE, GQA, KV-cache │                   │ • Focus: SFT, DPO, RLVR     │
        │ • Benchmark AdamW vs Muon   │                   │ • Execution Docker Sandbox  │
        │ • Understand neural math    │                   │ • Real-world coding agent   │
        └─────────────────────────────┘                   └─────────────────────────────┘

Part 1: How the Two Tracks Complement Each Other

| Dimension        | Track A: Learn Track (`MiniFrontier`) | Track B: Use Track (`FrontierCoder`)             |
| :--------------- | :------------------------------------ | :----------------------------------------------- |
| **Objective**    | Neural mechanics & conceptual mastery | Real-world coding & software engineering         |
| **Model Size**   | 50M – 500M (optional 1.5B)            | 80B-Total / 3B-Active (or 7B Dense)              |
| **Foundation**   | From scratch (`nn.Module`)            | Frozen base (`Qwen3-Coder-Next-Base` / `OLMo-3`) |
| **Core Work**    | Tensor shapes, RoPE, GQA, Cache, Muon | Data quality, SFT, RLVR (GRPO), Agent Sandboxing |
| **Compute Cost** | Single consumer GPU (hours/days)      | Multi-GPU / LoRA / PEFT + Docker cluster         |
| **Output**       | Educational repository & lecture code | High-throughput coding agent via `vLLM`          |

Part 2: Track A Deep-Dive (The Learn Track)

In the Learn Track, avoid exotic upstream kernels (like custom Triton or fused
CUDA kernels). The entire forward pass, KV cache, and optimizer must be readable
in an afternoon.

The Educational Progression

1.  Edu Preset (Classic Foundation):
      - Pre-Norm RMSNorm + RoPE + Multi-Head Attention (MHA) + SwiGLU + Tied
        Embeddings.
      - Lesson: How residual streams, attention heads, and autoregressive
        generation mathematically work.
2.  Modern Preset (Frontier Evolution):
      - Swap MHA \to Grouped Query Attention (GQA).
      - Add QK-Norm (RMSNorm on Query/Key projections before RoPE to stop
        gradient explosions).
      - Implement Fill-in-the-Middle (FIM) tokens (<|fim_prefix|>,
        <|fim_suffix|>, <|fim_middle|>).
      - Lesson: How modern models cut KV-cache memory and stabilize training.
3.  The Optimizer Lab (AdamW vs. Muon):
      - Benchmark standard torch.optim.AdamW against a pure-Python
        implementation of the Muon optimizer (which applies matrix
        orthogonalization via Newton-Schulz iterations on 2D linear weights).
      - Lesson: Why modern pretraining speedruns achieve higher sample
        efficiency.

Part 3: Track B Deep-Dive (The Use Track / FrontierCoder)

For the Use Track, the neural architecture is completely frozen. The code relies
on Qwen3-Coder-Next-Base (or OLMo-3-7B-Base) and uses official upstream
implementations (causal-conv1d and flash-linear-attention).

All engineering effort is redirected into Post-Training and Verifiable
Reinforcement Learning:

[ Pretrained Base Model ]
           │
           ▼
[ Phase 1: High-Quality SFT ] ──► (Conversations, FIM, Patch formatting, Tool calls)
           │
           ▼
[ Phase 2: Preference Alignment (DPO) ] ──► (Tone, instruction following, concise formatting)
           │
           ▼
[ Phase 3: Verifiable RL (RLVR / GRPO) ] ──► (Reward = Code compiles & unit tests pass in Sandbox)
           │
           ▼
[ Phase 4: Autonomous Agent Loop ] ──► (Filesystem, Grep, Git Diff, Terminal execution)

1. Why RLVR (Reinforcement Learning with Verifiable Rewards) Wins for Coding

General chat models require fragile LLM-as-a-judge scoring. Coding does not:

  - Binary Truth: The Python interpreter or compiler either passes or fails.
  - Test Suites: Unit test coverage directly defines the reward R \in \{0, 1\}.
  - Syntax/Security Verification: Malformed tool calls or dangerous shell
    commands receive an immediate penalty.

Using TRL's GRPOTrainer, the model generates multiple candidate rollout
trajectories for a programming issue. Candidates that pass compiler checks,
linting, and automated tests are reinforced without subjective human scoring.

2. The Agentic Tool & Sandbox Runtime

The agent loop does not require a bulky framework (like LangChain). It is a
clean, stateful loop wrapped around Docker containers:

# Minimal Conceptual Agent Loop
def run_agent_loop(model, tokenizer, task_env, max_steps=10):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task_env.prompt}]
    
    for step in range(max_steps):
        response = model.generate(messages, tools=task_env.available_tools)
        messages.append(response.message)
        
        if response.is_complete:
            break
            
        # Execute tool safely inside containerized sandbox
        obs = task_env.execute_tool(response.tool_name, response.tool_args)
        messages.append({"role": "tool", "name": response.tool_name, "content": obs})
        
    return task_env.verify_resolution()

Part 4: Which Should You Focus on First?

Follow this concrete, 4-stage roadmap:

Month 1                                       Month 2
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  STAGE 1: Learn Track        │              │  STAGE 3: Use Track SFT      │
│  • Write 150M MiniFrontier   │              │  • Prepare SFT dataset       │
│  • Verify RoPE, GQA, Cache   │              │  • Run LoRA SFT on Base      │
│  • Train on toy dataset      │              │  • Build Jinja2 ChatML       │
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  STAGE 2: Evaluation Suite   │              │  STAGE 4: Verifiable RL      │
│  • Set up lm-eval-harness    │              │  • Build Docker sandbox      │
│  • Set up HumanEval/FIM tests│              │  • Run GRPO test-passing loop│
│  • Benchmark Base checkpoint │              │  • Deploy on vLLM            │
└──────────────────────────────┘              └──────────────────────────────┘

1.  Weeks 1–2 (Build MiniFrontier-Edu): Write the 150M model in raw PyTorch.
    Understand every tensor transformation, RoPE rotation, and KV-cache index.
    Train it on a small sample of FineWeb-Edu/StarCoder.
2.  Weeks 3–4 (Establish Scoreboards): Set up the evaluation suite (lm-eval +
    executable code verification) and evaluate the unmodified
    Qwen3-Coder-Next-Base and Qwen3-Coder-Next checkpoints to establish your
    baseline scoreboards.
3.  Weeks 5–6 (Run Use-Track SFT): Implement your structured JSON/Jinja2 dataset
    schema. Run LoRA-based SFT on your domain-specific code and chat mixtures
    using TRL.
4.  Weeks 7+ (RLVR & Agent Sandboxing): Set up Docker execution sandboxes,
    connect GRPOTrainer with automated test runners, and optimize the model on
    executable bug-fix and repo-patching tasks.

Summary

  - Do not discard the educational model: It gives you the fundamental
    understanding needed to debug architectures, attention masks, and
    optimizers.
  - Do not try to pretrain a 7B+ model from scratch: Use Qwen3-Coder-Next-Base
    (or OLMo-3-7B) for Track B, and put 100% of your production efforts into
    data quality, verifiable RL (GRPO), and sandboxed execution.


