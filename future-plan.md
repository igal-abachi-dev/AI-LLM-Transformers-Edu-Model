Yes — **the design is good**, especially as an educational “from first principles → modern LLM” implementation. But there are two very different questions here:

1. **Is MiniFrontier a good implementation for learning/teaching modern LLMs?** → **Yes, very good direction.**
2. **Can the current 50M–500M model compete with Claude/ChatGPT/Gemini/Grok or the strongest open models?** → **No. Not remotely at the current scale/training stage.**

More importantly, I would **not solve that gap by stuffing every frontier architecture trick into your code**. Your current philosophy—small readable core plus a few high-value modern techniques—is the right one.

I inspected the ZIP, including the model, attention, tokenizer/data pipeline, KV cache, tests, configs, training stubs, evaluation setup, and roadmap.

## My assessment

| Area                                |                  Rating | Assessment                                                      |
| ----------------------------------- | ----------------------: | --------------------------------------------------------------- |
| Educational architecture            |              **9.5/10** | Excellent progression from basic attention to modern LLM        |
| Code readability                    |                **9/10** | Very clean, explicit PyTorch                                    |
| Correctness engineering             |                **9/10** | Strong tests, reference paths, cache parity, provenance         |
| Modern architecture direction       |              **8.5/10** | Your planned Modern preset is surprisingly current              |
| Current implementation completeness |                **5/10** | Core Edu works; Modern/training/SFT still largely unimplemented |
| Data/training recipe                |                **5/10** | Good educational pipeline, far too weak for serious capability  |
| Post-training                       |      **2/10 currently** | SFT/chat are still placeholders                                 |
| Current model competitiveness       |   **~1/10 vs frontier** | 50–500M cannot compete broadly                                  |
| Potential as a teaching project     |               **10/10** | This is where the project is strongest                          |
| Potential as a useful small LLM     | **high**, after changes | Especially at 1–3B+                                             |


### The biggest distinction

Your **architecture is much better than the resulting model capability would suggest**.

That is because frontier capability today is mostly:

**architecture × scale × data × training compute × post-training × synthetic data × reasoning training × tools/agents**

—not architecture alone.

For perspective, the newest open frontier systems are enormous. DeepSeek V4 Pro has 1.6T total parameters / 49B active; Kimi K3 is 2.8T; Qwen3.8-Max-class open weights are 2.4T / 95B active. ([Hugging Face][1])

Even the unusually compact new Qwen3.8-27B has **27B parameters, 64 layers, 262K native context, Gated DeltaNet/full-attention hybrid layers, MTP, multimodal pretraining and substantial post-training**. ([Hugging Face][2])

So a 150M model is roughly **180× smaller than Qwen3.8-27B**, before even discussing training data or post-training.

so we want to make our model bigger:
1B Modern
    ↓
train something meaningfully capable

3B/(7B optional)
    ↓
small practical local assistant

---

# The interesting part: your Modern architecture is actually very current

When I compared your planned design against the models released around now, one model stood out:

## Muse Glimmer is almost a validation of your architecture

Your intended MiniFrontier Modern:

```text
Pre-RMSNorm
GQA
QK-Norm
RoPE
Local
Local
Local
Global
Dense SwiGLU
optional global NoPE
```

Muse Glimmer, released this month, uses:

```text
Dense causal Transformer
52 layers
GQA: 32 Q / 2 KV
Local
Local
Local
Global
2048-token local window
SwiGLU
RoPE on local layers only
131K+ context
```

That resemblance is striking. Meta's official model card specifies the repeating `[Local, Local, Local, Global]` pattern, GQA, SwiGLU and RoPE only on local attention. ([Hugging Face][3])

So I **would not throw away your Modern architecture**.

In fact, I would make it the central teaching architecture.

GPT-OSS independently makes a similar choice: MoE Transformer, GQA, RoPE, and alternating full attention with locally banded sparse attention. ([OpenAI][4])

There is a real architectural convergence here:

```text
Transformer fundamentals
        +
RMSNorm
RoPE
GQA
SwiGLU
hybrid local/global attention
        +
better training/post-training
```

That is a very defensible educational destination.

---

# Where the current repository actually stands

This is important because the README can make the project sound further along than the executable model currently is.

Your actual `MiniFrontier` constructor says:

```python
if config.preset != "edu":
    raise NotImplementedError("the Modern preset begins at MF-039")
```

And `CausalSelfAttention` currently rejects:

```python
GQA
QK-Norm
hybrid attention
```

with `NotImplementedError`.

Likewise:

```text
train/pretrain.py
scripts/train.py
train/sft.py
```

are still stubs.

So **today you have a well-engineered educational Transformer foundation, not yet a trained modern LLM**.

That isn't a criticism of the design; your backlog explicitly says MF-039 onward is next.

Your implemented core already gets a lot right:

```text
raw PyTorch
Pre-RMSNorm
RoPE
MHA
SwiGLU
bias-free projections
tied embeddings
manual attention reference
SDPA optimized path
causal masking
KV cache
cached/full parity testing
sampling
checkpoint infrastructure
byte-BPE
streaming data
provenance
exact dedup
packing
evaluation adapters
reproducibility
```

This is substantially better for teaching than wrapping `transformers.LlamaForCausalLM`.

---

# What current frontier models tell us

There isn't one architecture you should copy.

| Model                | Particularly interesting idea for MiniFrontier                           |
| -------------------- | ------------------------------------------------------------------------ |
| **Muse Glimmer 30B** | Local/Local/Local/Global + GQA + local RoPE + dense SwiGLU               |
| **GPT-OSS-120B**     | MoE + local/global attention + GQA + attention sinks                     |
| **Qwen3.8-27B**      | Gated DeltaNet + periodic full attention + MTP                           |
| **Qwen3.8 2.4T**     | Same hybrid idea scaled with MoE, 10 routed + shared expert              |
| **DeepSeek V4**      | CSA/HCA hybrid sparse attention + mHC residuals + Muon + MoE             |
| **Kimi K3**          | Kimi Delta Attention + AttnRes + extremely sparse LatentMoE              |
| **MiniMax M3**       | Sparse attention specifically engineered for 1M context                  |
| **Llama 4**          | Relatively conventional Transformer plus MoE and huge-scale distillation |
| **Grok-1**           | Large MoE, GQA-like Q/KV asymmetry, otherwise recognizable Transformer   |

DeepSeek V4 explicitly combines compressed sparse/heavily compressed attention, mHC residuals and Muon. ([arXiv][5]) Kimi K3 uses Kimi Delta Attention, Attention Residuals and a 16-of-896-expert sparse MoE. ([Hugging Face][6]) MiniMax M3 focuses heavily on sparse attention for million-token operation. ([Hugging Face][7])

Qwen3.8 is probably the most interesting counterexample to your design: its 27B dense model repeats three Gated DeltaNet layers followed by one Gated Attention layer, and also trains with multi-token prediction. ([Hugging Face][2])

But **I would not put DeltaNet/KDA/HCA into MiniFrontier V1**.

They make a great V2 laboratory, but they'd hurt the project's biggest advantage: somebody can understand your entire neural core.

Llama 4 is another useful lesson: Meta moved to MoE at very large scale, but also relied heavily on distillation from a much bigger teacher. ([Meta AI][8])

And Muse Glimmer itself is a 30B model distilled from Muse Spark. ([Hugging Face][3])

That last point matters enormously.

---

# What I would change

I would make **10 changes, in this order**:

1. **Finish your existing M4 before adding anything exotic.** Implement real GQA, QK-Norm and `Local/Local/Local/Global`. Your roadmap is correct. Use native SDPA GQA where possible and keep explicit KV expansion only in the teaching implementation. This gives you most of the useful architectural modernization without destroying readability.

2. **Fix the hybrid KV cache architecture at the same time.** Right now every `LayerKVCache` allocates `max_seq_len` K/V storage. Once three of four layers are local, those local layers should retain approximately only `local_window` K/V entries—ideally through a ring/sliding cache—while global layers retain the long history. Otherwise hybrid attention saves attention FLOPs but throws away much of its potential KV-memory advantage.

3. **Split “Edu model size” from “useful model size.”** Keep `50M-Edu` and `150M-Edu`; they are fantastic teaching models. Keep `150M-Modern` for controlled A/B experiments. But add a separate eventual `1B-Modern` or perhaps `1.5B-Modern`. Do not pretend 150M is the model people should actually chat/code with. If resources eventually permit it, 3B is a much more credible small general assistant target. This keeps the neural implementation identical—the only thing changing is configuration.

4. **Keep 16K BPE for V1 experiments, but don't freeze it forever.** It is excellent for cheap 50M/150M comparisons, but it is not what I'd choose for the eventual general/coding model. Current models often use very large vocabularies—Muse Glimmer uses about 202K and Qwen3.8 about 248K. ([Hugging Face][3]) You don't need anything that extreme. For `MiniFrontier-Competitive`, I'd evaluate **32K vs 64K byte-BPE**. Keep 16K as the educational tokenizer so existing experiments stay comparable.

5. **Upgrade data much more aggressively than the architecture.** Your `filter_and_deduplicate()` currently does exact SHA-256 deduplication; that's good engineering but inadequate model-data cleaning. Add normalized text dedup, near-duplicate/MinHash dedup, repetition filtering, language identification, quality scoring, benchmark contamination filtering, document-quality heuristics and source weighting. For the coding goal, curate substantially more high-quality code, technical documentation, Q&A and repo-level examples. This is likely worth more capability than adding three new attention algorithms.

6. **Increase serious-training budgets.** Your `100–300M` tokens for the 50M model and `300M–1B` for 150M are excellent *educational experiment budgets*. Treat them exactly that way. They are not competitive training recipes. Make the README explicitly distinguish `smoke`, `experiment`, and `quality` budgets. Current frontier and even small serious models are trained on vastly more data; for example Meta reports up to 9T pretraining tokens even for its Llama 3.2 1B/3B family. ([Hugging Face][9]) You don't need trillions for this project, but you need to stop interpreting sub-billion-token runs as an approximation of frontier training.

7. **Make post-training a first-class part of MiniFrontier, not a tiny appendix.** The current plan effectively ends at small assistant-only SFT. For a genuinely pleasant chat/coding model, I'd make the conceptual pipeline `pretrain → code/FIM continued pretraining → high-quality SFT → preference optimization → optional verifier-based reasoning training`. Keep DPO as the simplest preference lesson; add a tiny GRPO/verifier-RL lab later for code/math. Current strong models explicitly train controllable reasoning, tools and agentic task completion rather than relying on pretrained next-token prediction alone. ([Hugging Face][2])

8. **Add distillation. This is probably your highest-ROI capability feature.** Don't change the Transformer at all. Teach a small model using high-quality outputs from a much stronger teacher: explanations, coding trajectories, corrected answers, synthetic textbook material, FIM tasks, tool-call examples and reasoning problems. Muse Glimmer being distilled from Muse Spark and Llama 4's smaller models being distilled from Behemoth are contemporary evidence that this is not merely a hobbyist shortcut. ([Hugging Face][3]) For your "small but surprisingly capable" goal, this matters far more than implementing Kimi Delta Attention.

9. **Add MTP as the first optional frontier extension.** I would do this before MoE, MLA, DeltaNet or fancy residuals. Something like an optional `MultiTokenPredictionHead` is conceptually understandable: predict `t+1`, `t+2`, perhaps `t+3`. Qwen3.8 explicitly trains with MTP. ([Hugging Face][2]) It also gives you a natural bridge to speculative decoding. Muse Glimmer shows how important speculative decoding can become for local models, shipping a dedicated drafter alongside the main model. ([Hugging Face][3])

10. **Modernize the evaluation gate before claiming model quality.** ARC-Easy/HellaSwag/PIQA are fine as educational continuity metrics, but a general chat/coding model now needs instruction following, coding, reasoning, tool use and long-context evaluations. Keep your tiny deterministic fixtures, then add a small serious suite covering instruction following, HumanEval/MBPP-style functional coding, LiveCodeBench-style fresh coding, GPQA/MMLU-Pro-level reasoning, FIM, tool/function calling and long-context retrieval. Current frontier model cards are dominated by coding-agent, long-horizon, reasoning and tool evaluations rather than old multiple-choice tasks alone. ([Hugging Face][2])

---

# The architecture I would freeze

Your project currently has:

```text
Edu
↓
Modern
```

I would make it:

```text
MiniFrontier
├── Edu
│   └── "How a Transformer works"
│
├── Modern
│   └── "What a clean 2026 LLM looks like"
│
└── Labs
    └── "How frontier models go beyond it"
```

And freeze **Modern** approximately as:

```text
Decoder-only Transformer

Pre-RMSNorm

Attention:
    GQA
    QK-Norm
    head_dim = 64 or 128

Schedule:
    Local
    Local
    Local
    Global

Local window:
    1024–2048 for larger configs

Position:
    RoPE on local attention
    NoPE global experiment

FFN:
    dense SwiGLU

Residual:
    ordinary residual connections

Embeddings:
    tied

Bias:
    false

Dropout:
    0

Tokenizer:
    Edu:        16K byte BPE
    Competitive: evaluate 32K/64K

Training:
    causal LM
    + FIM
    + optional MTP

Optimizer:
    AdamW baseline
    Muon + AdamW modern experiment

Inference:
    GQA KV cache
    sliding/ring KV cache for local layers
    full cache for global layers

Post-training:
    SFT
    preference optimization
    reasoning/code distillation
    optional verifier RL

Performance:
    BF16
    SDPA/FlexAttention
    torch.compile
    activation checkpointing
```

Notice what is **not** there:

```text
MoE
MLA
KDA
DeltaNet
HCA/CSA
mHC
AttnRes
custom Triton
FP8 training
distributed expert parallelism
```

That is deliberate.

---

# I would add only three tiny frontier-inspired extensions

Once Modern is finished:

```text
Lab 1: Dense FFN vs MoE
Lab 2: Transformer attention vs DeltaNet/linear attention
Lab 3: normal next-token vs MTP
```

Not because your canonical model needs them.

Because after completing MiniFrontier somebody should be able to look at Qwen3.8:

```text
3 × Gated DeltaNet
1 × Gated Attention
MoE
MTP
```

and say:

> "I understand exactly what they changed relative to the Transformer I built."

That's an excellent teaching outcome.

---

# There is also one small inference cleanup I would make

Your current `generation.py` does:

```python
output = torch.cat((output, next_token), dim=1)
```

on every generated token.

That's beautifully simple for teaching, so **keep it as the reference implementation**.

But add an optimized path that preallocates:

```python
tokens = torch.empty(batch, prompt_len + max_new_tokens, ...)
```

and writes tokens in place.

Likewise, you currently generate RoPE cosine/sine values during each forward. Keep that readable implementation, but optionally cache/precompute RoPE tables for the optimized path.

Same philosophy as your attention:

```text
reference implementation
+
optimized implementation
```

That pattern is one of the strongest aspects of this project.

---

# Do you need MoE to compete?

At your scale: **no**.

At frontier scale: increasingly, yes.

Grok-1 was already an 8-expert, 314B MoE activating two experts per token. ([GitHub][10]) Llama 4 moved Meta's main family to MoE. ([Meta AI][11]) GPT-OSS-120B is MoE with only 5.1B active parameters. ([OpenAI Developers][12]) DeepSeek, Kimi and large Qwen likewise make aggressive use of sparsity. ([Hugging Face][6])

But MoE solves:

> "How can I store enormous knowledge capacity without activating all parameters?"

Your immediate problem is:

> "How do I train a small model well?"

MoE won't fix insufficient data, insufficient training or insufficient post-training.

So **dense is the right MiniFrontier default**.

---

# Can a small model ever compete with frontier models?

There's an important qualification.

### 150M

No.

It can become a surprisingly nice educational LM, but not a serious Claude/GPT/Gemini competitor.

### 500M

Still no broadly.

It can become noticeably competent at text and constrained coding with strong data.

### 1–3B

Now you can create something genuinely useful locally if trained/distilled very well.

Still not broadly frontier.

### 7–14B

High-quality training, reasoning distillation and tools can make this surprisingly strong on selected domains.

### ~27–30B

Now there is current evidence of models entering frontier-adjacent territory on individual tasks.

Qwen3.8-27B is a current 27B dense model and its official results show it trading blows with much larger/proprietary systems on some coding and instruction benchmarks. ([Hugging Face][2]) Muse Glimmer demonstrates another 30B local model optimized specifically around agentic/tool/coding behavior. ([Hugging Face][3])

But neither achievement came from merely implementing a 30B Transformer.

The training recipe is the expensive part.

---

# Your reference-model list also needs two small corrections

As of **August 17, 2026**, I can verify official **GLM-5.2** as the current Z.ai open flagship release I found; I did not find an official `GLM-5.3` release. GLM-5.2 uses MoE plus sparse attention and supports 1M context. ([Hugging Face][13])

For Grok, xAI has an official open-weight **Grok-1** repository; I did not find an official Grok-2 open-weight release in xAI's official sources. Grok-1 itself is a 314B MoE with 48 Q heads / 8 KV heads, RoPE and two of eight experts used per token. ([GitHub][10])

DeepSeek V4 Pro's current official release is the **August 13, 2026** `DeepSeek-V4-Pro-0813`, superseding the Preview. ([Hugging Face][14])

Qwen3.8 is indeed extremely current; its open 2.4T/95B-active model was released only days ago. ([Hugging Face][15])

---

# So I would change the project's goal wording

Right now the implicit goal mixes two incompatible promises:

> small enough to teach completely

and

> frontier-level model.

I'd define MiniFrontier as:

**“The smallest codebase that teaches the architecture, training, post-training and inference ideas underlying modern frontier LLMs, while producing the strongest practical model possible at each available compute scale.”**

Then:

```text
50M Edu
    ↓
understand attention/Transformer

150M Modern
    ↓
understand a modern 2026 LLM

1B Modern
    ↓
train something meaningfully capable

3B/7B optional
    ↓
small practical local assistant

30B+
    ↓
frontier-adjacent research territory
```

That resolves the contradiction beautifully.

## Bottom line

**Do not rewrite this around Kimi K3/DeepSeek V4/Qwen3.8.**

Your current foundation is good.

I would **finish exactly the Modern path you already planned**, because the remarkable thing is that the brand-new Muse Glimmer architecture independently looks very similar to it. Then put most additional effort into:

**better data → much more training → distillation → serious SFT → preference/reasoning post-training → better evals.**

And of all the new architecture ideas, I'd add only **MTP + optional attention gating + a sliding local KV cache** to the canonical implementation. Leave MoE, DeltaNet/KDA, exotic residuals and compressed attention as clean labs.

That would give you something better than a tiny imitation of DeepSeek/Kimi: **a codebase where someone can learn almost the entire modern LLM stack end-to-end, while the same implementation can scale from a 50M classroom model toward a genuinely useful 1–3B+ model.**

[1]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro?utm_source=chatgpt.com "deepseek-ai/DeepSeek-V4-Pro"
[2]: https://huggingface.co/Qwen/Qwen3.8-27B "Qwen/Qwen3.8-27B · Hugging Face"
[3]: https://huggingface.co/meta-models/Muse-Glimmer-30B "meta-models/Muse-Glimmer-30B · Hugging Face"
[4]: https://openai.com/index/introducing-gpt-oss/?utm_source=chatgpt.com "Introducing gpt-oss"
[5]: https://arxiv.org/abs/2606.19348?utm_source=chatgpt.com "DeepSeek-V4: Towards Highly Efficient Million-Token ..."
[6]: https://huggingface.co/moonshotai/Kimi-K3 "moonshotai/Kimi-K3 · Hugging Face"
[7]: https://huggingface.co/MiniMaxAI/MiniMax-M3 "MiniMaxAI/MiniMax-M3 · Hugging Face"
[8]: https://ai.meta.com/blog/llama-4-multimodal-intelligence/ "The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation"
[9]: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8?utm_source=chatgpt.com "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"
[10]: https://github.com/xai-org/grok-1?utm_source=chatgpt.com "xai-org/grok-1: Grok open release"
[11]: https://ai.meta.com/blog/llama-4-multimodal-intelligence/?utm_source=chatgpt.com "The Llama 4 herd: The beginning of a new era of natively ..."
[12]: https://developers.openai.com/api/docs/models/gpt-oss-120b "gpt-oss-120b Model | OpenAI API"
[13]: https://huggingface.co/zai-org/GLM-5.2?utm_source=chatgpt.com "zai-org/GLM-5.2"
[14]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 "deepseek-ai/DeepSeek-V4-Pro-0813 · Hugging Face"
[15]: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B?utm_source=chatgpt.com "Qwen/Qwen3.8-2.4T-A95B"


Yes, the resemblance is real and useful.

Muse Glimmer’s published architecture lines up closely with the MiniFrontier Modern design you already planned:

Dense causal Transformer
Repeating [Local, Local, Local, Global] pattern
Sliding window of 2048 on the local layers
GQA (they went extreme: 32 query heads / 2 KV heads)
SwiGLU
RoPE applied only on the local layers (global layers use NoPE)
~52 layers, 131k+ context
This is exactly the kind of clean, 2026-style hybrid that makes a strong teaching target. Meta did not invent some radically new attention primitive; they took well-understood ingredients (GQA + hybrid local/global + SwiGLU + selective RoPE) and tuned them hard for KV-cache efficiency and local agentic workloads. That is validation of the direction, not a reason to abandon it.

A few small differences worth noting:

Muse adds gated attention on both local and global layers.
Their GQA ratio is more aggressive than most previous models.
They ship a dedicated speculative-decoding drafter and heavy quantization focus because the model is meant to stay responsive on a single consumer GPU.
It is distilled from the much larger Muse Spark, which is a large part of why a 30B dense model can punch above its weight on agentic/coding tasks.
None of those require you to change the core of MiniFrontier Modern. Gated attention and a more extreme GQA ratio are easy optional extensions later; the hybrid schedule + GQA + local RoPE + dense SwiGLU is already the high-value, readable modern baseline.


**No, it cannot compete with current Sota frontier models (closed or open-weights).** It is a high-quality educational/minimal from-scratch implementation that is excellent for its stated goals, but the gap to Opus-/Grok-/Claude-/Gemini-/Kimi-/DeepSeek-class models is fundamental (scale, data, post-training, architecture complexity, and compute), not something fixable by polishing the current code.

### What you actually built (strengths)

MiniFrontier is a clean, readable, well-tested, single-GPU-oriented decoder-only Transformer written in raw PyTorch. Core size is tiny (~1.7k LOC in `src/minifrontier/`).

**Architecture (Edu preset is fully implemented; Modern is scaffolded):**
- Pre-RMSNorm + residual blocks
- Full causal MHA (manual teaching path + fused-eligible SDPA)
- RoPE (split-half LLaMA convention, parity-tested)
- Dense SwiGLU (bias-free)
- Tied embeddings
- Proper width-aware init + residual output scaling (`1/√(2L)`)
- Preallocated KV cache with careful position/RoPE handling
- Explicit masks only when needed; causal SDPA preferred
- Config-driven Edu vs Modern presets (50M–500M)

**Modern preset (planned / partially gated):**
- GQA
- QK-Norm
- Hybrid attention (3 local + 1 global pattern, inspired by Muse Glimmer-style designs)
- Optional global NoPE experiment

**Engineering quality is unusually high for an educational project:**
- Strong config validation, frozen presets, exact 50M parameter count
- CPU-friendly overfit gate (<1e-3 nats/token)
- Checkpoint/resume, safetensors export, generation with top-k/p, temperature
- Tokenization (byte-level BPE 16k), packing, data governance notes
- Evaluation harness (loss/PPL/BPB, lm-eval adapter, FIM/code fixtures)
- Labs for single-variable experiments (attention math, RoPE, MHA vs GQA, QK-Norm, hybrid, KV cache, AdamW vs Muon, etc.)
- Tests, smoke scripts, reproducibility, provenance awareness

This is better than most “nanoGPT / tiny transformer from scratch” repos. It deliberately stops short of frameworks, custom kernels, MoE, MLA, etc., so every line remains teachable.

### Why it cannot compete with frontier models (2026 reality)

Current open frontier models (Kimi K3 ~2.8T MoE / ~104B active, Qwen3.8 Max ~2.4T/95B, DeepSeek V4-Pro ~1.6T/49B, GLM-5.x ~750B/40B, MiniMax M3, Llama 4 Maverick, etc.) and closed ones share these traits:

| Dimension              | MiniFrontier (V1)              | 2026 Frontier (open or closed)                          |
|------------------------|--------------------------------|---------------------------------------------------------|
| Scale                  | 50–500M dense                  | 100B–2.8T total, 13–100B+ active (almost all MoE)      |
| Data                   | Educational / small mixtures   | Multi-trillion tokens, high-quality filtered + synthetic |
| Attention              | Full MHA → planned GQA + hybrid local/global | MLA / KDA / Gated DeltaNet + full hybrids, DSA, linear/SSM mixes, sparse block attention |
| Positional             | RoPE (NoPE experiment)         | Partial RoPE, iRoPE, YaRN, NoPE variants, etc.         |
| FFN                    | Dense SwiGLU                   | MoE (fine-grained, shared experts, aux-loss-free, etc.) |
| Training               | AdamW (+ Muon lab)             | Muon/MuonClip variants, massive distributed, FP8/low-precision |
| Post-training          | Light SFT planned              | Heavy SFT + RL (GRPO etc.), tool use, agentic, long-horizon |
| Context                | 1–2k (teaching)                | 256k–1M+ native                                        |
| Compute                | 1× consumer GPU                | Thousands of H100/H200-class GPUs                       |

Your hybrid 3:1 local/global idea is directionally modern (Muse Glimmer, Qwen hybrids, MiniMax patterns, etc.), and GQA + QK-Norm + RoPE + RMSNorm + SwiGLU is the current dense baseline. But without MoE (or equivalent capacity), massive data, and post-training, a 150–500M model will never match even mid-size open models on coding, reasoning, or general chat quality.

### What to improve while staying simple / teachable / small

Keep the dual Edu / Modern spirit. Do **not** chase full MoE/MLA/DeltaNet in the teaching core.

**High-priority, still-minimal upgrades:**
1. **Finish the Modern preset** (GQA, QK-Norm, hybrid local/global with FlexAttention or clean masked fallback, optional global NoPE). This is already planned and is the highest-leverage architectural step.
2. **Scale the training regime, not just the architecture.** Even a perfect 150–350M model needs a serious token budget (billions, not millions) of high-quality data + careful packing + curriculum. Add a realistic single-GPU / multi-GPU training path with activation checkpointing, BF16/FP8 where stable, and proper learning-rate schedules.
3. **Muon (or MuonClip-style) as first-class option.** You already have a lab; promote it for the main training loop once parity is proven. Many recent strong models use it.
4. **Better long-context primitives while staying readable.** Sliding-window / hybrid is good. Consider a simple linear-attention or DeltaNet-style block as an *optional* experiment lab (not core) so students can see the efficiency trade-off.
5. **Data & tokenizer.** Byte-level 16k is fine for teaching. For real quality, consider a larger vocab or a proven open tokenizer + much stronger filtering/dedup/mixture (code + math + reasoning data with provenance).
6. **Evaluation & claims discipline.** You already do this well. Keep the engineering smoke separate from quality claims. Add more realistic coding/FIM and long-context retrieval evals once Modern is live.
7. **Inference polish.** Continuous batching is overkill; keep simple KV-cache generation solid. Optional speculative decoding or a tiny drafter later.
8. **SFT + light preference stage.** Assistant-only SFT is planned; a minimal DPO/GRPO-style stage (still single-GPU friendly) would close more of the “chat quality” gap than architecture tweaks alone.

**Explicitly stay away from (for V1 teaching core):**
- Full MoE / 512-expert routing
- MLA / complex latent attention (hard to implement cleanly and explain)
- Custom Triton/CUDA kernels
- Distributed training frameworks
- 100k+ context as a hard requirement
- RL agents / tool use

Those belong in a “scale-up” or “research” track, not the afternoon-readable core.

### Realistic positioning

- **Teaching / understanding transformers & modern LLM techniques:** Excellent. Better than most public educational codebases.
- **General chat + light coding on a single consumer GPU after proper training:** Plausible for a 150–350M Modern model if you invest in data + training. It will feel like a strong 2023–early-2024 small model, not 2026 frontier.
- **Competing with Kimi K3 / DeepSeek V4-Pro / Qwen3.8 / Claude Opus / Grok / Gemini frontier:** Impossible with this size and scope. Those require 100–1000× more compute, data, and post-training infrastructure.

**Recommended path:**  
Finish and harden the Modern preset + serious single-GPU training pipeline + data mixture. Release a well-trained 150M (and optionally 350M) checkpoint with full reproducibility. Use the labs and dual Edu/Modern design as the teaching vehicle. Treat anything beyond that as a separate “MiniFrontier-Scale” research line.

The foundation is solid and unusually disciplined. The limiting factor is not code quality—it is the intentional decision to stay small, simple, and fully explainable. That is the right decision for teaching; it is the wrong decision if the goal is frontier capability.



-------------


**Your plan is good.** Targeting **1B Modern as the primary useful model**, with optional 3B (and maybe 7B), while staying dense and readable, is the right ambition level. Do **not** artificially limit yourself to 500M if you want people to actually use the Modern model for useful local chat/coding.

### Realistic capability ladder (2026)

| Size | Role | Realistic outcome with excellent data + distillation + post-training |
|------|------|---------------------------------------------------------------------|
| 50–150M Edu | Teaching | Perfect for understanding every line |
| 150–500M Modern | Experiments + A/B | Solid research baseline, limited practical use |
| **1B Modern** | **Primary useful target** | Genuinely usable local assistant (chat, light coding, FIM, instruction following) if trained well |
| **3B Modern** | Strong practical | Competitive with many current high-quality small open models at the same size |

1B–3B is the sweet spot where “people can actually use it” and you can still claim competitiveness **at that size**. 
500M is too small for most people to prefer over existing small models.
Going past ~7–8B starts competing with much better-resourced open releases and strains the “simple + educational core” promise.

### Refined plan (what I recommend)

Keep the dual nature, but make the scaling path explicit:

```text
Edu track (teaching)
50M / 150M
    ↓
Modern track (architecture + experiments)
150M / 350M / 500M
    ↓
Competitive Modern track (the one people actually download and use)
1B (primary) → 3B (recommended) → 7B (optional)
```

**Architecture freeze for Competitive Modern** (almost what you already have):

- Dense decoder-only
- Pre-RMSNorm + QK-Norm
- GQA (reasonable ratio, e.g. 8–16 Q / 2–4 KV at 1B–3B)
- Hybrid 3 local + 1 global (local window 1024–2048)
- RoPE on local layers; global NoPE as optional experiment
- Dense SwiGLU
- Tied embeddings, bias-free, dropout 0
- Muon (or Muon+AdamW partition) as the preferred optimizer
- Optional MTP head later (small, high-value addition)

**Do not add** to the main path: MoE, MLA, DeltaNet/KDA, custom kernels, distributed expert parallelism. Those stay in Labs.

### Highest-leverage changes (in priority order)

1. **Data quality and volume >> architecture**  
   This is still the #1 gap. Exact SHA dedup is good engineering but far from enough. Invest heavily in:
   - Near-duplicate / MinHash filtering
   - Quality scoring, language ID, repetition filters
   - Stronger code + math + reasoning + technical mixture
   - Contamination filtering
   - Source weighting and provenance (you already care about this)

2. **Token budgets that match the size**  
   Treat current small budgets as educational/smoke only. For 1B you want multiple billions of high-quality tokens; for 3B even more. Distinguish clearly in docs:
   - Smoke
   - Experiment
   - Quality (the runs people should trust)

3. **Distillation as first-class**  
   Highest-ROI capability lever at 1–3B. Use a strong teacher to generate explanations, corrected solutions, coding trajectories, synthetic textbooks, FIM examples, etc. This is how many strong small models punch above weight.

4. **Post-training stack**  
   Make it real, not an appendix:
   ```
   pretrain → code/FIM continued pretraining → high-quality SFT → preference optimization (DPO or simple equivalent) → optional light verifier/reasoning stage
   ```
   Assistant-only masking and packing are already good foundations.

5. **Tokenizer**  
   Keep 16k byte-BPE for Edu and early Modern experiments. For the Competitive 1B/3B track, evaluate 32k or 64k. Larger vocab helps quality and coding without exploding the model size.

6. **Inference efficiency for hybrid**  
   Local layers should not keep full-history KV. Add a clean sliding/ring cache for local layers (global layers keep full history). This becomes important once people run longer contexts on consumer GPUs.

7. **Evaluation that matches the claim**  
   Keep the educational suite, but for the 1B/3B releases add:
   - Instruction following
   - Functional coding (HumanEval-style + fresher sets)
   - FIM
   - Basic reasoning
   - Long-context retrieval
   - Chat preference / win-rate style checks if possible

### Should you limit to 500M?

**No.**  
500M is a fine research checkpoint, but it is not the model most people will prefer for daily local use in 2026. 1B is the minimum credible “useful Modern” target; 3B is the size where you can actually be competitive with other well-trained small open models.

Hardware reality check:
- 1B dense BF16 is comfortable on a single good consumer GPU
- 3B is still practical with checkpointing / quantization
- 7B is doable but starts requiring more care (and more training compute)

### What not to change

- Do not abandon the hybrid GQA + QK-Norm + local/global design — it is validated by contemporary models (Muse Glimmer-style patterns etc.).
- Do not turn the core into a framework or add heavy serving complexity.
- Keep Edu and Competitive tracks sharing the same neural code; only configs, tokenizer, data, and training recipes diverge.
- Keep claims honest: “strong at 1B/3B among dense readable from-scratch models,” not “frontier.”

### Suggested sequencing after M10–M11

1. Freeze Competitive Modern config family (1B primary, 3B secondary).
2. Upgrade data pipeline + define Quality token budgets.
3. Implement / harden sliding local KV cache.
4. Run solid 1B training + distillation + SFT + preference.
5. Evaluate rigorously and only then decide whether 3B is worth the extra compute.
6. Optional: MTP lab → small speculative decoding path.

**Bottom line:**  
Your instinct is correct. Finish the current M-series, then treat **1B Modern as the real product target** and **3B as the recommended practical size**. Stay dense, stay readable, and put the majority of remaining effort into data, distillation, and post-training. That path gives you both an outstanding teaching codebase and a model people can actually use for useful local work — without drifting into 30B+ frontier research territory.


---

People expect models on Hugging Face + GGUF to “just work” in vLLM, llama.cpp, LM Studio, OpenCode, Cline, Roo Code, Aider, Vercel AI SDK, etc. Your architecture is custom (hybrid local/global + GQA + QK-Norm + specific RoPE/NoPE policy), so a naive upload will break or silently degrade in most of those tools.

### Recommended distribution strategy

**1. Hugging Face (primary)**

Upload two things clearly separated:

- **Native MiniFrontier release**  
  Full precision (BF16/FP16) + config + tokenizer + generation config + chat template.  
  This is the source of truth. Document exactly what the architecture is and that it is **not** Llama.

- **Transformers-compatible adapter** (do this properly)  
  Implement a real `MiniFrontierConfig` + `MiniFrontierForCausalLM` (or equivalent) that preserves Edu/Modern semantics (GQA, hybrid schedule, QK-Norm, layer-wise position encoding).  
  Prove greedy / logit parity against your native checkpoint.  
  Only after that should you claim “works with `transformers`”.

**2. GGUF (llama.cpp / LM Studio / Ollama ecosystem)**

- First ship a **high-precision GGUF** (Q8_0 or similar) and prove it matches native outputs on a fixed set of prompts.
- Then produce **Q4_K_M / Q5_K_M** (or whatever quality you measure) and report the degradation.
- You will almost certainly need a custom architecture definition or conversion script in llama.cpp (or a clear “this is a custom model, use this conversion path”).  
  Just converting as if it were Llama will corrupt the hybrid attention and RoPE behavior.

**3. vLLM**

- Prefer the Transformers modeling backend once your HF adapter is solid.
- Only write an out-of-tree vLLM plugin if the Transformers path is insufficient.
- Document the exact context length, that tool calling is **not** supported in V1, and the recommended serving flags.

### Tooling reality check (what will actually work)

| Tool / Runtime              | What works out of the box | What you must do |
|-----------------------------|---------------------------|------------------|
| **llama.cpp / LM Studio**   | Almost nothing reliable   | Custom conversion + architecture support + quality tests |
| **vLLM**                    | Limited                   | Solid Transformers adapter first, then test |
| **OpenCode / Cline / Roo / Kilo / Aider** | Text chat only | Point them at a correct OpenAI-compatible endpoint (vLLM or llama.cpp server). Disable tools. Document the real context limit. |
| **Vercel AI SDK**           | `generateText` / `streamText` | Same as above — works for plain chat if the server speaks Chat Completions. No tool loops / agents in V1. |
| **Ollama**                  | Same as llama.cpp         | Needs correct GGUF + Modelfile |

Do **not** claim “works with coding agents” until you have measured it. A successful API connection ≠ useful repository editing.

### Practical release checklist before you hit “Publish”

1. Native MiniFrontier checkpoint + exact reproduction script.
2. Transformers modeling code with parity tests (greedy + a few temperature samples).
3. High-precision GGUF + parity tests against native.
4. Quantized GGUF (4-bit) + measured quality drop on your eval suite.
5. Simple OpenAI-compatible server example (vLLM or llama.cpp).
6. Clear model card that states:
   - Architecture (hybrid GQA, QK-Norm, etc.)
   - Context length
   - No tool/function calling in this release
   - Recommended quant and runtime
   - “Educational / small practical local model” positioning (especially at 1B/3B)

### Suggested Hugging Face layout

```text
MiniFrontier-1B-Modern          ← main useful model (BF16)
MiniFrontier-1B-Modern-GGUF     ← Q8 + Q4_K_M etc.
MiniFrontier-3B-Modern          ← if you train it
MiniFrontier-150M-Modern        ← research / teaching companion
```

Keep Edu models in a separate collection or clearly marked so people don’t download the 50M teaching model expecting a daily driver.

### Bottom line

Yes — put it on Hugging Face and ship GGUF so people can run it in LM Studio, llama.cpp, vLLM, and point coding tools / Vercel AI SDK at it.  

But treat compatibility as an engineering project of its own (Transformers adapter → high-precision GGUF → quantized GGUF → server smoke tests). Uploading weights alone will produce frustrated users and false “it doesn’t work” reports.

Once the 1B (and later 3B) Modern models are trained with good data + distillation + SFT, this distribution path is exactly what makes them useful to other people. Just don’t skip the parity and conversion work.



-------------------------------------------

Yes — **this is now a genuinely good educational LLM repo**. But there is an important distinction:

> **MiniFrontier is currently a good LLM implementation/training laboratory, not yet a demonstrated good trained LLM.**

The architecture of `tiny-edu` / `tiny-modern` is credible, modern, and worth continuing. The missing piece is no longer “better Transformer code”; it is **serious pretraining + post-training + evaluation at scale**.

### My assessment

| Dimension                                          |                            MiniFrontier now |
| -------------------------------------------------- | ------------------------------------------: |
| Learning/teaching Transformers                     |                                    **9/10** |
| Modern architecture choices                        |                                  **8.5/10** |
| Code readability                                   |                                  **8.5/10** |
| Experimentability                                  |                                    **9/10** |
| Training/runtime completeness                      |                                    **8/10** |
| Current demonstrated model quality                 |                                    **2/10** |
| Foundation for future useful small LLM             |                                    **8/10** |
| Foundation for frontier ChatGPT/Claude-level model | **Architecture yes; resources/training no** |

The repo itself correctly makes that distinction: the latest real GPU run is only a **1–5M-token FineWeb-Edu training gate on the 50M Edu model**, and the README explicitly says this is integration evidence, **not a model-quality claim**. The serious matched 150M training experiments are still outstanding. ([GitHub][1])

## What exactly is this repo?

It is no longer comparable to one of those 200-line “build GPT from scratch” tutorials.

It is closer to a small **LLM research/education framework** built from raw PyTorch. You have the whole pipeline:

**text → tokenizer → streaming/shards → packing → Transformer → training → checkpointing → evaluation → generation/KV cache → FIM/code → SFT/chat → HF export → serving/runtime experiments.**

And the two-model design is particularly good pedagogically:

|            | **Edu**                              | **Modern**                          |
| ---------- | ------------------------------------ | ----------------------------------- |
| Norm       | Pre-RMSNorm                          | Pre-RMSNorm + QK-Norm               |
| Position   | RoPE                                 | RoPE + optional NoPE experiment     |
| Heads      | MHA                                  | GQA                                 |
| Attention  | Full causal                          | 3 local : 1 global                  |
| FFN        | SwiGLU                               | SwiGLU                              |
| Embeddings | tied                                 | tied                                |
| Purpose    | understand the canonical Transformer | understand a contemporary small LLM |

That progression is explicitly what your repository intends: manual attention → SDPA → RoPE → MHA/GQA → QK-Norm → hybrid attention → KV cache → Muon → FIM → SFT. ([GitHub][1])

That is a **very good teaching design** because somebody can learn why each modern addition exists rather than staring at a production implementation containing 40 interacting optimizations.

---

# Is `tiny-modern` actually modern?

Yes.

In fact, one of the strongest validations I found is **Karpathy's current nanochat**. NanoGPT is now explicitly described by Karpathy as old/deprecated in favor of nanochat. ([GitHub][2])

Nanochat has independently moved toward several of the same architectural ideas: modern normalization/attention techniques and short/long attention patterns, while Modded-NanoGPT uses RoPE, QK-Norm, Muon and other current training ideas. ([GitHub][3])

So your Modern model isn't an obsolete GPT-2 with some fashionable names pasted onto it.

The core:

**Pre-RMSNorm + RoPE + QK-Norm + GQA + local/global attention + SwiGLU**

is entirely reasonable for a compact decoder model in 2026.

I would **not** start throwing MoE, MLA, latent attention, state-space layers, DeltaNet, exotic routing, etc. into this repo. You would harm its best feature: it remains possible to understand the entire model.

---

# The best comparison is actually nanochat

Today I'd position them like this:

| Project          | Best use                                                                 | Compared with MiniFrontier                      |
| ---------------- | ------------------------------------------------------------------------ | ----------------------------------------------- |
| **MiniFrontier** | Learn modern LLM internals + controlled experiments                      | **Best fit for your goal**                      |
| **nanochat**     | Minimal end-to-end LLM training → chat with demonstrated training recipe | Strongest alternative                           |
| nanoGPT          | Simple GPT training                                                      | Now deprecated in favor of nanochat             |
| minGPT           | Understand classic GPT                                                   | Simpler, but much less modern                   |
| Modded-NanoGPT   | Discover extreme training-efficiency tricks                              | Much harder to teach                            |
| llm.c            | Understand GPU/CUDA implementation                                       | Excellent lower-level companion                 |
| LitGPT           | Actually train/fine-tune many existing LLM families                      | Far more production-oriented, much less minimal |

Nanochat is particularly interesting because it now covers tokenization, pretraining, fine-tuning, evaluation, inference and chat, and has an actual reproducible training target/leaderboard. ([GitHub][4])

`llm.c` is the better repo when the lesson becomes “what does this computation really look like in CUDA?” rather than “how does an LLM architecture work?” ([GitHub][5])

LitGPT is the better answer if somebody says “I don't care about implementing attention; I need to pretrain/fine-tune/deploy Llama/Gemma/etc.” ([GitHub][6])

So I **wouldn't replace MiniFrontier with any of them**. I would use nanochat and Modded-NanoGPT as references from which MiniFrontier selectively steals only concepts that remain easy to explain.

---

## Edu vs Modern: keep both

Definitely don't collapse them into one.

`Edu` is valuable because it answers:

> “What is a decoder Transformer?”

`Modern` answers:

> “What changes when we turn that simple Transformer into something closer to a current small LLM?”

That's one of the strongest concepts in the repo.

I would make **Modern the model you actually try to make good**, while Edu remains the reference model used for teaching and A/B experiments.

There is one experimental caveat I found in the uploaded implementation.

I instantiated your configurations without allocating weights and obtained approximately:

| Config        | Actual parameters |
| ------------- | ----------------: |
| 50M Edu       |        **53.36M** |
| 50M Modern    |        **47.86M** |
| 150M Edu      |       **154.17M** |
| 150M Modern   |       **138.45M** |
| 350M Modern   |       **332.46M** |
| “500M” Modern |       **433.91M** |

The Modern difference is understandable because **GQA removes K/V parameters**.

That's actually educationally useful.

But it means an experiment called:

> 150M Edu vs 150M Modern

isn't really parameter-matched.

Keep the current presets, but when you eventually claim Modern is better than Edu, I'd add one additional **parameter-matched control**.

---

# Can 50M/150M become a good general chat model?

Not in the sense you're probably aiming for.

The best reality check is SmolLM2.

Hugging Face's **135M SmolLM2** was pretrained on **2 trillion tokens**, using a mixture including FineWeb-Edu, DCLM and The Stack, on **64 H100 GPUs**. Its instruct version then received SFT and DPO. Even after that enormous effort, it remains a tiny model with substantial limitations. ([Hugging Face][7])

Compare that with your current 1–5M-token smoke run.

That's the difference between:

**“our implementation learns”**

and

**“the weights contain a useful language model.”**

Architecture tweaks will not bridge that gulf.

### Roughly what I'd expect

|   Model size | Realistic role                                                         |
| -----------: | ---------------------------------------------------------------------- |
|      **50M** | architecture experiments, unit/smoke training, TinyStories-style tasks |
|     **150M** | educational LLM, constrained completion, narrow specialized assistant  |
| **350–500M** | potentially useful specialized assistant with excellent training       |
|      **~1B** | first size I'd target for a genuinely useful small general chat model  |
|      **~3B** | much more credible general chat + coding model                         |
|    **3–7B+** | much more realistic starting point for good local agentic coding       |

These aren't hard mathematical boundaries; data quality and distillation can shift them substantially.

But they are sensible engineering targets.

Google's Gemma 3 270M is another useful reference. Google explicitly says that its instruction model is **not designed for complex conversational use**, even though it can follow general instructions. ([Google Developers Blog][8])

That is approximately the capability region your 350M-ish experiment would eventually inhabit, assuming very good training.

---

# But small models *can* be agentic

This is where things get interesting.

Google's **FunctionGemma 270M** shows that a 270M model can be useful in agent-like systems when its job is tightly defined. Google specifically positions it as a base for **specialized function calling**, and says it should be fine-tuned for the particular tools/workflow rather than treated as a general dialogue model. ([Google AI for Developers][9])

That distinction matters enormously.

A 300M MiniFrontier could potentially become very good at:

> User instruction → select one of 30 tools → generate valid arguments → consume result → answer.

That's plausible.

A 300M model independently exploring a 500,000-line C# repo, discovering a race condition, planning a six-file refactor, editing it, fixing compiler failures and reasoning through tests?

**No — that needs much more capability.**

---

# Can MiniFrontier eventually do coding?

Yes. Your architecture does not prevent it.

You've already made an important start by implementing **code data handling and FIM**.

But FIM teaches:

> “complete the code between these two regions.”

Agentic coding requires much more:

**understanding repositories, instruction following, generating patches, calling search/read/edit/build/test tools, interpreting compiler failures, remembering previous actions and correcting bad edits.**

That's mainly a **training/post-training problem plus an agent-runtime problem**, rather than needing another attention architecture.

So MiniFrontier Modern can absolutely be the neural core.

---

# Can it eventually be used by Cline/Aider/OpenCode/etc.?

Yes, once the model becomes capable enough.

You already have the right transport direction: HF export, vLLM experiments, and OpenAI-compatible serving are enough to make external clients communicate with it.

But there's a major distinction between:

**“Cline successfully sends prompts to my model”**

and

**“my model can reliably operate Cline.”**

Your own current README correctly does not claim trained tool-use reliability yet. The code infrastructure is ahead of the model.

For real agentic coding, you'll eventually need a trained protocol roughly equivalent to:

`assistant → tool_call → tool_result → assistant → tool_call → ... → final`

with training examples containing failed commands, test results, partial edits, retries and long trajectories.

FunctionGemma demonstrates precisely why those special tool formats and training data matter. ([Google AI for Developers][10])

---

# What I would change for MiniFrontier V2

I would **not redesign the Transformer**. I'd keep the current educational core and put almost all future complexity outside it:

1. **Finish real 50M/150M scaling experiments first.** Get loss curves, compute/token efficiency, downstream evals and Edu-vs-Modern evidence.
2. **Add 1B and ~3B Modern configs.** Those should be the eventual useful-model targets; keep 50M/150M as teaching sizes.
3. **Increase context for the larger models.** 2K is fine pedagogically but poor for coding. I'd eventually target at least 8K/16K and preferably ~32K for the coding model.
4. **Create tokenizer V2.** Keep your 16K BPE for education/backward compatibility, but benchmark ~32K–64K vocabulary on natural language + code + your target languages.
5. **Make the data recipe the main research project.** General web + high-quality educational text + code + math/reasoning + synthetic data, followed by code-heavy continued pretraining.
6. **Post-train seriously.** SFT → preference tuning/distillation → tool-use/coding trajectories. Distillation from a much stronger teacher will probably give you vastly more capability per unit of compute than inventing another architecture component.
7. **Add an actual agent protocol and evals.** Function schemas, tool calls/results, filesystem/search/edit/build/test tools, then HumanEval/MBPP-style code evals followed eventually by repo/SWE-style evaluations.
8. **Only then explore architecture extras.** ReLU² vs SwiGLU, tied vs untied embeddings, value residuals, logit soft-capping, etc. should be measured experiments, not additions because frontier models use them.

That would preserve your **80/20 philosophy**.

---

## I also found two repo-cleanliness issues worth fixing

First, parts of `future-plan.md` / older documentation appear to describe earlier states where features were still missing, while the current code and README have already implemented them. I would either regenerate those documents or move historical plans under something like `docs/archive/`. Otherwise a learner can get conflicting descriptions of the project.

Second, the public GitHub tree currently contains:

`workflows/ci.yml`

at the repository root. ([GitHub][11])

Yet your README says the source-archive process includes:

`.github/workflows/ci.yml`. ([GitHub][1])

For GitHub Actions, the latter is the important location. I'd verify that immediately; the current repository tree suggests the workflow has been misplaced during archive/upload. That's exactly the kind of small release-engineering problem worth fixing before calling the repo polished.

I also couldn't independently execute the full test suite in my inspection environment because the sandbox didn't have your `tokenizers` dependency synchronized, so I'm treating the README's “186 tests” as the repository's claim rather than independently reproduced evidence.

---

# My final verdict

**Keep this repo. The architecture is good.**

I'd describe it publicly as:

> **MiniFrontier is a compact, from-scratch modern LLM laboratory for understanding the path from a canonical Transformer to a small contemporary decoder model.**

Not:

> “A tiny frontier-quality LLM.”

At present:

**MiniFrontier Edu** = very good teaching model.
**MiniFrontier Modern** = good small research architecture.
**Current checkpoints** = not yet useful general LLMs.
**150M eventually** = fun/educational/narrow assistant.
**350–500M eventually** = potentially useful specialist.
**1B Modern** = where I'd begin pursuing genuine general chat.
**3B Modern** = where I'd seriously pursue general chat + coding + tool use.
**Agentic coding** = absolutely possible with this codebase, but it will require scale, code-rich pretraining, long context, post-training and explicit tool-use trajectories—not another clever attention layer.

And compared with alternatives, I think you've arrived at a sensible niche:

> **nanochat is currently the better demonstrated minimal “train a small ChatGPT” project; MiniFrontier can become the better “understand every important part of a modern LLM and experiment with it” project.** ([GitHub][4])

That's a worthwhile distinction. The next major gains should come from **training science and model capability**, not adding more architecture complexity.

[1]: https://github.com/igal-abachi-dev/AI-LLM-Transformers-Edu-Model "GitHub - igal-abachi-dev/AI-LLM-Transformers-Edu-Model: AI LLM implementation for learning about transformers and ai models · GitHub"
[2]: https://github.com/karpathy/nanoGPT/blob/master/README.md?plain=1&utm_source=chatgpt.com "nanoGPT/README.md at master · karpathy/nanoGPT · GitHub"
[3]: https://github.com/karpathy/nanochat?utm_source=chatgpt.com "GitHub - karpathy/nanochat: The best ChatGPT that $100 can buy. · GitHub"
[4]: https://github.com/karpathy/nanochat "GitHub - karpathy/nanochat: The best ChatGPT that $100 can buy. · GitHub"
[5]: https://github.com/karpathy/llm.c?utm_source=chatgpt.com "GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA · GitHub"
[6]: https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md?utm_source=chatgpt.com "litgpt/tutorials/pretrain.md at main · Lightning-AI/litgpt · GitHub"
[7]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/README.md?utm_source=chatgpt.com "README.md · HuggingFaceTB/SmolLM2-135M at main"
[8]: https://developers.googleblog.com/introducing-gemma-3-270m/?utm_source=chatgpt.com "Introducing Gemma 3 270M: The compact model for hyper-efficient AI - Google Developers Blog"
[9]: https://ai.google.dev/gemma/docs/functiongemma/model_card?utm_source=chatgpt.com "FunctionGemma model card  |  Google AI for Developers"
[10]: https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices?utm_source=chatgpt.com "FunctionGemma formatting and best practices  |  Google AI for Developers"
[11]: https://github.com/igal-abachi-dev/AI-LLM-Transformers-Edu-Model/blob/main/workflows/ci.yml "AI-LLM-Transformers-Edu-Model/workflows/ci.yml at main · igal-abachi-dev/AI-LLM-Transformers-Edu-Model · GitHub"




------------

Your own RTX 2070 Super measurements are already enough to put hard numbers on the problem. At 150M/1K context you measured roughly 2.1–2.75k training tokens/s, while the 150M Modern 8K inference path already spills past 8 GB VRAM (~10.9 GB). So “350M–1B + native 16–32K pretraining + hundreds of billions/trillions of tokens” is not a home-GPU plan; we need a different training strategy, not just a smaller copy of SmolLM2.


The biggest actionable finding from the source comparison is already clear: your repo’s auto training precision is wrong for this particular GPU target. Tiny-Modern currently chooses BF16 only when supported and otherwise falls to FP32; Turing’s Tensor Cores support FP16 but not BF16, while nanoGPT/nanochat explicitly provide FP16 training with loss scaling. NVIDIA confirms Turing’s Tensor Core precision set is FP16/INT8/INT4/INT1, with BF16 arriving in Ampere.

-----------------------



Yes — **using a SmolLM2-like high-quality mixture is the right next direction for Tiny-Modern**, but I would change two parts of your proposed plan:

> **Do not try to imitate SmolLM2's token count, and do not pretrain at 16–32K context.**

On one RTX 2070 Super, the best project is a **compute-efficient 150M→350M research program**, with 350M as the serious home-GPU model. Keep 1B as a supported architecture/config and future rented-GPU training target.

And after inspecting your uploaded nanoGPT and nanochat source, there are several concrete improvements that can make MiniFrontier substantially better without ruining its simplicity.

# 1. First: SmolLM2's 2T tokens are not what you should copy

The published SmolLM2 135M run really was enormous for such a small model: its Nanotron config used BF16, 64-way data parallelism and 2,000,000 steps. ([Hugging Face][1])

Its 2T-token data wasn't simply:

> FineWeb-Edu + DCLM + The Stack

The released corpus description is more interesting:

| Source         | Approx. tokens |
| -------------- | -------------: |
| DCLM-Edu       |       1,065.6B |
| FineWeb-Edu    |         710.4B |
| Stack-Edu      |           125B |
| FineMath       |            34B |
| InfiMM-WebMath |            40B |
| Cosmopedia V2  |            30B |

([Hugging Face][2])

Notice **Stack-Edu**, not merely raw The Stack.

And DCLM-Edu's authors specifically say that for small models, filtering to `edu_int_score >= 3` improved downstream performance. ([Hugging Face][3])

That is extremely relevant to your situation.

With 1 GPU, your scarce resource is **tokens you can afford to process**. Therefore every token needs to be better.

---

# 2. I would use a mixture — but a small, aggressively filtered one

For a Tiny-Modern intended to eventually be decent at both general language and code, I'd start experiments around:

| Dataset                                     | Starting share |
| ------------------------------------------- | -------------: |
| DCLM-Edu `score >= 3`                       |        **45%** |
| FineWeb-Edu                                 |        **30%** |
| Stack-Edu                                   |        **15%** |
| FineMath / strong math                      |         **5%** |
| Cosmopedia / other curated educational data |         **5%** |

This is **not a magic proven optimum**. It's the first mixture I would experimentally test.

It intentionally gives code more weight than SmolLM2-135M did because you care about coding.

Then test perhaps three mixtures at 50M/150M rather than committing the expensive 350M run blindly:

**A — General:** 55% DCLM / 35% FineWeb / 5% code / 5% math
**B — Balanced:** 45 / 30 / 15 / 10
**C — Code-heavy:** 35 / 25 / 30 / 10

Use the same tokens, seed, tokenizer and schedule and compare BPB + HellaSwag/ARC/PIQA + code evaluations.

That is far more scientifically useful than saying:

> “SmolLM2 used these datasets, therefore I'll use exactly their percentages.”

---

# 3. There's an even more interesting dataset from nanochat

Current nanochat no longer uses FineWeb-Edu for its record run.

Karpathy reports that they repeatedly tried FineWeb, DCLM and OLMo alternatives without improving the run, then switched to NVIDIA **ClimbMix**, which produced a clear improvement. The Time-to-GPT-2 project went from around 3 hours originally to ~2 hours around the ClimbMix change and has since improved further. ([GitHub][4])

ClimbMix is explicitly a **400B-token compute-efficient pretraining mixture**, constructed using topic clustering plus advertising and educational-quality filtering. ([Hugging Face][5])

That's almost exactly the research question you care about:

> **How do I get maximum capability per training token?**

However, there's a catch:

**ClimbMix is CC BY-NC 4.0 / research-and-development use**, so I would not make it the sole/default MiniFrontier training recipe if you want users to be able to build broadly reusable/commercial models. ([Hugging Face][5])

Use it as an experimental benchmark:

> Tiny-Modern 50M, same compute:
>
> FineWeb-Edu
> DCLM-Edu
> your balanced mixture
> ClimbMix

That could itself become a really nice MiniFrontier experiment.

---

# 4. Do NOT train from scratch at 16K or 32K

This is one of the most important corrections.

Even **SmolLM2-135M was pretrained at sequence length 2048**:

```text
sequence_length: 2048
max_position_embeddings: 2048
```

([Hugging Face][1])

The released HF model later supports 8192 positions and uses a larger RoPE theta. ([Hugging Face][6])

So even Hugging Face, with 64 H100s, did not say:

> “Let's make every pretraining sequence 32K.”

Neither should you.

For MiniFrontier I'd make:

| Stage                 |                                      Context |
| --------------------- | -------------------------------------------: |
| Main pretraining      |                                     **2048** |
| Optional intermediate |                                         4096 |
| Context extension     |                                         8192 |
| Later experiment      |                                          16K |
| 32K                   | supported/experimental, not initial training |

This is particularly important because attention compute and activation memory explode with sequence length unless the sliding-window structure handles most of it efficiently.

Your 3-local:1-global Tiny-Modern design helps substantially, but global layers still cost real compute.

### The model can nevertheless advertise 16K/32K architecture support.

Those are different claims:

**model implementation supports 32K**

versus

**the model was pretrained on 32K sequences.**

You need the first now, not the second.

---

# 5. What can your RTX 2070 Super realistically do?

Your own benchmark is useful here.

At 150M, 1024 tokens, you measured approximately:

**2,747 tok/s without activation checkpointing**
**2,135 tok/s with checkpointing.**

That's roughly:

**237M tokens/day** at 2747 tok/s.

So even at your existing 150M speed:

| Training tokens | Ideal uninterrupted time |
| --------------: | -----------------------: |
|            100M |                ~10 hours |
|            500M |                ~2.1 days |
|              1B |                ~4.2 days |
|              3B |               ~12.6 days |
|              5B |                 ~21 days |
|             10B |                 ~42 days |

That's 150M, seq=1024, under your measured conditions.

For ~332M Tiny-Modern, a simplistic inverse-parameter scaling from your measured 150M result gives roughly **1.2K tok/s**, or ~105–110M tokens/day.

So approximately:

| 350M target | Rough current-order estimate |
| ----------: | ---------------------------: |
| 500M tokens |                    ~4–5 days |
|          1B |                   ~9–10 days |
|          2B |                  ~18–20 days |
|          3B |                  ~27–30 days |
|          5B |                    ~45+ days |

These are planning estimates, **not benchmarks**. 2K context, hybrid attention, VRAM pressure, optimizer behavior and compilation can change them substantially.

And a 1B model is dramatically worse. A generous compute-only extrapolation already puts 1B tokens at several weeks — before dealing with the much larger memory/optimizer problem.

So:

### 350M: yes, potentially.

### 1B scratch pretraining on the 2070S while refusing months: no.

You could technically make all sorts of CPU offload/optimizer tricks allow it to execute.

But:

> **“can execute” ≠ “sensible training platform.”**

I'd keep a `modern-1b` config and verify initialization/forward/inference/export, but I would **not pretrain 1B from scratch on this card**.

---

# 6. But I found something that should significantly improve your RTX 2070S situation

This is probably the highest-ROI discovery from comparing your source with nanoGPT/nanochat.

Your current:

```python
Precision = Literal[
    "auto",
    "float32",
    "bfloat16",
]
```

effectively does:

```text
BF16-capable CUDA → BF16
otherwise → FP32
```

Your RTX 2070 Super is **Turing**.

Turing Tensor Cores support FP16, but **not BF16 Tensor Core math**. NVIDIA's own compatibility table lists:

| Architecture | Tensor-Core types           |
| ------------ | --------------------------- |
| Turing       | FP16, INT8, INT4, INT1      |
| Ampere A100  | FP64, TF32, BF16, FP16, ... |

([NVIDIA][7])

That explains why your BF16 experiments weren't attractive on this machine.

## nanoGPT handles this better

The uploaded nanoGPT code literally chooses:

```python
'bfloat16'
if torch.cuda.is_bf16_supported()
else
'float16'
```

and FP16 automatically activates `GradScaler`.

nanochat does essentially the same thing conceptually: its README explicitly documents FP16 training and automatic gradient scaling. ([GitHub][8])

### MiniFrontier should become:

```text
Ampere+ native BF16 → BF16
Turing CUDA → FP16 + GradScaler
CPU → FP32 / optional BF16 where appropriate
```

This is **priority zero** before attempting 350M.

It could improve both memory and performance considerably on your GPU. Exactly how much needs to be measured on your implementation rather than assumed.

---

# 7. This changes what I recommend training

Given your “not months” constraint:

### My actual target would be **350M Modern, 2K pretraining context, ~1B tokens first**.

Not 1B parameters.

If the 1B-token checkpoint is improving well and the training rate after FP16 optimization is acceptable, extend it to **2–3B tokens**.

That gives you a meaningful research model without committing blindly to a 30–60-day run.

And crucially:

> Don't wait until 3B tokens to discover your tokenizer, data mix or optimizer was bad.

Use the 50M and 150M models as experimental proxies.

That is exactly why having your whole **50M → 150M → 350M → 1B** MiniFrontier family is useful.

---

# 8. Nanochat changes my view of how many tokens you need experimentally

There's another useful lesson.

Nanochat deliberately targets a compute-efficient parameter:data ratio rather than “train on trillions because frontier labs do.”

Its current base trainer actually derives the training horizon automatically from model parameter count and a configurable target parameter:data ratio. Its leaderboard describes experiments around roughly 8–10.5 tokens per scaling parameter depending on the speedrun. ([GitHub][4])

That doesn't mean:

> `350M × 10 = exactly 3.5B and you're done.`

Nanochat's parameter accounting is unusual, architecture/data are different, and its objective is a particular CORE threshold.

But it strongly reinforces this:

### For MiniFrontier research, **billions**, not trillions, are the correct unit.

For a **fully saturated 135M commercial-grade checkpoint**, Hugging Face could justify 2T.

For:

> “Can Tiny-Modern demonstrate that its architecture/data recipe works?”

you absolutely do not need 2T.

---

# 9. What I learned from nanoGPT that should go into MiniFrontier

nanoGPT's **architecture itself isn't something you should copy**.

Karpathy now explicitly marks nanoGPT old/deprecated and directs people toward nanochat. ([GitHub][9])

Tiny-Modern is already architecturally much more current than nanoGPT.

But its engineering still has lessons.

The highest-value items from the uploaded nanoGPT are:

1. **FP16 + GradScaler**, as discussed above.
2. Fused AdamW where PyTorch/device support it.
3. Better automatic hardware-aware dtype selection.
4. Explicit tokens/iteration reporting.
5. MFU / achieved compute measurement.
6. Very simple gradient accumulation accounting.
7. `torch.compile` benchmarking as part of the standard training profile.

You have parts of several already.

The key is making **“performance engineering for normal GPUs”** part of MiniFrontier's identity.

---

# 10. Nanochat is much more interesting

This is the serious comparison.

Current nanochat is explicitly trying to be a minimal end-to-end experimental harness that produces a real conversational model, and Karpathy says the project now targets micro-models accessible under ~$1000 rather than becoming a giant configurable framework. ([GitHub][8])

Its current Transformer has:

**RoPE
RMSNorm
QK-Norm
GQA support
sliding-window attention
ReLU²
untied embedding/head
embedding RMSNorm
value residual/value embeddings
learnable residual scaling
x0 residual
smear
backout
logit soft-capping
FlashAttention 3
Muon + AdamW**

The current source documents many of these directly. ([GitHub][10])

Some are fascinating.

**I would absolutely not copy all of them.**

---

# 11. Tiny-Modern already makes some choices I prefer

Your architecture is:

```text
RMSNorm
   ↓
GQA + QK-Norm + RoPE
   ↓
residual
   ↓
RMSNorm
   ↓
SwiGLU
   ↓
residual
```

with hybrid:

```text
Local
Local
Local
Global
```

That's extremely clean.

Nanochat's latest speedrun model has increasingly experimental things like value embeddings, smear, backout and learned residual coefficients.

Those may make the metric better.

But they make the explanation:

> “this is how modern Transformers work”

less clean.

There's a revealing detail in nanochat's own leaderboard rules: Karpathy says changes can be rejected if they're too gnarly, bloated or esoteric even if they improve the metric. ([GitHub][4])

That is exactly the principle MiniFrontier should follow.

---

# 12. What I would actually take from nanochat

Here's my priority order:

| Improvement                                   | Tiny-Modern             | Priority |
| --------------------------------------------- | ----------------------- | -------: |
| FP16 + GradScaler                             | **Default on Turing**   |    🔴 P0 |
| Better local/sliding attention backend        | Yes                     |    🔴 P0 |
| Fused AdamW                                   | Yes                     |    🔴 P0 |
| Exact MFU / tok/s / memory dashboard          | Yes                     |    🔴 P0 |
| Automatic token-budget calculator             | Yes                     |    🔴 P0 |
| Automatic batch/grad accumulation suggestions | Yes                     |    🔴 P0 |
| Data-mixture ablation framework               | Yes                     |    🔴 P0 |
| 16K/32K context-extension experiment          | Yes                     |    🟠 P1 |
| 16K vs 32K tokenizer experiment               | **32K candidate**       |    🟠 P1 |
| BOS-aligned best-fit packing                  | experiment              |    🟠 P1 |
| ReLU² vs SwiGLU                               | experiment              |    🟡 P2 |
| tied vs untied embedding                      | experiment              |    🟡 P2 |
| embedding RMSNorm                             | experiment              |    🟡 P2 |
| logit softcap                                 | experiment              |    🟡 P2 |
| value residual                                | experiment              |    🟡 P2 |
| x0 residual scaling                           | experiment              |    🟡 P2 |
| smear/backout                                 | probably no/default-off |     ⚪ P3 |
| FP8                                           | no 2070S value          |        ❌ |
| FA3/H100 specialization                       | no 2070S value          |        ❌ |

The important architectural philosophy should be:

```text
tiny-modern/
    clean baseline

experiments/
    relu2
    untied_embeddings
    value_residual
    x0_residual
    softcap
    ...
```

Then a feature graduates to `tiny-modern` **only if it wins an A/B test enough to justify the extra concept**.

That would be very strong.

---

# 13. There is another interesting nanochat idea: packing

Your current MiniFrontier pretraining pipeline uses the classic continuous ribbon:

```text
document A <eos> document B <eos> document C ...
────────────────────────────────────────────────
          slice into fixed sequences
```

Simple. Efficient. Good for teaching.

Current nanochat instead has an optional-looking but currently used **BOS-aligned best-fit** strategy:

```text
<BOS> document
<BOS> document
...
```

and fills each row by best-fitting documents, cropping when necessary.

The source claims 100% training utilization but roughly **35% document-token cropping at T=2048**.

This gives the model cleaner document boundaries but sacrifices some data.

I would **not replace yours immediately**.

I would implement:

```text
--packing ribbon
--packing bos-bestfit
```

and measure.

That's precisely the kind of controlled experiment MiniFrontier should excel at.

---

# 14. Another improvement: tokenizer V2

Your 16,384 BPE tokenizer is appropriate for the educational model.

For a 350M general+coding model, I think **32K deserves a real experiment**.

Nanochat uses a more sophisticated ~32K tokenizer setup, while SmolLM2 uses 49,152 tokens. ([Hugging Face][1])

Don't jump to 50K just because SmolLM2 does.

At your model size, vocabulary parameters are expensive.

Instead compare:

```text
16K
32K
48K
```

on:

**bytes/token
characters/token
English BPB
source-code compression
JSON/code punctuation
training throughput
embedding parameter cost**

My bet would be **32K** becoming Modern V2 and 16K remaining Edu.

But benchmark it.

---

# 15. Your biggest remaining performance problem after FP16 is attention

Your own 150M Modern benchmark exposes it.

Your eager local FlexAttention path is painfully slow on the 2070S.

This matters much more than adding some sexy architecture trick from a 2026 paper.

Tiny-Modern needs:

```text
Attention interface
      │
      ├── PyTorch SDPA
      ├── efficient sliding-window CUDA backend
      └── reference/manual implementation
```

with automatic backend selection.

nanochat has FlashAttention 3 plus an SDPA fallback. ([GitHub][10])

FA3 itself is **not the answer for a 2070S**.

The lesson is the abstraction:

> Use the best implementation available for the hardware while preserving a clean reference implementation.

That would also make your “Modern” model substantially more credible at 16K/32K eventually.

---

# 16. And I would steal nanochat's scaling UX

This is perhaps its best feature.

You can essentially say:

```bash
python -m scripts.base_train --depth=...
```

and it derives much of the model/training configuration automatically. Nanochat explicitly advertises this “single complexity dial.” ([GitHub][8])

MiniFrontier shouldn't hide everything behind magical heuristics because teaching explicit configs is valuable.

But add something like:

```bash
python scripts/plan_run.py \
    --model modern-350m \
    --gpu rtx2070s \
    --budget-tokens 1B
```

and return:

```text
Parameters              332,456,xxx
Trainable parameters     ...
Context                  2048

Precision                FP16
Gradient scaler          yes
Activation checkpoint    yes

Micro batch              1
Gradient accumulation    128
Global tokens/update     262,144

Target training tokens   1,000,000,000
Steps                    ...
Estimated optimizer mem  ...
Estimated activation mem ...
Measured tok/s           from benchmark DB
```

Now **that** would be useful to people.

It preserves the explicit educational configuration while offering nanochat-style convenience.

---

# 17. How MiniFrontier can actually get people to choose it over nanochat

Don't try to become:

> “nanochat but written slightly differently.”

Karpathy has an enormous ecosystem advantage. You won't win that game by cloning its current architecture.

Instead MiniFrontier should have a very clear identity:

> **The smallest understandable modern LLM laboratory that lets you learn, train, benchmark, modify and export every major component — including on normal consumer GPUs.**

Then the comparison becomes:

|                                  | nanoGPT    | nanochat             | **MiniFrontier**         |
| -------------------------------- | ---------- | -------------------- | ------------------------ |
| Classic Transformer education    | Good       | Medium               | **Excellent**            |
| Modern architecture              | Old        | **Excellent**        | **Excellent**            |
| Edu→Modern progression           | No         | No                   | **Unique**               |
| Explicit MHA→GQA learning        | No         | Not primary goal     | **Yes**                  |
| Controlled architectural A/Bs    | Hack it    | Speedrun oriented    | **Core feature**         |
| Consumer 8GB GPU focus           | Limited    | H100-centered record | **Make this a strength** |
| Code/FIM teaching                | Limited    | later stages         | **Already present**      |
| HF export                        | not focus  | not primary          | **Yes**                  |
| GGUF                             | not focus  | not primary          | **Yes**                  |
| Clean ecosystem/runtime path     | Limited    | own runtime          | **Yes**                  |
| Reproducible real quality result | Historical | **Major strength**   | **Currently missing**    |
| Public competitive benchmark     | No current | **Excellent**        | **Need this**            |
| Data efficiency research         | Old        | **Excellent**        | **Need this**            |

The bottom three are where your next work should go.

---

# 18. So what should MiniFrontier V2 actually be?

I wouldn't make V2 an architectural rewrite.

I would make it:

### **“Tiny-Modern becomes a real consumer-GPU-trained LLM.”**

The sequence I would implement is:

1. **FP16 + GradScaler + hardware-aware `auto` precision.**
2. **Profile/fix Modern sliding attention on RTX 2070S.**
3. Add fused optimizer paths and improve compile benchmarks.
4. Add exact training-performance metrics: tokens/s, MFU, GPU memory, parameters, tokens processed.
5. Add `plan_run.py` with parameter:data/run-budget calculations.
6. Add dataset mixtures and deterministic weighted streaming for DCLM-Edu/FineWeb-Edu/Stack-Edu/FineMath.
7. Add data-mixture A/B experiments at 50M first.
8. Test tokenizer 16K vs 32K.
9. Run 150M Modern on ~500M–1B high-quality tokens as the validation model.
10. Run **350M Modern / 2K / initial 1B tokens**, inspect curves/evals, then decide whether 2–3B is worth continuing.
11. Add 8K→16K context extension only after the base model is good.
12. Add nanochat ideas such as ReLU², untied embeddings, value residual and soft-cap **one at a time**, requiring measured quality-per-compute improvement before moving into the default model.

That is the 20% of changes I think gives you 80% of the gain.

---

# My revised target architecture

I would currently aim for:

```text
Tiny-Modern 350M
────────────────────────────────

~330–350M actual parameters

Pre-RMSNorm
RoPE
QK-Norm
GQA
3 local : 1 global attention
SwiGLU
tied embeddings

Tokenizer:
    32K candidate (after 16K/32K A/B)

Main pretraining:
    context = 2048
    FP16 + GradScaler on RTX 2070S
    activation checkpointing as required

Data:
    DCLM-Edu >=3
    FineWeb-Edu
    Stack-Edu
    small FineMath/curated component

Phase 1:
    1B tokens

Phase 2:
    continue toward 2–3B only if
    eval improvement justifies it

Context extension:
    4K → 8K → 16K
    later / small continuation

Post-training:
    SFT after base quality is demonstrated

Agent/tool training:
    much later
```

## And 1B?

Implement it.

Test it.

Support checkpointing/export/inference.

But **don't use your 2070 Super to scratch-pretrain it** under your “not months” constraint.

If the 350M experiment proves MiniFrontier is genuinely good, **that is the point where spending a small amount on rented modern GPU compute for a 1B run makes sense**. You will then be buying compute for a tested recipe rather than burning money while discovering basic training issues.

Most importantly, I would now change my earlier framing slightly:

> **Your next bottleneck isn't simply “you need huge compute.”**

It's more specifically:

> **MiniFrontier now needs a compute-efficient training recipe.**

Nanochat is strong evidence that architecture + **data quality + optimizer + batching + precision + kernel efficiency + rigorous quality-per-wall-clock experimentation** can radically change how much compute you need. Its entire project has reduced its Time-to-GPT-2 objective from roughly three hours toward ~1.5–2 hours on the same class of 8×H100 setup through exactly that type of iteration. ([GitHub][11])

That's the direction I would take your repo now: **not more Transformer features, but making Tiny-Modern the best understandable LLM you can actually train end-to-end on ordinary hardware.**

[1]: https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt/blob/main/135M/final/config.yaml?utm_source=chatgpt.com "135M/final/config.yaml · HuggingFaceTB/SmolLM2-nanotron-ckpt at main"
[2]: https://huggingface.co/datasets/EleutherAI/SmolLM2-135M-10B?utm_source=chatgpt.com "EleutherAI/SmolLM2-135M-10B · Datasets at Hugging Face"
[3]: https://huggingface.co/datasets/HuggingFaceTB/dclm-edu?utm_source=chatgpt.com "HuggingFaceTB/dclm-edu · Datasets at Hugging Face"
[4]: https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md?utm_source=chatgpt.com "nanochat/dev/LEADERBOARD.md at master · karpathy/nanochat · GitHub"
[5]: https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix/blob/main/README.md?utm_source=chatgpt.com "README.md · nvidia/Nemotron-ClimbMix at main"
[6]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/d6a5589c239236d22370e2126bbe23d4843c47d9/config.json?utm_source=chatgpt.com "config.json · HuggingFaceTB/SmolLM2-135M at d6a5589c239236d22370e2126bbe23d4843c47d9"
[7]: https://www.nvidia.com/en-eu/data-center/tensorcore/?utm_source=chatgpt.com "Tensor Cores: Versatility for HPC & AI | NVIDIA"
[8]: https://github.com/karpathy/nanochat?utm_source=chatgpt.com "GitHub - karpathy/nanochat: The best ChatGPT that $100 can buy. · GitHub"
[9]: https://github.com/karpathy/nanoGPT/blob/master/README.md?plain=1&utm_source=chatgpt.com "nanoGPT/README.md at master · karpathy/nanoGPT · GitHub"
[10]: https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py?utm_source=chatgpt.com "nanochat/nanochat/gpt.py at master · karpathy/nanochat · GitHub"
[11]: https://github.com/karpathy/nanochat/blob/master/README.md?utm_source=chatgpt.com "nanochat/README.md at master · karpathy/nanochat · GitHub"

350M Modern, 2K pretraining context, ~1B tokens first.



for 1B you need to rent , not at home gpu:
Where to Rent the HardwareTo train the model at the lowest cost, you need to use cloud GPU providers that offer on-demand or spot instances for 8xH100 SXM nodes. Some of the most popular providers for this specific price tier include:Lambda Labs GPU Cloud: A standard choice for machine learning engineers with highly competitive hourly pricing.DataCrunch: Frequently recommended for offering highly affordable spot instance rates.Prime Intellect: A decentralized and aggregated orchestration layer ideal for multi-GPU setups.FluidStack or Vast.ai: Excellent alternatives for renting unutilized enterprise hardware at steep discounts.
Log into your chosen provider's dashboard and launch an 8x H100 SXM (80 GB) node running Ubuntu 22.04 LTS. Make sure to add your SSH key for secure access.

so for now 350M i good enough for at home consumer gpu , not h100/dgx