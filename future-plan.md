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