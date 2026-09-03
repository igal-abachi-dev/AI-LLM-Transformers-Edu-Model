# AI-LLM-Transformers-Edu-Model
AI LLM implementation for learning about transformers and ai models
inspired by Muse-Glimmer / Llama4 / Qwen3/ GPT oss...

the idea is similar to: nanogpt/nanochat/llm.c , 
but this is a 150-350M model that is designed to train and run on home consumer pc hardrware like single nvidia geforce rtx gpu , and not just H100/DGX and more advanced hw/server-farm

# MiniFrontier

[![CI](https://github.com/igal-abachi-dev/AI-LLM-Transformers-Edu-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/igal-abachi-dev/AI-LLM-Transformers-Edu-Model/actions/workflows/ci.yml)

MiniFrontier is a from-scratch, educational decoder-only language model built with raw PyTorch for a single consumer GPU. Its goal is to make the path from classic Transformer fundamentals to a small set of modern LLM techniques visible, testable, and measurable.

> **task Status:** M4 Modern, the CPU-verifiable M5 path, M6 code/FIM, M7 Muon, and M8 assistant-only
> SFT/chat are implemented. M9 protocol/export/release validation tooling is implemented, while the
> real matched 150M training artifacts remain open. M10 preflight and the M11 Transformers/export,
> external-runtime, and GGUF orchestration paths are implemented; hardware/upstream-runtime gates
> remain unmeasured. The CPU suite passes all 191 tests (187 in the
> default non-slow gate). The first home-GPU pass (RTX 2070 Super, 8GB) landed 2026-08-21: real CUDA
> BF16/accumulation/activation-checkpointing parity, an initial 50M/150M profiling matrix, and a real
> 1-5M-token FineWeb-Edu 50M-Edu training gate with decreasing loss/validation — see
> `tasks/evidence/MF-046-050-063-home-rtx.md`. On 2026-08-25, real FP16+GradScaler support was added
> (this GPU has no native BF16 Tensor Cores, so prior BF16 evidence measured emulated BF16) —
> real FP16 trains ~2.7x faster than emulated BF16 on this card — see
> `tasks/evidence/MF-075-fp16-gradscaler.md`. This is engineering/integration evidence on a real but
> undertrained checkpoint, not a model-quality claim; the real matched 150M runs (MF-064/065) remain
> open.

## Introduction (read first)
- [Introduction to LLM / Transformers / Attention](introduction.md)
- [Architecture / Diagrams](minifrontier-architecture-diagrams.md)

If your goal is to understand how modern decoder-only transformers actually work by reading and running real, well-structured code (attention → RoPE → GQA → training loop → decoding → SFT),
Recommended starting path (as the repo itself suggests):

Read introduction.md
Look at the architecture diagrams
Run the tiny overfit / labs and the CPU test suite
Explore src/minifrontier/ (especially model.py, attention.py, rope.py, etc.)


## What we are building

One compact codebase exposes two presets:

| | MiniFrontier Edu | MiniFrontier Modern |
| --- | --- | --- |
| Normalization | Pre-RMSNorm | Pre-RMSNorm + QK-Norm |
| Position | RoPE | RoPE; global NoPE as an experiment |
| Attention heads | MHA | GQA |
| Attention span | Full causal in every layer | 3 local layers, then 1 global layer |
| Feed-forward | Dense SwiGLU | Dense SwiGLU |
| Embeddings | Tied input/output | Tied input/output |

The learning progression is deliberate:

```text
manual attention -> SDPA -> RoPE -> MHA -> GQA -> QK-Norm
                 -> full vs hybrid attention -> KV cache
                 -> AdamW vs Muon -> FIM -> SFT
```

The neural architecture and decoding machinery should remain small enough to read in an afternoon and clear enough to explain on a whiteboard.





### What it is
clean “from first principles to modern small LLM” project

**MiniFrontier** (the core of the repo) is a clean, from-scratch PyTorch implementation of a decoder-only language model designed for learning. It targets a single consumer GPU and deliberately keeps the code small and readable enough to study in an afternoon.
**It's a solid, high-quality educational repo** — especially if your goal is to deeply understand how modern decoder-only transformers (LLMs) work under the hood.

It offers two presets that share the same codebase:
- **Edu**: Classic modern baseline (pre-RMSNorm, RoPE, full causal MHA, SwiGLU, tied embeddings).
- **Modern**: Adds practical upgrades — GQA, QK-Norm, hybrid local/global attention (3 local + 1 global), optional NoPE experiments.

Supported sizes go from tiny toy models (~29k parameters for teaching) up to planned 50M / 150M (canonical target) / 350M / 500M presets( and 1B-3B model later for modern preset). It includes a 16k-token byte-level BPE tokenizer, full training loop (AdamW + experimental Muon), KV-cache generation, FIM/code data handling, assistant-only SFT, evaluation tooling, and export paths.

### Strengths (why it’s good)
- **Excellent teaching material**. The `introduction.md` is one of the best plain-language walkthroughs I’ve seen. It explains next-token prediction, tokens, residual streams, RoPE, attention, etc., with zero math prerequisites and maps every concept directly to the source files.
- **Clean, intentional architecture**. Explicit teaching paths (manual attention) sit alongside optimized ones (SDPA/GQA). Residual stream, pre-norm, etc., are treated as first-class concepts.
- **Strong engineering hygiene**. 182 CPU tests (most in the default gate), rigorous parity checks, deterministic data pipelines, provenance tracking for code data, checkpoint/resume, safetensors, overfit proofs, and detailed task backlog with acceptance criteria.
- **Modern techniques without bloat**. GQA, hybrid attention, QK-Norm, FIM, Muon lab, SFT — all implemented in a focused way. Explicitly excludes MoE, multi-GPU, custom kernels, agents, etc., so it stays educational.
- **Well-organized**. Clear structure (`src/minifrontier/`, labs, tests, configs, tasks/), good docs (`plan.md`, architecture diagrams, AGENTS.md), and a sensible learning progression.

in addition to /src folder there is /scripts and /labs folders , and documentation


Current limitations
- Single-GPU / educational scope only — not a production framework or high-performance training stack.

 If you want to understand *why* modern LLMs are built the way they are (attention, RoPE, GQA, residuals, training loop, decoding, SFT, etc.) 
 by reading and running real code rather than high-level frameworks, this is one of the cleaner and more thoughtfully designed options available right now. 

Start with `introduction.md`, then the tiny models and labs — that’s clearly the intended path.

## Frozen V1 targets

| Preset size | Layers | Width | Q heads | Modern KV heads | SwiGLU width | Context | Role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50M | 14 | 512 | 8 | 2 | 1,408 | 1,024 | Development, tests, experiments |
| 150M | 20 | 768 | 12 | 4 | 2,048 | Canonical Edu/Modern release |
| 350M | 28 | 1,024 | 16 | 4 | 2,816 | Optional scale check |
| 500M | 24 | 1,280 | 20 | 4 | 3,456 | Optional stretch target |

All sizes use a single 16,384-token byte-level BPE tokenizer, 64-dimensional attention heads, bias-free linear layers, tied embeddings, RMSNorm epsilon `1e-6`, and dropout `0` by default.

The required hardware target is one NVIDIA GPU with 24 GB for comfortable 50M/150M work; larger presets require profiling and may need smaller microbatches or activation checkpointing. CPU mode supports setup, correctness tests, and small labs.

8GB will work on 50M-150M , for 500M+ you need 24GB gpu

Labs should construct tiny_edu(n_layers=4) whenever comparing architectures with tiny_modern so it will be comparable


also,
Real validation results — and this directly answers the overfitting question the external feedback raised:
 488,280/488,280 updates, 999.0M tokens, final train loss 3.43
┌───────────────┬────────┬───────────────────┬───────────────────────────────────┐
│               │ Before │ After (1B tokens) │ vs. earlier 150M-token checkpoint │
├───────────────┼────────┼───────────────────┼───────────────────────────────────┤
│ Cross-entropy │ 10.17  │ 3.49              │ 3.91 → 3.49                       │
├───────────────┼────────┼───────────────────┼───────────────────────────────────┤
│ Perplexity    │ 26,194 │ 32.8              │ 50.1 → 32.8 (34% better)          │
├───────────────┼────────┼───────────────────┼───────────────────────────────────┤
│ Bits/byte     │ 3.45   │ 1.18              │ 1.32 → 1.18                       │
└───────────────┴────────┴───────────────────┴───────────────────────────────────┘

Validation loss dropped right alongside training loss — no overfitting signal. The extra ~850M tokens produced a real, substantial generalization improvement, not just memorization of recent batches.

## Model architecture assessment

MiniFrontier has a sound architecture and an unusually good educational neural core. 
Edu is a clean LLaMA-style baseline; 
Modern adds coherent, relevant changes without turning the repository into a framework.

From configs/350m-modern.toml:
~350M parameters (28 layers, d_model=1024, 16 query heads / 4 KV heads → GQA, SwiGLU d_ff=2816, tied embeddings, 16k vocab)
Context: 2048

Coding completion / FIM: Promising. The project already has a deliberate, provenance-aware code + FIM path (15% rate, PSM-style, evaluation harnesses). A well-trained 350M Modern can become a useful local autocomplete / infill model for simple-to-medium tasks, especially if you keep feeding it good code data. It will not match 7B–14B specialized coding models.
General chat: Possible after solid pretraining + SFT, but limited. Expect something closer to a lightweight local assistant (short context, weaker reasoning, narrower knowledge) rather than a daily driver that competes with current small open models (1–3B class) or anything larger. The README itself notes that 1B–3B already requires rented multi-GPU hardware.

Practical ceiling on a single consumer GPU:

350M is a reasonable sweet spot for training from scratch at home.
Going much beyond that (or training for tens of billions of tokens) quickly becomes painful without cloud rentals.
Inference of a finished 350M (quantized) will be fast and pleasant on the same hardware.


As a future general-chat + coding model: Good foundation. A carefully trained 350M Modern can be a useful local tool (especially for code completion), but it will remain in the “small model / educational / specialized local” tier, not a replacement for larger open or closed models.
If you want to push it toward usable chat/coding, prioritize: longer high-quality pretraining on a good mixture, stronger code/FIM weighting, solid SFT, and then quantization


  The core implementation appears mathematically correct from static review and existing evidence:

  - Correct pre-RMSNorm residual ordering.
  - Correct bias-free SwiGLU.
  - Split-half LLaMA RoPE with an independent Transformers parity test.
  - Proper causal and offset masking.
  - Manual FP32 attention as a readable reference.
  - Fused full-context SDPA.
  - Native compact GQA without repeating K/V on optimized paths.
  - Correct shifted next-token loss and token-weighted gradient accumulation.
  - Tied embeddings and depth-scaled residual initialization.
  - Well-designed linear and ring KV caches with rollback and absolute-position handling.

  I would describe it as “ready for hardening and real training,” not “proven release-ready.” The canonical 150M models are still untrained, 
  


   Model          Layers    Width    Q/KV heads      FFN    Attention              Parameters
  ━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━
   tiny Edu            2       32           4/4       96    Full MHA                   28,832
  ─────────────  ────────  ───────  ────────────  ───────  ────────────────────  ─────────────
   tiny Modern         4       32           4/2       96    3 local + 1 global         51,552
  ─────────────  ────────  ───────  ────────────  ───────  ────────────────────  ─────────────
   50M Edu            14      512           8/8    1,408    Full MHA               53,361,152
  ─────────────  ────────  ───────  ────────────  ───────  ────────────────────  ─────────────
   50M Modern         14      512           8/2    1,408    Hybrid GQA             47,857,920
  ─────────────  ────────  ───────  ────────────  ───────  ────────────────────  ─────────────
   150M Edu           20      768         12/12    2,048    Full MHA              154,172,160
  ─────────────  ────────  ───────  ────────────  ───────  ────────────────────  ─────────────
   150M Modern        20      768          12/4    2,048    Hybrid GQA            138,446,080


These proportions are sensible:

  - Head dimension 64 in production presets is conventional and efficient.
  - d_ff ≈ 2.67 × d_model is appropriate for a three-matrix SwiGLU and roughly preserves the parameter cost of a traditional 4× two-matrix MLP.
  - A 16K vocabulary is a strong choice for models this small because it limits embedding and softmax cost.
  - GQA ratios of 4:1 at 50M and 3:1 at 150M are reasonable.
  - Tied embeddings are particularly valuable at this scale.
  - RoPE everywhere by default is the safe choice; global NoPE should remain experimental.


- [Future plan for 350M Modern model, 32K tokenizer - 2048 context](future-plan.md)
better for general chat and coding completions

(1B-3B need to lease paid servers to train the model , not feasable on home pc single gpu,
7B+ needs massive resources)

## V1 scope

V1 includes:

- model and tokenizer training from scratch;
- an explicit teaching attention path and optimized PyTorch SDPA/GQA path;
- streamed, filtered, deduplicated, tokenized, and packed data;
- a conservative, provenance-preserving code mixture and FIM examples;
- an explicit AdamW training loop with BF16 and optional compilation;
- KV-cached generation, checkpoint/resume, and safetensors export;
- validation loss, perplexity, bits-per-byte, a small lm-eval suite, and code/FIM evaluations;
- a first-party Muon versus AdamW laboratory;
- small assistant-only supervised fine-tuning and a simple chat template.

V1 explicitly excludes MoE, MLA, DeltaNet, MTP, RL/GRPO, tool-using agents, vision, distributed training, custom CUDA/Triton kernels, serving frameworks, and production long-context scaling.

## Roadmap

Implementation is organized into dependency-ordered tasks with measurable acceptance criteria:

- [Human-readable backlog](tasks/backlog.md)
- [Jira issue-export-style XML](tasks/jira-issues.xml)
- [Task workflow and completion gates](tasks/README.md)
- [Full architecture rationale](plan.md)
- [Canonical V1 evaluation release gate](docs/EVALUATION_RELEASE_GATE.md)

The critical path is:

```text
Foundation -> M0 math -> M1 Edu -> M2 tokenizer/data -> M3 inference
           -> evaluation gate -> M4 Modern -> M5 performance
           -> M6 FIM/code -> M7 Muon -> M8 SFT -> M9 150M release
```

M10 (350M/500M scale checks) and M11 (Transformers/vLLM/GGUF adapters) are post-V1 and cannot block the educational release. Their software harnesses may land before the deferred RTX session; measured decisions and compatibility claims remain gated on real canonical artifacts and external-runtime runs.

## Planned repository layout

```text
configs/                 Frozen model/training presets
src/minifrontier/        Readable neural core and runtime
train/                   From-scratch pretraining and SFT loops
scripts/                 Tokenizer, data, train, eval, sample, chat, export
src/minifrontier/evaluation/  Language, code, FIM, and benchmark runtime
eval/                    Versioned suites and evaluation-only fixtures
labs/                    Single-variable educational experiments
tests/                   Primitive, model, cache, generation, and I/O tests
templates/               Simple chat serialization
artifacts/               Ignored local run outputs and checkpoints
tasks/                   Ordered Markdown and Jira-form backlog
docs/                    Research references
```

## Bootstrap

MiniFrontier requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). On Windows, install `uv`
with any one of these methods.


- [Install instructions](install.md)

Official PowerShell installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Windows Package Manager:

```powershell
winget install astral-sh.uv
```

Or, when Python and `pip` are already available:

```powershell
python -m pip install --user --upgrade uv
```

also you can install triton-windows, for possible 16% speedup

Open a new terminal if the installer changed `PATH`, then verify the prerequisites:

```powershell
python --version
uv --version
```

Windows users can then initialize the checkout with the idempotent bootstrap:

```bat
init.cmd cu130
```

Available PyTorch backends are `cpu`, `cu126`, `cu130`, and `cu132`; `cu130` is the project default. Choose the wheel supported by the installed NVIDIA driver, or use `cpu` for correctness work. Backends are mutually exclusive locked extras, so `init.cmd cpu` on a Dev Box and `init.cmd cu130` on an RTX machine use the same project metadata safely. The default sync installs the core and development groups. Add `--all-groups` when evaluation and plotting dependencies are needed. The script creates missing directories and package markers but never overwrites an existing file.

The normal developer loop is:

```bat
uv sync --extra cpu --group dev
uv run --extra cpu ruff check .
uv run --extra cu130 python -m pytest
```

Do not start a serious training run until primitive tests, Edu overfit, data tests, checkpoint resume, and baseline evaluation gates pass.

test with
https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu
dataset



to chat/test it you can run this:
sample.py is exactly right for that — that's the plain prompt-completion tool. A couple of notes:

Use sample.py for this checkpoint, not chat.py. 
chat.py exists (template-aware, multi-turn, interactive REPL) but it's designed for an SFT'd/instruction-tuned model — 
it wraps your input in the chat template. Your mf065-150m-modern-1b-release is a base model (no SFT ever ran on it), 
so it was never trained to follow that template; feeding it through chat.py would likely just produce confused output. 
sample.py with a plain prompt (like you already used) is the right tool for a base model — no new script needed, this is exactly what it's for.

Your own prompt:
```bat
./.venv/Scripts/python.exe scripts/sample.py --model artifacts/mf065-150m-modern-1b-release --prompt "<your prompt here>" --device cuda --precision float16 --temperature 0.7 --top-k 40 --max-new-tokens 100
```

A few tweaks from your example worth knowing:
- Drop --seed 42 --temperature 0.0 unless you specifically want deterministic greedy output — temperature 0 is why the earlier samples looked repetitive ("the world was created in the first place..." looping). 

or prompt like "The history of science shows that"

Try --temperature 0.7 --top-k 40 (or --top-p 0.9) for more natural variety.

- if GPU is busy with training/eperiment . Running sample.py concurrently would contend for the same 8GB — either wait a bit, 
or add --device cpu for a quick test now (slower, but a 150M model on CPU for a short completion is still fine, just not instant).


Given the model was trained on FineWeb-Edu (educational web text — explanatory articles across science, history, math, health, etc.), the best prompts are short, factual sentence-starters in that same register — not questions or instructions, since this is a base completion model, not a chat model(needs SFT'd/instruction-tuned model). Here's a set of 10 that span the dataset's actual content mix:

1. "The history of science shows that" (baseline — already tested, good for before/after comparison)
2. "Photosynthesis is the process by which plants" — biology/textbook
respond like:
"Photosynthesis is the process by which plants use a chemical process to generate light and light. Since the process involves the use of light, the plant is able to produce light and light in a similar manner as the plant itself. The process of plants producing light has been a significant part of modern agriculture."

3. "The French Revolution began in" — history/factual
4. "In mathematics, a prime number is defined as" — math/definition
5. "Climate change is caused primarily by" — science/current topics
6. "The human digestive system consists of" — biology/health
7. "a <name> War ended in the year" — history/dates
8. "The water cycle describes how" — earth science
9. "To bake a loaf of bread, you first need to" — practical/how-to
10. "The theory of atoms was first proposed by" — science history

 Set expectations honestly: this checkpoint is at ~6.7 tokens/param (1B tokens ÷ 150M params),
 well short of Chinchilla-optimal (~20 tokens/param, ~3B tokens) — expect grammatically coherent 
 but factually shaky/generic completions, especially past the first sentence or two.

## Implemented CPU checks

The current code supports the Edu/Modern path, resumable training, code/FIM, Muon, and assistant
SFT/chat experiments:

- validated 50M/150M Edu and Modern configuration files;
- reference-tested RMSNorm and SwiGLU;
- explicit causal and exact-width local masks;
- readable scaled dot-product attention and fused-eligible causal SDPA parity;
- split-half RoPE with parity against the Transformers Llama primitive;
- pre-norm Transformer blocks, width/depth-aware initialization, and tied embeddings;
- a frozen byte-level BPE contract, provenance validation, deterministic splitting, and EOS-aware packing;
- preallocated KV-cached prefill/decode, robust sampling, exact checkpoint resume, and safe release export;
- validation cross-entropy/perplexity/BPB, lm-eval adapter/config, code/FIM fixtures, and comparable benchmark records;
- exact 53,361,152-parameter construction for the frozen 50M Edu preset;
- native compact-K/V GQA, pre-RoPE QK-Norm, cached FlexAttention local masks, the 3:1
  local/global schedule, and isolated global NoPE;
- token-weighted gradient accumulation, update-indexed warmup/cosine AdamW, CPU-side batch
  validation, exact deterministic checkpoint/data-cursor resume, optional compile and whole-block
  activation checkpointing;
- disk-backed exact/near deduplication, immutable hashed memory-mapped token shards, deterministic
  resumable shard/row shuffle, direct pinned FineWeb-Edu preparation, stable splits across FIM
  transforms, code-license/sensitive-data admission, deterministic 15% PSM FIM, and versioned
  code/FIM scoring;
- mixed-capacity KV caching with bounded local ring storage, full-history global storage, absolute
  RoPE positions, wrap/rollback parity, and mask-free single-token local SDPA decode;
- an annotated Newton-Schulz lab plus first-party `torch.optim.Muon`/AdamW disjoint parameter
  partitioning, checkpointable combined optimization, and separate-LR matched A/B tooling;
- deterministic Jinja chat serialization, provenance-complete conversation ingestion,
  assistant-only token masks, whole-turn truncation, packing, resumable SFT, a versioned regression
  prompt set, and bounded multi-turn chat generation;
- draft/frozen training-protocol validation, release generation metadata, complete SHA-256
  manifests with tamper detection, custom model cards, and matched Edu/Modern load audits;
- a CPU-friendly 100-example overfit harness that reaches below `1e-3` nats/token;
- a one-step packed-real-text 50M CPU integration scorecard in
  [`reports/50m-edu-cpu-smoke.md`](reports/50m-edu-cpu-smoke.md).

Additional bounded engineering records are
[`reports/m4-50m-cpu-smoke.json`](reports/m4-50m-cpu-smoke.json),
[`reports/m5-50m-cpu-inference-smoke.json`](reports/m5-50m-cpu-inference-smoke.json), and
[`reports/m6-50m-cpu-smoke.json`](reports/m6-50m-cpu-smoke.json).

Run the current verification suite with:

```bat
init.cmd cpu
uv run --extra cpu pytest
uv run --extra cpu ruff check .
```

The overfit proof can also be run directly:

```bat
uv run --extra cpu minifrontier-overfit --device cpu --steps 700
```

Run the bounded full-model integration gate or evaluate an exported release with:

```bat
uv run --extra cpu python scripts/smoke_50m.py
uv run --extra cpu python scripts/smoke_modern_50m.py
uv run --extra cpu python scripts/profile_model.py --help
uv run --extra cpu python train/pretrain.py --help
uv run --extra cpu python scripts/prepare_code.py --help
uv run --extra cpu python scripts/compare_fim.py --help
uv run --extra cpu python scripts/eval_code.py --help
uv run --extra cpu python scripts/smoke_muon.py
uv run --extra cpu python scripts/smoke_sft.py
uv run --extra cpu python scripts/compare_optimizers.py --help
uv run --extra cpu python train/sft.py --help
uv run --extra cpu python scripts/eval_sft.py --help
uv run --extra cpu python scripts/freeze_protocol.py --help
uv run --extra cpu python scripts/audit_release.py --help
uv run --extra cpu python scripts/preflight_scale.py --help
uv run --extra cpu python scripts/assemble_scale_measurement.py --help
uv run --extra cpu python scripts/decide_scale.py --help
uv run --extra cpu python scripts/export_huggingface.py --help
uv run --extra cpu python scripts/create_serving_fixture.py --help
uv run --extra cpu python scripts/smoke_vllm_api.py --help
uv run --extra cpu python scripts/convert_gguf.py --help
uv run --extra cpu python scripts/quantize_gguf.py --help
uv run --extra cpu python scripts/build_source_archive.py --output tmp/minifrontier-source.zip
uv sync --extra cpu --group dev --group eval
uv run --extra cpu python scripts/eval.py --release artifacts/my-release --help
```

The scorecard is intentionally an engineering result. Standard lm-eval datasets and the longer
FineWeb-Edu smoke are not run or claimed on the CPU-only Dev Box.

The source-archive command is the supported way to prepare a review or GitHub upload ZIP. It
includes `.github/workflows/ci.yml`, fails instead of overwriting an existing archive unless
`--force` is supplied, and excludes bytecode/caches, corpora, checkpoints, local artifacts, and
third-party research/reference bundles. Model release artifacts are built and audited separately.

## Development principles

- Tests precede serious training.
- Eager FP32 is the correctness baseline; optimized modes must prove parity.
- Experiments change one variable and keep tokenizer, data, token budget, batch tokens, context, and evaluation matched.
- Training and benchmark claims include their seeds, hardware, dependency versions, and raw run records.
- Code training data is admitted only with explicit license and source provenance.
- No important model line should depend on a high-level architecture framework.

Repository instructions for coding agents are in [`AGENTS.md`](AGENTS.md); `CLAUDE.md` points to the same canonical rules.

## Runtime compatibility status

The current M9 release format is a safe, self-loading MiniFrontier/PyTorch artifact. M11 now provides
a separate Transformers repository exporter with local Auto-class parity tests and a vLLM
Transformers-backend validation harness. Canonical trained exports, a pinned Hub smoke, and the WSL2
CUDA runtime gate remain open. llama.cpp/GGUF support is stricter: conversion still fails closed until
a distinct upstream MiniFrontier C++ graph exists. Uploading files to Hugging Face makes them
downloadable, but only the matching adapter and runtime gates earn a compatibility claim.

The post-V1 [ecosystem-adapter tasks](tasks/backlog.md#m11--ecosystem-adapters-post-v1-not-part-of-the-neural-core)
make each claim independently: MF-071 adds and parity-tests the Transformers/Hugging Face export;
MF-072 validates vLLM on the Windows 11 machine through WSL2 CUDA; MF-073 adds high-precision
GGUF/llama.cpp architecture support; and MF-074 produces and evaluates `Q4_K_M` artifacts on native
Windows CUDA. The raw-PyTorch implementation and canonical checkpoints remain the reference.

See the [ecosystem compatibility matrix](docs/ECOSYSTEM_COMPATIBILITY.md) for the exact Hugging Face,
vLLM, OpenCode, Cline, Roo Code, Kilo Code, Aider, and tool-calling gates. Interactive native chat uses
the concise [default general/coding system prompt](templates/system_prompt.md); it is guidance, not a
technical requirement, and the CLI allows overriding or disabling it.

### vLLM and Vercel AI SDK usage after the external-runtime gate

> The export and client harness now exist, but this example is not a verified runtime claim until
> MF-071 exports the canonical checkpoint and MF-072 passes native-versus-vLLM parity and
> OpenAI-compatible API tests under WSL2 CUDA. See
> [`docs/VLLM_WSL2.md`](docs/VLLM_WSL2.md).

Install the [Vercel AI SDK OpenAI-compatible provider](https://ai-sdk.dev/providers/openai-compatible-providers):

```bash
npm install ai @ai-sdk/openai-compatible
```

Point it at the [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/):

```ts
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText, streamText } from "ai";

const vllm = createOpenAICompatible({
  name: "vllm",
  baseURL: "http://localhost:8000/v1",
  apiKey: "local-key",
  includeUsage: true,
});

const model = vllm.chatModel("minifrontier-150m-modern");

const { text } = await generateText({
  model,
  system: "You are MiniFrontier, a helpful coding assistant.",
  prompt: "Write a TypeScript binary search function.",
});

console.log(text);
```

`local-key` must match the key passed to `vllm serve --api-key`. Streaming uses the same provider:

```ts
const stream = streamText({
  model,
  prompt: "Explain binary search one step at a time.",
});

for await (const chunk of stream.textStream) {
  process.stdout.write(chunk);
}
```

Conversation messages use the same model and MiniFrontier chat template:

```ts
const result = await generateText({
  model,
  messages: [
    { role: "system", content: "You are a concise coding assistant." },
    { role: "user", content: "Explain this function." },
  ],
});
```

Ordinary application loops are also valid:

```ts
for (const prompt of prompts) {
  const { text } = await generateText({ model, prompt });
  console.log(text);
}
```

`generateText`, `streamText`, prompts, system messages, conversation messages, and application-managed
loops are transport-compatible with vLLM's Chat Completions API. Vercel AI SDK automatic tool loops
are a different capability: MiniFrontier V1 has no trained tool/function-call protocol or vLLM tool
parser, so `ToolLoopAgent`, tool-driven `stopWhen` loops, and agent tool use are not claimed.

After MF-072, OpenCode, Cline, Roo Code, Kilo Code, and Aider may be pointed at the same base URL and
served model ID for text-only chat/completion smokes. Configure the real 1K/2K context limit, disable
tool calling where the client permits it, and do not infer reliable repository editing from a
successful API connection: the canonical 150M model is an educational model with a small context, not
a production coding-agent model.

## Research background

The local [`docs/`](docs/) collection and [`more-context.md`](more-context.md) contain the research trail. The frozen design in [`plan.md`](plan.md) draws on the original Transformer, RoPE, LLaMA-style RMSNorm/SwiGLU, GQA, modern QK normalization, hybrid local/global attention, FIM, and small-model training practice. The [inference reference review](docs/INFERENCE_REFERENCE_REVIEW.md) records the exact local Grok-1, vLLM, SGLang, and Llama-oriented snapshots used to audit M3–M5. These references motivate the implementation; they are not copied as a framework dependency.

The [research source review](docs/RESEARCH_SOURCE_REVIEW.md) identifies every local PDF/context
snapshot by hash, separates historical foundations from out-of-scope frontier mechanisms, and
records which conclusions changed the remaining tasks.

## License

MiniFrontier source is available under the [Apache License 2.0](LICENSE). Research references, training datasets, trained weights, and other third-party artifacts retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md) and the [data-governance policy](docs/DATA_GOVERNANCE.md).
