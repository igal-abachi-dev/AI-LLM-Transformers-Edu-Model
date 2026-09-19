# MiniFrontier labs

Ten small, runnable scripts, each isolating exactly one concept from
[`introduction.md`](../introduction.md) and putting a real number or a real printed
tensor behind it. None of these take command-line arguments — just run the file. None
need a GPU; all are sized to run on CPU in seconds.

```bash
uv run --extra cpu python labs/00_attention_math.py
```

Read `introduction.md`'s matching section first for the plain-language explanation, then
run the lab to see the actual numbers — that's the intended order, not the other way
around. Every lab's own module docstring has a longer "What this lab shows" writeup if you
want more than the one-line summary below.

| # | Lab | What it shows | `introduction.md` section |
|---|-----|----------------|-----------------------------|
| 00 | `00_attention_math.py` | Manual attention derived by hand, on tensors small enough to trace with a pencil — every intermediate step printed, nothing hidden inside a matmul. | **2.4 Attention, explained with a classroom** |
| 00b | `00b_attention_pure_python.py` | The same attention math with zero libraries at all — no PyTorch, no NumPy, just Python lists and loops — plus a mapping from each function to the real ATen/cuBLAS/CUDA source file that does the same job at production speed. | **2.4 Attention, explained with a classroom** |
| 01 | `01_rope.py` | RoPE's rotation angle growing with position, and that what actually matters between two tokens is the *difference* in their rotations. | **2.2 Step two: stamping positions (RoPE)** |
| 02 | `02_mha_vs_gqa.py` | The real parameter and KV-cache numbers MHA versus GQA produce — not just the idea, the actual memory difference. | **3.1 GQA — four askers share two note-takers** |
| 03 | `03_qk_norm.py` | The same Modern forward pass run twice, once with QK-Norm on and once off, side by side. | **3.2 QK-Norm — a volume limiter on the question and the name tag** |
| 04 | `04_full_vs_hybrid.py` | The exact attention-mask density difference between full and local/hybrid attention, with no speed or quality claim attached. | **3.3 Hybrid attention — three near-sighted layers, one far-sighted** |
| 05 | `05_kv_cache.py` | Cached decode producing identical results to uncached recomputation, plus how cache memory actually grows with context length. | **2.8 The notebook that makes chat fast (KV cache)** |
| 06 | `06_adamw_vs_muon.py` | The educational Newton–Schulz iteration used to explain first-party Muon's orthogonalization step. | **3.7 Muon — an optimizer that treats matrices as matrices** |
| 07 | `07_rope_vs_global_nope.py` | The global RoPE-versus-NoPE flag isolated on identical random weights, so the only thing that differs is that one setting. | **3.4 NoPE on global layers (the experiment)** |
| 08 | `08_mtp.py` | What Multi-Token Prediction's extra heads are actually graded on, beyond the ordinary next-token loss. | **3.8 MTP — Multi-Token Prediction (optional auxiliary loss)** |
| 09 | `09_chunked_vs_fused_cross_entropy.py` | Why the final vocabulary-sized projection, not the transformer body, is what actually breaks memory at scale. | **Chunked cross-entropy (CCE) — computing the loss without exploding memory** |

## Scope

These labs stay inside MiniFrontier's own frozen architecture (see `AGENTS.md`) — every one
of them either exercises a real, implemented piece of this repository directly (importing
from `minifrontier`), or, like `00b`'s zero-dependency reimplementation, derives the exact
same algorithm independently for teaching purposes. They are not a survey of techniques from
other model families; nothing here demonstrates an architecture this codebase doesn't
actually build.

## If you want the broader survey: rasbt/LLMs-from-scratch , is a great repo/book

For architectures and techniques *outside* MiniFrontier's own frozen scope, the strongest
resource found while benchmarking this repo against comparable educational projects is
Sebastian Raschka's [`LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) and
its companion book (*Build a Large Language Model (From Scratch)*). What makes it worth
knowing specifically:

- **A real book, not just a repo** — chapter-by-chapter prose paired one-to-one with
  runnable notebooks, which is the single biggest reason it reads as more polished
  curriculum than most educational LLM repos, this one included.
- **Actually kept current** — it has grown past the original GPT-2-style walkthrough to
  include from-scratch implementations of Qwen3, Gemma 3, GQA, DeepSeek Sparse Attention,
  and Muon, so it tracks the field rather than freezing at a 2022-era architecture.
- **Consumer-hardware friendly**, same spirit as this repo's own reference-GPU discipline.

it's a broad tour across many architecture families (useful precisely because it covers the families this repo
deliberately doesn't), where MiniFrontier is a single, narrower, from-scratch model family
taken deep enough to fully explain and to actually train and release on one consumer GPU.
its great Complementary reading.
