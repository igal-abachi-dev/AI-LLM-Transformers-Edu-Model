# Inference reference review

Reviewed on 2026-08-18 before MF-039. This is a local-source comparison, not a claim that
MiniFrontier implements or is compatible with any external serving engine.

## Reviewed snapshots

| Reference | Local artifact | SHA-256 | Scope inspected |
|---|---|---|---|
| xAI Grok-1 | `docs/grok-1-main.zip` | `23e0ffa2d89073473b0e7ef32c94c3a7067dcad53827247438f7732aba68357c` | Entire archive: all Python, configuration, README, and checkpoint/inference code |
| vLLM | `inference/vllm-main.zip` | `4ce4f720ffcdfb8e3948cac3d12da0cba36a091d5d3ff7aab0a40d2f4885d26` | Archive inventory plus model, sampling, logits, KV-cache specification, worker, and benchmark paths relevant to V1 |
| SGLang | `inference/sglang-main.zip` | `cfe2abf70611abe8eb7696d5e4547c3b7080515ca45ba6bfc0cdc665e18124b2` | Archive inventory plus model, sampling, precision, sliding-window cache, scheduler, and benchmark paths relevant to V1 |
| Llama 4 text model | `inference/modeling_llama4.py` | `4a35a606e1a12bb1f5df6ead8b214f41e11d363db8c024bbe682873d9743bb85` | Text attention, RoPE, cache, hybrid masks, Q/K normalization, and selective logits |
| Minimal PyTorch sketch | `inference/pytorch.py` | `f9d5d7f27e6624a4c5a7b93f9aa476c6595e2104dc0cc686ee73fd7564b094bd` | Architecture and uncached generation teaching sketch |
| Inference notes | `inference/inference.md` | `842b1f41a24c95346ac5b0f1bfb3130852cb2397fa0fda271e77987f98ed761b` | Proposed educational/deployment lifecycle; treated as non-authoritative commentary |

The ZIP filenames do not encode immutable upstream revisions. Their hashes identify the exact
local inputs to this review. They remain reference material and are not imported, executed, or
packaged with MiniFrontier model releases. MF-068 must make an explicit redistribution decision
before the GitHub release; the preferred public form is a pinned upstream link plus these hashes,
not duplicated third-party source archives.

## Verdict

MiniFrontier's M3 inference math is correct for its documented fixed-shape, single-stream scope.
No model redesign is justified. The strongest existing choices are:

- split-half RoPE with absolute cached offsets;
- fused-eligible full prefill, unrestricted single-query decode, and offset-masked chunks;
- preallocated per-layer K/V storage without history concatenation;
- cache rollback after a failed layer, capacity errors instead of position restart, and
  full/chunk/token cached parity;
- FP32 sampling probabilities, seeded generation, per-row EOS handling, and restored model mode;
- safetensors release weights and an explicit trust boundary around local pickle training state.

Grok-1 independently reinforces split-half RoPE, grouped Q/KV heads, FP32 normalization/attention
softmax, and valid-length masking of preallocated cache storage. Its MoE, JAX sharding, attention
logit soft-cap, embedding/output multipliers, unusual residual normalization, bucketed multi-device
runner, and pickle checkpoint loader are specific to that checkpoint and are not MiniFrontier V1
requirements.

## Work mapped to existing tasks

### MF-041 / MF-043 — correctness before compact storage

M4 uses a full-history local cache as the simple reference and enforces the local window in the
attention mask. FlexAttention owns the optimized local compute path and native GQA. This mirrors
the useful separation in production engines between attention semantics and cache allocation.
M4 must not claim a local-cache memory saving.

### MF-046 — mixed-precision inference contract

The cache dtype must follow projected K/V tensors under the selected inference precision. It
cannot be inferred from embedding output under autocast. Sample, chat, and evaluation entry points
will share explicit `auto|float32|bfloat16` selection and capability checks.

### MF-048 — compile claims are path-specific

Training/prefill and token decode have different shapes and mutation behavior. A successful
training compile does not prove cached decode compiled; graph breaks and fallbacks are recorded
separately.

### MF-050 — single-stream inference hardening

Before performance claims, M5 will:

- request only the last logit row for ordinary prefill/decode instead of constructing unused
  `[batch, sequence, vocabulary]` logits;
- avoid copying the entire generated prefix with `torch.cat` on every token;
- reject non-finite temperature clearly, test top-k/top-p boundary behavior against hand-computed
  cases, reuse one softmax for nucleus filtering, and keep non-finite-logit checks out of the normal
  CUDA hot path unless validation/debug mode is enabled;
- implement an optional bounded local ring/window cache only after the full-history reference is
  correct, preserving absolute RoPE positions and chronological initialized reads across wrap;
- report time-to-first-token, prefill throughput, inter-token latency, decode throughput, peak
  allocated/reserved VRAM, and logical/allocated cache bytes at meaningful context lengths.

These are single-stream or fixed-shape batch measurements. Continuous batching, paged allocation,
prefix/Radix caching, speculative decoding, request scheduling, cache offload, quantized cache,
distributed serving, and OpenAI-compatible APIs remain outside V1.

The bounded implementation now exists alongside the full-history reference. Local layers allocate
only the configured window, global layers retain the full requested capacity, and wrap/chunk/reset/
rollback parity is tested. Token-by-token local decode uses SDPA over the bounded chronological
view, so Flex block masks remain shape-cached for training/prefill instead of growing with every
absolute decode position. CUDA throughput and VRAM conclusions still require MF-050/MF-063.

### MF-062 / MF-067 — honest interface and release metadata

The later SFT chat CLI preserves whole message/template boundaries during truncation and documents
its single-user scope. Final releases add a generation configuration and complete SHA-256 manifest,
but do not claim Hugging Face, vLLM, SGLang, GGUF, or server compatibility without a separately
tested adapter.

### MF-071–074 — ecosystem compatibility is an adapter deliverable

The inspected `vllm/model_executor/models/llama.py` is not a generic loader for every
Llama-shaped decoder. It reads a Hugging Face Llama configuration (`hidden_size`,
`num_attention_heads`, `num_key_value_heads`, `num_hidden_layers`, and related fields), expects
Llama module/tensor names, and explicitly maps separate Q/K/V checkpoint tensors into vLLM's packed
projection. MiniFrontier currently exports its own configuration names (`d_model`, `n_heads`,
`n_kv_heads`, `n_layers`) and its own state-dict layout. More importantly, Modern adds QK-Norm,
per-layer local/global attention, and optional global NoPE, so labeling it `LlamaForCausalLM` would
misdescribe the graph.

The first serving route is therefore a real MiniFrontier Transformers adapter followed by vLLM's
Transformers modeling backend. The adapter must expose base-model `auto_map` metadata, forward
backend kwargs through attention, use the Transformers attention interface, and pass native/vLLM
parity tests. A vLLM out-of-tree plugin is the fallback if the frozen architecture cannot be
expressed correctly through that backend. The reviewed vLLM snapshot's `setup.py` supports Linux
(including WSL) and macOS rather than native Win32, so the Windows 11 NVIDIA acceptance environment
is WSL2 CUDA.

GGUF is a second, independent port. llama.cpp conversion supports registered architectures; a new
graph requires converter metadata/tensor mapping plus loader and compute-graph support. MF-073 first
proves BF16/F16 GGUF parity for both presets. Only then does MF-074 create `Q4_K_M` artifacts and
measure their quality regression. llama.cpp itself has native Windows CUDA releases, so that path is
tested directly on Windows 11 rather than through WSL2.

## Reference cautions

`inference/inference.md` is useful brainstorming, but several broad runtime and performance claims
are unsourced or overgeneralized. It is not a source of truth. In particular, a raw-PyTorch teaching
model does not need to inherit `PreTrainedModel`, and MiniFrontier should not copy a production
serving engine into its core.

The loose Llama 4 file demonstrates shared position computation, distinct full/chunk masks, native
cache objects, and last-logit slicing. Its NoPE attention-temperature tuning and parameter-free
Q/K normalization are architecture-specific. MiniFrontier keeps its frozen QK-Norm-before-RoPE
experiment and does not silently add temperature tuning to the NoPE flag.

## Extension (2026-09-12): llama.cpp and TensorRT-LLM, plus a re-check of newer code

Reviewed on 2026-09-12, prompted directly by the user, extending the 2026-08-18 review above
with two references not covered then, and re-reading MiniFrontier's own inference code as it
exists now (`generation.py`, `cache.py`, `chat.py`, `speculative_decoding.py` -- the last of
these, self-speculative decoding via MTP heads, postdates the original review entirely). Same
framing as above: not a claim of compatibility, not a case for adopting production-serving
complexity into MiniFrontier's core.

| Reference | Local artifact | SHA-256 | Scope inspected |
|---|---|---|---|
| llama.cpp | `inference/llama.cpp-master.zip` | `4e0084bb1870c8ff55a12967fd20e7c01dae9b0319a7e2f48347a347c86df7cb` | `src/llama-kv-cache.h`, `common/sampling.cpp`, `src/llama-batch.h` |
| TensorRT-LLM | fetched live, `github.com/NVIDIA/TensorRT-LLM` `main` branch | not a local artifact -- upstream revision is whatever `main` was on 2026-09-12, not pinned | `tensorrt_llm/_torch/speculative/mtp.py`, `.../pyexecutor/resource_manager.py`, `.../pyexecutor/cuda_graph_runner.py` (file listing only for `_torch/` and `_torch/speculative/` as a whole) |

### llama.cpp: sound convergence, nothing MiniFrontier is missing

`llama_kv_cache` is a much more general structure than MiniFrontier's `KVCache` needs to be --
it supports multiple simultaneous sequences sharing one cache, sequence removal/copy/shift, and
state save/restore, all addressed through an explicit `slot_info` (a vector of *cell indices* a
batch should write to, computed by `find_slot`). Stripped of the multi-sequence/multi-stream
machinery MiniFrontier's single-stream scope doesn't need, the *core* mechanism is the same
pattern `LayerKVCache.append` already uses: compute target write positions, then copy K/V into
them. This is real, independent confirmation that MiniFrontier's simpler design isn't a
naive shortcut -- it's the same idea, correctly scoped down.

`common/sampling.cpp`'s real sampler chain (`llama_sampler_init_top_k/top_p/min_p/penalties`,
plus frequency/presence penalties, XTC, typical-p, top-n-sigma) overlaps substantially with
`generation.py`'s `sample_next_token` (temperature, top-k, top-p, min-p, repetition penalty,
no-repeat-ngram) -- every mechanism MiniFrontier has, llama.cpp also has, under the same names.
The extras llama.cpp has and MiniFrontier doesn't (frequency/presence penalty as a finer-grained
alternative to a flat multiplicative penalty; XTC and typical-p as newer creative-writing-focused
samplers) are real but optional refinements, not a correctness gap -- consistent with this
project's stated engineering value of covering the standard, well-understood mechanisms clearly
rather than every sampler variant that exists.

**Verdict: no gap found.** MiniFrontier's cache and sampling code converge with llama.cpp's own
on every mechanism actually in scope.

### TensorRT-LLM: sound convergence on speculative decoding, plus real, direct confirmation of a finding already made

TensorRT-LLM's `_torch/speculative/` has a dedicated `mtp.py` -- the same MTP (Multi-Token
Prediction) mechanism MiniFrontier's own `speculative_decoding.py` (MF-093/MF-109) already uses,
independently arrived at. Its accept/reject cycle is the same shape: verify a draft token against
the target model's own verdict in one batched pass, keep state for accepted tokens, undo the
rest (`"Keep the state written for accepted draft tokens, undo the rest"`, its own comment) --
functionally identical to MiniFrontier's `KVCache.truncate` rollback on a rejected draft. **One
real, deliberate design difference worth naming, not a gap**: TensorRT-LLM offers an optional
`use_relaxed_acceptance_for_thinking` mode that accepts draft tokens even without an exact match
for reasoning-mode content, trading strict correctness for a higher acceptance rate.
MiniFrontier's own exactness guarantee (`speculative_generate` is provably bit-identical to plain
greedy decode, tested in `tests/test_speculative_decoding.py`) deliberately has no such relaxed
mode -- a stricter, simpler, more conservative choice, consistent with this project's
correctness-first stance elsewhere (eager execution as the baseline, `AGENTS.md`'s own framing),
not a missing feature.

**Direct, real confirmation of [[MF-115]]'s own finding.** [[MF-115]] (CUDA graphs on the decode
loop, since Dropped) reasoned from first principles that CUDA graph replay requires the "current
position" to live in a persistent tensor updated in place, not a Python int -- MiniFrontier's
`LayerKVCache.append(start_pos: int)` isn't built that way. TensorRT-LLM's real
`cuda_graph_runner.py` does exactly this: `self.shared_static_tensors["position_ids"] =
torch.zeros(...)`, updated every graph replay via `.copy_()` in place, and comments explicitly
noting speculative decoding "updates kv_lens_cuda in-place during every forward." Independent
confirmation of a diagnosis already made and already recorded, not a new finding on its own --
appended to [[MF-115]]'s own backlog note as a citation.

**Verdict: no gap found in the newer speculative-decoding code.** MiniFrontier's rollback design
holds up against a real, production analog of the same mechanism, and where the two differ
(relaxed vs. exact acceptance) it's a documented, deliberate choice on MiniFrontier's side, not
an oversight.

### Jinja templates: wider survey, one real fix already applied, nothing further found

Beyond the three templates the original comparison sampled (Llama-3.1-8B-Instruct, SmolLM3-3B,
Qwen3-0.6B), pattern-scanned all 69 templates in `jinja-templates/` for conventions worth
adopting that aren't tool-calling or reasoning-mode machinery (both already, correctly, out of
`AGENTS.md`'s frozen V1 scope). 31/69 trim or strip message content -- `templates/chat_template.jinja`
now does too (`| trim`, already fixed this session, not a new finding). 41/69 reference
`bos_token` as an externally-supplied Jinja variable rather than a hardcoded literal; MiniFrontier
hardcodes `<|bos|>` -- a real difference, but not a gap: those templates need this because they're
reused across many different base-model configs with different tokenizers, while MiniFrontier
has exactly one frozen tokenizer contract, so hardcoding is simpler and equally correct for its
actual use case. 21/69 use `raise_exception` for input validation inside the template itself;
MiniFrontier's `ChatMessage.__post_init__` already validates role/content at construction time,
in Python, before the template ever runs -- a different but equally sound place to put the same
guard. **No further real gap found** -- reporting this honestly rather than manufacturing one.
