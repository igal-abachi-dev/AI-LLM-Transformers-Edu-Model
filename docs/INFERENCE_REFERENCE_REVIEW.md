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
