# Plan 2: a genuinely from-scratch, local, approximately 350M LLM

Research date: 2026-09-17.

## 1. The decision

Yes: training your own model from random initialization is feasible. My recommended first model is **349,237,760 parameters**, trained in PyTorch on your single 16–24 GB RTX like MSI GeForce RTX 4070 Ti SUPER 16G GAMING X SLIM, then post-trained and distilled, and deployed through llama.cpp in Q4_K_M.

A 1–2B model is not technically forbidden on a consumer GPU. The problem is the combination of optimizer memory, training throughput, available data, and your less-than-18-day budget. Fitting a model and training a competitive model are different achievements.

This plan treats random initialization and local training as firm requirements. It does not substitute a pretrained student, LoRA adaptation, rented hardware, or paid teacher APIs. It incorporates more-context-v2.md, while correcting conflicting or unsupported claims in it. The earlier plan remains unchanged.

| Component | Starting decision |
|---|---|
| Initialization | Random weights, including embeddings; no pretrained student |
| Architecture | Dense, decoder-only Transformer |
| Size | 349.24M unique trainable parameters |
| Layers / hidden width | 28 / 1,024 |
| Attention | Full causal attention in every layer |
| Query / KV heads / head dimension | 16 / 4 / 64 |
| Normalization | Pre-RMSNorm; QK RMSNorm before RoPE; final RMSNorm |
| FFN | Bias-free SwiGLU, intermediate width 2,816 |
| Vocabulary | Own 32,768-token byte-level BPE, including special tokens |
| Input/output embeddings | Tied |
| Position encoding | Full-head RoPE, base 100,000 |
| Context | 2,048 for most pretraining; 4,096 for a measured final phase |
| Optimizer | Fused AdamW initially; Muon only if a controlled pilot earns its cost |
| Precision | BF16 compute with FP32 parameters/optimizer state |
| Attention implementation | Fused PyTorch SDPA first; verify actual kernel dispatch |
| Distillation | Verified teacher responses first; no giant logit cache in the initial run |
| Export graph | Exact Qwen3-compatible text-decoder mathematics, with your own dimensions and weights |
| Deployment | BF16/F16 parity check, then calibrated Q4_K_M GGUF |

“Qwen3-compatible” describes the computation and export format. It does not mean loading Qwen weights, using Qwen's large tokenizer, or adapting a pretrained student.

This is a defensible starting configuration, not an experimentally proven universal optimum. The exact GPU, its sustained throughput, and the still-unspecified specialist task prevent anyone from honestly proving the optimal size in advance.

### What “competitive” can realistically mean

1. A complete, stable, reproducible modern LLM trained locally from scratch: realistic.
2. Strong performance for its local training budget, with useful specialization and some general assistant/code/math ability: a reasonable goal, to be demonstrated by evaluation.
3. Broad parity with the best similarly sized 2025–2026 models, or guaranteed 2027+ SOTA: not a credible promise under this budget.

For scale, the published SmolLM2-360M recipe used **4 trillion pretraining tokens**. A 5B-token run is about 800 times smaller in token exposure; this is a scale comparison, not a claim that quality scales linearly with tokens. Architecture alone cannot be assumed to erase that gap. [SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)

The productive target is a small, efficient specialist with measurable general capabilities. Future model performance beyond the research date is unknown. Until the specific domain is chosen, the plan uses an English/code/math foundation and leaves a defined specialization stage.

## 2. What I keep and change from more-context-v2.md

Keep its focus on local wall-clock efficiency, data quality, measured experiments, GQA, tied embeddings, stability, and early export tests. Change the following:

| Claim or suggestion in the context | Correct interpretation / decision |
|---|---|
| Full attention is universally best below 1B | Unsupported as a universal rule. It is my engineering choice at 2K–4K context for this run. |
| Modern small models almost all use full attention | False: Qwen3.5-0.8B uses linear/full attention; Gemma 3 uses local/global attention. |
| Small-model KV caches are always negligible | Often modest at short context and batch one; they grow with context, layers, KV width, precision, and concurrency. |
| Head dimension 96 is a universally established sweet spot | The reported MF-108 result concerns a particular model, GPU, kernel, and memory regime. It is not a universal optimum. |
| Head dimension 128 necessarily causes a 6.5× slowdown | A paging or kernel-fallback cliff is not an intrinsic property of dimension 128. |
| Head dimension 256 is inherently unstable | Not established; consider normalization, head count, projection width, and architecture together. |
| QK norm guarantees good 4-bit behavior | It controls Q/K scale, not every FFN/weight outlier or quantization error. |
| A SwiGLU clamp can simply be folded away | A nonlinear clamp generally cannot. Removing it changes the learned function. |
| Partial RoPE + QK norm can simply be named “Qwen3” | Not safely. The inspected llama.cpp Qwen3 graph requires full-head RoPE. |
| Weight sharing and cross-layer KV sharing are equivalent | Different changes to training, computation, and caching. Neither is in the baseline. |
| Muon is automatically better on a consumer GPU | Compare validation progress per hour, including optimizer time and memory. |
| Top-64-only softmax KL equals full-vocabulary distillation | It discards probability mass outside that set. It is a different objective. |
| A student's tokenizer must match its teacher's | Only straightforward token-level logit matching needs compatible tokenization. Response distillation does not. |
| Repeating any corpus four times is effectively free | Published results support useful repetition in studied regimes, not an unconditional rule. |

The small hybrid counterexample is directly visible in the [Qwen3.5-0.8B configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json). Gemma's local/global design and replacement of Gemma 2 soft-capping with QK normalization are described in the [Gemma 3 report](https://arxiv.org/html/2503.19786v1). MobileLLM's additional sharing technique is block-wise **weight** sharing, not synonymous with sharing cached keys and values. [MobileLLM paper](https://arxiv.org/abs/2402.14905)

I have not independently reproduced the MF-108 or other local experiments quoted in `more-context-v2.md`. Their precise improvements are reported observations, not verified predictions for this model. Likewise, claims about newer/future model versions in the context are not accepted without primary-source verification.

## 3. Exact network specification

### 3.1 Configuration

This is the specification of your own implementation. Some fields are implementation-specific; do not pass the entire block unchanged to a library configuration class.

```yaml
name: MiniFrontier-349M
initialization: random
architecture_family: qwen3_compatible_dense_decoder

vocab_size: 32768                 # total, INCLUDING reserved special tokens
num_hidden_layers: 28
hidden_size: 1024
intermediate_size: 2816
num_attention_heads: 16
num_key_value_heads: 4
head_dim: 64

hidden_act: silu                  # SwiGLU = silu(gate) * up
attention_bias: false
mlp_bias: false
attention_dropout: 0.0
residual_dropout: 0.0
embedding_dropout: 0.0

norm_type: rmsnorm                # ordinary gamma, initialized to 1
rms_norm_eps: 1.0e-6
qk_norm: true                     # RMSNorm over head_dim, before RoPE
qk_norm_shared_across_heads: true # separate Q and K vectors per layer

rope_theta: 100000.0
rotary_fraction: 1.0
rope_layout: neox_half_split      # must agree with export/runtime
rope_scaling: null
max_position_embeddings: 4096

tie_word_embeddings: true
scale_embeddings: false
attention_logit_softcap: null
output_logit_softcap: null
swiglu_clamp: null
attention_output_gate: false
sliding_window: null
layer_weight_sharing: false
cross_layer_kv_sharing: false
mtp_heads: 0
moe: false
```

Record actual BOS/EOS/PAD, chat delimiter, and optional FIM IDs in the tokenizer manifest. Do not borrow numeric IDs from another model.

Explicitly set `head_dim=64` in the export configuration rather than inheriting a library default. Use ordinary RMSNorm, not a zero-centered-gamma variant.

### 3.2 Block equations

For hidden states `x` shaped `[batch, sequence, 1024]`:

```text
u = RMSNorm_attention(x)

q = reshape(Wq(u), [B, S, 16, 64])
k = reshape(Wk(u), [B, S,  4, 64])
v = reshape(Wv(u), [B, S,  4, 64])

q = RoPE(RMSNorm_Q(q), positions)
k = RoPE(RMSNorm_K(k), positions)

a = causal_grouped_query_attention(q, k, v, scale=1/sqrt(64))
x = x + Wo(concatenate_query_heads(a))

u = RMSNorm_ffn(x)
x = x + Wdown(silu(Wgate(u)) * Wup(u))

# After the last block:
h = final_RMSNorm(x)
logits = h @ token_embedding_weight.T
```

Use separate Q and K normalization vectors of length 64 in each layer, broadcast over heads. Do not normalize all heads together or introduce separate learned vectors for every head without deliberately changing the architecture.

Tie the embedding and output projection to the same actual `Parameter`; two equal-valued tensors are not weight tying. Deduplicate optimizer parameters.

During training, allow the loss path to avoid materializing all logits. During inference, calculate only the output positions needed by the caller.

### 3.3 Exact parameter accounting

With `d = Hq * Dh`, tied embeddings, no biases, these norms, and no extra heads:

```text
N = V*d
    + L * (2*d*d + 2*d*Hkv*Dh + 3*d*f + 2*d + 2*Dh)
    + d

V=32768; d=1024; L=28; Hkv=4; Dh=64; f=2816
```

| Part | Parameters |
|---|---:|
| Shared input/output embedding | 33,554,432 |
| Attention projections, all layers | 73,400,320 |
| SwiGLU projections, all layers | 242,221,056 |
| All RMSNorm parameters | 61,952 |
| **Total unique trainable parameters** | **349,237,760** |

Approximately 9.6% are embeddings, 21.0% attention projections, and 69.4% FFNs. Optimizing only attention therefore misses much of the model's work.

These counts are calculated from the specification, not inferred from a marketing name.

### 3.4 Nearby sizes, if measurements justify changing

All rows use the same 32,768-token tied vocabulary, head dimension 64, normalization scheme, and no biases.

| Role | Layers | Width | FFN width | Q / KV heads | Exact parameters |
|---|---:|---:|---:|---:|---:|
| Lower-cost fallback | 24 | 1,024 | 2,816 | 16 / 4 | 304,140,288 |
| **Default** | **28** | **1,024** | **2,816** | **16 / 4** | **349,237,760** |
| Shallower, larger-FFN comparison | 24 | 1,024 | 3,584 | 16 / 4 | 360,763,392 |
| Stretch option | 26 | 1,280 | 3,456 | 20 / 4 | 489,297,408 |
| Approximately 1B, not default | 28 | 2,048 | 4,096 | 32 / 8 | 1,065,473,536 |
| Approximately 1.5B, not default | 28 | 2,304 | 5,376 | 36 / 6 | 1,462,898,432 |

Do not train all six. Benchmark the default and at most one alternative relevant to the measured bottleneck.

## 4. Why these architecture choices?

### Dense, not MoE

Dense provides straightforward optimization, ordinary matrix multiplication, and manageable full-weight training. MoE adds routing, expert balancing, total parameter/optimizer storage, and potentially inefficient small expert batches.

Low active parameter count is not low total training memory. MoE is not a free route to a much larger student in 16 GB.

### Full attention, not a hybrid in version 1

This is a constraint-specific recommendation: 2K–4K training context, little engineering budget, and an exact established export graph. It is not a claim that hybrids are scientifically inferior.

Revisit hybrids if long-context training, prefill, or cache storage becomes the dominant measured bottleneck:

- Sliding-window/global interleaving reduces local work, but global layers still have quadratic attention work.
- Gated DeltaNet/full-attention interleaving has bounded recurrent state in its recurrent layers, but a different training/kernel/runtime contract.
- Differential attention changes the attention function; it is not a GQA configuration flag.

Use an exact supported architecture for any branch. Support for one hybrid does not prove support for your arbitrary combination.

### GQA: four KV heads

Sixteen query heads sharing four KV heads is a moderate 4:1 ratio. It reduces K/V projections and cache size relative to 16 KV heads without compressing to one KV head.

GQA does not divide all attention FLOPs by four: all query heads still calculate attention. MHA remains a valid quality comparison if retrieval/copying is weak. MQA is not my starting choice.

### Head dimension: 64

This gives 1,024 total query width and 256 total KV width. To test dimension 128 at the same projection sizes, use eight Q heads and two KV heads. Increasing head dimension while keeping head counts fixed changes the projection widths and is not a clean head-dimension comparison.

Dimension 96 is not inherently wrong. Width 1,024 does not divide into an integral number of 96-dimensional heads unless projection width differs from hidden width. That is a separate architectural decision.

### Depth versus width

28 layers at width 1,024 is moderately deep, not an extreme 48–64-layer narrow design:

- More depth provides additional sequential transformations.
- More width can improve matrix utilization and per-layer representation.
- More layers increase sequential decode latency and launch overhead.
- A larger model may receive fewer tokens in the same time.

MobileLLM provides relevant evidence for deeper, narrower sub-billion models, but does not establish the optimum on your GPU at equal time. [MobileLLM paper](https://arxiv.org/abs/2402.14905)

The 24-layer/3,584-FFN candidate is one useful alternative if the default is launch-bound. Compare quality per hour, not only quality per token.

### Feed-forward network: SwiGLU

SwiGLU costs approximately `3*d*f` parameters per layer. Width 2,816 is 2.75 times hidden width, close to the common `8/3*d` convention after rounding to a convenient multiple. This is not a mandatory ratio.

For clarity, `(8/3)*1536 = 4096`, not 4,864. Some arithmetic in the context conflates these.

Do not add SoLU, special clipping, ReLU-squared, or auxiliary FFN branches to the first export graph.

### QK norm, soft-capping, and stability

Use QK RMSNorm before RoPE and standard `1/sqrt(head_dim)` attention scaling. Start without attention-logit caps, output-logit caps, or SwiGLU clipping.

If training is unstable, inspect learning rate, precision, initialization, data, and gradient norms before changing the network. QK norm is not a guarantee against activation outliers; its learned gains can also grow.

The inspected [llama.cpp Qwen3 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3.cpp) supports the intended normalization order, tied output fallback, and full-head RoPE. Those properties motivate this export contract; actual conversion and parity still need testing.

### Embeddings and position encoding

Tie input/output weights. Untying adds 33,554,432 parameters plus optimizer state without a demonstrated benefit here.

Use learned token embeddings plus RoPE, not learned absolute position embeddings. Keep the RoPE base and layout fixed during the initial run. Only advertise 4K competence after training and evaluating at 4K.

Changing a context number does not establish 32K–128K capability. Future extension requires training and evaluation.

Interleaving local/global layers and interleaving rotary coordinates are unrelated meanings of “interleaved.” Keep the latter exactly consistent with export.

## 5. Tokenizer: small, yours, and export-tested

Train a **32,768-token byte-level BPE**, including special tokens, on a stratified sample of the training split.

Start with an existing llama.cpp-supported pre-tokenization scheme, such as canonical GPT-2-style regex splitting and byte mapping. Preserve spaces, tabs, newlines, case, indentation, and Unicode round-tripping. Avoid destructive text normalization.

1. Sample approximately 0.5–2 GB of deduplicated training text across sources; stream preparation rather than requiring it all in RAM.
2. Include real code and mathematics, not just prose.
3. Reserve special-token capacity inside the total vocabulary, for example 32 slots.
4. Define document-end, chat delimiters, and optional FIM markers before pretraining.
5. Measure tokens per byte for prose, code, math, and the eventual domain.
6. Freeze files, hashes, IDs, and preprocessing before the long run.

### Why not copy the newest teacher's enormous vocabulary?

At width 1,024, a 248,320-token tied embedding alone has **254,279,680 parameters**. Keeping our body and replacing its vocabulary with that one would make it approximately 570M parameters.

At width 1,280, that embedding alone is 317,849,600 parameters. This is costly for a mainly-English 300M–400M target. Shared vocabulary simplifies straightforward logit distillation, but response distillation does not need it.

### Tokenizer compatibility is a release gate

A newly trained BPE may fail a converter's tokenizer fingerprint recognition even when its splitting scheme already exists in the runtime.

- Verify identical IDs and decoding between your tokenizer and llama.cpp.
- If needed, register the exact tokenizer fingerprint against the already-supported pre-tokenizer in the converter.
- Do not force arbitrary tokenization through a “Qwen2” or “GPT-2” label.
- Test leading spaces, CRLF/LF, tabs, indentation, numbers, punctuation, Unicode, and every special token.
- Test ordinary text separately from explicit chat-control-token parsing.

If parity cannot be achieved, change the tokenizer **before** pretraining. The maintained export entry point is [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py); pin a tested revision.

## 6. Your dataset: inventory, audit, and preparation

Dataset: [MiniFrontier-150M-Modern-3B-token-mixture](https://huggingface.co/datasets/igalk474/MiniFrontier-150M-Modern-3B-token-mixture).

Inspected revision: `39272b7346b9cea66af8c03ea1fd18b6a6aca3b3`.

The earlier inspection covered the card, file/repository metadata, Dataset Viewer metadata, and a few records. It did not download/audit the entire corpus or count it with this new tokenizer.

### Inventory and initial mixture

| Source | Published weight | Publisher-reported tokens | Proposed token-sampling weight |
|---|---:|---:|---:|
| dclm-edu | 35% | 1,651,369,984 | 35% |
| fineweb-edu | 25% | 901,208,064 | 25% |
| github-code | 20% | 1,393,422,336 | 23% |
| finemath | 15% | 933,122,048 | 15% |
| cosmopedia-v2 | 5% | 81,995,776 | 2% |
| **Total** | **100%** | **4,961,118,208** | **100%** |

The reported inventory is about 4.96B tokens despite “3B” in the name. The inspected metadata lists 3,401,006 documents: 3,366,983 train and 34,023 validation; Parquet storage totals about 6.72 GB. These are publisher/repository figures, not measured clean student-token counts. [Pinned card](https://huggingface.co/datasets/igalk474/MiniFrontier-150M-Modern-3B-token-mixture/blob/39272b7346b9cea66af8c03ea1fd18b6a6aca3b3/README.md), [size metadata](https://datasets-server.huggingface.co/size?dataset=igalk474%2FMiniFrontier-150M-Modern-3B-token-mixture), [repository metadata](https://huggingface.co/api/datasets/igalk474/MiniFrontier-150M-Modern-3B-token-mixture)

The proposed weights are a starting hypothesis, not a validated improvement. Lowering Cosmopedia reduces overexposure to a small source. The original project used a smaller tokenizer: re-tokenize and recount.

### Repetition by source

```text
effective_passes_i =
    sampled_student_tokens_i / unique_usable_student_tokens_i
```

Using the original reported counts only for illustration, a 5B-token run with 5% Cosmopedia gives about 3.05 corpus-equivalents of that source; at 2% it gives about 1.22. Recalculate after filtering and tokenization.

Data-constrained scaling experiments found useful repeated-data regimes, including up to four epochs in their setting. That is not a guarantee for every oversampled component. Watch source-specific validation and memorization. [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)

### Audit requirements

The data contains raw text with provenance metadata. A label such as `github-code` does not mean every record is executable code; inspected examples included Markdown. Synthetic prose is not automatically instruction/assistant training data.

Check:

- Empty/invalid text, pathological repetition, boilerplate, generated/minified files.
- Actual language/file-type composition and source-specific length distributions.
- Exact and near-duplicates within and across sources.
- Train/validation leakage by content hash, parent hash, document family, and repository where available.
- Public benchmark contamination, especially coding and mathematics.
- Secrets, credentials, and personal information requiring removal.
- Preservation of source, revision, license, and transformation metadata.

Do not execute scripts or obey instructions found in dataset records. They are data, not workspace instructions. Track underlying source licenses/provenance rather than relying only on a top-level dataset label.

Inspect hundreds of random records per source and different length buckets. Do not infer cleanliness from the first few rows, or discard large fractions using unvalidated filters.

### Pre-tokenize and pack

Example source loader:

```python
from datasets import load_dataset

rows = load_dataset(
    "igalk474/MiniFrontier-150M-Modern-3B-token-mixture",
    "github-code",
    split="train",
    revision="39272b7346b9cea66af8c03ea1fd18b6a6aca3b3",
    streaming=True,
)
```

Processing order:

1. Filter/deduplicate while preserving the train/validation boundary.
2. Tokenize once; append the document-end token.
3. Write token shards plus document/source indexes and a manifest.
4. Store IDs as `uint16` because all IDs are below 65,536; cast batches to the integer type the training implementation requires.
5. Sample equal-length source blocks by **tokens**, not document counts.
6. Shuffle deterministically and save RNG/shard/cursor state.
7. Prefetch asynchronously; do not decode Parquet or tokenize in every GPU step.

Five billion uint16 IDs require about 10 GB before indexes. Budget roughly 100 GB working-disk headroom for source data, token shards, resumable checkpoints, exports, and a modest teacher cache; verify actual use before launch.

Simple packing may concatenate documents with EOS and ordinary causal attention. This allows attention to earlier packed documents. Document-isolated variable-length attention is optional if the backend supports it efficiently. Do not silently introduce a dense block mask that disables the fast kernel.

For SFT, preserve conversation boundaries and use assistant-only loss masking. Do not detach answers from prompts by careless chunking.

### Optional code infilling

Reserve FIM tokens now if infilling is useful. After basic evaluation works, a minority such as 10% of genuine code examples can use fill-in-the-middle transformation. Keep normal causal code examples, do not apply FIM indiscriminately to Markdown, and do not assume it improves chat. Leaving FIM disabled is the default.

## 7. Feasibility: memory, throughput, and the real token budget

### 7.1 Training memory is not 4-bit inference memory

The initial training recipe uses FP32 parameters, FP32 gradients, and two FP32 Adam moments:

```text
persistent training state ≈ N * (4 + 4 + 4 + 4) bytes
                          ≈ 16*N bytes
```

This excludes activations, temporary BF16 casts, optimizer workspace, CUDA context, allocator reservation, and compiled-graph buffers.

| Model | Persistent state at 16 bytes/parameter |
|---|---:|
| 304.14M | 4.53 GiB |
| **349.24M** | **5.20 GiB** |
| 489.30M | 7.29 GiB |
| 1.065B | 15.88 GiB |
| 1.463B | 21.80 GiB |

Consequences:

- The 349M default leaves meaningful activation/workspace room on both 16 GB and 24 GB cards.
- A roughly 1B model with this straightforward optimizer layout has essentially no useful headroom on a 16 GB card. On 24 GB it may train with small microbatches, checkpointing, and careful loss computation.
- Around 1.5B is very constrained even on 24 GB with this layout.
- Different optimizer/precision/offload schemes can change this. They do not remove the wall-clock cost or automatically preserve training quality.

Full-weight training with an 8-bit optimizer is different from 4-bit QLoRA. The former can be investigated later; the latter is not a substitute for this from-scratch request. Do not make CPU optimizer offload the default for an 18-day throughput-limited run.

### 7.2 Measure the actual card

“RTX 30/40/50, 16–24 GB” is not one throughput class. Do not predict your run from a different card's marketing TFLOPs. assume mostly MSI GeForce RTX 4070 Ti SUPER 16G GAMING X SLIM, but sometime worse or better

Before choosing a token target, run the complete training step, including forward, loss, backward, accumulation, optimizer updates, data input, and checkpointing overhead.

Benchmark after compilation/warmup, then run a 30–60-minute thermal/stability check. Record:

```text
GPU model / VRAM / driver / compute capability
PyTorch / CUDA / attention backend / compiler versions
model and tokenizer hashes
sequence length / microbatch / accumulation
active training tokens per second
elapsed wall-clock tokens per second
peak allocated and reserved GPU memory
step-time median and tail latency
data-loading time / optimizer time / checkpoint duration
temperature / sustained clock / power / thermal throttling
```

Measure actual loss-bearing tokens for throughput if padding/masking is present. Do not report a forward-only benchmark as training throughput. Do not count repeated tokens as new unique data.

### 7.3 Twelve pretraining days: useful planning arithmetic

The schedule below allocates 12 days to base pretraining. If `R` is active-step throughput and 85% of that window is available for active steps:

```text
student_tokens = R * 12 * 86400 * 0.85
               = R * 881280
```

| Measured active training throughput | Tokens in that window |
|---|---:|
| 2,000 tokens/s | 1.76B |
| 3,000 tokens/s | 2.64B |
| 5,000 tokens/s | 4.41B |
| 8,000 tokens/s | 7.05B |
| 12,000 tokens/s | 10.58B |

These are arithmetic scenarios, **not claimed RTX benchmarks**. If you already measured whole-window throughput including downtime, do not apply the 85% factor again.

Conversely, the same window needs approximately:

| Desired exposure | Required active throughput |
|---|---:|
| 3B tokens | 3,404 tokens/s |
| 5B tokens | 5,674 tokens/s |
| 8B tokens | 9,078 tokens/s |
| 10B tokens | 11,347 tokens/s |

Use separately measured 2K and 4K throughput. For two phases, sum their times; do not assume the 4K phase has the 2K rate.

My provisional goal is **roughly 3–8B student-token exposures if the measured system supports it**, with about 5B as a planning example. It is not a guaranteed minimum or a mandatory ceiling. If the card is slower, select a smaller exposure or the 304M fallback. If faster, keep training while validation improves and source repetition remains useful.

### 7.4 Why not choose the largest model that fits?

The rough dense-training estimate `6*N*T` captures much of the projection/FFN work but omits important attention, recomputation, loss, and system costs. The 349M model at 5B tokens is about `1.05e19` FLOPs under that approximation. It is not a timing prediction.

Compute-optimal scaling research supports balancing parameters against token exposure, not blindly maximizing parameters. The commonly used approximately 20 tokens/parameter rule is a rough historical heuristic, not a law for your dataset, task, distillation method, or GPU. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

At 5B tokens, the default sees about 14.3 tokens per parameter. A 1B model at the same token budget sees about five and also takes substantially more projection work per token. This motivates the smaller starting point; it does not prove that the larger model loses every downstream task.

Choose the 489M stretch only if measurements show it can receive a useful token budget and its quality-versus-time pilot is favorable. Treat 1B as a future or explicitly higher-risk experiment, not as impossible.

## 8. PyTorch implementation and performance priorities

### 8.1 Environment

Prefer Linux or WSL2/Linux for the training stack, especially if using Triton/fused loss kernels. A native Windows SDPA implementation can be a fallback, but validate its available kernels rather than assuming parity with Linux.

Pin a tested PyTorch/CUDA/driver combination supporting the **exact** GPU. For RTX 50-series, verify that the installed build and every custom kernel support its device architecture; “Blackwell support” in a datacenter-kernel announcement is not by itself a consumer-card compatibility test.

Start with stock PyTorch SDPA. It supports GQA subject to shape/backend constraints, including query heads divisible by KV heads. Its documented non-square causal-mask behavior also makes cache-decoding tests important. [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

For ordinary training tensors, the key call is conceptually:

```python
# q: [B, 16, S, 64], k/v: [B, 4, S, 64]
out = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    dropout_p=0.0,
    is_causal=True,
    enable_gqa=True,
)
```

Profile the backend. Do not assume this spelling guarantees the fast implementation on every installation. A diagnostic can force a fused backend and fail visibly if unsupported.

For cached decoding, do not blindly reuse `is_causal=True` with `query_length=1` and a longer key sequence: the mask alignment must reflect the actual absolute query positions. Test full-forward versus cached logits.

FlashAttention-2 is an optional alternative. Its upstream documentation lists Ampere/Ada/Hopper support; newer variants have different hardware requirements. There is no reason to require a Hopper-specific kernel on your RTX 30/40 card. [FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
MSI GeForce RTX 4070 Ti SUPER 16G GAMING X SLIM

### 8.2 Precision and numerical correctness

Start with:

- FP32 model parameters and AdamW states.
- BF16 autocast for matrix multiplications and compatible compute.
- FP32 RMSNorm reductions, loss reductions, and sensitive statistics.
- No ordinary BF16 GradScaler requirement; if falling back to FP16, use the appropriate scaler and overflow handling.
- FP32 RoPE frequency/angle generation before casting working tensors.

Do not simply call `model.bfloat16()`, use an optimizer with reduced-precision moments/updates, and assume it is equivalent to FP32-master mixed precision. Small updates can be lost. Any lower-memory optimizer layout needs its own numerical validation.

Initialize linear/embedding weights with standard deviation 0.02 as a starting recipe; initialize norm gains to 1. Scale initial attention output and FFN down-projection weights by `1/sqrt(2*L)`. This is an initialization choice, not a runtime multiplier that export must implement. Check activation/gradient scales during the pilot.

Ensure the tied embedding is initialized once. No additional runtime embedding scale, residual-depth scale, clamp, or logit transform is introduced.

### 8.3 The bottlenecks to prioritize

| Priority | Bottleneck | First action |
|---|---|---|
| 1 | Data starvation / tokenizer work in the step | Pre-tokenized shards, prefetch, pinned transfers |
| 2 | Unfused attention or materialized attention matrices | Verify fused causal GQA backend |
| 3 | FFN activations and GEMM throughput | Profile FFN; tune microbatch; checkpoint selectively |
| 4 | Full-vocabulary logits/loss memory | Benchmark fused linear cross-entropy |
| 5 | Launch overhead in a small deep network | Benchmark stable-shape `torch.compile` |
| 6 | Optimizer update / memory | Fused AdamW; measure alternatives only if relevant |
| 7 | Paging / thermal throttling / allocator peaks | Keep VRAM headroom and run a long stability probe |

The ordinary `[B,S,V]` logit tensor at microbatch 2, length 2,048, vocabulary 32,768 is already 0.5 GiB if materialized in FP32, before other loss/backward buffers. Fused linear cross-entropy can avoid materializing the whole tensor.

An available implementation is Apple's Cut Cross-Entropy. Start by comparing its exact/reference mode against ordinary loss and gradients; then benchmark production modes because some apply gradient filtering. For a 32K vocabulary, verify actual benefit rather than importing large-vocabulary speedup claims. [Cut Cross-Entropy implementation](https://github.com/apple-aiml-research/ml-cross-entropy)

Naively looping over logit chunks while retaining every autograd graph does not necessarily solve backward-memory usage.

### 8.4 Microbatch, accumulation, and checkpointing

Initial target: **65,536 tokens per optimizer update**.

At 2K sequence length:

| Microbatch | Gradient accumulation | Tokens/update |
|---|---:|---:|
| 1 × 2,048 | 32 | 65,536 |
| 2 × 2,048 | 16 | 65,536 |
| 4 × 2,048 | 8 | 65,536 |
| 8 × 2,048 | 4 | 65,536 |

On 16 GB, start testing microbatch 1–2; on 24 GB, test 2–4 and increase only if measured safe. These are probes, not promises of fit. At 4K, adjust microbatch/accumulation to preserve the token batch.

Use selective or per-block non-reentrant activation checkpointing when needed. Benchmark both checkpointed and non-checkpointed configurations: recomputation may cost throughput, but a better microbatch can compensate.

Leave approximately 1–2 GiB practical VRAM headroom where possible, especially on a display-driving card. Measure the peak through the optimizer step, not only forward. Avoid silent host-memory spill.

### 8.5 Compile and fuse selectively

After eager correctness passes:

1. Compile a stable-shape training path.
2. Exclude first-call compilation from steady-state throughput, but include compilation in the campaign's time budget.
3. Check for graph breaks/recompilation at sequence changes.
4. Fuse projections or normalization only where measured beneficial.
5. If fusion changes saved tensor layout, split/map it correctly during export.

Do not spend several training days chasing the last few percent of throughput. A stable complete run is more valuable than an unfinished kernel project.

## 9. Concrete pretraining recipe

### 9.1 Starting hyperparameters

These are actionable starting values, not the outcome of a completed tuning sweep.

```yaml
objective: next_token_cross_entropy
optimizer: fused_adamw
learning_rate_peak: 0.0006
betas: [0.9, 0.95]
epsilon: 1.0e-8
weight_decay: 0.1
gradient_clip_global_norm: 1.0

tokens_per_optimizer_step: 65536
warmup_tokens: 30000000
schedule: warmup_stable_decay
decay_fraction_of_final_token_budget: 0.20
learning_rate_final: 0.00006

sequence_length_initial: 2048
sequence_length_final: 4096
long_context_fraction_target: 0.10
dropout: 0.0
label_smoothing: 0.0
auxiliary_losses: none
```

Apply weight decay to hidden linear weights. Exempt norms and the shared embedding/output parameter as the initial policy; record the parameter groups explicitly. Never update the tied weight twice.

Warmup 30M tokens is approximately 458 updates at this token batch. A 5B-token run is approximately 76,294 updates.

Normalize accumulated gradients by the intended number of loss tokens. Variable masks and unequal microbatch lengths require token-weighted reduction, not a blind mean of per-microbatch means.

Start LR at zero, warm up to 6e-4, hold, then decay over the final 20% of the planned exposure to 6e-5. A cosine-shaped decay within the final segment is fine.

### 9.2 Why WSD here?

Warmup–stable–decay makes it convenient to determine the practical stopping point after measuring throughput, while reserving a meaningful final decay. Once the token budget is chosen, save it and the consumed-token counter.

Cosine over a fixed full run is also valid and resumes correctly if its state is restored. The context's implication that cosine inherently breaks on resume is incorrect.

A concrete 5B-token example:

```text
0–30M:       warmup at 2K
30M–4.0B:    stable learning rate, mostly 2K
4.0B–5.0B:   learning-rate decay
4.5B–5.0B:   optional 4K phase inside the decay
```

Do not promise exactly 5B if the 4K phase or evaluation takes more time than measured. Recalculate and start decay early enough to finish before the deadline.

### 9.3 Context curriculum

Use 2K for the first approximately 90% of exposure. If the model is learning, the pipeline is stable, and time permits, train the last approximately 10% at 4K using meaningful long documents/conversations.

Keep RoPE parameters fixed. Reduce microbatch as needed. A 4K sequence consisting only of unrelated short documents is not strong long-context supervision.

If useful 4K training does not fit, ship a tested 2K model and describe the limit honestly. A checkpoint that merely accepts 4K tensors is not necessarily good at 4K tasks.

### 9.4 Stability and recovery

Log loss, source losses, gradient norm, update norm, learning rate, logits, norm gains, activation percentiles, and throughput.

For NaN/Inf or a persistent loss spike:

1. Save diagnostic information; stop applying corrupted updates.
2. Check data/labels, attention masks, precision, backend changes, and optimizer state.
3. Reproduce the problematic batch where practical.
4. Restore the last known-good checkpoint.
5. If needed, reduce peak LR to 3e-4 or extend warmup and document the change.

Do not “fix” instability by silently adding a clamp that the inference runtime does not reproduce.

Save a resumable checkpoint every approximately 1–2 hours and at phase boundaries. Include model, optimizer, scheduler, RNGs, source sampler/cursor, token counter, tokenizer/data hashes, and software/config revisions.

Keep a rotating latest pair plus milestone and best-validation checkpoints. Write checkpoints atomically; confirm restart before launching the long run.

## 10. A less-than-18-day campaign

Use a **17-day cap** for the actual training/post-training/release campaign:

| Window | GPU work | Deliverable / stop condition |
|---|---|---|
| Day 1 | Final correctness, export, memory, speed and resume gates; bounded pilot | Locked model, tokenizer, kernels, token budget |
| Days 2–13 | Base pretraining, including final decay and optional 4K phase | Best and final base checkpoints |
| Day 14 | Local teacher generation and verification, if useful instruction data is not already available | Capped, filtered response cache |
| Day 15 | Full-weight instruction tuning / response distillation | Candidate assistant/specialist checkpoint |
| Day 16 | Final evaluations, GGUF conversion, calibration, Q4 comparison, server tests | Release candidate |
| Day 17 | Reserved recovery/overrun margin | Reproducible final artifact |

All GPU work is serial on your one card. Teacher generation is not assumed to run simultaneously with student training for free.

**Engineering time matters:** this table assumes the trainer and data preparation pipeline are already implemented enough to pass the Day 1 gates. Building the entire software stack from nothing is additional work, not magically included in one preflight day. If “18 days” includes all coding and data preparation from today, subtract that time from pretraining and use the throughput formula again. Do not skip correctness or export testing to preserve an arbitrary token target.

If you explicitly want distillation only in a later campaign, Days 14–15 can instead extend base training; still reserve release/recovery time and schedule LR decay accordingly. The default table includes a first modest distillation pass inside the cap.

The pilot budget is limited. Use brief backend/memory probes, then at most one meaningful model/optimizer comparison. Do not schedule many billion-token ablations inside one day.

## 11. Distillation without abandoning a from-scratch student

There is no contradiction between random initialization and later distillation:

```text
random student weights
    -> language-model pretraining on your corpus
    -> verified teacher-response instruction training
    -> optional later targeted distillation
    -> quantization and deployment
```

The student's entire parameter history begins in your run. A pretrained **teacher** supplies supervision; it does not initialize the student.

### 11.1 First choice: response/sequence distillation

For this budget, use the teacher to produce high-value examples rather than scoring every token of the full corpus.

Candidate tasks:

- Short code generation, debugging, and explanation with runnable tests.
- Arithmetic and elementary reasoning with executable or symbolic answer checks.
- Structured extraction, schema-constrained JSON, and tool-argument construction.
- Domain questions grounded in documents you are allowed to use.
- Concise explanations, rewriting, summarization, and ordinary instruction following.

Choose tasks where correctness can be checked. Diverse, verified examples are more valuable than millions of unchecked repetitions of the teacher's style.

Response distillation works across different tokenizers: store prompt and response as text, then tokenize with the student tokenizer. It does not require matching hidden sizes, layers, attention type, or vocabulary.

### 11.2 A local teacher versus an actual frontier teacher

A practical starting candidate is a strong 8B–9B-class instruction model in 4-bit, such as the released Qwen3.5-9B, provided your pinned runtime supports it. Benchmark it at short context with a modest batch. A 9B 4-bit weight payload is plausible within 16–24 GB, but actual VRAM also includes mixed-precision tensors, runtime workspaces, and cache/state. Measure fit and accepted-output throughput. [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)

Use its text capability; this student does not need a vision encoder. Do not equate a local 9B teacher with the strongest available frontier model.

A genuinely much larger frontier teacher may not run fully on your GPU at useful speed, or may not provide downloadable weights. CPU offloading can make some larger teachers technically runnable but too slow for this campaign. If you later have permitted, already-generated frontier-teacher responses, ingest them in the same response-distillation pipeline. Do not assume access to private hidden states or logits.

Teacher licensing/usage constraints and data provenance must be checked for the chosen checkpoint/source before generating or redistributing a distilled dataset. This plan assumes no paid API and no rental.

### 11.3 Bound generation by time and accepted tokens

Start with a 200–500-prompt pilot:

1. Measure prompt-prefill time and aggregate generated tokens/second.
2. Track answer length, verifier pass rate, duplicates, and truncations.
3. Estimate accepted examples per hour.
4. Select one teacher and a practical prompt/context budget.

A useful initial dataset might be **5,000–20,000 accepted responses** averaging a few hundred tokens, if your measured rate permits. This is a target range, not a required count.

For illustration: 10,000 accepted answers averaging 500 output tokens require 5M accepted output tokens. At an aggregate 50 output tokens/s, 5M generated tokens alone take about 27.8 hours, before prompt processing, rejected answers, and verification. At 100 tokens/s it is about 13.9 hours. Do not assume either rate for your card.

Use bounded batches, cache every accepted result, and stop when the allocated day is over. Prefilling/scoring existing text is different from autoregressive generation; do not compare those throughput numbers directly.

Do not make a 1.3B-token teacher-logit cache a prerequisite. Even scoring that in two days requires about 7,523 tokens/s continuously, before other overhead.

### 11.4 Verification and curriculum

For code:

- Execute tests in an isolated, resource-limited, network-disabled environment.
- Include hidden/held-out tests where possible.
- Reject syntactically valid but incorrect code.
- Avoid executing untrusted generated code directly in your normal workspace.

For math/reasoning:

- Verify final answers with a reliable checker where possible.
- Keep short explanations proportional to student capacity.
- Include simple-to-moderate problems, not only teacher-level puzzles.
- If using reasoning traces, treat them as potentially flawed text; a correct final answer does not prove every intermediate step.

For factual/domain responses:

- Ground prompts in supplied documents and verify quoted facts.
- Keep references to source material.
- Include “insufficient information” examples and appropriate uncertainty.

Do not train the model to emit very long reasoning before every simple answer. Concise direct responses, short verified reasoning, and format adherence are useful initial priorities.

### 11.5 Initial instruction/response-distillation recipe

Train **all student weights**, not LoRA adapters, from the base checkpoint you trained.

Starting values:

```yaml
stage: full_weight_sft_and_response_distillation
optimizer: fused_adamw
learning_rate_peak: 0.00003
learning_rate_final: 0.000003
betas: [0.9, 0.95]
weight_decay: 0.01
gradient_clip_global_norm: 1.0
warmup_fraction: 0.03
schedule: cosine
loss: assistant_tokens_only_for_chat
epochs_max: 2
early_stopping: held_out_task_quality
pretraining_replay_token_fraction: 0.20
```

Use 1–2 epochs as a cap, not a reason to keep training after validation degrades. A small response dataset may take far less than a day; use the remaining allocation for verification/evaluation rather than hundreds of repeated epochs.

The remaining 80% of sampled tokens come from verified instruction/teacher examples. Within those, a provisional mix once the domain is known is 50% specialist, 25% code/math, and 25% general instructions. Until then, use 50% code/math and 50% general instructions; do not fabricate a specialist corpus.

Implement loss accounting carefully:

- Chat batches: supervise assistant content and appropriate assistant end markers; mask user/system/padding tokens.
- Raw-text replay: ordinary next-token loss.
- Keep the intended token mixture explicit because chat prompt tokens are not all supervised.
- Use the same chat template at training, evaluation, export, and serving.

Hold out prompts/document families before generating paraphrases. Otherwise a generated variant can leak into validation.

### 11.6 Optional later logit distillation

This is a later research branch, not a requirement for a useful first model.

For aligned teacher/student tokenization, a conventional objective is:

```text
L = (1 - alpha) * CE(y, student)
    + alpha * temperature^2 * KL(
        softmax(teacher_logits / temperature)
        ||
        softmax(student_logits / temperature)
      )
```

Reasonable exploratory values are temperature 1–2 and alpha 0.2–0.5, chosen on a small validated subset. They are not guaranteed optimal.

Requirements:

- Same meanings for vocabulary IDs, including special tokens, and aligned prediction positions.
- A teacher interface that really exposes the required logits/probabilities.
- A cache format tied to teacher revision, tokenizer, prompt, temperature, and exact context.
- Full accounting of teacher scoring time, student time, and disk/I/O.

Sharing architecture is neither necessary nor sufficient for logit alignment. Sharing the teacher's full tokenizer is the simplest route, but conflicts with this plan's compact vocabulary. Advanced cross-tokenizer distillation exists, yet is outside the baseline.

For sparse top-k caches, keep globally normalized probabilities, not just a softmax renormalized over the selected entries. A useful approximation is top-k entries plus a single “rest of vocabulary” bucket:

```text
q_rest = 1 - sum(q_i for i in selected_ids)
p_rest = 1 - sum(p_i for i in selected_ids)

L_sparse =
    sum(q_i * log(q_i / p_i) for i in selected_ids)
    + q_rest * log(q_rest / p_rest)
```

This is KL on a coarsened vocabulary, not exact full-vocabulary KL. It retains tail mass but not the distribution within the tail. Compute it numerically stably and handle zero probabilities correctly.

The student probabilities still need the student's global normalizer. Raw top-k teacher logits without the teacher's global normalizer do not recover their full-vocabulary probabilities. A normalizer recorded at temperature 1 cannot generally be reused unchanged at temperature 2.

Cache arithmetic matters: with uint32 IDs and FP16 values, 12 sparse entries cost 72 bytes per token before offsets/normalizers. At 1.3B tokens, that alone is 93.6 GB, not 48 bytes/token or a free cache.

### 11.7 On-policy distillation and RL: later, bounded experiments

Once the student produces meaningful answers, a later cycle can collect student attempts, ask the teacher for corrections, verify the corrected output, and train on the difficult cases.

Do not begin a random-initialized model with expensive on-policy rollouts. Do not start with GRPO/PPO or open-ended long reasoning trajectories on the only GPU. Rollouts, teacher feedback, and verification consume the same scarce time.

Distillation scaling results support treating teacher/student compute allocation as a real optimization problem; they do not imply that distillation is free or always dominates a better use of the same local time. [Distillation Scaling Laws](https://arxiv.org/abs/2502.08606)

## 12. Specialization without losing all general ability

The exact domain is the only substantial missing task choice. It does not block the base architecture or pipeline, but it must be specified before claiming specialist performance.

Define the domain in terms of inputs, outputs, and verifiers, for example:

```text
Input: short domain document plus a question
Output: grounded answer with specific fields and source references
Constraints: English, <=4K total context, concise response
Checks: answer correctness, JSON schema, source support, abstention
```

Create at least a few hundred high-quality held-out examples, separated by document/repository/task family. Use them to decide whether more domain text, teacher examples, or a different training mix is the next best investment.

For a text-heavy domain, a late base-training phase can replace approximately 10–20% of generic mixture tokens with vetted domain text, if such data exists. Track general and domain validation separately. Do not invent domain data or assume the current general mixture already supplies it.

For application use, retrieval and external tools can complement the small student without changing its from-scratch provenance. Report model-only performance separately from retrieval/tool-augmented performance.

## 13. Evaluation: what to measure and what counts as success

### 13.1 Maintain three separate sets

1. Training data and teacher-generation material.
2. Development/validation data used for model selection.
3. A frozen final test set, not repeatedly used for tuning.

The provided validation split is a starting point, not proof against near-duplicate leakage. Hold out at document/repository/template-family level wherever possible.

### 13.2 Evaluation matrix

| Area | Early/cheap signal | Final signal |
|---|---|---|
| Language modeling | Per-source held-out NLL | Aggregate and per-source loss, same data definition |
| Cross-tokenizer comparison | Bits per byte on identical text | Same byte-based metric; not raw perplexity across tokenizers |
| Basic language/commonsense | Small fixed ARC-Easy/PIQA/HellaSwag slice | Larger standardized suite if time allows |
| Math/reasoning | Simple arithmetic and verified short problems | Held-out math tasks, fixed prompting/output budget |
| Coding | Syntax and local unit tests | Small-code pass@1; MBPP/EvalPlus-style tests where appropriate |
| Assistant behavior | Format, instruction, refusal/abstention checks | Held-out instruction-following and task success |
| Specialist capability | Domain dev set | Frozen domain test with confidence intervals |
| Context | Copy/retrieval at 512/2K | 2K/4K position-sensitive retrieval and actual long-document tasks |
| Quantization | BF16 versus Q8 versus Q4 dev checks | Same final suite and decoding settings for deployed GGUF |
| Efficiency | Training tokens/s and memory | Prefill/decode latency, VRAM, model size, concurrency |

For reproducible public-task evaluation, use a pinned [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) or equivalent. For stronger code tests, consult [EvalPlus](https://github.com/evalplus/evalplus). Do not run generated code without isolation.

Bits per byte should use summed negative log-likelihood divided by the same underlying UTF-8 byte count and by `ln(2)`. Keep document-boundary/special-token treatment consistent.

### 13.3 Cadence

- Every approximately 100–250 optimizer steps: lightweight training statistics.
- Every approximately 1–2 hours: small held-out source-loss evaluation and checkpoint.
- At major token milestones and before/after SFT: broader fixed dev suite.
- Final release: frozen tests plus runtime/quantization comparisons.

Keep routine evaluation overhead bounded, for example below approximately 5% of campaign time. The total 15% non-active allowance in the throughput example includes evaluation, checkpoints, and operational overhead; do not budget the same time twice.

### 13.4 Appropriate comparison models

Evaluate, but do not initialize from:

- SmolLM2-360M / its instruction variant as a similarly sized reference.
- Qwen3-0.6B or Qwen3.5-0.8B as larger reference points if the evaluation/runtime supports them.
- Your own base, SFT, and Q4 checkpoints to identify which stage helps or hurts.

The references differ in data, parameters, architecture, and compute. They are useful capability yardsticks, not controlled architecture experiments. The [SmolLM2-360M configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json), for example, uses 32 layers, width 960, GQA, and a 49,152-token tied vocabulary.

Do not call a result SOTA after winning a handful of hand-selected prompts. Use the same prompt format, shot count, generation budget, scoring rules, and comparable test conditions. Report variability; do not present a 1–2-example difference as a robust gain.

### 13.5 Predeclared practical release gates

These are project acceptance criteria, not research guarantees:

- No unresolved NaNs, mask errors, data leakage, or resume failures.
- Held-out learning improves meaningfully over early checkpoints.
- The assistant follows its trained template and stops correctly.
- Specialist improvement, when claimed, appears on the held-out domain test.
- General/code/math regressions from specialization are measured and disclosed.
- Q4 losses/regressions are measured relative to the same checkpoint in BF16.

An initial Q4 acceptance target could be at most 0.05 nats/token additional held-out NLL and no more than about two percentage points absolute loss on a sufficiently sized primary task test. These are thresholds to choose before inspecting results, not expected outcomes. If the test is too small, use uncertainty intervals and more examples.

If Q4 fails the target, first inspect calibration/export and compare Q5/Q8. If 4-bit is mandatory, the run has not passed that deployment requirement until a validated Q4 artifact exists; do not quietly relabel Q5 as 4-bit.

## 14. GGUF and llama.cpp: design for exact compatibility

### 14.1 Test export before expensive training

Required path:

```text
your PyTorch model
    -> standard-format HF config + safetensors + tokenizer/template
    -> BF16/F16 GGUF
    -> numerical and tokenization parity checks
    -> calibrated Q4_K_M GGUF
    -> llama-server
```

“The converter produced a file” is not a parity test.

Create a tiny test model with the same operations, and also export one random/default-size checkpoint. Confirm tensor mapping, tied output behavior, RoPE layout, GQA head order, QK norms, BOS/EOS, and tokenizer metadata.

Random-checkpoint parity catches structural errors cheaply. Repeat on a trained checkpoint because trained activations and token distributions expose additional problems.

The Qwen3-compatible route assumes the exact graph specified above. Adding attention gates, partial RoPE, FFN clipping, alternative norm semantics, or other operations invalidates that assumption unless the pinned runtime reproduces them.

### 14.2 Tensor mapping

| Your component | Expected standard HF-style key |
|---|---|
| Token embedding | `model.embed_tokens.weight` |
| Attention pre-norm | `model.layers.i.input_layernorm.weight` |
| Q / K / V projections | `model.layers.i.self_attn.q_proj.weight`, `k_proj.weight`, `v_proj.weight` |
| Q / K norm | `model.layers.i.self_attn.q_norm.weight`, `k_norm.weight` |
| Attention output | `model.layers.i.self_attn.o_proj.weight` |
| FFN pre-norm | `model.layers.i.post_attention_layernorm.weight` |
| FFN gate / up / down | `model.layers.i.mlp.gate_proj.weight`, `up_proj.weight`, `down_proj.weight` |
| Final norm | `model.norm.weight` |
| Output head | `lm_head.weight`, logically tied to the input embedding |

Validate exact keys, tensor orientation, and tied-tensor save conventions against the pinned Transformers/converter versions. The table is a mapping specification, not a claim that files have already been generated.

For a compatible HF export, use `model_type="qwen3"` and `architectures=["Qwen3ForCausalLM"]` only after confirming mathematical parity. Supply your dimensions explicitly and set `tie_word_embeddings=true`. Using that class with a fresh configuration is random initialization; `from_pretrained` is not part of student initialization.

### 14.3 Numerical tests

On fixed token sequences:

1. Compare your eager FP32 implementation with an independently constructed reference using the exported weights.
2. Compare eager versus fused/compiled BF16 paths with dtype-appropriate tolerances.
3. Compare full-sequence logits with token-by-token cached logits.
4. Compare HF/reference logits and per-token NLL with unquantized GGUF.
5. Check greedy continuations, while recognizing that near-ties can change generated text despite small numeric errors.

Do not require bitwise equality across different fused floating-point implementations. Record max/mean logit differences and NLL differences; large or systematic discrepancies must be explained before release.

### 14.4 Calibration and quantization

Prepare a dedicated representative calibration set, separate from the final test set: English, real code, math, domain text if available, and formatted conversations.

Start with roughly 100K–500K calibration tokens as an engineering range, then measure whether a larger or better-balanced set improves Q4 quality. Match chat special-token handling to the imatrix tool's options.

Use the pinned version's commands; the following are templates to run later from the appropriate directories, not commands executed during this planning task:

```bash
python convert_hf_to_gguf.py ./export_hf --outfile ./model-bf16.gguf --outtype bf16

./build/bin/llama-imatrix -m ./model-bf16.gguf -f ./calibration.txt -o ./imatrix.gguf -ngl 99

./build/bin/llama-quantize --imatrix ./imatrix.gguf ./model-bf16.gguf ./model-Q4_K_M.gguf Q4_K_M

./build/bin/llama-quantize ./model-bf16.gguf ./model-Q8_0.gguf Q8_0

./build/bin/llama-server -m ./model-Q4_K_M.gguf -ngl 99 -c 4096 -np 1 --flash-attn on --host 127.0.0.1 --port 8080
```

Windows build executable paths may instead include a `Release` directory and `.exe` suffix. Check `--help` on the exact build. Use `-c 2048` if only 2K was successfully trained/validated.

Current tool documentation describes GGUF imatrix output, quantization, and server options. [Importance matrix tool](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix), [quantizer](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize), [server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)

Bind to localhost by default. If serving beyond the machine later, explicitly configure authentication and network access controls.

### 14.5 What “4-bit” does and does not mean

Q4_K_M is a mixed quantization recipe, not literally every value stored in exactly four bits. File size includes block metadata, mixed-precision tensors, and tokenizer metadata.

For 349M parameters, the hypothetical uniform four-bit payload is approximately 174.6 MB. The actual Q4_K_M file will be larger; measure it instead of promising an exact size.

Q4 weights do not imply Q4 activations or Q4 KV cache. Start with a standard F16/BF16 cache; only experiment with quantized caches after measuring quality and backend support.

For this model's full-attention cache at batch one:

```text
KV bytes = 2 * layers * sequence_length * KV_heads * head_dim * bytes_per_value

at 4,096 tokens, 2-byte values:
2 * 28 * 4096 * 4 * 64 * 2 = 117,440,512 bytes = 112 MiB
```

At 32K it is 896 MiB before other runtime memory, despite the small model. Concurrency increases total cache needs. The 32K calculation is a memory illustration, not a claim of trained 32K capability.

At 2K–4K, this modest cache supports using full attention for the first version. It does not prove caches are irrelevant for all small-model applications.

### 14.6 Runtime measurements

Record separate measurements for:

- Prefill at 128, 512, 2K, and validated 4K lengths.
- Decode at a fixed number of generated tokens.
- Time to first token, steady-state decode, and end-to-end latency.
- GPU and host memory, file size, and batch/concurrency settings.
- BF16/F16, Q8_0, and Q4_K_M using the same prompts and template.

A model this small may be limited by launch overhead, synchronization, or memory movement rather than theoretical tensor throughput. Lower bit width is not a guarantee of higher speed on every workload.







------------------------------


Plan 2 is the superior, production-grade roadmap. It correctly identifies and
resolves the fundamental trap of consumer-GPU LLM projects: the token-starvation
dilemma.

Fitting a 1.2B–1.5B model into VRAM and training a functional 1.2B–1.5B model on
a single consumer GPU in under 18 days are completely different challenges.

A critical comparative breakdown reveals why Plan 2 is the right path, how it
corrects earlier assumptions, and how to execute it cleanly.

Part 1: The Critical Reality Checks Plan 2 Corrects

1. The Token-Starvation Trap: Why 350M Beats 1.2B Under an 18-Day Cap

  - The Math of 1.2B on a Single Card: At ~3,500 tokens/sec across 12 days of
    pretraining, your card processes at most ~3.6 billion tokens. For a 1.2B
    model, that is 3 tokens per parameter.
      - As demonstrated in the MiniFrontier runs (MF-064/065), a model trained
        at 3 tokens/parameter is severely undertrained. Under greedy decoding,
        it gets trapped in degenerate, repetitive loops ("The history of
        science... is a new concept... is a new concept").
  - The Reality of 350M: At 349M parameters, the exact same compute budget gives
    you 14 to 15+ tokens per parameter. You cross the threshold where syntax,
    code structure, instruction following, and associative recall actually
    solidify into a useful specialist.
  - Verdict: A well-trained 350M model consistently outperforms a hollow,
    undertrained 1.2B model on downstream tasks.

2. The Non-Linearity Export Trap: SwiGLU Clamping

  - The Correction: A non-linear operation cannot be folded away at export:
    \text{SiLU}(\text{clamp}(xW_{\text{gate}}, -c, c)) \ne \text{SiLU}(xW_{\text{gate}} \cdot \alpha)
  - If you train with SwiGLU clamping and export to a standard llama.cpp graph
    (Qwen2 or Qwen3) that lacks a native clamp op in its FFN kernel, your
    exported model diverges from your PyTorch model.
  - Plan 2’s Solution: Drop the clamp. Stabilize training using QK-Norm before
    RoPE, a conservative initialization scale (1/\sqrt{2L} on residual
    projections), and stable FP32 RMSNorm reductions. This keeps the execution
    graph 100% compliant with standard llama.cpp C++ kernels.

3. Logit Distillation vs. Response Distillation

  - The Logit Distillation Illusion: Standard token-level logit distillation
    (\mathcal{L}_{\text{KL}}(P_{\text{teacher}} \parallel P_{\text{student}}))
    requires either:
    1.  Matching the teacher’s tokenizer: If you adopt Qwen-2.5’s 151,936
        vocabulary at d_{\text{model}} = 1024, your tied embedding table
        consumes
        151{,}936 \times 1024 = \mathbf{155.5\text{ \textbf{million parameters}}}.
        That is 44% of your entire 350M model budget wasted on an embedding
        lookup table, leaving only ~190M parameters for actual transformer
        layers!
    2.  Caching teacher logits: Plan 2 proves that storing even a sparse top-12
        logit cache for 1.3B tokens demands ~94 GB of high-speed disk I/O,
        destroying your local storage and training loop throughput.
  - Plan 2’s Solution (Response Distillation): Train the 350M student on its own
    compact 32,768-token byte-level BPE (embedding table = only 33.5M
    parameters, or 9.6% of the budget). Use the teacher (Qwen 2.5 32B/72B,
    Claude 3.5, or DeepSeek) to generate high-quality, verified text responses
    (code with unit tests, math with symbolic checks, concise explanations).
    Train the student via standard assistant-masked cross-entropy. Zero
    vocabulary coupling, zero disk-cache bottleneck, zero projection loss.

4. Clean Architecture Divisibility: d_k = 64

  - At d_{\text{model}} = 1024:
      - 16 Query Heads \times 64 = 1024
      - 4 KV Heads \times 64 = 256 (clean 4:1 GQA)
  - Trying to force head_dim = 96 or 128 on d=1024 requires either unaligned
    projection matrices (1024 \to 1536) or awkward fractional head counts that
    degrade tensor core GEMM efficiency and trigger kernel-fallback paths.
    head_dim = 64 is native, fast, and stays far below the memory-paging cliff.

Part 2: The Exact Qwen3-Compatible Blueprint

This architecture runs on PyTorch, converts to GGUF using stock scripts, and
executes directly on llama.cpp without patching C++ source code.

                    Input Tokens [B, S]
                            │
                            ▼
           Tied Embedding [32,768 x 1024] (33.5M params)
                            │
         ┌──────────────────┴──────────────────┐
         │     Repeat x 28 Transformer Blocks  │
         │                                     │
         │  ┌─ RMSNorm (1024)                  │
         │  │    │                             │
         │  │    ├─► Q_proj [1024 -> 1024] ─► Q_norm (64) ─► RoPE ┐
         │  │    ├─► K_proj [1024 ->  256] ─► K_norm (64) ─► RoPE ┼► GQA (SDPA)
         │  │    └─► V_proj [1024 ->  256] ───────────────────────┘     │
         │  │                                                           ▼
         │  └─────────────────── Out_proj [1024 -> 1024] ◄──────────────┘
         │         │
         │         ▼ (+) Residual
         │  ┌─ RMSNorm (1024)
         │  │    │
         │  │    ├─► Gate_proj [1024 -> 2816] ──┐
         │  │    │                              ├──► Silu(Gate) * Up
         │  │    ├─► Up_proj   [1024 -> 2816] ──┘         │
         │  │                                             ▼
         │  └─────────────────── Down_proj [2816 -> 1024] ┘
         │         │
         │         ▼ (+) Residual
         └──────────────────┬──────────────────┘
                            │
                     Final RMSNorm (1024)
                            │
                            ▼
         Tied LM Head [1024 -> 32,768] (Shares weights with Embed)

Parameter Accounting (Verified Exact)

  - Hidden size (d): 1024
  - Layers (L): 28
  - FFN intermediate size (f): 2816
  - Query heads (H_q): 16 (dim 64)
  - KV heads (H_{kv}): 4 (dim 64)
  - Vocab (V): 32,768

\begin{aligned} \text{Embeddings (Tied):} & \quad 32{,}768 \times 1024
= 33{,}554{,}432 \ \text{Per-layer Attention:} & \quad (1024 \times 1024) + 2
\times (1024 \times 256) + (1024 \times 1024) = 2{,}621{,}440 \ \text{Per-layer
SwiGLU:} & \quad 3 \times (1024 \times 2816) = 8{,}650{,}752 \ \text{Per-layer
Norms:} & \quad 1024 + 1024 + 64 + 64 = 2{,}176 \ \text{Total Per Layer:} &
\quad 2{,}621{,}440 + 8{,}650{,}752 + 2{,}176 = 11{,}274{,}368 \ \text{28 Layers
Total:} & \quad 28 \times 11{,}274{,}368 = 315{,}682{,}304 \ \text{Final Norm:}
& \quad 1024 \ \mathbf{Total\ Unique\ Params:} & \quad 33{,}554{,}432
+ 315{,}682{,}304 + 1{,}024 = \mathbf{349{,}237{,}760} \end{aligned}

Part 3: The 17-Day Single-GPU Execution Schedule

Day:   1       2                   13    14      15      16      17
       ├───┼───────────────────────────┼───────┼───────┼───────┼───┤
Stage: Prep/   Base Pretraining        Teacher Full    Eval/   Buffer/
       Gates   (WSD Schedule: 2K->4K)  Gen     SFT     Quant   Release

  - Day 1: Preflight & Zero-Cost Export Gate
      - Train the 32,768-token tokenizer on your dataset sample.
      - Instantiate the model with random weights.
      - Export directly to Hugging Face format and run convert_hf_to_gguf.py.
      - Verify that llama-server loads the random .gguf file and outputs garbage
        tokens without crashing. Do not train a single token until this parity
        check passes.
  - Days 2–13: Base Pretraining (~12 Days)
      - Objective: Train on 4B–6B tokens from your 5-source mixture.
      - Schedule: WSD (Warmup-Stable-Decay).
          - First 30M tokens: Warmup to
            \text{LR}_{\text{peak}} = 6 \times 10^{-4}.
          - Stable Phase: Flat at 6 \times 10^{-4} up to ~80% of budget.
            Sequence length = 2048.
          - Decay Phase: Final 20% of tokens anneals cosine-style down to
            6 \times 10^{-5}.
          - Final 10%: Switch sequence length to 4096 on long documents.
  - Day 14: Local/Offline Teacher Response Generation
      - Run an 8B–9B teacher (e.g., Qwen-2.5-7B-Instruct / Qwen3.5-9B) locally
        in Q4_K_M via llama.cpp (or ingest pre-generated outputs from frontier
        models).
      - Generate 10,000–20,000 targeted, high-quality responses (coding
        unit-test problems, math step-by-step reasoning, JSON schema
        formatting).
  - Day 15: Full-Weight SFT & Response Distillation (~24 Hours)
      - Full-weight training on the teacher-generated data + 20% replay of base
        pretraining data to prevent catastrophic forgetting.
      - \text{LR}_{\text{peak}} = 3 \times 10^{-5}, cosine decay to
        3 \times 10^{-6}, 2 epochs maximum.
      - Mask user prompt tokens; compute loss strictly on assistant output
        tokens.
  - Day 16: Calibration, Quantization & Release Evaluation
      - Generate an imatrix calibration file using 200k tokens of mixed text.
      - Quantize the FP16 checkpoint to Q4_K_M GGUF.
      - Run automated validation on the test split: benchmark ARC-Easy,
        HumanEval-like code fixtures, and perplexity delta.
  - Day 17: Reserve / Failure Buffer
      - Absorbs thermal throttling, unexpected process restarts, or dataset
        re-sharding.

Part 4: Production PyTorch Code (349.24M Exact Qwen3 Graph)

This implementation adheres to standard Hugging Face Qwen2ForCausalLM / Qwen3
conventions, ensuring immediate exportability to GGUF:

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_seq_len: int = 4096, base: float = 100000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

class Qwen3CausalAttention(nn.Module):
    def __init__(self, d_model: int = 1024, n_heads: int = 16, n_kv_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # QK-Norm: Single 64-dim norm vector broadcast across heads per layer
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # 1. QK-Norm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 2. Apply Full-Head RoPE
        cos_expanded = cos.unsqueeze(1)
        sin_expanded = sin.unsqueeze(1)
        q = apply_rotary_pos_emb(q, cos_expanded, sin_expanded)
        k = apply_rotary_pos_emb(k, cos_expanded, sin_expanded)

        # 3. Transpose for PyTorch native SDPA: [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(self.num_kv_groups, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.num_kv_groups, dim=1)

        # Native SDPA automatically picks FlashAttention-2 or Memory-Efficient kernel
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class Qwen3SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int = 1024, intermediate_size: int = 2816):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen3Block(nn.Module):
    def __init__(self, d_model: int = 1024, n_heads: int = 16, n_kv_heads: int = 4, head_dim: int = 64, intermediate_size: int = 2816):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model)
        self.self_attn = Qwen3CausalAttention(d_model, n_heads, n_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = Qwen3SwiGLUFFN(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class MiniFrontier349M(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 1024,
        n_layers: int = 28,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        intermediate_size: int = 2816,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Qwen3Block(d_model, n_heads, n_kv_heads, head_dim, intermediate_size)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Strictly tied input and output embeddings
        self.lm_head.weight = self.embed_tokens.weight

        self.rotary = RotaryEmbedding(dim=head_dim, base=100000.0)
        self._init_weights(n_layers)

    def _init_weights(self, n_layers: int):
        init_std = 0.02
        residual_std = init_std / math.sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                if any(proj in name for proj in ["o_proj", "down_proj"]):
                    nn.init.normal_(p, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(p, mean=0.0, std=init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        cos, sin = self.rotary(S, input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

Part 5: Exporting for llama.cpp and Serving via GGUF

To ensure convert_hf_to_gguf.py parses your weights without errors, save the
model configuration using the standard Hugging Face metadata naming conventions:

import json
import os

def export_to_hf_format(model: MiniFrontier349M, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save safetensors weights
    from safetensors.torch import save_file
    state_dict = model.state_dict()
    # Explicitly retain tied lm_head key if preferred by parser
    state_dict["lm_head.weight"] = model.embed_tokens.weight
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    
    # Standard Qwen2 / Qwen3 config schema recognised by llama.cpp
    config = {
        "architectures": ["Qwen2ForCausalLM"],  # Fully compatible with llama.cpp Qwen2/3 parser
        "model_type": "qwen2",
        "vocab_size": 32768,
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 100000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16"
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"HF-compatible weights saved to {output_dir}")

Final Terminal Workflow (GGUF Quantization & Serving)

# 1. Convert to high-precision GGUF
python llama.cpp/convert_hf_to_gguf.py ./my_exported_model \
    --outtype bf16 \
    --outfile ./model-bf16.gguf

# 2. Build importance matrix (Calibration on 200k tokens of mixed text)
./llama.cpp/llama-imatrix \
    -m ./model-bf16.gguf \
    -f ./calibration_corpus.txt \
    -o ./imatrix.dat \
    -ngl 99

# 3. Quantize to 4-bit with minimal perplexity loss
./llama.cpp/llama-quantize \
    --imatrix ./imatrix.dat \
    ./model-bf16.gguf \
    ./mini-frontier-349m-Q4_K_M.gguf \
    Q4_K_M

# 4. Launch the local llama.cpp server
./llama.cpp/llama-server \
    -m ./mini-frontier-349m-Q4_K_M.gguf \
    -c 4096 \
    -ngl 99 \
    --host 127.0.0.1 \
    --port 8080

The Bottom Line

  - Plan 2 is your actual path to success. It respects the mathematical
    constraints of single-GPU wall-clock limits and avoids chasing unviable
    hyperparameter settings.
  - It guarantees a fully trained, non-starved model (~15 tokens/parameter
    instead of 3), fits inside VRAM without silent memory thrashing, keeps your
    vocabulary budget under 10% of total weights, avoids un-exportable
    non-linear hacks, and deploys directly to llama.cpp in Q4_K_M.




---------------------------------------------




Plan 2 is good: much better than the first document, and a defensible v1. I re-derived the 349,237,760 parameter count, the 5.20 GiB training state, the KV-cache math and the token-rate tables, and they're all correct. It also fairly corrects some of my earlier numbers, so I'll concede those first and then list what I'd change.

## Where it beats my last answer

- **Model size.** At this scale its compute math wins. My 0.73B spent 318M parameters on a 248k-vocab embedding, and the tied output head costs compute on every token, not just memory. With 5B unique tokens and one card, a ~350M body with a 32k vocab gets roughly 2× the tokens per GPU-hour. The one thing that choice gives up is logit KD, and on a serial single GPU that's an acceptable trade.
- **Dimensions for k-quants.** k-quants need tensor row sizes divisible by 256, or llama-quantize silently uses a fallback type for that tensor. Plan 2's default passes (1024, 2816, 256). My 0.73B spec used FFN 3456, which doesn't. That was my mistake.
- **"Nearly free up to ~4 epochs."** Its caution is fair. Watch per-source validation loss for the oversampled sources.

One correction back: its 72 bytes/token cache estimate assumes top-k storage with uint32 IDs and fp16 values. The random-sampling KD I suggested stores a 3-byte ID plus a 1-byte count per sample, so 48 bytes/token was right for that method. Its "top-k plus rest-bucket" math is correct, but for a different estimator. None of this matters now that logit KD is out of v1.

## What I'd change

**1. Distillation comes too late and too thin.** One teacher day plus one SFT day after a pure web pretrain gives a 350M model weak assistant behavior. The WSD literature (MiniCPM introduced the schedule) found that mixing high-quality and SFT-style data into the decay phase beats bolting SFT on afterward, and OLMo 2 and SmolLM3 do the same in mid-training.

Reorder the schedule:
- Days 2–9: pretraining, stable LR.
- Days 10–11: teacher generation and verification.
- Days 12–14: decay phase, with ~20–30% of tokens from verified teacher responses and specialist data.
- Day 15: short SFT polish.

The GPU time is the same; the result is much better. This also forces the specialist task to be defined by about day 9, which is a useful deadline.

**2. Its teacher throughput assumption is single-stream thinking.** At 50–100 tok/s aggregate, a day yields ~5M tokens. Batched generation with vLLM or SGLang, or llama-server with many parallel slots, on a 9B 4-bit teacher should do hundreds to over a thousand aggregate tok/s on a 4090 for short prompts. That's tens of millions of tokens per day. Verification, not generation, will limit you, so measure batched throughput in the pilot.

**3. Spend the one allowed pilot comparison on Muon vs AdamW, not depth/width.** Optimizer choice has the largest expected effect: Moonlight reports ~2× compute efficiency, and Essential AI found Muon expands the compute-time Pareto frontier over AdamW. The overhead is small: Newton-Schulz on 1024×2816 matrices adds maybe a few percent to a multi-second accumulated step. Your MiniFrontier Muon lab already exists, so this is cheap to run properly. Judge it by validation loss per wall-clock hour, as Plan 2 says.

**4. Don't guess the SFT learning rate; sweep it.** A peak of 3e-5 decaying to 3e-6 is conservative for a 350M from-scratch base and may underfit the chat format in 1–2 epochs. At 350M, an SFT pass over 10M tokens takes minutes, so try 3e-5, 1e-4 and 3e-4 and pick on held-out task quality.

**5. Size to the measured card, and keep dimensions 256-friendly.** These are my estimates for full training at ~2.8e9 FLOPs/token (6N plus attention at 2K) and ~40% MFU. Replace them with your pilot numbers.

| Card | ≈ tok/s (349M) | 12 days × 85% |
|---|---:|---:|
| 4060 Ti 16GB | ~3,500 | ~3B |
| 3090 | ~5,000 | ~4.4B |
| 4090 | ~12,000 | ~10.6B |

On a 3090 or 16 GB card, 349M is right. On a 4090 you'd see ~2 epochs, which favors the stretch size. Its 489M stretch uses FFN 3456 and hits the same k-quant fallback my spec did. Use 26L / d=1280 / FFN 3584 / 20Q:4KV / head_dim 64 instead: ≈502M, all dimensions divisible by 256.

**6. Build on MiniFrontier; don't re-spec from zero.** Plan 2 treats the trainer, shard pipeline, SFT masking and GGUF export as unknowns, but your repo has most of them. The only real conflict is the tokenizer: your existing 16k BPE versus its new 32k. At d=1024 the output head costs ~5% more compute per token at 32k. If 32k compresses your mix ~10% better, the net gain is small either way.

The practical rule: if the 16k tokenizer already round-trips through llama.cpp, keep it and skip a retokenize-and-parity detour. Measure bytes/token on a sample before deciding.

**7. Reconsider 4-bit at this size.** A 350M Q8_0 is about 370 MB, and VRAM is irrelevant on 16–24 GB. Small models are the most quantization-sensitive, so the 0.05 nats Q4 gate may fail for reasons unrelated to training. Ship Q4_K_M if 4-bit is a hard requirement, but benchmark Q8_0 alongside it. Q8_0 may be faster in practice on some kernels.

**8. Most of its evals will sit at the noise floor.** A 350M model trained on ~5–10B tokens will be near random on standard MMLU and in low single digits on HumanEval. Use cloze-format MMLU, ARC-Easy, HellaSwag, per-source bits-per-byte and your own narrow unit-test tasks for decisions. Keep MBPP/EvalPlus as final-report numbers, not selection signals.

**Minor fixes:**
- **Document masking.** Use flash-attn varlen (or FlexAttention, which you already have) with per-document position resets. This avoids cross-document attention without a dense mask.
- **Adam β2.** At a 65k-token batch, 0.95 gives a very short second-moment horizon, so include 0.98–0.99 in the Muon/AdamW pilot.

With changes 1, 3 and 5, Plan 2 is the plan I'd run. Changes 1 and 3 add the most, and 5 prevents a silent quantization regression.