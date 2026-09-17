# Small LLM research and training plan

Research date: September 17, 2026.

This document preserves the full original research plan and adds a revised plan for the subsequently clarified constraints.

Confirmed requirements:

- A single consumer NVIDIA RTX GPU, Ampere or newer, for both student training and inference.
- Hardware range clarified: RTX 30/40/50 series with **16–24 GB VRAM**. like MSI GeForce RTX 4070 Ti SUPER 16G GAMING X SLIM
- A model below approximately 1–2B parameters.
- A specialized domain or task, while retaining some general assistant, reasoning, and coding capability.
- Mainly English.
- PyTorch training and eventual 4-bit GGUF deployment through llama.cpp server.
- Later distillation from a larger, stronger teacher.
- Available corpus: [MiniFrontier-150M-Modern-3B-token-mixture](https://huggingface.co/datasets/igalk474/MiniFrontier-150M-Modern-3B-token-mixture).
- Training limit: **less than 18 days**.

Still to specify: the exact GPU within that range, the specialized task, any specialist instruction/answer data beyond the linked raw-text corpus, and whether teacher-generated data can come from an API or an existing dataset.

**Planning update:** Given the single-GPU training constraint, adapting a pretrained student is the recommended production route. The custom architecture below remains a research design. The original plan is preserved with its earlier assumptions; the [current hardware and dataset plan](#current-hardware-and-dataset-plan) incorporates the latest requirements and supersedes conflicting earlier recommendations.

---

## Original research plan

My recommendation is to build a **dense, decoder-only model around 1.6–1.8B parameters**, using a Qwen3-compatible Transformer as the first implementation. Use GQA, SwiGLU, RMSNorm, QK normalization, RoPE, and tied embeddings.

Then evaluate **one hybrid challenger: Gated DeltaNet interleaved with full attention**. That becomes especially attractive if long-context inference is central to your product.

The biggest determinant of competitiveness will be your **data, training budget, and distillation pipeline**. Architecture matters, but a clever attention mechanism cannot compensate for a severely undertrained model.

I researched this against sources available on **September 17, 2026**. Nobody can identify a guaranteed winner for 2027+. The practical objective is an architecture you can train well, evaluate fairly, and improve without repeatedly rebuilding your deployment stack.

I’ll assume a text-only assistant focused on English, reasoning, and code; initially 8K context, later 32K; and primarily one user at a time. Training hardware remains unspecified, so the plan includes both a custom pretraining route and a much cheaper pretrained-student route.

1. Establish the competitive target before designing the network.

   These are useful reference points:

   | Reference | What it contributes to your design |
   |---|---|
   | Qwen3-1.7B | A conventional dense Transformer with GQA, QK normalization, SwiGLU, and tied embeddings |
   | Qwen3.5-0.8B / 2B | Small hybrid models using three Gated DeltaNet blocks followed by one full-attention block |
   | LFM2.5-1.2B | A different hybrid: short convolutions mixed with GQA |
   | SmolLM2-1.7B | An unusually transparent small-model pretraining recipe |

   Those descriptions come from the published [Qwen3 configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json), [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-2B), [LFM2.5 model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct), and [SmolLM2 paper](https://arxiv.org/abs/2502.02737).

   Benchmark these on your intended RTX and tasks. Published scores use different reasoning budgets, prompts, and evaluation settings; they do not establish which model is best for your application.

   Also define “under 2B” as **total deployed parameters**, including embeddings and any auxiliary modules. “Active” or “effective” parameters can be misleading for this constraint. For example, Google lists Gemma 4 E2B as 2.3B effective but 5.1B with embeddings. It is therefore a useful architectural reference, but outside a strict 2B total-parameter budget. [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)

2. Start with this exact custom architecture.

   This is my proposed starting configuration, not a claim that this exact combination has already demonstrated SOTA.

   | Component | Starting choice |
   |---|---|
   | Architecture | Autoregressive decoder-only Transformer |
   | Total parameters | Approximately **1.661B** |
   | Transformer layers | **28** |
   | Hidden width | **2,048** |
   | Query heads | **16** |
   | KV heads | **4** |
   | Head dimension | **128** |
   | Attention | Full causal attention in every layer |
   | FFN | **SwiGLU** |
   | FFN intermediate width | **7,168** |
   | Normalization | Pre-RMSNorm; final RMSNorm |
   | QK normalization | RMSNorm over each Q/K head, before RoPE |
   | Norm epsilon | `1e-6` |
   | Positional encoding | Full-head RoPE |
   | Initial RoPE theta | `1_000_000`, subject to context validation |
   | Vocabulary | **65,536**, including reserved tokens |
   | Input/output embeddings | **Tied** |
   | Linear-layer biases | None |
   | Dropout | `0.0` for large-scale pretraining |
   | Attention soft-capping | Off |
   | Output-logit soft-capping | Off |
   | Initial training length | 4,096 |
   | First supported context target | 8,192 |
   | Later context target | 32,768 after explicit training and evaluation |
   | Training precision | BF16 with appropriate FP32 reductions/state |
   | Deployment target | Calibrated GGUF `Q4_K_M` |

   The block should have this structure:

   ```text
   h = RMSNorm(x)

   q = QKNorm(Wq(h))
   k = QKNorm(Wk(h))
   v = Wv(h)

   q, k = RoPE(q, k)
   a = causal_grouped_query_attention(q, k, v)

   x = x + Wo(a)

   h = RMSNorm(x)
   x = x + Wdown(SiLU(Wgate(h)) * Wup(h))
   ```

   After the last block:

   ```text
   logits = RMSNorm(x) @ token_embedding.weight.T
   ```

   The parameter allocation is approximately:

   | Component | Parameters |
   |---|---:|
   | Shared embedding/output matrix | 134M |
   | Attention projections | 294M |
   | Feed-forward networks | 1,233M |
   | Normalization | 0.12M |

   This explains an important optimization priority: **about 74% of the parameters are in the FFNs**. Attention is only one part of the cost.

   The computational structure closely follows a supported Qwen3 graph. llama.cpp already implements its QK normalization, GQA, SwiGLU, and tied-output behavior. Custom dimensions and tokenizer metadata still need conversion tests. [llama.cpp Qwen3 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3.cpp)

3. Make the attention decision according to workload.

   First, these terms describe different axes:

   - **Dense versus MoE:** how the FFN parameters are used.
   - **Full versus sliding-window versus recurrent/linear attention:** how information moves between tokens.
   - **GQA:** how query heads share keys and values.

   A model can have dense FFNs, hybrid attention, and GQA simultaneously.

   | Choice | My recommendation for your project |
   |---|---|
   | Dense FFNs | **Yes.** Straightforward training, predictable memory use, mature kernels |
   | MoE | Defer under a strict 1–2B total-parameter budget |
   | Full attention + GQA | **First implementation** for an 8K–32K target |
   | Sliding-window + global attention | Good challenger when KV memory dominates |
   | Gated DeltaNet + full attention | **Main hybrid challenger**, especially for frequent long prompts |
   | Pure recurrent/linear architecture | Specialized research branch; carefully test retrieval and copying |
   | Differential attention | Research ablation after the baseline works |
   | MLA / aggressive cross-layer KV sharing | Defer unless profiling identifies a compelling need |

   **Full GQA:** Start with 16 query heads and four KV heads. Relative to 16-head MHA with the same head dimension, this cuts KV storage by four. Test eight KV heads as the quality-oriented alternative. Treat two KV heads as an additional compression experiment.

   **Sliding-window interleaving:** A reasonable experiment is three local-attention layers followed by one global layer, with a 1K or 2K local window. Keep the final layer global. These ratios are proposed experiments; Gemma 3 provides evidence for the broader approach with its published 5:1 local/global design. [Gemma 3 report](https://arxiv.org/html/2503.19786v1)

   Real savings require window-aware kernels and bounded local caches. Applying a local mask to an otherwise full attention implementation does not automatically produce the desired speed or memory savings.

   **Gated DeltaNet interleaving:** Use the established pattern:

   ```text
   [Gated DeltaNet → Gated DeltaNet → Gated DeltaNet → Full attention] × N
   ```

   Qwen3.5 already demonstrates this at small sizes, and llama.cpp has a corresponding implementation. Its recurrent layers avoid a token-growing KV cache, while the remaining full-attention layers preserve direct access to earlier tokens. The overall model still has context-dependent memory and attention cost. [Qwen3.5 configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json), [llama.cpp implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)

   For this experiment, copy the complete published block semantics—including gates, normalization, convolution state, and recurrent-state precision. Do not assemble a partially compatible variant from isolated ideas.

   My decision rule: promote the hybrid if it improves **quality at your latency and memory limits**. A better asymptotic complexity alone is insufficient, particularly at short contexts.

4. Keep depth, width, gating, and normalization experiments controlled.

   **Depth versus width:** I would start at 28 layers and width 2,048. Deeper models can use parameters effectively, but each additional layer adds sequential work during generation.

   MobileLLM provides evidence for deeper, narrower architectures at small sizes. That evidence does not establish the fastest configuration for an RTX running quantized llama.cpp kernels. [MobileLLM paper](https://arxiv.org/abs/2402.14905)

   Compare approximately equal-parameter candidates:

   | Candidate | Layers | Width | FFN width | Q/KV heads | Approx. parameters |
   |---|---:|---:|---:|---:|---:|
   | Wider, shallower | 20 | 2,560 | 7,680 | 20 / 4 | 1.662B |
   | Recommended starting point | 28 | 2,048 | 7,168 | 16 / 4 | 1.661B |
   | Deeper, smaller FFN | 32 | 2,048 | 6,144 | 16 / 4 | 1.678B |

   All use a 65,536-token tied vocabulary and 128-dimensional heads. Compare both training efficiency and actual decode latency.

   **SwiGLU:** Keep it as the default. Its three projection matrices make FFN width a major parameter-budget decision. Do not assume the same “4× expansion” convention used for a two-matrix GELU FFN.

   **Soft-capping:** Leave both attention and output caps disabled initially. Use QK normalization and monitor attention logits, gradient norms, and activation outliers. Gemma 3 explicitly replaced Gemma 2’s soft-capping with QK normalization; that supports testing normalization first, without proving that soft-capping is universally harmful. [Gemma 3 report](https://arxiv.org/html/2503.19786v1)

   **Attention output gating:** This deserves an early ablation. Research including 1.7B dense models found benefits from sigmoid gating after attention. However, headwise and elementwise gates have different costs, and adding either changes the deployment graph. Include those costs in the experiment. [Gated Attention paper](https://arxiv.org/abs/2505.06708)

   **Differential attention:** Prefer investigating DIFF V2 over reproducing V1. V2 simplifies the design and can use standard FlashAttention kernels, but PyTorch kernel compatibility does not establish compatibility or speed in your quantized llama.cpp graph. Treat the authors’ reported improvements as motivation to test it, not a guarantee for your model. [Microsoft’s DIFF V2 description](https://huggingface.co/blog/microsoft/diff-attn-v2)

   **RoPE/NoPE interleaving:** Lower priority than the above. SmolLM3 removes RoPE from every fourth layer and reports useful long-context results, but that is a positional-encoding choice; it does not itself remove attention’s KV cache. [SmolLM3 recipe](https://huggingface.co/blog/smollm3)

5. Decide the tokenizer together with the distillation strategy.

   This decision materially changes the model.

   For an English/code model trained from scratch and distilled from teacher-generated text, I would start with **64K byte-level BPE**, using pre-tokenization behavior that you can reproduce exactly in llama.cpp.

   Validate tokenization on code indentation, Unicode, numbers, mathematical notation, JSON, and any important non-English languages.

   For direct token-distribution distillation from an open teacher, retaining its tokenizer can be more valuable than having a smaller vocabulary.

   Conveniently, you can keep approximately the same total parameter count:

   | Variant | Vocabulary | FFN width | Approx. parameters |
   |---|---:|---:|---:|
   | English/code-oriented | 65,536 | 7,168 | 1.661B |
   | Qwen3-tokenizer-aligned | 151,936 | 6,144 | 1.662B |

   Both retain the proposed 28-layer, 2,048-wide, 16/4-head structure.

   The larger-vocabulary version spends more capacity on embeddings and the output projection. The smaller-vocabulary version spends more on FFNs, but may require more tokens to represent the same text. Compare **bytes processed, task quality, and wall-clock cost**, not just tokens per second.

   Keep embeddings tied initially. Untying the 64K vocabulary adds about 134M parameters. Test whether untying beats spending the same parameter budget on additional FFN capacity.

   If adapting an existing checkpoint, preserve its architecture and tokenizer initially. Changing either creates a separate adaptation problem.

6. Optimize the bottlenecks that actually matter on the RTX.

   | Workload | Likely bottlenecks | First interventions |
   |---|---|---|
   | Short-context, single-stream decode | Weight traffic, small operations, sequential layers | GGUF quantization, suitable dimensions, fewer unnecessary operations |
   | Long-context decode | KV-cache reads and attention work | GQA, cache quantization, local/global or recurrent hybrid |
   | Prompt processing | Matrix multiplication and attention computation | Efficient attention kernels; benchmark prompt lengths separately |
   | Training | Optimizer state, activations, output logits, data delivery | BF16, checkpointing, fused loss, efficient packing and loading |

   For the proposed model, FP16 KV memory per sequence is:

   \[
   M_{\mathrm{KV}}=2 \times L \times T \times H_{\mathrm{KV}}\times d_{\mathrm{head}}\times 2\text{ bytes}.
   \]

   | Cached tokens | FP16 KV cache |
   |---:|---:|
   | 8,192 | **448 MiB** |
   | 16,384 | **896 MiB** |
   | 32,768 | **1.75 GiB** |

   These numbers exclude weights, runtime buffers, CUDA overhead, and other concurrent sequences.

   The mathematical 4-bit weight payload is approximately 0.83 GB. Actual `Q4_K_M` files are larger because of block metadata and mixed tensor precision; use roughly **1.0–1.2 GB as a planning estimate**, then measure the converted artifact.

   Weight quantization and KV-cache quantization are separate decisions. Start with FP16 cache for quality measurements, then evaluate Q8 cache.

   Also measure **time to a correct answer**. A model that emits twice as many reasoning tokens can feel slower despite a higher raw generation rate.

7. Build the PyTorch implementation and GGUF export together.

   My implementation order would be:

   1. Implement the reference block and causal loss.
   2. Match an established implementation numerically.
   3. Add cached decoding and verify it against full-sequence evaluation.
   4. Export an untrained or tiny-trained checkpoint.
   5. Verify tokenizer and logits in llama.cpp.
   6. Only then optimize and start expensive training.

   The main correctness checks should cover:

   - Next-token label shifting and loss masking.
   - Causal masking, especially when cached query/key lengths differ.
   - RoPE positions during incremental decoding.
   - GQA head mapping.
   - Tied-weight serialization.
   - Packed-document boundaries.
   - Checkpoint resume, including optimizer, RNG, and data position.
   - PyTorch → HF-compatible checkpoint → GGUF numerical agreement.

   For performance, use PyTorch SDPA or FlashAttention-2 on Ampere, then profile. Verify which kernel actually runs with your shapes and masks. Current PyTorch documents explicit GQA support and constraints; FlashAttention-2 supports Ampere, while newer FlashAttention versions target different hardware generations. [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [FlashAttention repository](https://github.com/Dao-AILab/flash-attention)

   Add `torch.compile`, selective activation checkpointing, and a fused or chunked output loss after correctness is established.

   Avoid materializing the full `[batch, sequence, vocabulary]` logits tensor unnecessarily during ordinary cross-entropy training. Cut Cross-Entropy is one relevant implementation, although distillation losses may need a different computation path. [Cut Cross-Entropy](https://github.com/apple-aiml-research/ml-cross-entropy)

   For the training environment, I would use Linux or WSL2 with pinned PyTorch/CUDA/kernel versions. Keep a simple unfused implementation available for debugging.

8. Set the compute budget before committing to training from scratch.

   A 1.66B model fitting in VRAM for inference says little about its training cost.

   Conventional mixed-precision Adam training commonly requires approximately **12–16 bytes per parameter for model/gradient/optimizer state**, depending on implementation. Here that is roughly **20–27 GB before activations and temporary buffers**. Memory-saving optimizers and offloading change this, but introduce tradeoffs.

   A rough dense-training estimate is:

   \[
   \text{FLOPs}\approx6NT.
   \]

   For 1.66B parameters:

   | Training tokens | Approximate FLOPs | At a hypothetical sustained 100 TFLOP/s |
   |---:|---:|---:|
   | 10B | \(9.96\times10^{19}\) | 11.5 days |
   | 100B | \(9.96\times10^{20}\) | 115 days |
   | 1T | \(9.96\times10^{21}\) | 3.2 years |

   This is an illustrative compute estimate, **not an RTX throughput benchmark**. It excludes teacher inference and additional costs such as evaluation, recomputation, and long-context attention.

   Modern small models often receive enormous training budgets: SmolLM2 reports about 11T tokens; LFM2.5-1.2B reports 28T. These are examples of the scale of competition, not mandatory token targets for every useful student. [SmolLM2](https://arxiv.org/abs/2502.02737), [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)

   Therefore:

   - **One consumer GPU for training too:** start from an existing pretrained student; focus on distillation and specialization.
   - **A modest rented-GPU budget:** prioritize a strong existing base, data experiments, and selective continued pretraining.
   - **A substantial pretraining budget:** pursue the custom architecture with controlled ablations and staged scaling.

   For your goal of a competitive usable model, I would choose the pretrained-student route unless architectural research itself is a primary objective.

9. Build a data pipeline with explicit quality controls.

   For custom pretraining, a reasonable initial English/code mixture to test is:

   | Data category | Initial proportion |
   |---|---:|
   | General and educational text, documentation, books | 70% |
   | Code and code-related technical text | 20% |
   | Mathematics and worked problems | 10% |

   These percentages are proposed starting points, not an established optimum. Later, test increasing code and math while retaining sufficient general-language data.

   Useful public starting resources include [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu), and [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath).

   The pipeline should:

   - Track provenance, permitted use, source, language, and quality.
   - Remove exact and near duplicates.
   - Filter boilerplate, corrupted text, repetitive material, and low-quality synthetic content.
   - Split by document/source before producing training chunks.
   - Decontaminate against your evaluations.
   - Monitor repetition of small high-quality subsets.
   - Retain real held-out data for each important domain.

   Tokenize ahead of training and use reproducible, efficiently readable shards. At a 65,536-entry vocabulary, unsigned 16-bit token storage is possible if every ID fits; larger vocabularies require wider storage.

   Teacher data should be selected for what the student needs to learn: clear explanations, accurate answers, executable code, grounded extraction, and realistic conversations. More synthetic tokens are not automatically better.

10. Use staged training with explicit exit conditions.

    For custom pretraining, I would use this progression:

    | Stage | Purpose | Exit condition |
    |---|---|---|
    | Tiny correctness run | Validate implementation and resume | Stable loss; correct export and decoding |
    | Small proxy experiments | Screen data mixtures and architecture choices | Eliminate clearly inferior candidates |
    | Target-size pilot | Validate scaling, stability, and throughput | Quality improves predictably within budget |
    | Main pretraining | Learn broad capabilities | Continue while measured gains justify cost |
    | Quality-focused annealing | Emphasize stronger data | Better held-out and downstream results |
    | Context extension | Move from 4K/8K toward 32K | Pass retrieval and long-context task tests |
    | Distillation/post-training | Teach useful behavior and reasoning | Improvement without unacceptable regressions |
    | Quantization validation | Produce the deployed artifact | Acceptable task quality in llama.cpp |

    A starting optimizer configuration for **training from scratch**:

    ```text
    optimizer: AdamW
    learning rate: start testing around 3e-4
    LR sweep: 2e-4, 4e-4, 6e-4
    betas: (0.9, 0.95)
    epsilon: 1e-8
    weight decay: 0.1 on selected matrix weights
    gradient clipping: global norm 1.0
    scheduler: warmup → stable → decay
    precision: BF16; FP32 where required for stability
    ```

    Apply these as pilot settings. Learning rate depends on batch size, initialization, data, and training duration. Specify effective batch size in **tokens**, then choose microbatching and accumulation to fit hardware.

    Start context training at 4K. Extend to 8K, then 16K/32K with appropriate data and a lower learning rate. Changing the maximum-position configuration alone does not establish usable context length.

    During training, monitor domain losses, gradient norms, activation/logit outliers, throughput, memory, and downstream evaluations. Keep checkpoints before data-mixture and context transitions.

11. Distill through progressively stronger supervision.

    A strong teacher does not require the student to share its architecture. The important distinction is whether you have **text outputs** or **token-level distributions**.

    **First: response distillation.**

    Generate teacher answers, filter them, and train the student on accepted responses. This works across tokenizers and architectures.

    Begin with a carefully inspected pilot of roughly 100K examples, then expand based on measured gains. Include direct answers, concise worked solutions, code with tests, structured outputs, grounded QA, and multi-turn conversations.

    Train on assistant-response tokens while masking prompts and padding. Preserve general capabilities through a suitable mixture of existing instruction data and replay.

    Distillation results such as DeepSeek-R1’s small students demonstrate the value of this route, but those students started from pretrained models. They were not trained from random initialization using only the distillation set. [DeepSeek-R1 repository](https://github.com/deepseek-ai/DeepSeek-R1)

    **Second: token-distribution distillation, when available.**

    With aligned tokenization, a starting objective is:

    \[
    \mathcal L
    =\alpha\,\mathcal L_{\mathrm{CE}}
    +(1-\alpha)\tau^2
    D_{\mathrm{KL}}\!\left(p_{\mathrm{teacher}}^\tau
    \parallel p_{\mathrm{student}}^\tau\right).
    \]

    Pilot temperatures of 1 and 2 and a small sweep of the CE/KD mixture. These are tuning candidates.

    Direct KL requires compatible token identities and sequence alignment. If you only receive text from a frontier API, use response distillation. Limited top-k log probabilities are not equivalent to a complete teacher distribution.

    Do not store full-vocabulary logits for an enormous corpus. Use online scoring, carefully designed sparse targets with tail handling, or other memory-conscious methods.

    **Third: on-policy distillation.**

    Let the student generate responses and obtain teacher feedback on the prefixes the student actually visits. This addresses errors that are poorly represented in teacher-generated demonstrations.

    Qwen3 describes an off-policy stage followed by on-policy logit alignment. Its published comparison favoring distillation over direct RL was conducted at 8B, so its exact speedup should not be assumed for your model. [Qwen3 report](https://arxiv.org/html/2505.09388v1)

    Use an established GKD implementation or objective rather than improvising a sequence-level KL estimator. [GKD documentation](https://huggingface.co/docs/trl/gkd_trainer)

    **Finally: selective preference training or RL.**

    Add DPO for demonstrated preference problems, or verifiable-reward RL for tasks such as tested code and mathematics. Make this conditional on an evaluation showing a remaining gap.

    Teach both direct-answer and bounded reasoning behavior. Long traces consume training capacity and deployment latency; measure whether they improve correctness at the allowed output budget.

12. Run a small, disciplined experiment program.

    I would prioritize experiments in this order:

    | Priority | Experiment | What it resolves |
    |---:|---|---|
    | 1 | Data quality and mixture | Whether better inputs provide the largest gain |
    | 2 | Response distillation quality and curriculum | Whether the student learns useful teacher behavior |
    | 3 | Four versus eight KV heads | Quality versus cache cost |
    | 4 | Depth/width candidates above | Quality versus sequential latency |
    | 5 | Attention output gating | Whether the added computation pays off |
    | 6 | Full attention versus one hybrid | Whether long-context efficiency justifies complexity |
    | 7 | Vocabulary/FFN allocation | Token efficiency versus model capacity |
    | 8 | Quantization sensitivity | Whether improvements survive deployment |
    | 9 | Differential attention / NoPE / MTP | Additional research after stronger baselines |

    Compare architectures at approximately equal parameter budgets and separately at equal training compute. Keep data order, evaluation settings, and training recipes consistent where possible.

    Small proxies can eliminate bad ideas cheaply, but finalists need target-size tests. Do not treat a result at 100M parameters as proof of the ordering at 1.7B.

    Your evaluation suite should include:

    - Held-out language modeling by domain.
    - Instruction following, such as IFEval.
    - General knowledge and reasoning, such as MMLU-Pro.
    - Mathematics with independently checked answers.
    - Code evaluated by execution.
    - JSON/schema adherence and tool-call correctness.
    - Long-context retrieval, copying, and multi-document tasks.
    - A private set of realistic target-user requests.

    Compare BF16 and Q4 outputs. Report prefill speed, generation speed, peak VRAM, and task accuracy at fixed reasoning budgets.

    Use bits per byte or comparable text-level measures when comparing different tokenizers; raw perplexity values are not directly comparable across them.

13. Treat GGUF deployment as part of model development.

    Your deployment pipeline should be:

    ```text
    PyTorch checkpoint
        → HF-compatible weights, config, tokenizer, chat template
        → BF16/F16 GGUF
        → numerical and task checks
        → calibration importance matrix
        → Q4_K_M GGUF
        → llama-server evaluation
    ```

    GGUF conversion does not make arbitrary PyTorch operations executable. The destination architecture must implement the same computation. Changes such as differential attention, extra gates, or a custom hybrid require corresponding runtime support.

    For a supported model, representative Linux/WSL commands are:

    ```bash
    python convert_hf_to_gguf.py ./student-hf \
      --outfile student-bf16.gguf --outtype bf16

    ./build/bin/llama-imatrix \
      -m student-bf16.gguf \
      -f calibration.txt \
      -o imatrix.gguf \
      -ngl 99

    ./build/bin/llama-quantize \
      --imatrix imatrix.gguf \
      student-bf16.gguf student-Q4_K_M.gguf Q4_K_M

    ./build/bin/llama-server \
      -m student-Q4_K_M.gguf \
      -ngl 99 \
      -c 8192 \
      -np 1 \
      --flash-attn on \
      --host 127.0.0.1 \
      --port 8080
    ```

    Pin and validate the llama.cpp revision used for release. These command forms follow its current [conversion tool](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py), [importance-matrix documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix), and [server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

    Use representative calibration text covering your intended tasks, with evaluation data kept separate. Compare `Q4_K_M` against BF16 and a higher-precision quantization to diagnose losses.

    If Q4 loses too much quality, investigate sensitive tensors and calibration first. Quantization-aware training is a later option, but its simulated quantization must be relevant to the deployed format: training with NF4 or another int4 scheme does not automatically optimize GGUF `Q4_K_M`.

14. Make the first milestone a complete, measurable pipeline.

    I would organize the initial work as follows; these are engineering milestones, not promises about pretraining duration.

    | Milestone | Concrete deliverable |
    |---|---|
    | First week | Existing small models benchmarked on the RTX; fixed task suite; baseline llama-server |
    | Second week | PyTorch reference model, cached decoding, checkpointing, and working GGUF export |
    | Third week | Data pipeline and a reviewed teacher-distillation pilot |
    | Fourth week | First trained/distilled checkpoint, with BF16-versus-Q4 quality and speed report |
    | Following stage | Evidence-based choice between continued adaptation, custom pretraining, and the hybrid challenger |

    **For a custom network, I would implement the 28-layer, 2,048-wide, 1.66B GQA model above first. For a competitive model on a limited training budget, I would begin by distilling into an existing pretrained student and preserve its original architecture.** In either case, the first substantial training run should start only after the model already converts correctly and runs through your intended llama.cpp server pipeline.

---

## Revised plan for one RTX and a specialized English task
MSI GeForce RTX 4070 Ti SUPER 16G GAMING X SLIM
This section incorporates the clarification that the same RTX must handle both student training and inference. It supersedes the production-route recommendation and training scale assumptions in the original plan where they conflict.

### 1. Use a pretrained student as the production starting point

The practical target is a strong specialist that retains useful general capabilities. Training a new 1–2B foundation model from random initialization to broad small-model SOTA on one consumer GPU is not a realistic default project budget.

My provisional first student is **Qwen3-1.7B with its released pretrained and post-trained weights**, preserving its architecture, tokenizer, and chat template. This is a practical implementation baseline, not a declaration that it is the best model for the still-unspecified domain.

Benchmark it against LFM2.5-1.2B and a small Qwen3.5 candidate on the actual task before choosing the final student. For multimodal checkpoints, count any deployed vision or auxiliary modules against the size budget.

If writing the network yourself is part of the goal, you can implement the existing architecture in PyTorch and load the released weights after numerical parity tests. That still gives you ownership of the implementation while retaining the expensive pretraining.

The released Qwen3-1.7B differs from the proposed custom model:

| Component | Released Qwen3-1.7B |
|---|---:|
| Layers | 28 |
| Hidden width | 2,048 |
| FFN width | 6,144 |
| Query heads | 16 |
| KV heads | **8** |
| Head dimension | 128 |
| Vocabulary size | 151,936 |
| Embeddings | Tied |
| Parameter count calculated from these dimensions | Approximately 1.721B |

These dimensions are from the [official configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json). Do not change the KV-head count or vocabulary merely to match the earlier custom design.

Its FP16 cache is also larger than the custom four-KV-head model:

| Cached tokens | FP16 KV cache per sequence |
|---:|---:|
| 4,096 | 448 MiB |
| 8,192 | 896 MiB |
| 32,768 | 3.5 GiB |

These are calculated cache payloads, excluding weights and runtime overhead.

### 2. Choose LoRA or QLoRA after checking VRAM

LoRA trains additional low-rank weight updates while freezing the original weights. QLoRA combines this with a quantized frozen base to reduce memory use. The trainable adapters still use higher-precision computation. [PEFT LoRA documentation](https://huggingface.co/docs/peft/developer_guides/lora), [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)

Use the following as initial planning choices, not guaranteed memory fits:

| RTX VRAM | Initial approach |
|---|---|
| 4–6 GB | QLoRA; short sequences and microbatch 1; consider a smaller student if necessary |
| 8 GB | QLoRA as the starting point; begin around 1K–2K sequence length |
| 12–16 GB | Test BF16 LoRA first; use QLoRA if activations or sequence length require it |
| 24 GB or more | BF16 LoRA first; consider full fine-tuning only after measuring its benefit and memory requirements |

BF16 LoRA may be preferable when it fits because it avoids training against quantized base weights. QLoRA is useful when it buys needed memory headroom; it is not automatically the fastest choice.

Do not use the inference GGUF file as the training checkpoint. Train through PyTorch-compatible weights and adapters, then create the GGUF release.

### 3. Use a small, explicit adaptation configuration

A provisional Qwen3 LoRA pilot:

```yaml
student: Qwen/Qwen3-1.7B
method: lora  # use qlora when required by measured memory
rank: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
learning_rate: 0.0001
lr_candidates:
  - 0.00005
  - 0.0001
  - 0.0002
microbatch_size: 1
initial_sequence_length: 2048
gradient_checkpointing: true
assistant_response_loss_only: true
initial_epochs: 1
max_grad_norm: 1.0
```

These are experimental starting values. Set gradient accumulation from a target effective token batch after measuring throughput. Inspect validation results before additional epochs.

At rank 32, adapting the seven listed matrices across 28 layers adds approximately **34.9M trainable parameters**. The original embeddings and LM head remain frozen in this first experiment.

For QLoRA, start with NF4, double quantization, and BF16 compute if supported by the selected stack. This is the configuration pattern documented by [PEFT](https://huggingface.co/docs/peft/developer_guides/quantization).

Inspect actual loss masks. TRL supports assistant-only loss, but the chat template must expose the assistant generation spans correctly. [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer)

### 4. Make specialization the main data objective

Start with a small, high-quality supervised dataset. A useful first experiment is 5K–20K reviewed task examples, followed by expansion only when the learning curves and held-out tasks justify it.

A proposed sampling mixture, measured by supervised response tokens:

| Category | Initial share |
|---|---:|
| Specialized task and domain | 70% |
| General assistant and instruction following | 15% |
| Coding | 10% |
| General reasoning and mathematics | 5% |

These proportions are a starting hypothesis. If the domain is coding or mathematical, redefine the categories so they do not overlap.

Include difficult cases, missing information, malformed inputs, realistic user phrasing, and examples requiring a concise answer. Reserve a source-separated evaluation set before generating variants of training examples.

If the task depends on changing facts or a large document collection, evaluate retrieval alongside adaptation. Fine-tuning can teach the model how to use evidence; it does not guarantee reliable recall of an entire knowledge base.

Continued pretraining on raw domain documents is optional. Add it only if baseline evaluation shows a material domain-language or knowledge gap that supervised examples do not resolve efficiently.

### 5. Generate teacher data separately from student training

Student training and serving can stay on the single RTX. A frontier teacher that does not fit must be accessed through an API, external hardware, or an existing teacher-generated dataset.

If everything must remain local, use the strongest suitable teacher that fits the machine, generate and save its responses, then unload it before student training. That teacher may be considerably smaller than a frontier system.

For the first distillation run:

1. Collect realistic prompts and source material from the specialized task.
2. Generate teacher responses grounded in that material.
3. Verify code with tests, structured outputs with schemas, and factual answers against source evidence.
4. Remove duplicates, unsupported claims, and unnecessarily long explanations.
5. Train the student with supervised response loss.
6. Evaluate the resulting errors and request targeted additional examples.

Text-output distillation is the initial recommendation because it avoids simultaneously hosting a large teacher and student and works across tokenizers.

Direct logit distillation and on-policy methods remain later experiments. API text alone does not supply the distributions required for ordinary token-level KL.

### 6. Benchmark training time before scaling the dataset

Run a short throughput test with the intended model, sequence lengths, loss implementation, precision, and checkpointing settings.

Estimate:

```text
training time ≈ total processed tokens / measured training tokens per second
```

Count all processed tokens, including prompt tokens that are masked out of the loss. Include repetition across epochs.

For illustration only, 50M processed tokens would take approximately 27.8 hours at 500 tokens/second or 9.3 hours at 1,500 tokens/second, before evaluation and checkpoint overhead. These rates are hypothetical, not predictions for the unspecified RTX.

Use this measurement to choose an affordable dataset size. Avoid promising a training duration from the GPU family or model size alone.

### 7. Preserve general capability with explicit regression tests

Evaluate the unmodified student first. Keep the same prompts, decoding settings, and output budgets when evaluating adaptations.

The release criteria should include:

- Measurable improvement on the specialized task.
- Acceptable general instruction-following retention.
- Acceptable coding and reasoning retention.
- Correct output formatting and tool schemas when relevant.
- Acceptable performance after GGUF Q4 quantization.
- Acceptable latency and memory on the actual RTX.

Choose the checkpoint using task metrics and regression results, not training loss alone. Increase general replay or reduce training intensity if specialization causes unacceptable regressions.

### 8. Merge and quantize the selected student

For ordinary LoRA, merge the trained updates into a higher-precision copy of the original base, then export.

For QLoRA, a common deployment route is to reload the original BF16/FP16 base, merge the adapter, and evaluate that merged model before GGUF quantization. The merged model can differ from the quantized-base training model, so this comparison is necessary.

```text
Original pretrained weights
    + selected adapter
    → merged BF16/FP16 student
    → quality and regression evaluation
    → GGUF conversion
    → representative calibration
    → Q4_K_M
    → llama-server evaluation on the RTX
```

NF4 training and Q4_K_M deployment are different quantization schemes. Passing the PyTorch evaluation does not replace evaluating the final GGUF.

Preserve the original tokenizer, special-token IDs, chat template, and stopping behavior throughout export.

### 9. Keep custom architecture research as a separate experiment

If architectural research is itself a goal, use small models and tightly budgeted experiments to study depth, GQA, gating, and hybrid attention.

Loading pretrained weights into a faithful PyTorch reimplementation is the most direct way to combine implementation learning with a useful model. A new architecture trained from scratch is a separate research investment whose quality should be compared against the adapted pretrained student.

Before selecting a final batch size, context length, data budget, or student checkpoint, obtain:

- Exact GPU model and available VRAM.
- A concrete example of the specialized input and desired output.
- Existing dataset size, format, and quality.
- Maximum acceptable training duration.
- Availability of teacher APIs or teacher-generated datasets.

No models have been trained or benchmarked as part of this planning document. Configuration values are proposed starting points, and all deployment/training estimates require measurement on the selected hardware.

---

## Current hardware and dataset plan

This update applies to a single RTX 30/40/50-series card with 16–24 GB VRAM,MSI GeForce RTX 4070 Ti SUPER 16G GAMING X SLIM, the supplied MiniFrontier dataset, and a run lasting less than 18 days.

The recommendation is to adapt a pretrained approximately 1.7B student, use the existing corpus selectively for general replay or a measured continued-pretraining experiment, and spend most of the useful training budget on the specialized task and verified teacher responses.

The initial architectural choice is therefore the released student architecture, including its tokenizer and eight KV heads. A new 1–2B model trained from scratch on this corpus is technically a possible research experiment, but broad competitiveness with heavily pretrained small models is not a realistic expectation for that experiment.

### 1. What was verified about the supplied dataset

Dataset: [igalk474/MiniFrontier-150M-Modern-3B-token-mixture](https://huggingface.co/datasets/igalk474/MiniFrontier-150M-Modern-3B-token-mixture).

Inspected repository revision: `39272b7346b9cea66af8c03ea1fd18b6a6aca3b3`.

Inspection covered the dataset card, repository metadata and file listing, Dataset Viewer size metadata, and the first two training rows of the GitHub-code and Cosmopedia configurations. The complete Parquet corpus was not downloaded, tokenized, or audited for quality or contamination.

| Property | Observed value |
|---|---:|
| Configurations | 5 |
| Documents across train and validation | 3,401,006 |
| Training documents | 3,366,983 |
| Validation documents | 34,023 |
| Parquet files | 10, one train and one validation file per configuration |
| Total Parquet bytes | 6,715,118,865, approximately 6.72 GB |
| Exported columns | 12 |
| Published “real tokens” total | 4,961,118,208 |

The row and file counts were verified through the [Dataset Viewer size endpoint](https://datasets-server.huggingface.co/size?dataset=igalk474%2FMiniFrontier-150M-Modern-3B-token-mixture) and [repository metadata](https://huggingface.co/api/datasets/igalk474/MiniFrontier-150M-Modern-3B-token-mixture). The token counts below are publisher-reported figures from the [pinned dataset card](https://huggingface.co/datasets/igalk474/MiniFrontier-150M-Modern-3B-token-mixture/blob/39272b7346b9cea66af8c03ea1fd18b6a6aca3b3/README.md), not a new count under the proposed student's tokenizer.

| Source | Published mixture weight | Published source tokens | Train documents | Validation documents |
|---|---:|---:|---:|---:|
| dclm-edu | 35% | 1,651,369,984 | 1,121,762 | 11,391 |
| fineweb-edu | 25% | 901,208,064 | 788,019 | 7,921 |
| github-code | 20% | 1,393,422,336 | 764,296 | 7,796 |
| finemath | 15% | 933,122,048 | 590,991 | 5,863 |
| cosmopedia-v2 | 5% | 81,995,776 | 101,915 | 1,052 |
| **Total** | **100%** | **4,961,118,208** | **3,366,983** | **34,023** |

The published source-token inventory totals about 4.96B, despite “3B” in the name. These are different concepts: available corpus inventory and the token budget selected for a particular training run.

The card describes the linked original project as using a 16,384-token BPE vocabulary. Its token figures must not be reused as exact Qwen3 token counts. Retokenize or estimate from a stratified sample using the actual selected tokenizer, then count the final tokenized training shards.

The “150M” label describes the original project target. It does not restrict these text documents to training 150M-parameter models.

### 2. Important ingestion and evaluation details

The exported schema is:

```text
text
source
revision
license
language
record_id
content_hash
path
source_type
split
parent_content_hash
transform
```

This is a document-oriented corpus, not an explicitly structured chat dataset. It is appropriate input for next-token continued pretraining. It does not by itself provide the specialist instruction/response supervision required for the stated assistant goal.

Load each configuration explicitly. For example:

```python
from datasets import load_dataset

repo_id = "igalk474/MiniFrontier-150M-Modern-3B-token-mixture"
revision = "39272b7346b9cea66af8c03ea1fd18b6a6aca3b3"

code_train = load_dataset(
    repo_id,
    "github-code",
    split="train",
    revision=revision,
    streaming=True,
)
```

The card's generic example omitting the configuration should not be copied as the complete mixture loader. There are five configurations, and loading one does not load or mix the other four.

The published mixture weights also do not automatically apply when loading the data. Concatenating all documents produces the inventory distribution. Sampling documents with probabilities 35/25/20/15/5 produces approximately those document proportions, not necessarily those token proportions.

To implement token-weighted mixing:

1. Tokenize and pack each source into fixed-length blocks using consistent boundary rules.
2. Sample those blocks with the intended source probabilities.
3. Log actual consumed tokens, source repetitions, and validation exposure.

Alternatively, use a token-aware scheduler that compensates for differing document lengths.

At a hypothetical 3B-token run with a 5% Cosmopedia allocation, that source would supply 150M training tokens. Relative to the card's approximately 82M-token inventory, this is already about 1.83 passes, even before excluding validation or accounting for tokenizer differences. Repetition can be intentional, but it should be measured. Do not assume a 3B-token run samples every source without replacement.

In the inspected sample rows:

- The first two GitHub-code records were Markdown repository guidance files from Gradle, with `language="Markdown"` and `source_type="code"`. The source therefore includes documentation as well as executable code. This observation does not establish the proportion of documentation across the full source.
- The Cosmopedia examples were synthetic educational prose; they were not stored as role-separated conversations.
- The row-level `split` field was null even though the records came from the actual `train` split. Use the dataset's real split selection and file organization rather than trusting the optional row field.

Before training, check exact and near-duplicate overlap between splits. For code, add a repository-disjoint evaluation set using source repository identity. For text, group related documents and any transformed variants where their metadata permits it. Existing validation files alone do not prove independence.

Preserve source, revision, and per-example license metadata in the data manifest. The card explicitly uses mixed per-example licensing.

### 3. Choose between two adaptation routes

**Default route: preserve a competent assistant and specialize it.**

Start with `Qwen/Qwen3-1.7B` as the instruction-tuned student baseline. Build specialist demonstrations and teacher responses. Use the provided corpus for a carefully selected general replay objective or a short continued-pretraining pilot only if it demonstrates an advantage.

Avoid spending the entire allowance on broad raw text by default. A pretrained student may already be competent on similar distributions, and more broad-language training does not necessarily improve the specialized task.

**Conditional route: domain adaptation before instruction tuning.**

Use a base checkpoint, such as the corresponding Qwen3 base model, when the task involves a large new domain-language distribution and there is enough specialist instruction data to establish assistant behavior afterward.

Run a small continued-pretraining experiment, followed by the same supervised distillation evaluation used for the default route. Promote this route only if its final task gains justify the extra compute.

The currently supplied mixture is broad education, web text, code, and mathematics. The specialized domain remains unspecified, so there is not yet evidence that a long domain-pretraining stage is required.

A useful first comparison is:

| Arm | Training | Decision |
|---|---|---|
| A | Released assistant → specialist SFT/distillation | Establish the lowest-cost useful improvement |
| B | Short raw-text adaptation pilot → the same specialist SFT/distillation | Continue only if final task quality improves enough to pay for the extra stage |

Do not run a large architecture sweep within this time budget.

### 4. Separate the two training objectives

For raw documents:

```text
objective: next-token prediction over document text
loss: all valid next-token targets, excluding padding and inappropriate cross-document targets
input: the text column
```

For teacher demonstrations and conversations:

```text
objective: supervised response distillation
loss: assistant response tokens
input: correctly formatted messages or prompt/completion pairs
```

Do not apply the earlier assistant-only loss configuration to unstructured raw documents; those records do not contain assistant spans.

If mixing raw replay with supervised examples, use explicit task-specific loss masks and sample weights. Track both processed tokens and supervised loss tokens.

For the main supervised stage, keep the initial 70% specialist / 15% general assistant / 10% coding / 5% reasoning response-token mixture as a provisional hypothesis. The raw-text source mixture is a different distribution used for a different objective.

A first instruction-data target is 20K–100K well-verified examples, expandable if evaluation shows useful gains. Decide by total lengths and quality rather than row count alone. Reserve independent task examples before generating paraphrases, variants, or teacher answers.

### 5. Initial settings for the specified VRAM range

| Setting | 16 GB starting point | 24 GB starting point |
|---|---|---|
| Student | Approximately 1.7B pretrained | Approximately 1.7B pretrained |
| Main method | Benchmark BF16 LoRA; use QLoRA if memory requires it | BF16 LoRA |
| Initial context | 2,048 | 2,048 for throughput; test 4,096 if task needs it |
| Initial microbatch | 1, then increase after measuring | 1, then increase after measuring |
| LoRA rank | 32 | 32; compare 64 only if validation motivates it |
| Adapted modules | Q/K/V/O and three FFN projections | Same |
| Activation checkpointing | Enable initially | Enable initially; test its throughput cost |
| Attention backend | Verified efficient backend | Verified efficient backend |
| Output loss | Fused or chunked when needed | Fused or chunked when needed |
| Optimizer | AdamW over adapter parameters | AdamW over adapter parameters |

The table does not promise a memory fit for every kernel, batch, or sequence distribution. Inspect actual peak allocated and reserved memory.

For supervised LoRA adaptation, start with the earlier `1e-4` learning rate and a small `5e-5 / 1e-4 / 2e-4` pilot sweep if affordable. For continued pretraining of pretrained weights or adapters, start more conservatively and validate separately; do not reuse the scratch-pretraining learning rate automatically.

Use ordinary adapter AdamW first; the optimizer state for tens of millions of adapter parameters is much smaller than for all 1.7B weights. Add optimizer quantization only if measured memory savings justify it.

RTX 30/40/50-series cards with equal VRAM can have very different throughput. For a 50-series card, verify that the pinned PyTorch/CUDA and optional attention/loss kernels support that exact GPU architecture. Do not assume a package working on Ampere will run unchanged on every newer card.

### 6. Convert the time limit into a measured token budget

Use less than 18 days as a hard limit. Plan completion within **17 calendar days**, including evaluation and export, to preserve margin.

Allocate approximately 12 calendar days to adaptation experiments and training. Use an initial 85% training-duty factor for validation, checkpointing, transitions, and interruptions. This factor is a planning assumption to replace with measurements.

```text
usable training tokens
    = measured training tokens/second
    × available training days
    × 86,400
    × training-duty fraction
```

For 12 days and an 85% duty fraction:

| Measured training speed | Approximate usable processed tokens |
|---:|---:|
| 500 tokens/second | 441M |
| 1,000 tokens/second | 881M |
| 2,000 tokens/second | 1.76B |
| 4,000 tokens/second | 3.53B |

These speeds are examples, not measured predictions for any RTX model.

Processing 3B tokens in 18 uninterrupted days requires approximately 1,929 training tokens/second. Processing them inside the more conservative 12-day training allocation at 85% duty requires approximately **3,404 tokens/second**.

Processing the card's entire 4.961B-token inventory under that conservative allocation would require approximately **5,629 tokens/second**, if those token counts applied to the chosen tokenizer. They have not yet been verified under that tokenizer.

Those requirements concern processed tokens; they do not imply that using the entire corpus is the best adaptation strategy.

Measure throughput separately for raw-text training, masked response SFT, and any generation-based method. When taking a long-run throughput measurement that already includes evaluation/checkpoint downtime, do not apply the duty factor a second time.

Run a warmed-up 30–60 minute pilot with the actual training stack, context lengths, and loss masks. Record processed tokens/second, peak VRAM, step-time variation, GPU utilization, and data-loader stalls. Use that result to set maximum steps and a wall-clock deadline.

### 7. A concrete schedule that finishes before day 18

| Day | Work | Required output |
|---|---|---|
| 1 | Pin data/model revisions; audit source samples and splits; establish evaluations | Data manifest and unmodified-student scores |
| 2 | Profile LoRA/QLoRA with the actual GPU; verify save/resume and GGUF export | Measured throughput, memory, and a feasible token budget |
| 3–4 | Small specialist pilot; optional short raw-text adaptation comparison | Select one route based on held-out task results |
| 5–12 | Main specialist SFT/response distillation; general capability replay | Regular task/regression scores and resumable checkpoints |
| 13–14 | One targeted failure-correction pass, only if justified | Selected final checkpoint; avoid starting unrelated experiments |
| 15 | Merge adapter; verify higher-precision model; calibrate and quantize | BF16/FP16 reference and Q4_K_M GGUF |
| 16 | Test final GGUF through llama-server on the GPU | Quality, latency, context, and memory report |
| 17 | Contingency, final checks, and documentation | Final artifacts; stop before the 18-day limit |

If teacher data does not already exist, prepare it early enough to fit this schedule. Teacher generation can dominate the total effort and is not free merely because student fine-tuning fits on the RTX.

If no raw-text pilot improves the task, move its remaining allocation to specialist data and supervised training. If validation saturates early, stop training and use the time for evaluation and dataset correction.

Configure maximum steps from the measured token budget, save on a reasonable time interval, and preserve enough time for a final checkpoint and export. A nominal epoch count is insufficient to enforce the deadline across different hardware.

### 8. Release criteria and what remains unknown

The final model must beat the unmodified student on an independent specialized-task evaluation while retaining acceptable general assistant, coding, and reasoning behavior. Evaluate the merged higher-precision model and final Q4_K_M artifact separately.

A lower perplexity on this general corpus alone is not sufficient evidence of success.

The next missing specification is the **specialized domain/task and a concrete input/output example**. That determines which documents to select, which demonstrations to generate, whether retrieval is useful, and what metric should control checkpoint selection.

The exact GPU still determines the training budget: the 16–24 GB range establishes plausible memory options but does not establish throughput.

This update is a reviewed plan and metadata inspection. No training, full-corpus download, tokenizer recount, quality audit of the full dataset, or GPU benchmark has been performed.
