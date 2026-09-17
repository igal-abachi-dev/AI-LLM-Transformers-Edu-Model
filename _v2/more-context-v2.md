For a dense language model **up to 1B parameters**, **Full Attention (specifically paired with Grouped-Query Attention / GQA)** is currently the superior architectural choice for general-purpose use.

While **Hybrid architectures** (combining attention with SSMs like Mamba-2 or linear RNNs like RWKV/Griffin) are theoretically elegant, the practical realities of small-scale models heavily favor standard attention.

---

### 1. The KV-Cache Advantage of Hybrids is Negligible at $\le$1B

The primary reason to use hybrid architectures in larger models (7B–70B+) is to cut down the massive KV-cache memory footprint and the quadratic prefill cost at long contexts.

At the $\le$1B scale, the KV cache is already tiny:
* Consider a typical **1B model** (e.g., $\sim 16$ layers, hidden dimension $d = 2048$, using GQA with 8 KV heads of dimension 64):
  $$\text{KV Cache per Token} = 2 \times 16 \times (8 \times 64) \times 2\text{ bytes (FP16)} \approx 32\text{ KB}$$
* At **4,096 tokens**, the KV cache is only **$\sim 130$ MB**.
* Even at **8,192 tokens**, it is only **$\sim 260$ MB**.

Because models under 1B are rarely effective at reasoning over massive context windows (e.g., 64k–128k+ tokens) due to parameter capacity limits, the KV cache is rarely the system bottleneck.

---

### 2. On-Device Deployment and Hardware Friction

Sub-1B models are almost always deployed on **edge devices**: smartphones, laptops, IoT boards, and microcontrollers.

* **NPU / Mobile Acceleration:** Edge runtimes (Apple Core ML/ANE, Qualcomm Hexagon, ExecuTorch, TFLite/LiteRT, ONNX Runtime) are hyper-optimized for standard Matrix Multiplications (GEMM) and Softmax operations.
* **The SSM / Linear-RNN Kernel Problem:** SSMs like Mamba rely on associative parallel scans. While fast on modern NVIDIA GPUs using custom Triton/CUDA kernels, they often lack mature, fused kernel implementations on mobile NPUs or CPU vector engines, frequently falling back to slow un-fused CPU execution.
* Full attention with GQA or FlashAttention derivatives works out-of-the-box everywhere, from embedded microcontrollers to flagship GPUs.

---

### 3. Representational Capacity & Associative Recall

Parameters are scarce in sub-1B models. Every layer must pull its weight:

* **In-Context Retrieval:** Standard softmax attention acts as an explicit key-value lookup table, enabling strong associative recall, precise copy-pasting, multi-hop reasoning, and structured output formatting (JSON/code).
* **Fixed-State Compression:** Linear RNNs and SSMs compress history into a fixed-size hidden state. In sub-1B models, this compression creates a noticeable bottleneck in complex in-context learning tasks unless paired with frequent full attention layers.

---

### 4. What the State of the Art Does

Leading AI labs have almost universally opted for **Full Attention with GQA** for sub-2B models:

| Model | Size | Architecture |
| :--- | :--- | :--- |
| **Llama 3.2** | 1B | Full Attention + GQA |
| **Qwen 2.5** | 0.5B, 1.5B | Full Attention + GQA |
| **SmolLM2** | 135M, 360M, 1.7B | Full Attention + GQA |
| **MobileLLM** | 125M, 350M | Deep & Thin Full Attention + Weight Sharing |
| **Gemma 2** | 2B | Full Attention (alternating sliding window & global) |

Experiments with small hybrid/recurrent models (like Google’s *RecurrentGemma 2B* or Zyphra’s *Zamba2 1.2B*) demonstrated competitive quality, but they have seen limited community adoption primarily due to deployment stack friction and minimal latency gains at typical edge context lengths (1k–4k).

---

### When *Should* You Choose a Hybrid Architecture at $\le$1B?

A hybrid architecture (e.g., 1 Full Attention layer every 4 to 6 Mamba-2 layers) makes sense only under specific constraints:

1. **Continuous Streaming / Always-On Edge Inference:** If your 1B model is processing audio, sensor data, or continuous robotics control signals where generation runs indefinitely and latency must remain strictly $O(1)$ per step without periodic cache flushes.
2. **Extreme Context on Very Low RAM:** If you genuinely need a 1B model to ingest 32k–64k tokens on a device with strict memory limits (e.g., $<512\text{ MB}$ total system RAM).
3. **Controlled Server-Side Deployment:** If the model runs exclusively on NVIDIA hardware with custom Triton/TensorRT-LLM kernels, where high-concurrency batch throughput can fully exploit Mamba-2's linear prefill.

---

### Recommended Recipe for a $\le$1B Model

If you are training a 1B dense model today, the most battle-tested architecture is:

* **Transformer Structure:** Deep and thin (e.g., 20–24 layers, smaller hidden dimension $\approx 1536–2048$) rather than wide and shallow (Meta’s MobileLLM paper showed depth matters more than width at this scale).
* **Attention Mechanism:** Full Attention with **GQA** (e.g., 16 query heads, 2 to 4 KV heads).
* **Context Strategy:** Consider **alternating local sliding window attention** (e.g., window of 1024 or 2048) with **global attention** (every 2nd or 3rd layer), similar to Gemma 2. This delivers near-linear prefill benefits while maintaining full hardware compatibility.
* **Core Components:** RMSNorm, SwiGLU, and RoPE.





### 1. The `head_dim` Question: Can Savings Buy a Larger Head Dimension?

**Yes, mathematically—but there is a strict architectural trade-off.**

If you reduce KV heads (via GQA) or reduce attention layers (via a hybrid setup), you free up memory bandwidth and FLOPs. You *could* spend that budget increasing the head dimension ($d_k$) from the standard $64$ or $128$ up to $256$. Gemma 2 (2B) famously did this, pairing $d_{\text{model}} = 2304$ with a massive $d_k = 256$.

However, at the **$\le$1B scale**, increasing $d_k$ creates three severe problems:

1. **Head Count Starvation (Loss of Multi-Relation Tracking):**
   In standard attention, $d_{\text{model}} = H \times d_k$. If your 1B model has a hidden dimension of $d_{\text{model}} \approx 1536$ to $2048$:
   * With $d_k = 64 \implies 24\text{ to }32$ query heads.
   * With $d_k = 128 \implies 12\text{ to }16$ query heads.
   * With $d_k = 256 \implies \text{only } 6\text{ to }8$ query heads.
   
   Small models need distinct heads to act as independent pattern detectors (tracking subject-verb agreement, syntactic trees, quoting, positional offsets, formatting). Dropping to only 6–8 heads severely degrades the model’s ability to track multiple relational dependencies simultaneously.

2. **Attention Logit Instability (Entropy Collapse):**
   The dot-product variance scales with $\sqrt{d_k}$. At $d_k = 256$, dot-product magnitudes explode. Without specialized architectural fixes (like Gemma 2's $\tanh$ logit soft-capping), softmax outputs collapse into one-hot distributions, degrading representation quality.

3. **SRAM Register Spilling on Edge Hardware:**
   On mobile NPUs and edge GPUs, fast on-chip SRAM (tile cache) is extremely limited. Standard FlashAttention and fused kernels are compiled for $d_k = 64$ or $128$. Setting $d_k = 256$ causes local register spilling, often rendering the kernel **slower** despite having fewer heads.

---

### 2. Modern & Cutting-Edge Alternatives for $\le$1B Dense Models

Standard GQA (e.g., 16 query heads, 2–4 KV heads) is the 2023 industry baseline. To move beyond this for a dense sub-1B model, several architectural alternatives offer higher quality and hardware efficiency:

---

#### A. Differential Attention (Diff-Transformer)
* **What it is:** Instead of a single softmax attention map, compute two separate attention maps per head and subtract them:
  $$\text{DiffAttn}(X) = \left( \text{Softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \, \text{Softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right) \right) V$$
* **Why it shines at $\le$1B:** Small models suffer heavily from **attention dilution**—they assign residual probability mass to irrelevant context (noise), causing hallucinations and catastrophic needle-in-a-haystack failures.
* **The Impact:** Differential attention cancels out common-mode noise like a noise-canceling headphone, acting as a dynamic sparse router. At sub-1B scales, it significantly narrows the retrieval gap between small models and 7B+ models without inflating parameter count.

---

#### B. Cross-Layer KV Sharing (The MobileLLM Paradigm)
* **What it is:** Instead of every layer computing and caching its own K and V projections, **adjacent layers share the exact same KV cache**.
  * Layer $2i$ computes $K, V$ and caches them.
  * Layer $2i+1$ computes its own queries $Q$, but attends directly over the $K, V$ computed by Layer $2i$.
* **Why it beats vanilla GQA:**
  * If you push GQA to the extreme (Multi-Query Attention / 1 KV head), model quality drops sharply.
  * Cross-layer sharing cuts KV cache memory in **half (50%)** while allowing you to keep a healthier number of KV heads per layer (e.g., 4 or 8 heads instead of 1). Meta demonstrated in *MobileLLM* that this delivers much higher perplexity per byte than aggressive head pruning.

---

#### C. Multi-Head Latent Attention (MLA / Matrix Absorption)
* **What it is:** DeepSeek’s core innovation, adapted for smaller dense models. Instead of storing multi-head Keys and Values in memory, Keys and Values are jointly compressed into a low-rank latent vector $c_t^{KV}$ of dimension $d_c \ll (H \times d_k)$.
* **Why it's cutting-edge for 1B:**
  * It decouples the KV cache size from both the number of heads and the head dimension.
  * You can give your 1B model **many heads** and a **large head dimension ($d_k = 128$)**, while your KV cache footprint during generation behaves as if you had only a fraction of a single head.
  * At generation time, the projection matrices can be mathematically absorbed into the query projections, meaning zero runtime compute overhead for decompression.

---

#### D. Weight-Shared Attention over an SSM Backbone (The Zamba-2 Paradigm)
* **What it is:** If you *do* go hybrid, alternating unique attention layers every $N$ blocks is an inefficient use of parameters. Zamba-2 stacks high-throughput Mamba-2 blocks, but routes them through a **single, shared, global Attention block** reused across depth.
* **Why it’s better than standard Hybrids at 1B:**
  * A 1B model cannot afford parameter waste. Dedicating unique parameters to 6 or 8 separate attention blocks consumes 20–30% of your parameter budget.
  * Reusing a high-capacity global attention layer gives you full in-context associative recall while reinvesting those hundreds of millions of saved parameters into wider Feed-Forward Networks (SwiGLU) or deeper state spaces.

---

#### E. Local-Global Interleaved Attention + Logit Soft-Capping (Gemma 2 Style)
* **What it is:**
  1. Alternates **1 sliding-window local attention** layer (e.g., window size $W = 1024$) with **1 full global attention** layer.
  2. Implements **logit soft-capping**:
     $$\text{AttnLogits} = C \cdot \tanh\left(\frac{Q K^T}{\sqrt{d} \cdot C}\right) \quad (\text{typically } C = 50.0)$$
* **Why it works:**
  * Halves the average context compute budget by using local windows.
  * Soft-capping prevents activation spikes and entropy collapse during training, allowing higher learning rates and stable FP8/INT8 post-training quantization on edge hardware.

---

### Suggested Architectural Blueprint for a State-of-the-Art 1B Model

A design integrating these modern techniques would outperform a standard LLaMA-style 1B baseline:

1. **Topology:** Deep & Thin ($\sim 24$ layers, $d_{\text{model}} \approx 1536$).
2. **Attention Type:** **Differential Attention** or **Alternating Local/Global Attention**.
3. **KV Cache Management:** **Cross-Layer KV Sharing** (Layers $2i$ and $2i+1$ share KV) with 4 KV heads ($d_k = 128$).
4. **Stabilization:** Logit soft-capping on attention logits ($C=50$) and output logits ($C=30$).
5. **Feed-Forward:** SwiGLU with tied input-output embeddings to preserve capacity for hidden layers.



-----



‫בתאריך יום ה׳, 17 בספט׳ 2026 ב-13:58 מאת ‪Igal Abachi‬‏ <‪igal.abachi.workmail@gmail.com‬‏>:‬

Igal Abachi
14:07 (לפני שעתיים (2))
אני

Designing a competitive sub-2B parameter modern LLM that quantizes cleanly to
4-bit GGUF and runs at hundreds of tokens per second in llama.cpp requires
balancing parameter allocation, hardware alignment, and architecture
compatibility.

1. High-Level Architectural Decisions

+-----------------------------------------------------------------------+
| Sub-1.5B Modern Architecture Blueprint                                |
|                                                                       |
|  [Input Tokens] ---> [Tied Embedding Matrix (Shared with LM Head)]    |
|                             |                                         |
|    +------------------------v-----------------------------+           |
|    | Repeat x 24 to 28 Layers (Deep & Thin)               |           |
|    |                                                      |           |
|    |  +--> RMSNorm -> GQA (QK-Norm + RoPE) -> Residual --+|           |
|    |  |                                                  ||           |
|    |  +--> RMSNorm -> SwiGLU FFN (8/3 d_model) -> Residual|           |
|    +------------------------+-----------------------------+           |
|                             |                                         |
|  [Final RMSNorm] ---> [Tied LM Head] ---> [Top-K Logits Distillation] |
+-----------------------------------------------------------------------+

Dense vs. MoE

  - Verdict: Pure Dense.
  - Rationale: A 1–2B MoE (e.g., 2B total with 400M active) lacks sufficient
    active capacity for deep reasoning. In contrast, a pure dense 1.2B–1.5B
    model fits completely within 700MB–1GB VRAM at 4-bit (Q4_K_M), leaving
    practically your entire VRAM free for a 32k–128k context window.

Attention Paradigm: Full GQA vs. MLA vs. Hybrid (SSM/Mamba)

  - Verdict: Grouped-Query Attention (GQA) with 2 or 4 KV heads.
  - Why not MLA (Multi-Head Latent Attention)? MLA (used in DeepSeek) compresses
    KV cache via low-rank projections, which is vital for 60B+ models where KV
    cache dominates memory. In a sub-2B model, GQA already compresses the KV
    cache to a few megabytes. MLA adds matrix-decompression latency on small
    models and has limited, brittle support in quantization kernels.
  - Why not Hybrid/Mamba? While SSM/Transformer hybrids are fast, llama.cpp
    support for non-standard hybrid architectures is often experimental, lacks
    optimized fused kernels, and complicates GGUF conversion. Full GQA is
    universally supported, rock-solid, and achieves native FlashAttention/cuBLAS
    speeds.

Depth vs. Width: The "Deep and Thin" Rule

  - Verdict: Deep and Thin.
  - Rationale: MobileLLM research established that for sub-2B models, depth
    contributes significantly more to capacity and reasoning than width.
      - Bad configuration (Shallow & Wide): 16 layers, d_{\text{model}} = 2048.
      - Good configuration (Deep & Thin): 24 to 28 layers,
        d_{\text{model}} = 1536.

2. Core Components & Sub-Modules

Embeddings & Vocabulary: Tied vs. Untied

  - Verdict: Tied Embeddings (tie_word_embeddings = True).
  - The Math: Modern tokenizers (Llama-3, Qwen-2.5) use V \approx 128{,}000 to
    152{,}000. If V = 151{,}936 and d_{\text{model}} = 1536:
    \text{Embedding weights} = 151{,}936 \times 1536 \approx 233.3\text{M parameters}
    If embeddings are untied (separate input and lm_head), the vocabulary
    consumes 466\text{M} parameters—nearly 40% of a 1.2B model’s entire
    capacity, leaving almost no budget for transformer blocks. Tying the weights
    cuts this in half, saving \approx 233\text{M} parameters that can be
    redirected into 8 additional transformer layers (which is why LLaMA-3.2-1B
    and Gemma-2-2B both use tied embeddings).

Normalization & Attention Stabilization: QK-Norm

  - Verdict: Pre-RMSNorm + QK-RMSNorm.
  - Why QK-Norm is essential for 4-bit Quantization: Small LLMs suffer severely
    from attention entropy collapse and activation outliers in query/key
    projections. Applying RMSNorm directly to Q and K heads before computing
    dot-product attention keeps the activations bounded. This is the single most
    effective trick to prevent degradation when converting down to Q4_K_M or
    Q4_0.
  - What about Attention Soft-Capping (Gemma-style)? Gemma 2 applies
    \text{tanh}(\cdot) logit-capping. While effective, it slows down attention
    decode kernels and can break standard FlashAttention-2 implementations
    unless specialized kernels are used. QK-Norm achieves the same outlier
    stabilization without breaking native FlashAttention.

Feed-Forward Network (FFN)

  - Verdict: SwiGLU FFN with \approx \frac{8}{3} d_{\text{model}} expansion.
  - For d_{\text{model}} = 1536, set intermediate dimension to 4608 (or 4864 to
    align with 256/128 hardware strides):
    \text{FFN}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}} \right) W_{\text{down}}

Positional Embeddings

  - Verdict: Rotary Position Embedding (RoPE) with high base frequency.
  - Set \theta = 500{,}000 or 1{,}000{,}000 (like LLaMA-3 / Qwen-2.5). This
    allows context window extension from 4k/8k to 32k/64k tokens via simple YaRN
    / RoPE scaling during inference without fine-tuning.

3. Golden Architectural Specification (~1.25B Parameters)

This specification mirrors the Qwen2 / LLaMA-3 architectural family,
guaranteeing zero friction when converting to GGUF using standard tools.

| Parameter                           | Value              | Rationale                                                                     |
| :---------------------------------- | :----------------- | :---------------------------------------------------------------------------- |
| **Total Parameters**                | **\~1.25 Billion** | Fits in \< 900MB at 4-bit; ultra-fast on RTX cards.                           |
| **Number of Layers**                | **28**             | Deep-and-thin design for maximum parameter efficiency.                        |
| **Hidden Dim ($d_{\text{model}}$)** | **1536**           | Perfectly divides into tensor core strides (multiple of 128).                 |
| **FFN Intermediate Dim**            | **4864**           | $\approx 2.67 \times d_{\text{model}}$, multiple of 256 for Ampere alignment. |
| **Query Attention Heads**           | **12**             | Head dimension: $1536 / 12 = 128$ (ideal for FlashAttention).                 |
| **KV Attention Heads**              | **2 or 4 (GQA)**   | 4 heads (3:1 ratio) keeps accuracy high while slashing KV cache.              |
| **Vocabulary Size**                 | **151,936**        | Exact match with Qwen-2.5 (enables direct logit distillation).                |
| **Tie Word Embeddings**             | **True**           | Saves \~233M parameters for transformer layers.                               |
| **QK-Norm**                         | **Yes (RMSNorm)**  | Completely prevents activation outliers for 4-bit quantization.               |
| **Layer Norm**                      | **RMSNorm**        | $\epsilon = 1e\text{-}6$, no learned bias.                                    |
| **RoPE Base ($\theta$)**            | **500,000**        | Stable long-context extrapolation.                                            |

4. PyTorch Implementation

Write this model using standard Hugging Face naming conventions
(Qwen2ForCausalLM or LlamaForCausalLM). Doing so enables llama.cpp's built-in
convert_hf_to_gguf.py script to recognize the model without requiring custom C++
kernels.

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
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
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

class ModernAttention(nn.Module):
    def __init__(self, d_model=1536, n_heads=12, n_kv_heads=4, head_dim=128):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # QK-Norm: Critical for 4-bit quantization stability
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Apply QK-Norm per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q = apply_rotary_pos_emb(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = apply_rotary_pos_emb(k, cos.unsqueeze(1), sin.unsqueeze(1))

        # Transpose for SDPA: [B, heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV for GQA using repeat_interleave
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled Dot-Product Attention (PyTorch fused FlashAttention kernel)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model=1536, intermediate_dim=4864):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model=1536, n_heads=12, n_kv_heads=4, intermediate_dim=4864):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = ModernAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, intermediate_dim)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Sub2BModernLLM(nn.Module):
    def __init__(self, vocab_size=151936, d_model=1536, n_layers=28, n_heads=12, n_kv_heads=4, intermediate_dim=4864):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.rotary = RotaryEmbedding(dim=128, base=500000.0)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, intermediate_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Tied lm_head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Parameter sharing!

    def forward(self, input_ids):
        B, S = input_ids.shape
        cos, sin = self.rotary(S, device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

5. Training & Distillation Strategy

A sub-2B model cannot develop emergent reasoning from raw next-token prediction
alone unless trained on tens of trillions of tokens. Distillation from a
frontier teacher (e.g., Qwen-2.5-72B, Llama-3.3-70B, or DeepSeek-V3) provides a
significantly more data-efficient path.

       [Teacher: Frontier Model (e.g. Qwen 2.5 72B)]
                             |
             Compute Soft Logits (Top-K=64)
                             |
   +-------------------------+-------------------------+
   |                                                   |
   v                                                   v
[Student Model]                              [Hard Token Targets]
   |                                                   |
   +---> Loss = (1 - alpha) * CE + alpha * KL_div <----+

Step 1: Matching the Tokenizer

  - Adopt the teacher’s tokenizer completely.
  - If distilling from Qwen-2.5-72B, use the exact Qwen2Tokenizer
    (V = 151{,}936). This guarantees identical sequence representations and
    avoids cross-entropy matrix transformations when comparing logit
    distributions.

Step 2: Top-K Logit Distillation

Computing full forward KL divergence over a 152k vocabulary is
memory-prohibitive. Instead, store or stream the teacher's Top-64 logits:

\mathcal{L} = (1 - \alpha)\mathcal{L}_{\text{CE}}(y, P_{\text{student}}) + \alpha \tau^2 \mathcal{L}_{\text{KL}}\left(\text{Softmax}\left(\frac{Z_{\text{teacher}}^{\text{top64}}}{\tau}\right) \parallel \text{Softmax}\left(\frac{Z_{\text{student}}^{\text{top64}}}{\tau}\right)\right)

  - Set temperature \tau = 1.5 to 2.0 to expose probability mass relationships.
  - Set \alpha = 0.7 (distillation should heavily outweigh raw hard targets).

Step 3: Reverse KL / MiniLLM-Style Sampling (for Instruction Fine-Tuning)

For alignment and fine-tuning:

  - Standard forward KL \text{KL}(P \parallel Q) causes the student to
    overestimate low-probability regions, producing hallucinations.
  - Reverse KL \text{KL}(Q \parallel P) forces the student to capture the modes
    of the teacher, producing sharper, more decisive outputs at smaller
    parameter counts.

6. Deployment to llama.cpp Server (4-bit GGUF)

Step 1: Save with Hugging Face Config

Format the PyTorch config.json using the standard Qwen2ForCausalLM or
LlamaForCausalLM schema so llama.cpp tools automatically parse the architecture:

from transformers import PretrainedConfig

config = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 1536,
    "intermediate_size": 4864,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "num_hidden_layers": 28,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16"
}

Step 2: Convert to GGUF

Run the llama.cpp converter directly on your output directory:

python llama.cpp/convert_hf_to_gguf.py ./my_student_model \
    --outfile ./models/model-f16.gguf \
    --outtype f16

Step 3: Quantize to 4-bit (Q4_K_M)

Q4_K_M uses 4-bit weights for attention and feed-forward layers while preserving
critical tensors (like early normalization and embeddings) at higher precision:

./llama-quantize ./models/model-f16.gguf ./models/model-Q4_K_M.gguf Q4_K_M

Step 4: Run the Inference Server

./llama-server \
    -m ./models/model-Q4_K_M.gguf \
    -c 16384 \
    --n-gpu-layers 99 \
    --host 0.0.0.0 \
    --port 8080

Expected Hardware Performance (RTX 3060 / 4060 12GB)

  - Model Footprint in VRAM: \approx 850\text{ MB}
  - KV Cache Footprint at 16k Context: \approx 256\text{ MB} (thanks to GQA)
  - Decode Speed: 140\text{ to }190+\text{ tokens/second}
  - Quantization Quality: Minimal perplexity degradation from FP16 baseline due
    to QK-Norm suppressing activation outliers.







**Recommended modern architecture for a competitive 1–2B dense LLM (2025–2026 SOTA small-model practice)**

Aim for a **dense decoder-only Transformer** that is fully compatible with Hugging Face → GGUF conversion and llama.cpp (no exotic kernels required). This is the recipe used by the current strongest open models in the size class: **Qwen2.5-1.5B**, **Llama-3.2-1B**, **SmolLM2-1.7B**, and similar.

### 1. Overall recipe (copy this)

| Component              | Recommendation                                      | Why / Notes |
|------------------------|-----------------------------------------------------|-------------|
| **Dense vs MoE**      | **Dense**                                           | MoE adds routing complexity, worse quantization behavior, and is overkill / harder to train well at <2B. Dense is simpler, more stable, and still SOTA in this size. |
| **Attention**         | **GQA** (Grouped-Query Attention)                   | Default in every competitive small model. Much better KV-cache than full MHA, almost no quality loss vs MHA at this scale. MLA is more complex to implement/train and currently preferred mainly at larger scales. |
| **GQA ratio**         | 4:1 to 6:1 (e.g. 12 Q / 2 KV or 16 Q / 4 KV)       | Qwen2.5-1.5B uses 12/2. Llama-3.2-1B uses 32/8. Stay in this range. |
| **Layers (depth)**    | **24–32**                                           | Prefer moderate depth. Too shallow hurts reasoning; too deep is hard to train at small width. |
| **Hidden size (width)** | **1536–2048**                                     | Classic trade-off: depth vs width. Qwen2.5-1.5B = 1536 × 28 layers. Llama-3.2-1B = 2048 × 16 layers. Both work; slightly deeper + narrower is currently preferred by many for small models. |
| **Head dim**          | **64 or 128**                                       | 128 is common and hardware-friendly. |
| **FFN**               | **SwiGLU** (SiLU-gated)                             | Still the clear winner. Intermediate size ≈ 2.5–6× hidden (Qwen2.5-1.5B uses ~5.8× = 8960). Do **not** use plain GELU/ReLU. |
| **Norm**              | **RMSNorm** (pre-norm)                              | Universal. |
| **Positional**        | **RoPE** (θ = 500k–1M)                              | Standard. Add YaRN / NTK scaling later if you need >32k context. |
| **Embeddings**        | **Tied** input/output                               | Almost everyone does this under ~3–7B. Saves parameters and improves quality at this scale. |
| **Biases**            | **None** (or only QKV bias if following Qwen)       | Modern default is bias-free. |
| **Soft capping**      | Optional (attention logits + final logits)          | Used by Gemma-2 (cap ≈ 50). Helps stability, especially if you push learning rate. Not required for Qwen/Llama-style, but cheap and safe. |
| **Vocab**             | 32k–150k (prefer 50k–128k)                          | Match a good existing tokenizer (Llama-3, Qwen2, or SmolLM tokenizer). |

### 2. Concrete starting configs (pick one)

**Option A – Qwen2.5-1.5B style (currently very strong)**  
- Layers: 28  
- Hidden: 1536  
- Heads: 12 Q / 2 KV  
- Intermediate: 8960 (SwiGLU)  
- Tied embeddings  
- RoPE θ = 1 000 000  
- Context target: 32k  

**Option B – Llama-3.2-1B style (shallower, wider)**  
- Layers: 16  
- Hidden: 2048  
- Heads: 32 Q / 8 KV  
- Intermediate: 8192  
- Tied embeddings  
- RoPE θ = 500 000  

Both are excellent. Start with Option A if you want maximum quality per parameter.

### 3. What to focus on (bottlenecks at 1–2B)

1. **Data quality + quantity** beats architecture tweaks.  
   Train on 2–10T+ high-quality tokens if possible (FineWeb-Edu + DCLM + math/code mixtures). Many SOTA small models are heavily over-trained relative to Chinchilla.

2. **KV cache & inference efficiency**  
   GQA is the biggest win. Soft capping + proper RoPE scaling help long context.

3. **Training stability**  
   Pre-norm RMSNorm + SwiGLU + (optional) soft capping + careful LR schedule (cosine or trapezoidal with warmup).

4. **Quantization friendliness**  
   Stay with standard linear layers, no exotic activations or routing → excellent 4-bit GGUF behavior on Ampere+ (RTX 30/40-series).

### 4. Training & distillation path

- Implement in pure PyTorch (or use `torchtune` / Hugging Face Transformers as base).  
- Pre-train (or continued pre-train) on your data.  
- **Distillation** is highly recommended:  
  - Use a strong teacher (Qwen2.5-7B/14B, Llama-3.1-8B/70B, or even a frontier model via API).  
  - Prefer **logit distillation** (KL or reverse-KL) + optional intermediate-layer matching.  
  - On-policy distillation (student generates, teacher scores) often works better than pure offline.  
  - Libraries: `torchtune` has ready KD recipes; NVIDIA NeMo / ModelOpt also support it.  

After base + distillation → SFT → (optional) DPO/ORPO/GRPO.

### 5. Inference / deployment

- Export to Hugging Face format → convert with `llama.cpp` tools to GGUF.  
- 4-bit (Q4_K_M or similar) runs very comfortably on a single consumer Ampere+ GPU (even 8–12 GB cards for short context).  
- Full support in `llama.cpp` server, llama-cpp-python, Ollama, etc.

### 6. Things to avoid (or treat as research)

- Full MHA at this size (wastes KV cache).  
- MLA unless you really want to experiment (harder kernels, less battle-tested at <10B).  
- Pure linear / SSM / hybrid attention for the first model (interesting research, but GQA dense is still the safe SOTA path for llama.cpp compatibility and quality).  
- Untied embeddings under ~3B.  
- Overly wide shallow models or extremely deep narrow ones without careful ablations.

### Practical next steps

1. Clone a Qwen2.5-1.5B or Llama-3.2-1B config and implement the exact architecture in PyTorch.  
2. Verify forward pass matches the reference (logits, attention maps).  
3. Train a tiny version (100–300M) first to debug.  
4. Scale to 1.5B with high-quality data + distillation.  
5. Convert → GGUF → test in llama.cpp.

This combination (dense + GQA + SwiGLU + RMSNorm + RoPE + tied embeddings + heavy data + distillation) is exactly what produces the current strongest sub-2B open models that run excellently on consumer hardware.





**Yes — this document is high-signal and improves the design.** It correctly prioritizes what actually matters at ≤1–2B on consumer hardware and llama.cpp, and it surfaces several practical upgrades that pure “copy Qwen/Llama” misses.

Here’s the refined architecture recommendation, incorporating the strongest points from the text while staying strictly practical (trainability in pure PyTorch, excellent 4-bit GGUF behavior, single RTX Ampere+ GPU, distillation-friendly, llama.cpp-native).

### Core Decision (unchanged + reinforced)
**Stay fully dense + full attention with GQA.**  
Hybrids (Mamba/linear) are not worth it at this scale for general-purpose models. KV cache is already tiny (~30–60 KB/token with GQA), edge/NPU kernels favor standard attention, and representational capacity (associative recall) is more valuable than linear-time savings when context is typically 2–8k.

### Refined Blueprint for a Competitive ≤1–2B Model (2026 practical SOTA)

| Component | Recommendation | Rationale / Upgrade from previous advice |
|-----------|----------------|------------------------------------------|
| **Topology** | Deep & thin: **24–28 layers**, \(d_\text{model} = 1536\)–2048 | Depth beats width at this scale (MobileLLM finding + document). Prefer 24–28 over the shallower Llama-3.2-1B (16 layers). |
| **Attention** | **GQA** (4:1 to 6:1) + **optional Differential Attention** or **local/global interleaving** | Standard GQA is the safe baseline. Differential Attention is the highest-upside modern upgrade for small models (reduces attention dilution / noise). Local/global (Gemma-2 style, e.g. 3 local : 1 global or 4:1) is the next-best practical improvement and is fully llama.cpp-compatible. |
| **Head dimension** | **64 or 96** (prefer 96 if VRAM allows with activation checkpointing) | Avoid 256 — too few heads + logit instability at ≤1B. 96 is a measured sweet spot (better than pure 64, still hardware-friendly). |
| **KV management** | GQA + **Cross-Layer KV Sharing** (every 2 layers share KV) | Strong practical win from MobileLLM. Cuts KV cache ~50% while keeping healthier number of KV heads than aggressive MQA. Easy to implement and quantize. |
| **Stabilization** | RMSNorm (pre-norm) + **QK-Norm** + **logit soft-capping** (attention \(C=50\), final logits \(C=30\)) | Soft-capping is cheap, improves stability, and helps post-training quantization. QK-Norm is now standard in modern small models. |
| **FFN** | SwiGLU, intermediate ≈ 2.7–5.5× \(d_\text{model}\) | Keep classic SwiGLU. No need for exotic variants at this scale. |
| **Embeddings** | **Tied** input/output | Still optimal under ~3B. |
| **Positional** | RoPE (θ = 10k–100k base; raise later only in a dedicated long-context stage) | Do **not** start with θ=1M. Match the real base-pretrain practice of Qwen/SmolLM. |
| **Optional high-ROI extras** | - Differential Attention (or local/global)<br>- Cross-layer KV sharing<br>- Soft-capping<br>- QK-Norm<br>- Slightly higher head_dim (96) with activation checkpointing | These are the concrete upgrades that move beyond a pure 2024 Llama/Qwen clone toward 2025–2026 small-model quality without breaking llama.cpp or edge deployment. |

### Concrete Starting Config (recommended)

```
layers: 26–28
d_model: 1536 or 2048
n_heads: 16–24 (query)
n_kv_heads: 4 (or 2–4 with cross-layer sharing)
head_dim: 96 (or 64 if extremely tight on VRAM)
FFN: SwiGLU, intermediate ≈ 4–5.5 × d_model
attention: GQA + 3-local/1-global schedule (or Differential Attention)
norm: RMSNorm + QK-Norm
soft-capping: yes (attention + final logits)
embeddings: tied
RoPE θ: 10000–100000
```

This stays fully dense, fully standard attention primitives (FlashAttention / SDPA / FlexAttention friendly), and produces excellent 4-bit GGUF.

### What to de-prioritize / avoid
- Pure hybrid SSM/Mamba as the main architecture (deployment friction + limited quality gain at ≤1B for general use).
- Extreme GQA / MQA (quality drop is real).
- head_dim = 256 (head starvation + instability).
- MLA at this scale unless you already have mature kernels (more complexity than benefit for ≤2B + llama.cpp goal).
- Untied embeddings or overly wide/shallow shapes.

### Training / Distillation Implications
The architecture above is deliberately distillation-friendly (standard dense transformer). Use a strong teacher (Qwen2.5-7B/14B or larger) with logit distillation + optional intermediate matching. Soft-capping and QK-Norm also improve training stability when you push learning rate or use BF16/FP16 on consumer cards.

### Expected Real-World Gains
- Better associative recall and less attention noise (Differential or local/global).
- Lower KV memory / better long-context behavior without hybrid complexity (cross-layer sharing + local windows).
- Higher training stability and better quantization robustness (soft-capping + QK-Norm).
- Still runs excellently in 4-bit GGUF on a single Ampere+ GPU and in llama.cpp server.

This is a clear step up from a pure 2024 “Llama-3.2 / Qwen2.5 clone” while remaining simple enough to implement, train, distill, and deploy cleanly.  






The empirical findings, negative results, and hardware-level failure modes from
your backlog provide critical insights that directly refine the theoretical
blueprint.

When training a sub-1.5B model on a single consumer GPU (8GB–12GB VRAM, RTX
Ampere/Turing) targeted for 4-bit GGUF in llama.cpp, abstract textbook theory
breaks down. The differences between theoretical design and real-world
implementation are summarized below.

1. The Hard Empirical Corrections to the Architecture

A. The head_dim Trap: Why 128 Causes Paging Collapses on Consumer GPUs

  - The Theory: Modern LLMs (Llama 3, Qwen 2.5) use head_dim = 128 because
    Tensor Cores and FlashAttention tile shapes love multiples of 128.
  - The Empirical Reality (MF-108): At a real context length of 2048, moving
    from head_dim = 64 to 128 increases VRAM by +44.6%, crossing the physical
    VRAM threshold on 8GB–12GB cards. Because Windows CUDA silently pages memory
    to system RAM instead of throwing an OOM error, throughput collapsed by 6.5x
    (from ~4,000 tok/s to ~625 tok/s). Zero-padding to 128 inside kernels
    backfired identically.
  - The Corrected Design:
      - Option 1 (Safe & Memory-Efficient): Use head_dim = 64 (e.g.,
        d_{\text{model}} = 1024, 16 heads, 4 KV heads). Fits comfortably on any
        consumer card without paging.
      - Option 2 (SOTA Quality Win): Use head_dim = 96 (e.g.,
        d_{\text{model}} = 1152, 12 heads, 4 KV heads; or decoupled
        head_dim_override = 96). In MF-108, this produced a -0.44% BPB quality
        win (~9.7x above the noise floor) with only a modest ~23% throughput
        cost—provided mandatory activation checkpointing is enabled to keep it
        off the silent-paging cliff.

B. Full GQA vs. Hybrid Local/Global Attention at Sub-1.5B Scale

  - The Theory: Hybrid attention (alternating 3 sliding-window layers with 1
    global layer) saves massive KV cache memory and linearizes attention
    compute.
  - The Empirical Reality (MF-066, MF-082, MF-140):
    1.  At \le 2048 context, the KV cache is already tiny
        (\approx 50\text{--}150\text{ MB}). Full attention compute is trivial
        compared to the FFN layers.
    2.  Sliding-window attention strips away associative memory and in-context
        copy-paste capability. A standard full-attention architecture often
        outperforms a windowed hybrid on the same token budget at short
        contexts.
    3.  Window ring buffers cause attention-sink collapse (position 0 eviction)
        unless fixed with output gating or sink tokens.
    4.  In llama.cpp, non-standard local/global repeating patterns complicate
        conversion.
  - The Corrected Design: Pure Full Attention with GQA (3:1 or 4:1 ratio). For
    example, 16 query heads and 4 KV heads. MF-081 proved that aggressive GQA
    (like 12:1 or MQA) actively degrades quality (+0.24% CE loss), whereas 4:1
    sits at the sweet spot.

2. "Free Lunch" Mathematical Enhancements (Zero GGUF Overhead)

These training-time additions improve loss and post-quantization stability, but
can be mathematically compiled away at export so llama.cpp sees a standard
LLaMA/Qwen graph.

       Training Phase:                          Export to GGUF:
+-------------------------------+      +-------------------------------+
|  x ---> RMSNorm(x)            |      |                               |
|           |                   |      |  Fold scale into weight:      |
|           v                   | ===> |  W_effective = W * (l + 1)^-0.5|
|  Multiply by (l + 1)^-0.5     |      |                               |
|  (Curse of Depth Damping)     |      |  Clean Standard RMSNorm       |
+-------------------------------+      +-------------------------------+

A. LayerNorm Scaling with Export-Time Weight Folding (MF-081)

  - The Problem: In deep, narrow networks, residual stream variances grow with
    depth, causing late-layer gradient instability.
  - The Mechanism: Multiply the output of each layer's RMSNorm by (l + 1)^{-0.5}
    (where l is the 0-indexed layer depth). In MF-081, this produced a
    consistent -0.35% CE / -1.8% PPL improvement.
  - The Zero-Overhead Export Trick: Scalar multiplication commutes with RMSNorm
    weights:
    \text{RMSNorm}(x) \cdot W \cdot (l+1)^{-0.5} = \text{RMSNorm}(x) \cdot \left( W \cdot (l+1)^{-0.5} \right)
    During export to GGUF, simply scale the static RMSNorm weights by
    (l+1)^{-0.5} and export a standard RMSNorm. llama.cpp executes zero extra
    operations.

B. SwiGLU Activation Clamping (MF-106)

  - The Problem: SwiGLU activations can produce unbounded magnitude spikes.
    In 16-bit training, this causes instability; during 4-bit quantization,
    these outliers blow out quantization grids, degrading Q4_K_M quality.
  - The Fix: Clamp the SwiGLU gate projection to [-7.0, 7.0] or [-10.0, 10.0]
    before applying SiLU:
    \text{FFN}(x) = \left( \text{SiLU}(\text{clamp}(x W_{\text{gate}}, -7.0, 7.0)) \odot x W_{\text{up}} \right) W_{\text{down}}
    This stabilizes training dynamics and prevents outliers, resulting in
    noticeably lower perplexity loss when quantized to 4-bit GGUF.

C. The Tied Embeddings & Vocabulary Balance (MF-087, MF-090, MF-102)

  - Vocabulary Trap: Training a 32k+ tokenizer on a sub-10B token budget
    underperforms a 16k tokenizer because of the Zipf tail: thousands of
    vocabulary rows are rarely seen, leaving them undertrained. Because tied
    embeddings share input and output weights, these undertrained rows inflate
    softmax entropy across the entire vocabulary.
  - The Rule: If training a custom tokenizer from scratch, use 16,384 to 32,768
    with tie_word_embeddings = True. If distilling from a frontier teacher
    (e.g., Qwen 2.5), keep the teacher's exact vocabulary for logit alignment,
    but tie the input and output embeddings.

3. Critical Training & Memory Engineering

A. Chunked Cross-Entropy Loss (MF-084)

On an 8GB–12GB GPU, materializing the unreduced logit tensor [B, S, V] in FP32
alongside its autograd graph consumes:
\text{Memory} \approx 2 \times 2048 \times 151{,}936 \times 4\text{ bytes} \approx 2.48\text{ GB}
This single tensor triggers the Windows CUDA system-memory paging cliff.

  - The Fix: Implement chunked cross-entropy. Project the hidden states and
    compute cross-entropy in chunks of tokens (e.g., chunk_size = 256 or 512)
    using a custom torch.autograd.Function. This reduces peak logit allocation
    by ~1.8x to 2x, allowing training to fit comfortably in VRAM.

B. WSD (Warmup-Stable-Decay) Scheduler (MF-083)

  - Traditional Cosine Annealing couples the learning rate curve to the exact
    step count (max_updates). If your workstation crashes, power blinks, or you
    decide to train on 5B tokens instead of 3B, resuming breaks the schedule.
  - Use WSD:
      - Warmup: Ramp up LR over the first 1–2% of tokens.
      - Stable Phase: Keep LR flat at peak. If training is interrupted, simply
        resume. If the model is still improving, let it run.
      - Decay Phase: When ready to finish, trigger a cosine or square-root
        cooldown over the final 15–20% of tokens to minimum LR
        (0.1 \times \text{LR}_{\text{peak}}). Most quality gains arrive during
        this cooldown.

Learning
Rate
  ^       +-----------------------------+
  |      /                               \
  |     /                                 \
  |    /                                   \
  |   /  Warmup        Stable               \  Decay (Cooldown)
  |  /  (~1-2%)       (~80%)                 \  (~15-20%)
  +--------------------------------------------------------> Steps

4. Inference Acceleration: Self-Speculative Decoding (MF-093, MF-109)

When serving the model in llama.cpp or a local server, small dense models are
heavily memory-bandwidth bound at batch size 1 (decode speeds around 30–60 tok/s
on mid-tier cards).

  - The MTP (Multi-Token Prediction) Advantage: Co-train a single auxiliary
    linear head predicting token t+2 alongside the main t+1 head:
    \mathcal{L} = \mathcal{L}_{t+1} + 0.3 \cdot \mathcal{L}_{t+2} This adds ~5%
    compute during pretraining and provides a slight regularization gain.
  - Inference Speedup: The auxiliary head acts as an internal draft model for
    self-speculative decoding. The model drafts token t+2 for free, then
    verifies both tokens in a single batched forward pass.
  - The Empirical Result (MF-093): Achieved a 1.21x wall-clock speedup at a
    45.3% draft acceptance rate on greedy decoding.
  - Warning (MF-118): Restrict self-speculation to greedy decoding. Rejection
    sampling for temperature-based speculation added probability math overhead
    that resulted in a 0.94x slowdown.

5. The Battle-Tested Specification (~1.1B Parameters)

This specification incorporates the lessons learned from our empirical tests. It
is sized to train on an 8GB–12GB GPU without memory paging and deploys natively
as a standard Qwen2ForCausalLM or LlamaForCausalLM in llama.cpp.

| Parameter               | Value                 | Reason / Empirical Grounding                                       |
| :---------------------- | :-------------------- | :----------------------------------------------------------------- |
| **Total Parameters**    | **\~1.15 Billion**    | Sized for fast decoding and 4-bit memory fit (\<900MB).            |
| **Layers**              | **24**                | Deep-and-thin topology for sub-2B capacity.                        |
| **Hidden Size ($d$)**   | **1536**              | Hardware aligned (multiple of 128/256).                            |
| **Attention Type**      | **Full GQA**          | Avoids windowing capacity loss and attention-sink traps.           |
| **Query Heads**         | **16**                | $16 \times 96 = 1536$ (or 24 heads if using $d_k=64$).             |
| **KV Heads**            | **4 (4:1 Ratio)**     | Optimal quality retention; avoids 12:1 regression.                 |
| **Head Dim ($d_k$)**    | **96** (or **64**)    | Prevents the 6.5x throughput collapse seen at $d_k=128$.           |
| **Activation**          | **SwiGLU**            | Intermediate size = 4096 ($\approx 2.67 \times d_{\text{model}}$). |
| **SwiGLU Clamping**     | **$[-7.0, 7.0]$**     | Suppresses outlier activations for clean 4-bit quantization.       |
| **Normalization**       | **RMSNorm + QK-Norm** | Head-wise QK-RMSNorm prevents attention entropy collapse.          |
| **Depth Scaling**       | **$(l+1)^{-0.5}$**    | Prevents residual explosion; folded into RMSNorm at export.        |
| **Embeddings**          | **Tied**              | Saves \~150M–230M parameters for transformer blocks.               |
| **Target Quantization** | **`Q4_K_M` GGUF**     | Native `llama.cpp` support with minimal perplexity loss.           |

6. PyTorch Implementation with Export-Ready Folding

Here is the updated implementation incorporating QK-Norm, SwiGLU clamping,
chunked loss integration, and the export-time weight folding helper:

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class ClampedSwiGLU(nn.Module):
    def __init__(self, d_model: int, intermediate_dim: int, clamp_val: float = 7.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
        self.clamp_val = clamp_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.clamp_val is not None:
            gate = torch.clamp(gate, -self.clamp_val, self.clamp_val)
        return self.down_proj(F.silu(gate) * self.up_proj(x))

class ModernGQAAttention(nn.Module):
    def __init__(self, d_model: int = 1536, n_heads: int = 16, n_kv_heads: int = 4, head_dim: int = 96):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Per-head QK-Norm: essential for post-training 4-bit quantization stability
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Apply QK-Norm per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE (rotate half)
        q = (q * cos.unsqueeze(1)) + (self._rotate_half(q) * sin.unsqueeze(1))
        k = (k * cos.unsqueeze(1)) + (self._rotate_half(k) * sin.unsqueeze(1))

        # Prep for SDPA: [B, heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.num_queries_per_kv, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, d_model: int = 1536, n_heads: int = 16, n_kv_heads: int = 4, head_dim: int = 96, d_ff: int = 4096):
        super().__init__()
        self.layer_idx = layer_idx
        # Training-time depth scaling factor
        self.depth_scale = (layer_idx + 1) ** -0.5

        self.attn_norm = RMSNorm(d_model)
        self.attn = ModernGQAAttention(d_model, n_heads, n_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = ClampedSwiGLU(d_model, d_ff, clamp_val=7.0)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Scale norm output during training (folded at export)
        norm_attn = self.attn_norm(x) * self.depth_scale
        x = x + self.attn(norm_attn, cos, sin)

        norm_ffn = self.ffn_norm(x) * self.depth_scale
        x = x + self.ffn(norm_ffn)
        return x

class Competitive1BModel(nn.Module):
    def __init__(self, vocab_size: int = 151936, d_model: int = 1536, n_layers: int = 24, n_heads: int = 16, n_kv_heads: int = 4, head_dim: int = 96, d_ff: int = 4096):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(i, d_model, n_heads, n_kv_heads, head_dim, d_ff) for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Tied embeddings

        # Precompute RoPE frequencies
        self.head_dim = head_dim
        inv_freq = 1.0 / (500000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_rotary_emb(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        cos, sin = self.get_rotary_emb(S, input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def fold_depth_scaling_for_export(self):
        """Folds depth scale into RMSNorm weights so llama.cpp requires zero code changes."""
        for layer in self.layers:
            scale = layer.depth_scale
            layer.attn_norm.weight.mul_(scale)
            layer.ffn_norm.weight.mul_(scale)
            layer.depth_scale = 1.0
        print("LayerNorm depth scaling successfully folded into RMSNorm weights for GGUF export.")

7. Export and Distillation Strategy

1.  Teacher Selection & Distillation:

      - Distill from Qwen-2.5-32B or 72B (or DeepSeek-V3 outputs).
      - Match the teacher's tokenizer (V = 151{,}936) to allow token-for-token
        logit distillation without projection errors.
      - Stream the teacher's Top-64 logits at \tau = 1.5 to compute the forward
        KL loss, bypassing the memory cost of saving 152k logits.

2.  GGUF Export Protocol: Before converting to GGUF:

    model.fold_depth_scaling_for_export()
    model.eval()
    # Save as standard Hugging Face safetensors format

    Save the config with "architectures": ["Qwen2ForCausalLM"] and
    "head_dim": 96.

3.  llama.cpp Serving: Run the converter and quantize directly to Q4_K_M:

    python convert_hf_to_gguf.py ./my_student_model --outtype f16 --outfile model_f16.gguf
    ./llama-quantize model_f16.gguf model_q4_k_m.gguf Q4_K_M
    ./llama-server -m model_q4_k_m.gguf -c 4096 --n-gpu-layers 99

Summary of What Changed

  - Discarded: head_dim=128 (prevents memory paging collapses), hybrid windowed
    attention (prevents attention-sink degradation and preserves capacity at
    \le 2048 context), untied embeddings, and sampling-based speculative
    decoding.
  - Added: head_dim=96 (with activation checkpointing), SwiGLU clamping to
    prevent activation spikes, export-time LayerNorm depth-scale folding,
    chunked cross-entropy to protect GPU memory, WSD scheduling for training
    resilience, and optional greedy self-speculative decoding via an auxiliary
    linear head.




‫בתאריך יום ה׳, 17 בספט׳ 2026 ב-14:07 מאת ‪Igal Abachi‬‏ <‪igal.abachi.workmail@gmail.com‬‏>:‬

Igal Abachi
14:14 (לפני שעתיים (2))
אני

**Review of MiniFrontier 150M-Modern (as an AI specialist)**

This is one of the cleaner educational + practical small-LLM codebases I’ve seen. It is deliberately readable, has strong test coverage, real hardware evidence collection, and honest documentation of both positive and negative results. The 150M-Modern config is already quite sophisticated for its size.

### Current 150M-Modern Spec (from `configs/150m-modern.toml`)

```toml
n_layers          = 20
d_model           = 768
n_heads           = 12
n_kv_heads        = 4          # 3:1 GQA
head_dim_override = 96         # real measured quality win
d_ff              = 2048       # ~2.67×
qk_norm           = true
attention_pattern = "hybrid"   # 3 local + 1 global
local_window      = 512
gated_attention   = true
layer_norm_scaling = true
tie_embeddings    = true
rope_theta        = 10000.0
vocab_size        = 16384
max_seq_len       = 2048
```

### Strengths (keep these)

| Area | Assessment |
|------|------------|
| **Code quality & readability** | Excellent. Clear residual-stream mental model, extensive comments, educational labs. |
| **QK-Norm + head_dim=96** | Correct modern choices. Real measured quality win and better 4-bit behavior. |
| **Gated attention** | Good attention-sink mitigation. |
| **Layer-norm scaling + residual damping** | Solid depth-stability techniques; folding at export is the right idea. |
| **Tied embeddings + small vocab (16k)** | Correct for this scale and token budget. |
| **Training engineering** | Chunked CE, WSD, activation checkpointing, FP16+GradScaler, careful resume — all well done for an 8 GB card. |
| **Honesty** | Negative results (hybrid vs full, local_window ablations, Muon vs AdamW wall-clock, etc.) are documented. Rare and valuable. |

### Weaknesses / Opportunities for Improvement

1. **Hybrid attention is currently the biggest quality risk at this scale**
   - Your own evidence (MF-066, MF-140 discussion, local_window ablations) shows that at 1–2k context the capacity cost of windowed layers is not clearly paid back.
   - Almost every strong open model ≤1.5B (Qwen2.5-0.5/1.5B, Llama-3.2-1B, SmolLM2) stays **full GQA**.
   - **Suggestion**: Make `attention_pattern = "full"` the default for the next 150M (or 350M) release run, and keep hybrid only as an experiment. Re-run the full-budget Edu vs Modern comparison with current recipe + full attention.

2. **Depth vs width is still a bit conservative**
   - 20 layers × 768 is fine, but the “deep & thin” literature (MobileLLM) and your own later thinking favor going deeper.
   - **Suggestion**: Try 24 layers × 768 or 22–24 layers × 896 while keeping total params ~150–160M. Depth usually helps reasoning more than width at this size.

3. **GQA ratio can be made less aggressive**
   - 3:1 (12Q/4KV) is good, but Qwen3-0.6B uses 2:1. Your own sweeps only tested more aggressive ratios.
   - **Suggestion**: Add a 2:1 arm (e.g. 12Q/6KV or 16Q/8KV) in the next matched run. Restoring KV capacity often helps more than the extra cache cost hurts at short context.

4. **Missing or under-used free stability/quantization wins**
   - SwiGLU activation clamping (`clamp(gate, -7, 7)` or similar) — strongly recommended for clean Q4_K_M.
   - Soft-capping on final logits (and optionally attention) is cheap and helps both training stability and post-quantization behavior.
   - Cross-layer KV sharing (every 2 layers share KV) is a simpler alternative to hybrid windows if you still want KV savings.

5. **Tokenizer / vocabulary**
   - 16 384 is correct for the current token budgets. Keep it.
   - If you ever distill from Qwen2.5, you will need to switch to their 151k vocab + tied embeddings, which changes the parameter budget significantly.

6. **Inference path**
   - MTP + self-speculative decoding is already implemented — excellent. Make it the default for greedy Modern releases (you already started this).
   - Multi-turn KV-cache reuse across chat turns is still missing (easy win, already discussed in the backlog).

### Concrete Priority Recommendations (ordered)

| Priority | Change | Expected Impact | Effort |
|----------|--------|-----------------|--------|
| **P0** | Switch Modern default to **full GQA** (drop hybrid for the main release line) | Likely quality gain at matched tokens; simpler GGUF path | Low |
| **P0** | Add SwiGLU gate clamping | Better 4-bit quality, more stable training | Very low |
| **P1** | Test 2:1 GQA ratio | Possible quality recovery of representational capacity | Medium |
| **P1** | Slightly deeper config (24L × 768 or similar) | Better reasoning depth | Medium |
| **P2** | Final-logit soft-capping | Training + quant stability | Low |
| **P2** | Cross-layer KV sharing (optional) | KV savings without windowing capacity loss | Medium |
| **P2** | Multi-turn cache reuse in chat | Real latency win for interactive use | Low–Medium |

### Overall Verdict

This is already a **high-quality educational + practical 150M codebase**. The main architectural risk is continuing to treat hybrid local/global attention as the default Modern path when the evidence at short context points the other way. Fixing that, adding the cheap SwiGLU clamp, and re-running a clean full-budget comparison would make the next 150M (or 350M) checkpoint noticeably stronger while keeping the excellent educational clarity and consumer-GPU focus of the project.

Would you like me to propose an exact revised `150m-modern.toml` + the minimal code changes for SwiGLU clamping and full-attention default?




**Yes — this is a solid, well-engineered dataset for training a ~150M model**, especially for an educational / consumer-GPU project like MiniFrontier.

### Quick Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Overall quality** | Good → Very Good | Carefully filtered, deduplicated, and documented mixture |
| **Composition** | Strong | Balanced mix of high-quality educational web + code + math + synthetic |
| **Provenance & hygiene** | Excellent | Full per-document metadata, licenses, content hashes, admission filtering |
| **Size** | Appropriate | ~5B real tokens (labeled “3B-token mixture” but actual measured tokens are higher) |
| **Suitability for 150M** | High | Matches the scale and goals of MiniFrontier perfectly |

### Mixture Composition (from the dataset card)

| Source | Weight | Approx. Tokens | Role |
|--------|--------|----------------|------|
| **dclm-edu** | 35% | ~1.65B | High-quality educational web data (strong signal) |
| **fineweb-edu** | 25% | ~0.90B | Educational subset of FineWeb (proven strong for small models) |
| **github-code** | 20% | ~1.39B | Permissively licensed code (with proper license tracking) |
| **finemath** | 15% | ~0.93B | Math-focused data |
| **cosmopedia-v2** | 5% | ~0.08B | Synthetic educational content (Cosmopedia style) |
| **Total** | 100% | **~4.96B** | |

This is a thoughtful, modern small-model mixture:
- Heavy on educational / high-quality web (DCLM-Edu + FineWeb-Edu)
- Meaningful code and math components
- Small synthetic boost (Cosmopedia-v2)
- Proper deduplication and filtering already applied

### Strengths

- **Excellent data hygiene**: Every row carries full provenance (`source`, `revision`, `license`, `content_hash`, etc.). This is rare and very good practice.
- **License awareness**: Especially important for the GitHub-code portion (mixed permissive licenses tracked per example).
- **Ready-to-use Parquet**: Easy to load with `datasets.load_dataset` or stream.
- **Aligned with current best practices**: FineWeb-Edu / DCLM-style educational filtering + code + math is exactly the style that works well for models in the 100–500M range (SmolLM2, Qwen2.5-0.5/1.5B, etc.).
- **Designed for the same project**: It was built specifically for MiniFrontier’s 150M-Modern training, so the filtering, packing assumptions, and token counts match the model’s tokenizer and context.

### Limitations / Caveats

1. **Still a relatively small total token count** (~5B) for a 150M model by 2025–2026 standards.  
   Strong small models are often trained on 1–2T+ tokens. This mixture is excellent for a solid educational / research checkpoint, but not “frontier-scale” overtraining.

2. **Synthetic portion is small** (only 5% Cosmopedia-v2).  
   Some recent recipes use higher synthetic ratios, but your mix stays conservative and safer.

3. **English-only** (as expected).

4. **Per-example licenses** mean you must respect the individual `license` column if you redistribute subsets.

### Verdict

**Yes, this is a great dataset** for the intended purpose (training or continuing pre-training a 150M-class model on consumer hardware).  

It is clean, well-documented, properly filtered, and uses a sensible modern mixture of educational web + code + math + a bit of synthetic data. It is significantly better than most ad-hoc small-model datasets people throw together.

**Best uses**:
- Continued pre-training / full fine-tune of MiniFrontier-150M-Modern
- Continued pre-training of similar-size bases (SmolLM2-135M, Qwen2.5-0.5B, etc.)
- LoRA / PEFT experiments

If you want to push quality further later, the highest-ROI next steps would be:
- Adding more high-quality math/code
- Mildly increasing the synthetic educational portion
- Or simply training longer on this same mixture (it is clean enough to support more epochs).

Would you like recommendations on how to best use this mixture (packing, curriculum, mixing ratios during training, etc.)?

Igal Abachi
16:41 (לפני 7 דקות)
אני

Short version: build a dense, ~1.45B, Qwen3.5-style hybrid. That means 3 Gated DeltaNet layers for every 1 gated GQA attention layer, tied embeddings, the teacher's tokenizer, SwiGLU, QK-norm, partial RoPE, one MTP head, and Muon. Make it an exact config of an architecture llama.cpp already runs, so you write zero C++. The architecture is the easy 10%. Tokens, teacher logits and post-training decide whether you get anywhere near "small SOTA."

0. Reality check
The current bar is Qwen3.5-0.8B/2B and Gemma 4 E2B. The Qwen3.5 small models are natively multimodal, trained from scratch on a hybrid architecture, and Apache 2.0. Behind them are tens of trillions of tokens and an industrial post-training stack.

Here are my rough estimates. They assume ~40k tok/s student throughput per H100, ~8k tok/s prefill for a 27B dense teacher, and $2–3/h rentals.

Tier	Student tokens	KD tokens / teacher	Rough cost	Realistic outcome
A: proof	100B	30B / 9B	$3–5k	Good teaching artifact, below Qwen3.5-0.8B on general evals
B: serious	1T	250B / 9B–27B	$25–40k	Qwen3-1.7B-class general; can win one narrow slice (e.g. C#/.NET code)
C: contender	3–5T	~1T / 27B	$150k+	Trades blows with Qwen3.5-2B on selected benchmarks, behind overall
Two uncomfortable facts:

VRAM isn't your constraint. A 1.45B model at Q4_K_M is about 1 GB, so on a 3090 the sub-2B limit is a choice you're making. That's fine for ownership and teaching. But if the goal were "best model on a 24 GB card," an ~8B-A1B MoE would be the right design.
The 2070 Super is your CI box, not a training box. Turing has no native bf16, and 8 GB caps you at roughly 150–300M ablations.
1. Architecture decisions
Question	Pick	Why
Dense vs MoE	Dense	At ≤2B total, MoE adds router instability and needs more tokens per total param for little gain
Attention layout	3:1 Gated DeltaNet : full attention, last layer full	Recurrent layers keep constant state; periodic exact attention handles retrieval; ~4× smaller KV
Full-attn flavour	GQA, 8 Q / 2 KV, head_dim 256	Only 7 of 28 layers carry KV, so keep them rich
Differential attention	No	Not in llama.cpp; gated attention fixes the same attention-sink/noise problem and is production-proven
Soft-capping	None; QK-norm plus optional z-loss 1e-4	The target arch has no softcap, so adding one breaks compatibility; QK-norm handles stability
Norm	Pre-RMSNorm, zero-centered QK-norm	Matches Qwen3.5
Position	Partial RoPE (64 of 256 dims) on attention layers only	GDN layers get order from recurrence and their short conv
FFN	SwiGLU, intermediate = 3×d	Still the best compute/quality default
Embeddings	Tied, teacher's tokenizer	Tying saves 381M params; a shared vocab gives exact logit KD
Depth/width	28 layers × d=1536	See below
MTP	1 module, loss weight ~0.3	Small sample-efficiency gain plus a built-in draft head
Optimizer	Muon (2D hidden) + AdamW (rest)	Section 3
Why this layout. It matches what Qwen shipped at exactly your scale. The 0.8B model card spells out 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)). That model uses 24 layers, a 1024 hidden size, 8 attention heads with 2 KV heads, and multi-token prediction. The bigger siblings use the same attention block. In Qwen3.6-35B-A3B the gated attention has 16 Q heads and 2 KV heads at head dim 256 with RoPE dim 64, and the DeltaNet has 32 V heads and 16 QK heads at dim 128.

The competing design is Gemma's sliding-window/global interleave. All Gemma 4 models alternate local sliding-window and global layers, with 512-token windows on the small models and a global final layer. It's a fine baseline, but GDN gives constant state instead of a growing global KV cache.

Proposed "MF-1.45H" spec:

Vocab	248,320 (Qwen3.5 tokenizer), tied
d_model / layers	1536 / 28 = 7 × (3 GDN + 1 GatedAttn)
GDN	12 QK heads, 24 V heads, head dim 128, conv kernel 4, output gate + norm
Gated attention	8 Q / 2 KV, head_dim 256, RoPE dim 64, high θ (≥1M)
FFN	SwiGLU, 4608
Params	≈1.07B non-embedding + 0.38B embedding ≈ 1.45B (verify with a counter)
KV cache	≈1.8 GB f16 at 128k (a 28-layer dense GQA 4×128 needs ≈7.3 GB); GDN state ≈35 MB total
A 2B variant is d=2048, 24 layers = 6 × (3+1), FFN 5504, for ≈1.8B total.

Depth vs width. Deep-thin wins below ~500M (MobileLLM). At 1–2B, shipped models cluster at 24–36 layers. Gemma 4 E2B has 35 layers at model_dim 1536. Batch-1 llama.cpp decode pays per-layer kernel overhead, so past ~36 layers you lose tok/s for little quality.

Gated attention block. Match HF's exact tensor layout, or the GGUF converter will silently permute weights wrong. In the Qwen3-Next code, the query projection emits both query features and gate features, and the attention output is multiplied by sigmoid(gate) before the output projection. Qwen also uses zero-centered QK-Norm and partial RoPE in this block.

class GatedAttention(nn.Module):
    def __init__(s, d=1536, n_q=8, n_kv=2, hd=256, rope_dim=64):
        super().__init__()
        s.n_q, s.n_kv, s.hd, s.rope_dim = n_q, n_kv, hd, rope_dim
        s.q = nn.Linear(d, 2 * n_q * hd, bias=False)      # query + gate
        s.k = nn.Linear(d, n_kv * hd, bias=False)
        s.v = nn.Linear(d, n_kv * hd, bias=False)
        s.o = nn.Linear(n_q * hd, d, bias=False)
        s.qn, s.kn = ZeroCenteredRMSNorm(hd), ZeroCenteredRMSNorm(hd)

    def forward(s, x, cos, sin, block_mask):
        B, T, _ = x.shape
        q, g = s.q(x).view(B, T, s.n_q, 2 * s.hd).chunk(2, dim=-1)
        k = s.kn(s.k(x).view(B, T, s.n_kv, s.hd))
        v = s.v(x).view(B, T, s.n_kv, s.hd)
        q, k = partial_rope(s.qn(q), k, cos, sin, s.rope_dim)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                           block_mask=block_mask, enable_gqa=True)   # causal + doc mask
        y = y.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(g.reshape(B, T, -1))
        return s.o(y)
GDN layer. The recurrence is S_t = α_t·S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ, with output o_t = S_t q_t. Don't hand-write the training kernel. Use flash-linear-attention's chunked gated-delta-rule Triton kernel with cu_seqlens, so state resets at document boundaries. Keep a naive for-loop reference implementation only for tests; it's also the version to show juniors.

Don't add yet:

Per-layer embeddings. Gemma 4's PLE gives each decoder layer its own small per-token embedding, and those large tables are only used for lookups. That solves phone flash-vs-DRAM limits; on a GPU it's just extra params.
Qwen's 2027 direction. Qwen3.8-Flash-Next adds a GDN + Qwen Sparse Attention hybrid, a Gated Residual that widens the residual stream into 4 gated branches, and N-gram embeddings whose table can be offloaded to host memory. Qwen positions it as an early preview of the Qwen4 architecture. N-gram lookup memory offloaded to system RAM is very attractive for consumer GPUs. Leave a hook for it, but don't build on it until llama.cpp supports it.
2. Distillation: where you actually win
Use a teacher ladder on one tokenizer. Qwen3.5 moved to a 250k vocabulary, which improves encoding efficiency across most languages. Hash tokenizer.json across Qwen3.5, 3.6 and 3.8 before committing.

Logit teacher: Qwen3.5-9B or Qwen3.6-27B dense. Don't use the biggest model. The distillation scaling-laws work (Busbridge et al., 2025) shows a capacity gap: too strong a teacher can hurt a small student. The 27B is still strong: it resolves 77.2% of SWE-bench Verified in a dense 27B package.

Data teacher: the giant open models, used for tokenizer-agnostic synthetic SFT and reasoning traces. GLM-5.2 and DeepSeek-V4-Pro are both MIT-licensed. Skip closed APIs. They don't expose full logits, and their terms generally forbid training competing models on outputs.

Pretraining KD. Use forward KL on sparse teacher targets plus CE on real labels. The sparse logit sampling work cut stored teacher logits down to 12 per token, and related work concluded forward KL beats other objectives and adding CE helps. Sampling beats plain top-k, which makes students overconfident.

Storage is about 48 bytes/token, so 250B KD tokens ≈ 12 TB. For the vocab-sized loss tensor, see the recent fused, chunked KL loss for sparse top-K teachers, which keeps peak memory linear in sequence length.

Post-training order:

SFT on assistant tokens (you already have this).
On-policy distillation: the student samples, the same-tokenizer teacher scores each token with reverse KL. OPD is now a standard post-training primitive at Qwen, DeepSeek, Zhipu and NVIDIA. It's the biggest lever for small models and far cheaper than RL.
A thin RLVR pass on code and math with unit-test rewards.
3. Training recipe
Optimizer. torch.optim.Muon is native, handles only 2D params (AdamW takes embeddings, norms and other non-2D params), and offers an adjust_lr_fn option including "match_rms_adamw". Use that option so the AdamW LR and weight decay carry over. The torch Muon doesn't yet support distributed training, so FSDP needs a custom implementation. The workaround: 1.45B fits on an 80 GB card, so run plain DDP.

Schedule. Use WSD (warmup-stable-decay), which lets you branch annealing runs off the stable phase. Batch ~2M tokens.

Context length. Train at 4k, then a 32k stage, then 128k+ by raising RoPE θ on the attention layers only.

Precision. bf16 plus torch.compile. FP8 via torchao on rented H100s is optional; Ampere can't do it.

Data. Mix FineWeb-Edu/DCLM-style web, FineWeb2 for multilingual, Stack-Edu with FIM for code, and FineMath. The decay phase is where high-quality, math/code and long-context data go.

4. Bottlenecks, ranked
Data quality and token count. This dominates everything else at this scale.
The 248k-vocab logit tensor in both CE and KD. Naively, 8×4096×248k in fp32 is about 32 GB. Fused/chunked losses are mandatory.
Teacher compute. Logit generation with a 27B costs roughly 3–6× the student's per-token training compute, so a 9B teacher is often the better deal.
GDN kernel MFU and document masking. Measure MFU on day 1 of rented time.
4-bit sensitivity. Small models lose more at Q4. Keep token_embd at Q6_K or Q8, use imatrix, and measure KLD against BF16 with llama-perplexity --kl-divergence. Add QAT for the last ~1% of tokens if KLD is bad.
5. Shipping to llama.cpp
The runtime pieces exist. A fused GATED_DELTA_NET op with CPU and CUDA backends was merged in b8233. MTP support was merged in May 2026, with around 75% steady-state acceptance at 3 draft tokens and over 2× speedup on Qwen3.6.

Two caveats:

MTP can be a net loss on hybrids. In one Qwen3.5-27B analysis, recurrent layers had to be checkpointed before speculation and restored on rejection, and verification batches couldn't run nearly free the way they do for attention-only models. At 1.45B on a 3090 you're already at hundreds of tok/s, so ship GGUFs with and without MTP and benchmark both.
The GDN CUDA path has had correctness bugs. One user traced garbage Qwen3.8 output to an older llama.cpp CUDA implementation of the Gated DeltaNet layers. Pin a known-good build.
6. Execution in MiniFrontier terms
Compat harness (do this first, on the 2070S). Random-init a ~20M model in the exact HF Qwen3.5 text format. Run it through convert_hf_to_gguf.py and llama.cpp, and assert PyTorch vs llama.cpp KLD ≈ 0. It costs nothing and kills the most expensive failure mode. Also add the tokenizer swap to the teacher vocab.

Ablations. At 150–400M on one rented GPU with 5–10B tokens each, compare:

dense GQA vs 3:1 SWA vs 3:1 GDN
attention gate on/off
MTP on/off
Muon vs AdamW
Judge by loss, HellaSwag/ARC, needle retrieval, and code pass@1.

Teacher cache. Offline sparse logits from the 9B, sharded to object storage.

Pretrain. Tier A first, then stop and look at the numbers before committing to Tier B.

Mid-train. Long-context stage, reasoning traces, chat template, tool calling.

Post-train and ship. SFT → OPD → RLVR, then GGUF quants with KLD reports, an MTP variant, and a Jinja chat template for llama-server --jinja.

The fastest route to a model that beats Qwen3.5-2B is still Qwen3.5-2B-Base plus your own post-training stack. The from-scratch path buys ownership and teaching value, not leaderboard wins, so decide which you're paying for before Tier B.


‫בתאריך יום ה׳, 17 בספט׳ 2026 ב-14:14 מאת ‪Igal Abachi‬‏ <‪igal.abachi.workmail@gmail.com‬‏>:‬

Igal Abachi
16:42 (לפני 6 דקות)
אני

The last section ("Current hardware and dataset plan") is solid, and I re-derived its arithmetic: the 1.661B and 1.721B parameter counts, the KV-cache tables, the 34.9M LoRA parameters, the 1,929 / 3,404 / 5,629 tok/s thresholds, and the 1.83 Cosmopedia passes. All correct.

The document as a whole isn't yet a plan you can execute. It stacks three plans, picks a student that blocks its own distillation strategy, and spends the 18 days on the wrong bottleneck.

The pivot itself is right. My previous answer assumed rented H100s; once training has to happen on one 16–24 GB RTX in under 18 days, from-scratch 1–2B is off the table, and adapting a pretrained student is correct.

1. Collapse it into one plan
Three layered plans with "supersedes where conflicting" means whoever reads it next picks the wrong default. The conflicts are real:

8K context vs 2,048.
A 64K custom tokenizer vs keeping the teacher's tokenizer.
5K–20K examples vs 20K–100K.
A 70/20/10 raw mixture vs 35/25/20/15/5.
A "still to specify" list that asks for training duration after it's already fixed.
Move the original from-scratch design into an appendix labeled "MiniFrontier research track" and keep one current plan.

2. The student choice breaks the distillation ladder
This is the biggest technical miss. Qwen3-1.7B uses the 151,936-token vocabulary. Every strong local Qwen teacher you'd actually run is on the 248,320 vocabulary: Qwen3.6 checkpoints share the architecture and model_type of Qwen3.5 and load with the same classes. One Qwen3.8-27B deployment note found that only same-vocab drafters line up, and a Qwen2.5-family drafter shares just 0.19% of id-to-token meanings with it.

So with Qwen3-1.7B, stages 2 and 3 of the document's own distillation plan (logit KD and on-policy distillation) are impossible with any current Qwen teacher. You're capped at text-only SFT.

Make Qwen3.5-2B (or 0.8B) the default candidate, with Qwen3-1.7B as the fallback. Qwen3.5's costs are real but manageable:

GDN layers need flash-linear-attention kernels to train at reasonable speed.
PEFT target module names differ.
The text backbone uses multimodal RoPE, and if you replace the rotary module you have to preserve that split.
The vision tower goes in a separate mmproj file in llama.cpp, so it doesn't inflate the text GGUF.
Settle this with a day-1 bake-off, not "benchmark later."

3. The time budget is allocated backwards
Here are ballpark student throughputs, since the document gives none. These are my estimates for BF16 LoRA with gradient checkpointing (≈6N FLOPs/token), packed 2K sequences, ~40% MFU, on a 1.7B model:

Card	≈ tok/s
4060 Ti 16GB	~900–1,000
3090	~1,400
4080 / 5080	~2,000
4090	~3,000–4,000
100K examples at ~1.5K tokens each is 150M tokens. That's about 13 hours on a 4090 or 30 on a 3090. The scheduled "days 5–12 main SFT" is one overnight run.

The from-scratch table's hypothetical 100 TFLOP/s is also about 3× what a 4090 delivers in practice and about 7× a 3090. The conclusion survives, but the numbers flatter the idea.

The actual bottleneck is teacher generation and verification. A 27B teacher at Q4 on a 24 GB card does tens of tok/s single-stream and a few hundred batched, so 50M teacher tokens is roughly 2–3 days before any filtering.

Restructure the schedule:

Days 1–7 are mostly data: generation, verification, eval set.
After that, run nightly train → morning eval loops, 5–8 iterations instead of one long run.
4. Drop raw-text continued pretraining on this corpus by default
Your corpus is DCLM, FineWeb-Edu, GitHub code, FineMath and Cosmopedia, which is exactly the distribution Qwen already pretrained on at trillion-token scale. At ~3.2k tok/s, the 3B-token arm takes about 11 days on a 4090. That's the whole budget spent re-teaching what the base already knows, and LoRA on raw text tends to erode chat behavior on an instruct checkpoint.

Keep Arm B only if the specialist domain has its own raw corpus. For forgetting mitigation, instruction-format replay beats raw-text replay. This dataset's real home is MiniFrontier from-scratch runs: 5B tokens is ~33 tokens/param for a 150M model.

5. Missing gotchas
Thinking mode. Both Qwen3 and Qwen3.5 are reasoning models. SFT on bare answers degrades the thinking path, so decide think/no-think per example and render the template accordingly. The Qwen3 template strips reasoning from earlier assistant turns, so multi-turn reasoning data needs per-turn samples to avoid a train/inference mismatch.
Assistant-only loss. TRL's assistant_only_loss depends on {% generation %} markers in the chat template, and the stock Qwen templates don't ship them. Expect to patch the template or build masks yourself, then decode the labels to verify.
Packing for SFT. Microbatch 1 without packing wastes most of the GPU. Use padding-free packing with position_ids / varlen attention, and cu_seqlens state resets for GDN layers.
LoRA learning rate. Thinking Machines' "LoRA Without Regret" found the optimal LoRA LR is roughly 10× the full fine-tuning LR, so 2e-4 is a likelier center than 1e-4. Drop lora_dropout. LoRA as the default is right given "retain general capability," since LoRA forgets less than full fine-tuning.
Teacher on a 16 GB card. A 27B Q4 teacher plus KV cache won't fit. Use Qwen3.6-35B-A3B with experts offloaded to system RAM, or Qwen3.5-9B, and verify the tokenizer hash matches the student.
API teachers. The document lists APIs as a teacher source without flagging that OpenAI, Anthropic and Google terms generally prohibit using outputs to build competing models. Local Apache or MIT teachers avoid the issue.
Runtime LoRA in llama.cpp. convert_lora_to_gguf.py plus --lora lets you A/B specialists without re-quantizing. Ship a merged model for the final release.
6. The actual blocker
There's still no task. The document flags this but schedules days 3–17 anyway. Make day 0 a hard gate: about 200 real inputs with gold outputs and a scoring metric. Without that, every other choice in the document is a guess, including student size, data mix and context length.

I can rewrite it as a single consolidated plan with these fixes once you name the task.


‫בתאריך יום ה׳, 17 בספט׳ 2026 ב-16:41 מאת ‪Igal Abachi‬‏ <‪igal.abachi.workmail@gmail.com‬‏>:‬

Igal Abachi
16:48 (לפני 0 דקות)
אני

You're right that "off the table" was too strong. From scratch on one RTX in under 18 days works fine. What doesn't work is 1–2B parameters on that compute: a from-scratch 1.5B trained in 18 days on one card ends up worse than a ~0.5–0.7B trained in the same 18 days, and it's slower to serve.

Why size, not "from scratch," is the problem
The compute you have fixes the tokens you can process, and at a fixed compute budget there's an optimal model size. These are my estimates for a 4090-class card: ~40% MFU, full training at 6N FLOPs/token, 14 training days, with the Chinchilla (Hoffmann et al.) loss fit used only to show the ordering. The absolute loss values don't transfer to your data or tokenizer.

Model	≈ tok/s	Tokens in 14 days	Tokens/param	Chinchilla-fit loss
1.5B	~3,700	~4.5B	3	2.81
680M	~8,000	~9.7B	14	2.75
480M	~11,400	~13.8B	29	2.74
300M	~18,000	~22B	73	2.75*
*Optimistic: your corpus is ~5B tokens, so 22B tokens means 4–5 epochs. Repetition is nearly free up to ~4 epochs (Muennighoff et al.) and degrades after that.

The 1.5B also has a memory problem. An fp32 master copy plus grads plus AdamW state is ~24 GB before activations, so on 24 GB it needs Muon or 8-bit optimizer state plus checkpointing, which costs another ~25% throughput. On a 16 GB card it's impractical.

On a 3090 or 16 GB card, effective compute is ~2–3× lower, so shift the sweet spot down to ~300–400M.

What from scratch can and can't win
Can: you own every weight and design decision. It can also be genuinely good at one narrow task, if that task's data goes into pretraining and SFT.
Can't: match Qwen3.5-0.8B on general benchmarks. That model has thousands of times more training compute behind it, and no architecture trick closes that gap.
If you want a from-scratch model closer to competitive, the only knob is compute. Two or three days on a rented 8×H100 node gives roughly 20× your single-RTX token budget. The RTX would then handle post-training and serving. Skip this if the single GPU is a hard rule.

Recommended design: MF-0.7B, trained with distillation from day one
Tokenizer	Qwen3.5, 248,320 vocab, tied embeddings
Body	24 layers, d=1280, 10 Q / 2 KV heads, head_dim 128, QK-norm, partial RoPE
FFN	SwiGLU, 3456
Params	≈413M body + 318M embedding ≈ 0.73B total
Optimizer	Muon on hidden 2D matrices, AdamW on everything else, WSD schedule
Sequence	2,048, extended to 4,096 in the decay phase
Q4_K_M	≈0.5 GB
Why use the teacher's tokenizer, even though the vocab eats ~43% of compute: Distillation scaling laws show distillation beats plain CE training when the teacher already exists and the student is compute-starved, which is exactly your regime. The shared tokenizer also enables the rest of the pipeline:

Pretraining KD from a Base checkpoint, which is better calibrated than instruct. Qwen3.5-2B has a Base checkpoint, which people are already using as a KD teacher.
SFT data and on-policy distillation later, from Qwen3.5-9B or Qwen3.6-27B running locally at Q4, all on the same vocab.
Attention layout: start with dense full attention in every layer. At 2–4K training sequences it's the fastest per token, and speed is the binding constraint here. The 3:1 GDN hybrid from your Modern preset becomes an ablation, not the default.

Go/no-go rule for KD: at ~120M scale on 1B tokens, compare KD+CE against CE-only with a 49k custom tokenizer. Compare at equal wall-clock including teacher cost. If KD doesn't win, drop it and use the 49k tokenizer; that frees about 280M params of embedding for the body. The ablation costs about a day, and it settles the biggest open question with evidence.

17-day schedule (one GPU, everything serial)
Days	Work
1	Tokenize a 1B ablation subset both ways (CPU); run the random-init → GGUF → llama.cpp logit round-trip test
2–3	Ablations at ~120M: KD vs CE, full attention vs 3:1 GDN. Retokenize the full corpus on CPU in parallel
4–5	Cache sparse teacher logits (~12 sampled per token) from Qwen3.5-2B-Base over the ~1.3B highest-quality tokens (FineWeb-Edu, FineMath, code). About 60 GB on disk
6–13	Main pretraining, ~4.5B tokens: KD+CE on cached shards (reused across epochs), CE on the rest. Put specialist raw data in the decay phase
14–15	Generate specialist SFT data with Qwen3.5-9B, then SFT with assistant-only loss
16	Short on-policy distillation pass if time allows; merge, imatrix, Q4_K_M, KLD against BF16
17	llama-server evals and contingency
GPU contention is the hidden constraint: the teacher cache, teacher generation, pretraining and SFT all share one card. Any day of teacher work is a day of pretraining lost, which is why the day-2–3 ablation has to justify KD.

Before any of this, measure real tok/s with your trainer on the actual card for 30 minutes. Every token count above scales linearly with that number, and consumer cards vary 3–4× within the 16–24 GB range.