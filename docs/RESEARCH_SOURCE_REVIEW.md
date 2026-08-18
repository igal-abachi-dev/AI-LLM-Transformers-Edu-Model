# Research source review

This review records the local research material used to audit MiniFrontier's frozen plan on
2026-08-18. The execution contract remains [`tasks/backlog.md`](../tasks/backlog.md), followed by
[`IMPLEMENTATION_DECISIONS.md`](IMPLEMENTATION_DECISIONS.md) and the normative portion of
[`plan.md`](../plan.md). `more-context.md` is a deliberation transcript containing mutually
exclusive proposals; it is evidence, not a specification. `muse-glimmer.md` is a dated reference
snapshot, not an instruction to reproduce a 30B multimodal production model.

## Source inventory and disposition

| Local source | SHA-256 | What it contributes | V1 disposition |
| --- | --- | --- | --- |
| `1409.0473v7.pdf` - *Neural Machine Translation by Jointly Learning to Align and Translate* | `84801c8410da51b449d379d2fa4939a416123f2c93991077a680f863026022a7` | Historical learned attention/alignment motivation | Background only; no encoder-decoder architecture |
| `1701.06538v1.pdf` - *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* | `2567713d198a376b65b92f0c60b9f58c8c5f66755530b92203533a17c4ec4b8e` | Conditional-compute and routing tradeoffs | MoE remains outside V1 |
| `1706.03762v7.pdf` - *Attention Is All You Need* | `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697` | Scaled dot-product attention, causal masking, heads, residual structure, tied output weights | Mathematical foundation; MiniFrontier deliberately uses later pre-RMSNorm/RoPE/SwiGLU choices |
| `1810.04805v2.pdf` - *BERT* | `5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e` | Pretraining/evaluation history | Encoder-only masked-LM design is not copied |
| `2005.14165v4.pdf` - *Language Models are Few-Shot Learners* | `97fd272f1fdfc18677462d0292f5fbf26ca86b4d1b485c2dba03269b643a0e83` | EOS-delimited packed documents, Adam-family recipe, global clipping, cosine/warmup, quality filtering, fuzzy deduplication, contamination analysis | Supports MF-045/MF-047 and the evaluation policy; its scale/results are not MiniFrontier claims |
| `2104.09864v5.pdf` - *RoFormer* | `e9a481fbe1c8a20b7b1fa566b13102a1896c7829fa9a8b4c80528452a5ddaf79` | Rotary Q/K position encoding and relative-position inner-product property | Implemented with an independent Llama-convention parity test |
| `2106.09685v2.pdf` - *LoRA* | `e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a` | Parameter-efficient adaptation | Not needed for the from-scratch V1 core or small full SFT |
| `2201.11903v6.pdf` - *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* | `7d9f878c23b460e4566aa4ec9201b1abfb3b8faefb2b1356e411cb90fef72a12` | Prompting/evaluation history at large scale | No architectural requirement and no reasoning-capability claim for 150M |
| `2205.14135v2.pdf` - *FlashAttention* | `ca7f9fda10b90fc05dd291a3accc85e9c1a4a860b99b31928dab03ed3fcb14e4` | Exact IO-aware attention and long-sequence memory motivation | Use first-party PyTorch fused paths; do not write custom kernels in V1 |
| `2302.13971v1.pdf` - *LLaMA* | `2e663675ae36ad12adb2f5a05281bac2747ecf8d23d92bedd9f937a89fee7136` | Pre-RMSNorm, SwiGLU, RoPE, AdamW/cosine/warmup/clipping recipe | Closest architectural foundation for Edu, without checkpoint compatibility claims |
| `2606.19348v1.pdf` - *DeepSeek-V4* | `55b2d72f772ac00de2e470b3ee08443c648d971c7f57c52d6202895665e5978d` | Million-token compression, hybrid attention, Muon, and system/algorithm co-design | CSA/HCA, mHC, MoE, million-token serving remain outside V1 |
| `k3_tech_report.pdf` - *Kimi K3* | `86fb82a63ced501f0c3f4f404c0c6fa88a7a6cfac17aae81fd1a8f455998067c` | KDA/MLA hybrid, stability mechanisms, per-head Muon, data-quality and independent hyperparameter-search lessons | Architecture is out of scope; fair optimizer/schedule tuning and data discipline remain relevant |
| `language_understanding_paper.pdf` - *Improving Language Understanding by Generative Pre-Training* | `eb81a65e0856bd38e855e1f1de6ee12e7c3eb92ae05c8270d345a475beb631d0` | Historical generative-pretraining/transfer evidence | Background only |
| `muse-glimmer.md` | `41f8b42d2d9a1aa37d7758680020e5b567d93d474c9cc43a0be8903cf7d99d8e` | Dense 3-local/1-global precedent, local RoPE/global NoPE, GQA, QK normalization, Transformers/vLLM/GGUF release paths | Motivates isolated Modern experiments and post-V1 adapters; extra query scaling, multimodality, and speculative decoding are not silently imported |
| `more-context.md` | `e16c48e8650376e50ef92120d6f24b4b50c6ace281d8d7cee47373ce9c92014c` | Planning alternatives, corrections, and review feedback | Non-normative; the final Edu-first decision and backlog win over earlier alternatives |

The local `2201.11903v6.pdf` must not be cited as the Chinchilla scaling paper. The relevant external
paper is Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv `2203.15556`.
The 3B-token canonical target is an evidence-informed project target and remains subject to the
measured MF-063 feasibility gate; it is not presented as a universal law or guaranteed quality
threshold.

## Cross-check conclusions

- The frozen Edu and Modern architectures are sound for the teaching goal. No reviewed source
  justifies adding MoE, MLA/KDA/CSA/HCA, multimodality, attention residuals, or speculative decoding
  to V1.
- The original Transformer and RoFormer math agree with the implemented scale, causal-mask, Q/K
  rotation, and shifted next-token objective. LLaMA supports the later pre-RMSNorm/RoPE/SwiGLU
  choices. The existing independent RoPE parity test remains essential because internal attention
  parity alone cannot prove the rotation convention.
- Muse Glimmer is a useful architectural analogy, not a matched ablation. MiniFrontier changes one
  flag at a time, keeps its own ratios, and measures hybrid efficiency only with a separately
  labeled 8K+ performance configuration. A 1K/2K teaching run may report semantics and cost but not
  long-context benefit.
- GPT-3's appendix reinforces that exact deduplication alone is insufficient for a release corpus.
  MF-047 must version near-duplicate and evaluation-contamination rules and preserve reason counts.
- Fused attention can be numerically backend-dependent. Exact control/data resume is distinct from
  bitwise CUDA math reproducibility; deterministic fixtures and production tolerance/metadata must
  be reported separately.
- Kimi K3's schedule and Muon comparisons reinforce the existing rule that optimizer arms require
  their own tuning. Fashionable defaults are hypotheses, not conclusions.
- Hugging Face hosting, Transformers loading, vLLM serving, high-precision GGUF, and four-bit GGUF
  remain separate compatibility claims with separate parity gates.

## Publication note

These local review copies retain their original rights. MF-068 must record a redistribution
decision for every file; the preferred public-repository form is a pinned upstream citation plus
the review hash above unless redistribution permission is explicit.
