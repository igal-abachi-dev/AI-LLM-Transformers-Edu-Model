# GGUF and llama.cpp integration

MiniFrontier must never be labeled `llama` merely to make conversion run. Edu is close to a Llama
decoder, but Modern adds per-head QK-Norm, interleaved local/global attention, and optional global
NoPE. Those semantics require a real `minifrontier` architecture in llama.cpp.

The upstream integration gate follows llama.cpp's own new-model process:

1. register MiniFrontier in GGUF constants and tensor mappings;
2. add a converter for the Transformers tensor/config/tokenizer metadata;
3. add `llm_arch`/tensor-name/metadata/RoPE handling;
4. implement the compute graph, including QK-Norm and each layer's local/global and RoPE/NoPE policy;
5. build CPU and CUDA targets and prove native-reference parity with `llama-cli` and `llama-server`.

`scripts/convert_gguf.py` fails closed unless a pinned llama.cpp checkout contains MiniFrontier
markers in its converter, GGUF constants, architecture registry, and graph. It then runs only F16 or
BF16 conversion, checks GGUF magic, and records all hashes. The current repository does not contain
or claim a completed upstream C++ patch; MF-073 remains open until that code and runtime evidence
exist.

The reviewed upstream baseline is recorded in
[`adapters/llama_cpp/upstream.json`](../adapters/llama_cpp/upstream.json). The closest reusable pieces
are Qwen3's GQA/QK-Norm/SwiGLU graph and the mixed sliding/full-attention infrastructure used by
Gemma3 and Muse Glimmer. Neither graph is an exact alias: MiniFrontier needs its own pre-norm residual
graph, split-half RoPE, optional QK-Norm for Edu/Modern, and the per-layer global RoPE/NoPE flag.

Only after MF-073 passes may `scripts/quantize_gguf.py` create a `Q4_K_M` candidate. It refuses
requantization, validates optional calibration provenance, and deliberately writes
`publish_ready=false`. MF-074 additionally requires:

- native Windows NVIDIA CUDA `llama-cli` and `llama-server` smokes;
- size and peak RAM/VRAM;
- prompt/decode throughput;
- validation/perplexity delta and fixed greedy changes;
- language, functional-code, FIM, tokenizer, and chat-template regression results;
- pinned quantizer/runtime/model revisions and SHA-256 hashes;
- an uncalibrated comparison when an importance matrix is used.

Loadability or a four-bit filename is not a quality result.
