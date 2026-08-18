# MiniFrontier llama.cpp upstream adapter contract

`minifrontier-architecture.json` is the reviewed tensor, metadata, tokenizer, and graph contract for
the future pinned llama.cpp patch. It is not itself a loader or compute graph.

The implementation must follow llama.cpp's `docs/development/HOWTO-add-model.md`, add a distinct
`minifrontier` architecture in Python and C++, and pass high-precision native-reference parity before
quantization. `scripts/convert_gguf.py` deliberately checks for those upstream markers and refuses to
run against an unmodified checkout.

`upstream.json` pins the checkout used to review the current converter and graph layout. It is a
review baseline, not a claim that the unmodified revision supports MiniFrontier. Current upstream
places model converters in `conversion/<model>.py` and compute graphs in `src/models/<model>.cpp`.

Do not register `MiniFrontierForCausalLM` as Llama. Do not omit QK-Norm, local windows, the per-layer
RoPE/NoPE policy, or the reserved chat/FIM tokens to get a file that merely loads.
