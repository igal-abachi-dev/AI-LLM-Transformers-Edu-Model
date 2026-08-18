# Ecosystem compatibility

This document separates transport compatibility from model, capability, and quality compatibility.
It is intentionally stricter than saying that an HTTP request returned `200 OK`.

## Current status

| Target | Current status | Gate that earns the claim |
| --- | --- | --- |
| Native MiniFrontier/PyTorch | Implemented and CPU load-tested | MF-067 publishes the real canonical weights |
| Hugging Face file hosting | Technically possible, but not a useful model release yet | MF-064–067 produce trained, documented weights |
| Transformers local repository | Implemented and parity-tested on tiny Edu/Modern fixtures | MF-071 canonical exports plus pinned-Hub clean-environment parity |
| vLLM adapter contract | Implemented through the Transformers modeling interface | MF-072 under Windows 11 + WSL2 CUDA |
| Vercel AI SDK, OpenCode, Cline, Roo Code, Kilo Code, Aider | Transport examples/harness implemented; real vLLM smoke unmeasured | MF-072 OpenAI-compatible client smokes |
| Tool/function calling | Outside V1 and not claimed | A later trained tool protocol plus parser and end-to-end evaluations |
| llama.cpp / high-precision GGUF | Fail-closed orchestration implemented; upstream graph support not implemented | MF-073 |
| 4-bit GGUF | Q4_K_M candidate/provenance runner implemented; artifacts/evaluation unmeasured | MF-074 |

The files currently under `artifacts/` are one-update CPU engineering smokes. They prove plumbing,
checkpointing, and data integration; they are not trained language models suitable for publication or
interactive use.

## Intended serving path

```text
canonical trained MiniFrontier checkpoint
  -> MF-067 safe native release
  -> MF-071 Transformers/Hugging Face adapter
  -> MF-072 vLLM under WSL2 CUDA
  -> OpenAI-compatible /v1/chat/completions
  -> coding client configured with the MiniFrontier model ID
```

vLLM's Chat Completions endpoint requires a chat template. MiniFrontier already has a deterministic
template, but MF-071 must embed it in the Transformers tokenizer metadata and MF-072 must prove that
vLLM applies it identically to the native implementation. vLLM does not support native Windows; the
supported project target is Windows 11 hosting WSL2 Linux with NVIDIA CUDA.

See [`VLLM_WSL2.md`](VLLM_WSL2.md) for the pinned export/serve/parity workflow and
[`GGUF_LLAMA_CPP.md`](GGUF_LLAMA_CPP.md) for the separate llama.cpp architecture gate.

## Coding-client expectations

Vercel AI SDK, OpenCode, Cline, Kilo Code, and Aider document custom OpenAI-compatible endpoints. Roo Code also
supports local/OpenAI-emulating providers, but its agent behavior depends strongly on model format
following. Once MF-072 passes, plain text chat/completion transport should work by selecting the
custom endpoint, model ID, and the real MiniFrontier context limit.

That does **not** make MiniFrontier a useful coding agent:

- V1 intentionally has no tool/function-call schema or tool-use SFT.
- The canonical 150M model is a teaching-scale model, not comparable to the large coding models these
  clients normally use.
- The 1K/2K V1 contexts are too small for substantial repository maps, tool transcripts, and long
  edit loops.
- Aider may be tried with a conservative whole-file edit format after MF-072, but reliable code edits
  are a model-quality result that must be measured rather than inferred from API compatibility.
- Client configurations must advertise `tools: false`, text-only input/output, and the actual context
  limit. They must not cause the client to send unsupported tool calls.

MF-072 therefore tests ordinary system/user/assistant chat with each client separately and records
unsupported agent features. Tool calling remains a distinct post-V1 capability decision.

## System prompt

Chat serialization does not technically require a system message: the template accepts a conversation
that begins with `user`. For a released SFT assistant, a short stable system prompt is recommended so
identity, honesty, coding style, and capability boundaries are consistent. The checked-in
`templates/system_prompt.md` is the default for `scripts/chat.py`, is copied into release artifacts,
and can be overridden with `--system` or `--system-file`, or disabled with `--no-system`.

The prompt is deliberately short. Large agent prompts from prompt archives are useful design examples,
but they describe environment-specific tools and consume a material fraction of MiniFrontier's small
context. V1 does not pretend to have tools it was not given.

## Upstream references

- vLLM OpenAI-compatible server: <https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/>
- vLLM custom model requirements: <https://docs.vllm.ai/en/latest/models/supported_models/>
- vLLM GPU/Windows support: <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- Vercel AI SDK OpenAI-compatible provider: <https://ai-sdk.dev/providers/openai-compatible-providers>
- Vercel AI SDK tool loops: <https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling>
- OpenCode providers: <https://opencode.ai/docs/providers/>
- Cline OpenAI-compatible provider: <https://github.com/cline/cline/blob/main/docs/provider-config/openai-compatible.mdx>
- Roo Code local models: <https://github.com/RooCodeInc/Roo-Code-Docs/blob/main/docs/advanced-usage/local-models.md>
- Kilo Code custom models: <https://github.com/Kilo-Org/kilocode/blob/main/packages/kilo-docs/pages/code-with-ai/agents/custom-models.md>
- Aider OpenAI-compatible APIs: <https://aider.chat/docs/llms/openai-compat.html>
- Prompt examples reviewed for synthesis: <https://github.com/elder-plinius/CL4R1T4S>
