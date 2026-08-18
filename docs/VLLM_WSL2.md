# vLLM on a Windows 11 NVIDIA host

MiniFrontier targets vLLM inside WSL2 Linux, not native Win32. The checked-in Transformers model
uses `auto_map.AutoModel`, forwards model kwargs through every decoder layer, calls
`ALL_ATTENTION_FUNCTIONS`, advertises `_supports_attention_backend`, and declares the interleaved
`sliding_attention`/`full_attention` layer types needed by vLLM's Transformers backend.

## Export

Export a load-tested repository from each canonical native release:

```powershell
uv run --extra cpu python scripts/export_huggingface.py `
  --release artifacts/minifrontier-150m-modern `
  --output artifacts/minifrontier-150m-modern-hf `
  --source-revision <full-git-commit> `
  --report reports/mf071-modern.json
```

Review `configuration_minifrontier.py` and `modeling_minifrontier.py` before enabling remote code.
Upload with a pinned Hub revision; hosting without the local/Hub parity reports is not compatibility.

## WSL2 environment and serving

Inside an Ubuntu WSL2 environment with NVIDIA CUDA visible, create a separate Python 3.12 vLLM
environment. Pin the exact vLLM wheel or commit in the evidence record; do not install it into the
native MiniFrontier training environment.

```bash
nvidia-smi
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install 'vllm==<pinned-version>'
bash scripts/serve_vllm_wsl2.sh <hub-id-or-local-export> <pinned-model-revision> minifrontier-150m-modern
```

The equivalent server command uses:

```bash
vllm serve <model> --revision <revision> --model-impl transformers \
  --trust-remote-code --dtype bfloat16 --api-key local-key \
  --served-model-name minifrontier-150m-modern
```

Create a native reference fixture before starting the server, then exercise both raw completion and
chat transport:

```powershell
uv run --extra cpu python scripts/create_serving_fixture.py --release <native-release> `
  --prompt "The capital of France is" --output reports/native-serving-fixture.json

uv run --extra cpu python scripts/smoke_vllm_api.py `
  --model minifrontier-150m-modern --parity-fixture reports/native-serving-fixture.json `
  --output reports/vllm-api-smoke.json
```

MF-072 closes only after the real server records BF16 prefill/decode parity, prompt logprobs, greedy
continuations, cache behavior, VRAM, TTFT, and decode throughput for Edu and Modern. OpenAI transport
success is distinct from code-edit quality. Tool/function calling and native Win32 vLLM remain
unsupported.

