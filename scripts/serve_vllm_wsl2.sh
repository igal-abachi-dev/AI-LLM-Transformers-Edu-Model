#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: serve_vllm_wsl2.sh MODEL_OR_PATH PINNED_REVISION [SERVED_NAME]" >&2
  exit 2
fi

model="$1"
revision="$2"
served_name="${3:-minifrontier}"

exec vllm serve "$model" \
  --revision "$revision" \
  --served-model-name "$served_name" \
  --model-impl transformers \
  --trust-remote-code \
  --dtype bfloat16 \
  --api-key local-key \
  --chat-template-content-format string
