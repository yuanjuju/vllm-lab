#!/usr/bin/env bash
set -euo pipefail

VLLM_LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$VLLM_LAB_DIR/env.sh"

exec vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.50
