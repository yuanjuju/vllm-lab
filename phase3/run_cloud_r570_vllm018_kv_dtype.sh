#!/usr/bin/env bash
set -euo pipefail

# Controlled KV dtype comparison on the R570 / CUDA 12.8 environment.
# Only the first argument changes between the two runs.
if [[ $# -ne 1 || ( "$1" != auto && "$1" != fp8 ) ]]; then
  echo "Usage: $0 auto|fp8" >&2
  exit 2
fi

lab_dir=/root/fsas/vllm-lab
python_env="$lab_dir/.venv-cu128-vllm018"
model_dir="$lab_dir/models/Qwen3-8B"

if [[ ! -x "$python_env/bin/vllm" || ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "Missing project-local vLLM environment or model" >&2
  exit 1
fi

mkdir -p "$lab_dir/cache/vllm-cu128" "$lab_dir/tmp"
export HF_HOME="$lab_dir/cache/huggingface"
export VLLM_CACHE_ROOT="$lab_dir/cache/vllm-cu128"
export TMPDIR="$lab_dir/tmp"
export TOKENIZERS_PARALLELISM=false

exec "$python_env/bin/vllm" serve "$model_dir" \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes 4G \
  --no-enable-prefix-caching \
  --calculate-kv-scales \
  --enforce-eager \
  --kv-cache-dtype "$1"
