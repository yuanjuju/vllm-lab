#!/usr/bin/env bash
set -euo pipefail

# Temporary CUDA 12.8 service for R570 hosts. Keep the vLLM 0.29.0 .venv intact.
lab_dir=/root/fsas/vllm-lab
python_env="$lab_dir/.venv-cu128-vllm018"
model_dir="$lab_dir/models/Qwen3-8B"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--no-enable-prefix-caching" ) ]]; then
  echo "Usage: $0 [--no-enable-prefix-caching]" >&2
  exit 2
fi

if [[ ! -x "$python_env/bin/vllm" ]]; then
  echo "Missing temporary vLLM environment: $python_env" >&2
  exit 1
fi
if [[ ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "Missing Qwen3-8B model index: $model_dir" >&2
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
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  "$@"
