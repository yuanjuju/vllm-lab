#!/usr/bin/env bash
set -euo pipefail

# Isolated R570/vLLM 0.18.0 comparison: only the N-gram speculation switch
# changes. Both variants use the same Qwen3-8B BF16 target and BF16 KV cache.
if [[ $# -ne 1 || ( "$1" != baseline && "$1" != ngram ) ]]; then
  echo "Usage: $0 baseline|ngram" >&2
  exit 2
fi

lab_dir=/root/fsas/vllm-lab
python_env="$lab_dir/.venv-cu128-vllm018"
model_dir="$lab_dir/models/Qwen3-8B"
spec_args=()
if [[ "$1" == ngram ]]; then
  spec_args=(--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":1,"prompt_lookup_max":4}')
fi

if [[ ! -x "$python_env/bin/vllm" || ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "Missing project-local vLLM or Qwen3-8B model" >&2
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
  --kv-cache-dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --enforce-eager \
  "${spec_args[@]}"
