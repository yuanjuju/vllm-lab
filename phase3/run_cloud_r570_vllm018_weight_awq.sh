#!/usr/bin/env bash
set -euo pipefail

# Weight-only comparison. The AWQ model is served through the BF16-capable
# AWQ-Marlin kernel; both variants keep BF16 activations. With --dtype BF16,
# kv-cache-dtype auto resolves to BF16 for this vLLM version. Explicit
# bfloat16 KV dtype failed in the FlashAttention cache-update op on R570.
if [[ $# -ne 1 || ( "$1" != bf16 && "$1" != awq ) ]]; then
  echo "Usage: $0 bf16|awq" >&2
  exit 2
fi

lab_dir=/root/fsas/vllm-lab
python_env="$lab_dir/.venv-cu128-vllm018"
model_dir="$lab_dir/models/Qwen3-8B"
quant_args=()
if [[ "$1" == awq ]]; then
  model_dir="$lab_dir/models/Qwen3-8B-AWQ"
  quant_args=(--quantization awq_marlin)
fi

if [[ ! -x "$python_env/bin/vllm" || ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "Missing project-local vLLM environment or complete model: $model_dir" >&2
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
  "${quant_args[@]}"
