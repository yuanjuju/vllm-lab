#!/usr/bin/env bash
set -euo pipefail

# Single-instance vLLM 0.29.0 Phase 5 comparisons on a compatible CUDA host.
# Keep the model, project caches, and Python environment inside vllm-lab.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 base|prefix_on|kv_auto|kv_fp8|awq|ngram [eager|graph]" >&2
  exit 2
fi

lab_dir=/root/fsas/vllm-lab
python_env="$lab_dir/.venv"
model_dir="$lab_dir/models/Qwen3-8B"
variant=$1
execution_mode=${2:-eager}
kv_dtype=auto
extra_args=()
execution_args=()
case "$execution_mode" in
  eager)
    execution_args=(--enforce-eager)
    ;;
  graph)
    ;;
  *)
    echo "Unknown execution mode: $execution_mode" >&2
    exit 2
    ;;
esac
case "$variant" in
  base)
    extra_args=(--no-enable-prefix-caching)
    ;;
  prefix_on)
    ;;
  kv_auto|kv_fp8)
    extra_args=(--no-enable-prefix-caching --kv-cache-memory-bytes 4G)
    if [[ "$variant" == kv_fp8 ]]; then
      kv_dtype=fp8
    fi
    ;;
  awq)
    model_dir="$lab_dir/models/Qwen3-8B-AWQ"
    extra_args=(--no-enable-prefix-caching --quantization awq_marlin)
    ;;
  ngram)
    extra_args=(--no-enable-prefix-caching --speculative-config
                '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":1,"prompt_lookup_max":4}')
    ;;
  *)
    echo "Unknown variant: $variant" >&2
    exit 2
    ;;
esac

if [[ ! -x "$python_env/bin/vllm" || ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "Missing project-local vLLM or complete model: $model_dir" >&2
  exit 1
fi
mkdir -p "$lab_dir/cache/vllm-v029" "$lab_dir/cache/huggingface" "$lab_dir/tmp" "$lab_dir/logs"
export HF_HOME="$lab_dir/cache/huggingface"
export VLLM_CACHE_ROOT="$lab_dir/cache/vllm-v029"
export TMPDIR="$lab_dir/tmp"
export TOKENIZERS_PARALLELISM=false

exec "$python_env/bin/vllm" serve "$model_dir" \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --kv-cache-dtype "$kv_dtype" \
  --max-model-len 4096 --gpu-memory-utilization 0.75 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  "${execution_args[@]}" "${extra_args[@]}"
