#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" ]]; then
  VLLM_LAB_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  VLLM_LAB_SOURCE="${(%):-%x}"
else
  echo "Unsupported shell: use bash or zsh." >&2
  return 1 2>/dev/null || exit 1
fi

VLLM_LAB_DIR="$(cd "$(dirname "$VLLM_LAB_SOURCE")" && pwd)"

source "$VLLM_LAB_DIR/.venv/bin/activate"

export HF_HOME="$VLLM_LAB_DIR/.cache/huggingface"
export VLLM_CACHE_ROOT="$VLLM_LAB_DIR/.cache/vllm"
export XDG_CACHE_HOME="$VLLM_LAB_DIR/.cache"

echo "vLLM lab environment activated: $VLLM_LAB_DIR"
echo "Python: $(command -v python)"
echo "Model cache: $HF_HOME"
