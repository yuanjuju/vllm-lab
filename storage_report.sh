#!/usr/bin/env bash
set -euo pipefail

VLLM_LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for path in \
  "$VLLM_LAB_DIR/.venv" \
  "$VLLM_LAB_DIR/.cache/pip" \
  "$VLLM_LAB_DIR/.cache/huggingface" \
  "$VLLM_LAB_DIR/.cache/vllm"; do
  if [[ -e "$path" ]]; then
    du -sh "$path"
  fi
done

du -sh "$VLLM_LAB_DIR"
