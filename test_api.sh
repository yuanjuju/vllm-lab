#!/usr/bin/env bash
set -euo pipefail

curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "请用一句简短的中文说明什么是大语言模型推理。"
      }
    ],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'

printf '\n'
