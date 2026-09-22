# vLLM 云端阶段：首次远程流式请求

记录日期：2026-09-21

## 环境与链路

- 客户端：MacBook Air M5，通过 SSH 本地转发访问 `127.0.0.1:18000`。
- 服务端：Vylai 单卡 RTX 4090，约 48 GB 显存；驱动 `590.44.01`。
- 云端环境：`/root/fsas/vllm-lab/.venv`；vLLM `0.29.0`；PyTorch `2.13.0+cu132`。
- 模型：`Qwen/Qwen3-8B`，存于 `/root/fsas/vllm-lab/models/Qwen3-8B`。
- 启动配置：BF16，`max_model_len=4096`，`max_num_seqs=4`，`gpu_memory_utilization=0.75`。
- 云端 API 仅监听 `127.0.0.1:8000`；本次通过 SSH 隧道访问，未直接向公网开放。
- 云盘：`/root/fsas`，80 GB；模型权重约 16 GB。

## 验收

- 云端 `/v1/models` 返回模型 ID `Qwen/Qwen3-8B`。
- Mac 的 `127.0.0.1:18000` 由 `ssh` 监听，转发后的 `/v1/models` 返回同一模型。
- Mac 经隧道发起的聊天请求得到正确答案 `42`，`finish_reason=stop`。

## 首次远程流式请求

原始输出：[cloud_stream_20260921-140745.txt](cloud_stream_20260921-140745.txt)

提示词：`请用约100字解释什么是大语言模型推理服务，以及它与训练的区别。不要使用列表。`

参数：`temperature=0`，`max_tokens=160`，`stream=true`，`stream_options.include_usage=true`，`enable_thinking=false`。

| 指标 | 结果 |
|---|---:|
| TTFT（Mac 到首个非空内容块） | 0.9299 秒 |
| 总响应时间 | 2.0052 秒 |
| 内容块数量 | 64 |
| prompt tokens | 37 |
| completion tokens | 65 |
| 首 token 后近似输出速度 | 59.52 tokens/s |
| `finish_reason` | `stop` |
| 收到 `[DONE]` | 是 |

TTFT 和总时间包含 Mac、SSH 隧道与云端服务端开销；这里的速度采用 `(completion_tokens - 1) / (总时间 - TTFT)` 估算。单次请求包含可能的预热成本，不能代表稳定性能。Mac 第二阶段使用的是 Qwen3-0.6B，而这里是 Qwen3-8B，也不能把两者的 tokens/s 视为同模型硬件跑分。

随后预热 2 次并重复 20 次远程流式请求：20/20 成功，TTFT P50 为 0.136 s、P95 为 0.187 s，总耗时 P50 为 1.214 s。逐请求记录与汇总见 [CSV](cloud_20run_20260921-141120.csv) 和 [JSON](cloud_20run_summary_20260921-141120.json)。
