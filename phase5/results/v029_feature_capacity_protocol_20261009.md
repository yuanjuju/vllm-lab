# vLLM 0.29.0 特性与容量复核：预注册协议（2026-10-09）

本文件先固定实验条件与验收口径；结果只在真实运行并保存原始数据后补入独立报告。旧 R570/v0.18.0 结果不并入本组。

## 环境与共同条件

- 云端单张 RTX 4090 49140 MiB；启动前记录驱动、vLLM、PyTorch/CUDA、模型目录及 GPU 空闲状态。
- 项目内 `.venv`，Qwen3-8B BF16 目标模型；AWQ 组使用官方 Qwen3-8B-AWQ 权重。服务只监听 `127.0.0.1:8000`；客户端运行在云端 loopback。
- 除实验变量外：`max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`、`--enforce-eager`、BF16 激活、固定输出长度；服务启动日志是配置证据。
- 每组单独启动服务，先独立预热，再至少 3 次正式运行。原始 JSON、启动日志与失败记录均保留；中位数、P50/P95、min/max 与跨次波动从正式运行计算，不与预热合并。

## 单变量配对

| 对照 | 控制组 | 实验组 | 主要要回答的问题 |
| --- | --- | --- | --- |
| Prefix Cache | `base`：关闭 | `prefix_on`：开启 | 同一共享前缀下命中率、TTFT、吞吐变化；改变前缀作为负对照 |
| KV dtype | `kv_auto`：固定 4 GiB KV 池、BF16 | `kv_fp8`：同样固定 4 GiB、FP8 | 在相同 KV 字节预算下，容量、延迟、吞吐与基础质量变化 |
| 权重量化 | `base`：BF16 权重 | `awq`：官方 AWQ 4-bit 权重 | 模型占用释放多少显存，KV 池与性能、冒烟质量如何变化 |
| N-gram | `base`：无投机 | `ngram`：`num_speculative_tokens=4` | 高/低接受率与混合负载的效果和代价 |

KV dtype 对照不沿用旧版 `--calculate-kv-scales`：v0.29.0 的参数表已无此项。若 FP8 日志提示使用未校准的 1.0 scale，质量结论必须按该限制解释，不能声称 FP8 普遍无损。

## 容量可信度复核

在 `base` 与 `ngram` 两种服务配置上，用相同 50:50 重复抄写/连续数字合成负载，等间隔 open-loop 到达；每请求固定 256 输出 token，关闭 Prefix Cache。计划考察 1.6、1.8、2.0 req/s，每档 10 秒预热 + 3 次 60 秒正式窗口；每个档位先确认客户端实际派发量与延迟，再判断是否继续升档。示例 SLO：成功、TTFT ≤1 秒且 E2E ≤10 秒，同时看每轮 TTFT P95≤1 秒。它只用于本实验教学，不是生产承诺。

同时记录 Running、Waiting、KV 使用率、preemption 增量、服务端队列时间和 TTFT/ITL histogram、客户端 TTFT/E2E P50/P95、输出吞吐、SLO goodput、成功率及到达时间偏差。KV≥75%、Waiting>16 或发生抢占时停止后续派发；任何失败或 drain 超时则不升下一档。2.0 req/s 仍只是本次网格上限；即使通过也不能称为长期最大容量。

## 结束条件

每次切换配置前先停当前服务，确认 `/health` 不可达与 GPU 显存释放；全部结束后在平台控制台确认实例停止计费。密码和访问令牌不入库。
