# 权重量化：Qwen3-8B BF16 vs 官方 AWQ 4-bit（R570 / vLLM 0.18.0 探索实验，2026-10-09）

> **版本边界**：这是平台分配驱动 570.124.06 的单卡 RTX 4090 48GB 上，项目内 vLLM 0.18.0 / PyTorch 2.10.0+cu128 的探索对照，**不计作 vLLM 0.29.0 的 Phase 5 正式 B3 完成**。服务只监听云端 `127.0.0.1:8000`，客户端也在云端 loopback 测量。未变动 Mac、系统 Python 或推理 `.venv`。

## 为什么做、比的是什么

模型权重常驻显存。权重占用减少后，自动显存预算可能把腾出的空间分配给 KV Cache；这改变的是**可存储的上下文 token 数**，不自动等于吞吐翻倍或最大并发翻倍。AWQ 是带缩放/零点的 4-bit 权重量化，实际速度还取决于所用矩阵乘内核、Prefill/Decode 比例与负载。对照使用 [Qwen 官方 Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ)，其模型卡标明 AWQ 4-bit；vLLM 0.18.0 的[量化支持文档](https://docs.vllm.ai/en/v0.18.0/features/quantization/)列出 AWQ/Marlin 在 Ada 架构上的支持。

| 固定项 | 两组实际条件 |
| --- | --- |
| 实例与环境 | 同一 RTX 4090 48GB、驱动 570.124.06；`/root/fsas/vllm-lab/.venv-cu128-vllm018`，vLLM 0.18.0 / PyTorch 2.10.0+cu128 |
| 模型家族与协议 | Qwen3-8B；相同 tokenizer、tokenizer_config、generation_config 文件哈希；同一 served model ID `Qwen/Qwen3-8B`、相同 chat 请求和提示词 SHA-256 |
| 启动配置 | `--dtype bfloat16`、`--kv-cache-dtype auto`（此配置下实测池容量对应 BF16）、`max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`、`--enforce-eager`、关闭 Prefix Cache |
| 唯一处理因素 | 原始 BF16 权重、无量化内核 vs 官方 AWQ 4-bit 权重、`--quantization awq_marlin`；量化格式及其内核是一个不可拆开的部署变体 |
| 负载 | 每轮同时释放 8 个**独立**长请求，实际输入 3102–3105 tokens/条，固定输出 128 tokens/条；`temperature=0`、关闭 thinking；每组另有相同的 6 个质量探针 |
| 重复与指标 | 每组 1 轮预热 + 3 轮正式；逐请求 SSE 首内容 TTFT、整批输出吞吐、`/metrics` 的 Running/Waiting/KV 使用率、`num_preemptions`、队列时间与服务端 ITL；正式轮与预热分开 |

AWQ 的两个 safetensors 分片来自 Qwen 同名仓库，最终文件大小为 4,853,922,024 与 1,244,659,840 bytes；SHA-256 分别是 `6e112429856bc65e3837a9f38d6f6b71ffdda832cb46299a12f4fa8f6352516e` 与 `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff`，与[官方文件清单](https://huggingface.co/api/models/Qwen/Qwen3-8B-AWQ/tree/main?recursive=true)一致。云端直连 Hugging Face 入口被重置，先用项目内独立下载环境从 ModelScope 的 Qwen 同名仓库获取，慢速大分片再从官方 CDN 按区间续传；[恢复脚本](../../phase3/finish_official_awq_shard.sh)只在整片大小和 SHA-256 都正确后发布最终文件。下载中的临时分片在校验成功后已移除，模型最终文件保留。

## 复现顺序与验收

云端项目目录为 `/root/fsas/vllm-lab`。先检查 `nvidia-smi`、项目环境与最终模型分片哈希；不要在另一进程下载权重时测性能，也不要把服务公开到 `0.0.0.0`。两个终端分别启动服务与测量；启动后用 `/health` 和日志配置行验收，结束时停止服务并确认 8000 端口不可达、GPU 显存回落。

```bash
cd /root/fsas/vllm-lab
nvidia-smi --query-gpu=driver_version,memory.used,memory.total --format=csv,noheader
sha256sum models/Qwen3-8B-AWQ/model-0000{1,2}-of-00002.safetensors
bash phase3/run_cloud_r570_vllm018_weight_awq.sh bf16
```

服务就绪后，在另一个云端终端运行：

```bash
cd /root/fsas/vllm-lab
python3 phase3/weight_awq_ab_exploratory.py --variant bf16 --experiment-id r570awq_20261009
```

停止 BF16 服务、确认 GPU 释放后，将上面两条命令的 `bf16` 均替换为 `awq`；两组要用**相同** `experiment-id`，新一轮实验则换新 ID，避免覆盖 JSON。服务脚本、客户端与证据：[启动脚本](../../phase3/run_cloud_r570_vllm018_weight_awq.sh)、[测量脚本](../../phase3/weight_awq_ab_exploratory.py)、[BF16 原始 JSON](../../phase3/results/weight_awq_ab_r570awq_20261009_bf16.json)、[AWQ 原始 JSON](../../phase3/results/weight_awq_ab_r570awq_20261009_awq.json)、[BF16 启动日志](../../phase3/results/weight_awq_bf16_r570_v018_20261009.log)、[AWQ 启动日志](../../phase3/results/weight_awq_awq_r570_v018_20261009.log)。两组正式轮的对应提示词哈希完全相同；脚本依赖同目录已有的标准库指标函数，无新的推理环境依赖。归档日志只规范化了回车和行尾空白，云端 `logs/` 保留原始运行日志。

首次尝试显式 `--kv-cache-dtype bfloat16` 时，vLLM 0.18.0 在本节点的 FlashAttention cache-update 算子报 `Unsupported data type of kv cache: bfloat16`，**未进入正式测量**，见[失败启动日志](../../phase3/results/weight_awq_bf16_explicit_kv_failed_r570_v018_20261009.log)。改为 `auto` 并固定 `--dtype bfloat16` 后，两组成功启动；按模型 36 层、8 个 KV heads、128 head dimension、BF16 每元素 2 bytes，KV 理论为 `2×36×8×128×2=147456 bytes/token`，日志给出的两组池容量与 token 数正好对应此值。这同时是对 KV 实际精度的交叉核验，不靠命令行字面推断。

## 原始结果

启动日志给出模型加载显存 **15.27 → 5.71 GiB**，自动 KV 池 **19.76 → 29.32 GiB**，GPU KV 容量 **143,872 → 213,488 tokens**。权重加载显存减少约 9.56 GiB，恰好表现为 KV 池增加约 9.56 GiB；KV 容量增加 **48.4%**。运行中 `nvidia-smi` 约 37,110 / 37,126 MiB，整卡占用几乎不变，因为两组都让 vLLM 使用同一个 0.75 自动预算。权重省下来的显存是**重新分配给 KV**，不是宣称整卡显存下降 9.56 GiB。

| 权重 | 正式轮 | TTFT P50 / P95 (s) | 整批 (s) | 输出吞吐 (tok/s) | 服务端 ITL 均值 (s) | 队列时间均值 (s) | KV 峰值 | Waiting 峰值 | 抢占增量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1.7615 / 2.4605 | 5.3602 | 191.037 | 0.0288 | 0.4321 | 17.97% | 4 | 0 |
| BF16 | 2 | 1.8136 / 2.5169 | 5.4264 | 188.707 | 0.0286 | 0.5707 | 17.97% | 5 | 0 |
| BF16 | 3 | 1.7986 / 2.5027 | 5.4023 | 189.550 | 0.0288 | 0.5720 | 17.97% | 5 | 0 |
| AWQ | 1 | 1.7160 / 2.3807 | 4.9995 | 204.821 | 0.0265 | 0.5497 | 12.11% | 5 | 0 |
| AWQ | 2 | 1.7369 / 2.4031 | 5.0120 | 204.312 | 0.0265 | 0.5505 | 12.11% | 5 | 0 |
| AWQ | 3 | 1.5870 / 2.3145 | 5.0156 | 204.162 | 0.0268 | 0.3854 | 12.11% | 4 | 0 |

三次正式轮中位数：输出吞吐 **189.550 → 204.312 tok/s（+7.8%）**，整批完成时间 **5.4023 → 5.0120 s**；客户端 TTFT P50 **1.7986 → 1.7160 s**，P95 **2.5027 → 2.3807 s**；服务端 ITL 均值 **0.0288 → 0.0265 s**。BF16 与 AWQ 各 **24/24 正式请求成功**，每条 128 输出 tokens；相同 6 个小型质量探针两组均 **6/6 精确匹配**。这里吞吐定义为 `1024 个成功输出 tokens / 整批耗时`，含排队和 Prefill；TTFT 从请求开始到首个非空 SSE 内容事件，含 HTTP/排队，不是纯 GPU Prefill。服务端 ITL 来自 `/metrics` 直方图增量 `Δsum/Δcount`，**不把 SSE 内容事件间隔冒充逐 token ITL**。

## 如何解释与局限

1. **容量变化有直接证据**：同样 BF16 KV 每 token 144 KiB，权重变小使自动 KV 池从 19.76 增至 29.32 GiB，token 容量随之增加 48.4%。同一负载的 KV 使用率从 17.97% 降至 12.11%，主要是**分母变大**；实际占用 token 数约相同，不是请求突然少用了 KV。`max-num-seqs=8` 和每请求 4096 上限未变，因此不能从池容量直接推断实用并发增加 48.4%。
2. **速度收益是这组负载的观察值**：三轮 AWQ 吞吐均高于三轮 BF16，服务端 ITL 均值也较低，支持此实例在“长输入、8 并发、128 输出、eager、关闭 Prefix Cache”的负载下 AWQ-Marlin 略快。它并不证明短请求、长输出、CUDA Graph、其他内核或其他驱动同样获益，更不能按 4-bit 比 16-bit 推出 4 倍速度。
3. **排队不是抢占**：两组 Running 峰值均为 8、Waiting 峰值 4–5，但正式轮的 `num_preemptions` 增量全为 0；KV 峰值仅 17.97% / 12.11%。这些 Waiting 不能解释为 KV 枯竭，长输入在 4096-token 单轮预算下的调度分批仍会引入排队。此实验并未测出 AWQ 的最大并发容量。
4. **质量只做冒烟检查**：6 个确定性题目包含四个短题和两个约 3.1k-token 长文验证码回忆；6/6 只能排除明显失效，不能证明量化后的知识、推理、生成风格或长上下文质量等价。`temperature=0` 有利于对照，但不是完整质量评估。
5. **统计与版本边界**：正式每轮只有 8 个请求、仅 3 轮且顺序固定为 BF16→AWQ；轮内 P95 只是描述性尾部数。云端本机 loopback 不含 Mac↔云网络。vLLM 0.18.0/R570 结果与旧 vLLM 0.29.0 或旧实例结果必须分开存放；待平台有兼容驱动，再按同协议补做 v0.29.0 正式 B3。

两组服务均已停止，`/health` 不可达，GPU 显存回到约 2 MiB。下载临时大分片和四个区间文件在最终权重 SHA-256 校验通过后被移除，保留的是完整 AWQ 模型。**停止 vLLM 不等于云实例关机；用户仍需在云平台控制台确认实例停止计费。**
