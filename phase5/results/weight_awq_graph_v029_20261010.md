# A2：默认执行模式下 BF16 与 AWQ 权重对照（2026-10-10）

## 结论

在**本次容器、vLLM 0.29.0、RTX 4090、默认 torch.compile + CUDA Graph、约 3100-token 输入 / 固定 128-token 输出 / 8 请求同步突发**下，AWQ 4-bit 相对 BF16 的三次正式轮输出吞吐中位数为 **272.364 vs 193.214 tok/s（+40.96%）**。服务端 ITL 均值中位数从 28.3 降至 17.3 ms（−38.9%），整批耗时中位数从 5.2998 降至 3.7597 s（−29.1%）。因此，先前 [eager 模式下 AWQ −6.0%](weight_awq_v029_20261009.md) 的吞吐方向**没有在本次默认模式配对中延续**。

这只回答了两种权重在本次默认执行模式下的配对差异。`--enforce-eager` 同时关闭 torch.compile 与 CUDA Graph，且旧 eager 与本轮不是同一容器内完成的四格配对；**不能把方向翻转单独归因于 CUDA Graph、torch.compile 或某个 kernel**。

## 做了什么

按[预注册协议](weight_awq_graph_protocol_20261010.md)，先确认 SSH、GPU、驱动、CUDA、项目 `.venv`、两套模型、服务和显存状态。当前容器主机名为 `positive-cedar-1105-cd5d67b6-hczk2`，GPU UUID 为 `GPU-5faea9f4-6d37-d847-e664-e9d2d4ab4fe3`，RTX 4090 49140 MiB、驱动 590.44.01；PyTorch 2.13.0+cu132（torch CUDA 13.2）、vLLM 0.29.0。`nvidia-smi` 报告的驱动 CUDA 兼容版本为 13.1；容器未安装 `nvcc`。开始时 GPU 占用 0 MiB，`/health` 不可达。[环境原始记录](../../phase3/results/phase5_v029_graph_env_20261010.txt)

仓库的[服务启动脚本](../../phase3/run_cloud_vllm029_features.sh)增加可选 `graph` 模式；原有单参数调用仍为 eager。先后启动 `base graph` 与 `awq graph`，每组分别通过 `/health`、`/v1/models` 和真实 Chat Completions 请求检查。两组均使用 `127.0.0.1:8000` loopback、`--dtype bfloat16 --kv-cache-dtype auto --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-seqs 8 --max-num-batched-tokens 4096 --no-enable-prefix-caching`，均不传 `--enforce-eager`；AWQ 组只改模型目录并追加 `--quantization awq_marlin`。

服务[BF16 日志](../../phase3/results/phase5_v029_graph_base_20261010.log)与[AWQ 日志](../../phase3/results/phase5_v029_graph_awq_20261010.log)均显示 `enforce_eager=False`、`VLLM_COMPILE`、`FULL_AND_PIECEWISE`、torch.compile 执行完成及 PIECEWISE/FULL CUDA Graph 捕获完成。客户端在云端运行相同脚本和 `experiment-id=v029awqgraph_20261010`：各组先做 6 题质量冒烟、1 次预热，随后做 3 次正式轮；每轮 8 条同步请求，`temperature=0`、关闭 thinking、输出固定 128 tokens。质量题和预热均不进入性能中位数。切换组别前停止 BF16 服务并确认显存为 0 MiB。

## 原始结果

| 指标（三次正式轮） | BF16 | AWQ 4-bit | AWQ 相对 BF16 |
| --- | ---: | ---: | ---: |
| 输出吞吐逐轮 tok/s | 194.769 / 193.214 / 190.920 | 272.364 / 272.370 / 271.654 | 各轮均更高 |
| **输出吞吐中位数 tok/s** | **193.214** | **272.364** | **+40.96%** |
| 整批耗时中位数 s | 5.2998 | 3.7597 | −29.1% |
| 客户端 TTFT P50 中位数 s | 1.8085 | 1.7241 | −4.7% |
| 客户端 TTFT P95 中位数 s | 2.5380 | 2.4133 | −4.9% |
| 服务端 ITL 均值中位数 ms | 28.3 | 17.3 | −38.9% |
| 服务端队列时间均值中位数 s | 0.4461 | 0.4923 | +10.4% |
| 正式轮成功 / 输出长度 | 24/24，均为 128 tokens | 24/24，均为 128 tokens | 相同 |
| 正式轮 preemption 增量 / 指标轮询错误 | 每轮 0 / 0 | 每轮 0 / 0 | 相同 |

两组质量冒烟均为 **6/6 精确匹配**；六题 prompt SHA-256 逐组相同，各预热及正式轮的八条 prompt SHA-256 也逐组相同。质量题只是短答冒烟检查，不能推断 AWQ 整体质量。BF16 与 AWQ 的正式轮 Running 峰值均达到 8；Waiting 峰值分别为 4/4/4 与 5/4/4，不能把 Waiting 单独解释成 KV 抢占。

| 服务启动日志指标 | BF16 | AWQ 4-bit | 解释 |
| --- | ---: | ---: | --- |
| 模型加载占用 GiB | 15.27 | 5.71 | AWQ −62.6% |
| 自动 KV 池 tokens | 137,360 | 206,704 | AWQ +50.5% |
| 正式轮 KV 使用比例峰值 | 18.83% | 12.51% | 两组绝对占用均约 25,858 tokens；比例不可直接比较 |
| torch.compile 耗时 s | 57.98 | 56.41 | 冷启动成本，不计入正式吞吐 |

原始证据：[BF16 JSON](../../phase3/results/weight_awq_ab_v029awqgraph_20261010_bf16.json) · [AWQ JSON](../../phase3/results/weight_awq_ab_v029awqgraph_20261010_awq.json) · [BF16 客户端日志](../../phase3/results/phase5_client_awq_graph_bf16_20261010.log) · [AWQ 客户端日志](../../phase3/results/phase5_client_awq_graph_awq_20261010.log) · 上述两组服务日志与环境记录。七份文件复制到本地后均与云端 SHA-256 逐一一致。

## 学到什么，重点是什么

本次差异主要落在 Decode：TTFT P50 仅下降约 4.7%，而 ITL 均值下降约 38.9%，使 8 条固定输出请求的整批完成时间明显缩短。**同一项权重量化在不同执行模式下，吞吐方向可以不同**；权重更小、KV 池更大与服务更快是三件需要分别量测的事。这里的服务端 ITL 是 histogram 增量的 `sum/count`，不是客户端 SSE 内容事件间隔。

BF16 的 KV 峰值比例较高，主要因为它的自动 KV 池较小：`0.188257 × 137360 ≈ 25859`，AWQ 为 `0.125097 × 206704 ≈ 25858` tokens。两组处理相同形状的请求时，绝对 KV 占用几乎相同。本次没有 preemption；不能把 AWQ 的吞吐优势归因于避免 KV 抢占。

## 局限与收尾

- 只测本次同容器、单张 RTX 4090、单一同步突发负载。结果不是开环容量、SLO goodput 或真实业务分布下的保证；更长输出、不同输入长度与并发可能改变差异。
- 旧 eager 的 −6.0% 与本次默认模式的 +40.96% 方向可对照，但旧数据不是本次同容器重跑，不能据此估计严格的“模式 × 权重”交互效应。要单独估计 CUDA Graph 与 torch.compile 的贡献，还需同节点额外执行模式对照。
- 3 次正式轮在各组内波动很小，足以回答本次负载的方向问题；仍不能外推 AWQ 普遍更快。6 题质量检查也不是正式质量评测。
- 两组服务均已停止；结束时 `/health` 不可达、GPU 显存 0 MiB。随后在云平台控制台确认实例 `positive-cedar-1105` 显示**“已关机 / 不计费”**。平台关机提示写明实例只保留 3 天，届时自动删除；后续实验需重新核对实例、数据与接入方式。
