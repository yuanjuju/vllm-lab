# vLLM 性能工程：token budget、chunked prefill 与延迟取舍

状态：**2026-09-23 已完成云端实测。** 512/4096 两个 budget、short/long/mixed 三类负载均完成预热和三次正式重复；完整结果见[对照报告](../results/token_budget_comparison_20260923.md)。下文保留实验设计，便于复跑。

## 为什么测这个

上一次 6 请求、每条固定输出 384 tokens 的对照中，`max-num-seqs=4` 时 Running/Waiting 峰值为 4/2，整批 13.672 s；设为 8 后为 6/0，整批 8.185 s，KV 使用率峰值分别只有约 1.22%/1.91%。这说明当时主要是**序列名额**限制，却还没有回答：长输入涌入时，同一调度轮的 token 预算如何影响已有输出的平滑度、后来请求的首 token 和整批吞吐量。本实验只改变 `max-num-batched-tokens`，用短、长、混合输入区分这些影响。

## 机制与预期（不是测量结论）

vLLM 0.29.0 的 V1 scheduler 每轮先处理 Running，再考虑 Waiting；单轮可调度 token 受 `max_num_batched_tokens`（以及其他可配置的 scheduled-token 限制）约束。开启 chunked prefill 后，超过剩余预算的输入可拆到后续轮；对同一长请求，较小预算通常意味着更多 prefill 轮次，首 token 可能更晚，但每轮较少的 prefill 工作也可能降低正在 decode 请求的输出间隔。较大预算可能改善长输入 TTFT，却可能让同时 decode 的 ITL 变差，吞吐量也不保证单调改善。实际方向取决于输入长度、并发、缓存命中、GPU、内核与调度时序。

参照本地安装的 `vllm==0.29.0`：`vllm/v1/core/sched/scheduler.py` 中 `schedule()` 对 Running/Waiting 使用 token budget，并在 KV 分配失败时才走 `_preempt_request()`；`vllm/config/scheduler.py` 校验了 chunked prefill 与小于 `max_model_len` 的 budget 的关系；`vllm/v1/metrics/loggers.py` 定义本页使用的指标。官方同版本文档：[优化与调优](https://docs.vllm.ai/en/v0.29.0/configuration/optimization/)、[服务参数](https://docs.vllm.ai/en/v0.29.0/cli/serve/)、[指标](https://docs.vllm.ai/en/v0.29.0/usage/metrics/)。文档中的一般趋势不是这台机器的性能承诺。

## 固定实验条件

同一 Vylai 节点、同一 Qwen3-8B 权重与 vLLM 0.29.0；BF16、`max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、开启 chunked prefill；云端 API 仅绑定 `127.0.0.1:8000`，Mac 通过 SSH 本地转发 `18000` 访问。每批同时发 6 个请求，固定 `min_tokens=max_tokens=128`、`temperature=0`、禁用思考。脚本提供 `short`、`long`、`mixed` 三种负载；前两者用于按输入长度独立对照，`mixed` 更容易看到长 prefill 与短请求 decode 的干扰。脚本在用户内容开头写入唯一运行 ID，减少跨批次前缀缓存复用；仍记录 prefix-cache hit/query 增量，不把它假定为零。

第一轮建议 budget **512 与 4096**。二者只是用来制造清晰差异的测试点，不是优化建议。如果 512 在当前环境不稳定或太慢，记录失败并停止，不要悄悄改变其他配置。`max-num-seqs=8` 高于请求数 6，目的是避免重现上一次的名额瓶颈。若初次长输入的 `prompt_tokens + 128 >= 4096`，应缩短脚本中的长输入并重新固定整个矩阵；不能只改某一组。每个配置至少预热一次后做 3 次正式重复，报告跨重复的中位数与范围；预算有限可先各跑 1 次探索，但不可据此宣称稳定结论。

## 开机后的运行顺序

1. 在 Vylai 控制台确认实例确实开机、GPU 型号/计费状态；SSH 登录后，在云端已有 `/root/fsas/vllm-lab` 内检查驱动、CUDA 与模型，不重装：

   ```bash
   cd /root/fsas/vllm-lab
   nvidia-smi
   .venv/bin/python -c 'import torch, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA", vllm.__version__)'
   test -d models/Qwen3-8B
   ```

   `torch.cuda.is_available()` 必须为 `True`；若节点驱动兼容性报错，换兼容节点或先排查，不直接重装项目环境。记录实际驱动、GPU、版本、启动时间。

2. 在云端用已有 `.venv` 启动服务。以下先运行 budget 512，第二组只把末尾的 `512` 改成 `4096`；每次要先正常停止旧服务、确认新服务启动日志中的**实际**配置。不要同时开两个服务。

   ```bash
   cd /root/fsas/vllm-lab
   .venv/bin/vllm serve /root/fsas/vllm-lab/models/Qwen3-8B \
     --served-model-name Qwen/Qwen3-8B \
     --host 127.0.0.1 --port 8000 \
     --dtype bfloat16 --max-model-len 4096 \
     --gpu-memory-utilization 0.75 \
     --max-num-seqs 8 --enable-chunked-prefill \
     --max-num-batched-tokens 512
   ```

3. 在 Mac 使用当前实例的 SSH 地址和端口建立已有方式的本地转发（不要把私密地址写入仓库）。确认 `/health` 与 `/v1/models` 后，从仓库根目录运行。`--budget` 只是**声明标签**，不会调整远端服务；必须与启动日志一致。

   ```bash
   cd '/Users/jinian/Documents/llm learning/vllm-lab'
   python3 phase3/token_budget_benchmark.py --budget 512 --shape short --dry-run
   for shape in short long mixed; do
     python3 phase3/token_budget_benchmark.py --budget 512 --shape "$shape" --warmup
     for repeat in 1 2 3; do
       python3 phase3/token_budget_benchmark.py --budget 512 --shape "$shape"
     done
   done
   ```

   更换服务 budget 为 4096 后，只把命令中的 `--budget 512` 改为 `--budget 4096`，重复同样三种 shape。预热批次保留文件并标为预热；正式批次不得筛掉慢请求。每组运行前确保 Running/Waiting 回到 0，避免上一批互相干扰。脚本只使用 Python 标准库，原始 JSON 写入 `phase3/results/`，包括请求级 SSE 非空内容块时间、`usage`、采样序列、指标前后快照。不要把这些 JSON 当成已完成实验前先提交性能结论。

## 指标与验收

| 指标 | 采集与解释 |
| --- | --- |
| 成功率、`finish_reason`、tokens | 逐请求 SSE 的 `[DONE]`、最终 `usage`、预期 `length`；每组 6/6 才进入性能比较。`length` 是固定输出实验的预期截断。 |
| 客户端 TTFT P50/P95 | 发送至首个非空内容块，包含 Mac、SSH、API、排队和推理；适合**同链路同模型**横向对照。 |
| 客户端 chunk 间隔 P50/P95 | SSE 内容块时间差，**不是严格的逐 token ITL**；一块可能包含多 token，网络缓冲也会影响它。 |
| 服务端 TTFT、ITL、排队时间 | `/metrics` 同名 histogram 在批次前后的 `_count/_sum/_bucket` 增量；脚本给均值及桶上界 P50/P95，桶上界不是精确分位数，也不能对应到单个请求。 |
| Running、Waiting、KV cache | 0.1 s 轮询的观察峰值，可能漏掉更短的尖峰；KV 值为 0–1 比例。 |
| `num_preemptions` | 累计 counter 的前后差值；**Waiting 非零不等于抢占**。 |
| 聚合输出吞吐量 | 全批 completion tokens / 全批墙钟时间；不等于单请求 tokens/s。 |

验收时检查：模型 ID 正确、服务日志记录配置、6/6 成功、`prompt_tokens` 的短/长差异足够明显且两者都满足 `prompt_tokens + 128 < 4096`、正式组的请求数和输出 tokens 一致、`/metrics` 可读、无其他客户端流量混入 histogram delta。若指标缺失、计数不匹配或部分请求失败，保留原始结果并标记无效，不填补缺失值。

## 如何判读瓶颈与局限

- **名额瓶颈**：Running 靠近 `max-num-seqs`，Waiting 增长，KV 远未满，抢占增量为 0。已有 4/8 对照符合这个解释；本实验固定名额为 8，若 6 请求仍只运行 4 个，要先核实真实启动配置。
- **token-budget 限制**：长输入超过或竞争单轮预算，小/大 budget 的长请求 TTFT、服务端排队/Prefill 指标、混合负载下的 chunk 间隔出现可重复变化；Running 不必到 8，KV 也不必高。单凭 Waiting 或 TTFT 不能证明预算是唯一原因，要结合输入 token 数和排除缓存/名额/KV 干扰。
- **KV 空间压力**：KV 使用率高并伴随抢占增量或明显等待，才考虑 KV 分配竞争；KV 高但无抢占不能推断发生重算，抢占增量也应对照日志和同批计数。不要为了“测出抢占”随意缩小 KV 或拉高并发导致 OOM。

每次服务重启会改变冷/热状态，首次请求、CUDA 图/内核预热和 prefix cache 都可能影响结果。Mac-SSH 客户端指标不是纯 GPU 时间；Prometheus histogram 是服务端全局累计，批次间必须没有其他业务请求。本次实测没有合并短/长与预热/正式，也没有与 Mac 0.6B 数字做硬件比较；汇总表、重复范围和原始文件入口均在[结果报告](../results/token_budget_comparison_20260923.md)。

本次实验结束时已中止 vLLM 服务和 SSH 隧道，远端无残留进程、GPU 显存回到约 1 MiB used。Vylai 控制台关机与计费状态仍必须由实例所有者刷新确认；仅关闭终端或 API 不代表停止计费。
