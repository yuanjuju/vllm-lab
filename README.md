# vLLM 推理服务与性能分析



## 入门实验

在云端 Qwen3-8B / vLLM 0.29.0 服务上，同时发送 6 个请求，每个请求固定生成 384 tokens。只调整 `max-num-seqs`，观察到：

| 配置 | Running 峰值 | Waiting 峰值 | 整批耗时 | 聚合输出吞吐量 |
| --- | ---: | ---: | ---: | ---: |
| `max-num-seqs=4` | 4 | 2 | 13.672 s | 168.52 tokens/s |
| `max-num-seqs=8` | 6 | 0 | 8.185 s | 281.49 tokens/s |

“把参数调大，GPU 就算得更快”的说法是错误的。这组负载下，改动让两个等待中的请求提前进入调度，整批任务更早结束；与此同时，原本先完成的 4 个请求各自稍慢。**吞吐量、排队时间和单请求延迟需要一起看。**实验只在这台机器上各运行一次，不能据此确定最佳生产配置。[完整条件、逐请求数据和局限](phase3/results/scheduler_capacity_comparison.md)

随后完成了 KV 容量与 Preemption 的安全对照：同样的 8 并发固定负载，在正常 137360-token KV 池中连续 3 批无抢占；仅把逻辑 KV 池限制为 1024 blocks（16384 tokens）后，连续 3 批各发生 1 次抢占。两组均 24/24 请求成功、没有 OOM，但受控组每批都有一条请求出现约 3 秒流式停顿，整批输出吞吐量中位数下降 6.1%。这证明 **Waiting 不能单独代表抢占，必须联合 KV 使用率、Preemption counter 和请求级延迟判断。**[完整对照与原始数据](phase3/results/kv_pressure_comparison_20260923.md)

性能工程主线也已完成第一轮闭环：Token Budget 对照证明 long/512 的 Waiting 来自单轮 token budget，而非序列名额或 KV；closed/open-loop 容量曲线用预先声明的 SLO 计算 Goodput；10 req/s 冲击测到 Waiting 峰值 21、KV 仅 1.55%、无抢占，说明这是普通排队过载。[Token Budget](phase3/results/token_budget_comparison_20260923.md) · [容量曲线](phase3/results/benchmark_capacity_curve_20260923.md) · [过载与恢复](phase3/results/overload_recovery_20260923.md)

## 项目做了什么

- **服务部署**：Mac M5 使用 vLLM-Metal 运行 Qwen3-0.6B；云端使用 CUDA/BF16 运行 Qwen3-8B。两端都通过 OpenAI 兼容 API 验证模型加载和聊天生成。云端 API 只监听回环地址，Mac 经 SSH 隧道访问，没有把 8000 端口直接暴露到公网。
- **测量与观测**：客户端记录流式 TTFT、总耗时、输出速度和成功率。`scheduler_capacity.py` 用 Barrier 同步发起请求、线程池并发提交，并轮询 vLLM `/metrics`；逐请求耗时与 Running、Waiting、KV Cache 使用率一起写入 JSON。
- **机制验证**：重复前缀请求时检查缓存命中计数，用 4/8 序列上限对照解释队列变化，再用受控 KV 池区分普通排队与真实抢占。请求链路写在[架构笔记](phase3/learning_notes/01-request-lifecycle.md)中，KV 容量和重算机制写在[Preemption 笔记](phase3/learning_notes/03-kv-cache-preemption.md)中。

云端请求链路：`Mac 客户端 → SSH 本地转发 → vLLM API Server → Scheduler / KV Cache → GPU Worker`。客户端从同一条隧道读取 `/metrics`，把 API 层的耗时和引擎层的状态放在一起分析。

两个刻意保留的取舍：云端 API 只开在回环地址，避免为了测试而直接暴露服务；调度对照把 `min_tokens` 和 `max_tokens` 都设为 384，减少回答长短带来的干扰。后者有助于解释排队现象，但并不代表真实用户流量的长度分布。

另有两组留存逐请求数据的基线：Mac 上补采的 20 次请求全部成功，TTFT P50 为 0.0301 s；云端的 20 次远程流式请求也全部成功，TTFT P50 为 0.136 s。两组模型、提示词和链路不同，不能直接当作硬件跑分比较。[Mac 数据](phase2/results/06_sequential_stability_summary_20260920-132447.json) · [云端数据](phase3/results/cloud_20run_summary_20260921-141120.json)

## 从哪里读起

| 内容 | 说明 |
| --- | --- |
| [Mac Metal 基线](phase2/README.md) | 生成参数、思考模式、流式输出与短请求负载 |
| [云端服务与实验脚本](phase3/README.md) | SSH 转发、客户端脚本和原始结果入口 |
| [调度配置对照](phase3/results/scheduler_capacity_comparison.md) | 本页表格背后的请求级数据与分析 |
| [请求链路笔记](phase3/learning_notes/01-request-lifecycle.md) | 用实测 Running/Waiting 解释连续批处理 |
| [KV 与 Preemption 对照](phase3/results/kv_pressure_comparison_20260923.md) | 区分排队、KV 压力、抢占与重算的完整证据链 |
| [Token Budget 对照](phase3/results/token_budget_comparison_20260923.md) | Chunked Prefill 对 TTFT、ITL、吞吐量的取舍 |
| [Benchmark 容量曲线](phase3/results/benchmark_capacity_curve_20260923.md) | Closed/open-loop、Goodput、SLO 与重复范围 |
| [过载与恢复](phase3/results/overload_recovery_20260923.md) | 排队增长、超时、取消、恢复与监控边界 |

## 运行方式

本地虚拟环境和模型缓存不在 Git 中。已有适配 Apple Silicon 的 `.venv` 时，可在两个终端分别运行：

```bash
source env.sh
./run_server.sh
```

```bash
curl -fsS http://127.0.0.1:8000/v1/models
./test_api.sh
```

云端实验脚本是 **Mac 客户端**，不是云端一键部署脚本。先按[云端服务说明](phase3/README.md)准备 Qwen3-8B 服务和 SSH 转发，确认 `http://127.0.0.1:18000/health` 可用，再运行：

```bash
python3 phase3/metrics_counter_walkthrough.py
python3 phase3/prefix_partial_reuse.py
python3 phase3/scheduler_capacity.py --requests 6 --tokens 384
python3 phase3/token_budget_benchmark.py --budget 4096 --shape mixed
python3 phase3/load_curve_benchmark.py --mode open --rate 2 --requests 12
```

## 当前边界

这是单实例推理服务原型及性能分析项目，不是长期在线的托管服务。仓库没有打包云端虚拟环境、模型权重或完整的服务端安装自动化；也尚未覆盖鉴权、长期稳态压测和多实例故障恢复。`.venv/`、`.cache/`、通用运行日志和密钥不提交；只有报告明确引用的少量实验服务日志作为原始证据保留。所有性能数字都应连同模型、硬件、请求形状和采样方式阅读。
