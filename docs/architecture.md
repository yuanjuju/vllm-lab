# 架构与工程实现

本文描述仓库已经存在的测量实现，以及它和 vLLM 引擎之间的边界。系统面向独占的单实例实验服务，便于把请求体验、调度状态和资源压力放到同一批次中分析。

## 组件边界

| 层次 | 仓库提供的实现 | 依赖的外部能力 |
| --- | --- | --- |
| 请求构造 | 固定输出长度、short/long/mixed 输入、运行与请求标识、禁用思考的实验配置 | Qwen Chat Template、vLLM OpenAI 兼容接口 |
| 负载生成 | Barrier 同步突发、线程池固定并发、按时钟释放固定到达率请求 | Python 标准库、主机线程调度 |
| 响应测量 | SSE 行解析、首内容块计时、usage / finish reason / DONE 检查、异常记录 | HTTP 连接与网络传输 |
| 引擎观测 | `/metrics` 采样、counter 差值、histogram 差值、逐批 JSON 归档 | vLLM Prometheus exposition endpoint |
| 推理执行 | 本地启动包装与服务访问约定 | vLLM API Server、Scheduler、KV 管理、GPU Worker；Mac 使用 Metal 后端 |
| 分析展示 | 对照报告、SLO / Goodput 汇总、从历史 JSON 生成可追溯图表 | 图表绘制使用 Matplotlib，推理测量不依赖它 |

仓库没有修改 vLLM 调度器、实现注意力算子或开发 GPU kernel。项目的工程工作集中在服务接入、实验工具、观测和诊断。

## 三种负载生成方式

### 同步突发

[`scheduler_capacity.py`](../phase3/scheduler_capacity.py) 与 [`token_budget_benchmark.py`](../phase3/token_budget_benchmark.py) 使用 `ThreadPoolExecutor` 和 `threading.Barrier`，让一批请求尽量同时进入服务。固定 `min_tokens=max_tokens` 控制输出工作量，方便比较不同配置下的队列和完成时间。

前者采用非流式响应做调度观察，后者采用 SSE 保留逐内容块到达时间。两者都把负载与引擎状态采样写入同一批结果。

### Closed-loop 固定并发

[`load_curve_benchmark.py`](../phase3/load_curve_benchmark.py) 启动指定数量的 worker。worker 完成一个请求后才领取下一个，共享索引由锁保护。响应变慢时发送速度自然下降，适合观察并发与吞吐关系。

### Open-loop 固定到达率

同一脚本按 `origin + index / rate` 计算计划发送时间，不等待前一个请求完成。结果记录 `scheduled_at_s`、`started_at_s` 和 `client_release_lag_s`，用于识别客户端未能按计划释放请求的情况。

当前实现给整批请求预留线程，适合归档实验的小规模负载；它不是可无限扩展的分布式压测器。这里的 open-loop 是确定性间隔，不是 Poisson 到达模型。

## 指标与结果契约

| 字段或指标 | 计算位置与含义 | 解释边界 |
| --- | --- | --- |
| `client_ttft_s` | HTTP 请求开始到第一个非空内容块 | 包含客户端、SSH、网络与服务端等待；不包含计划发送前的 release lag |
| `duration_s` | 单请求或整批计时，依所在字段层级区分 | 整批吞吐采用整批测量窗口，包含完成后的采样收尾开销 |
| `content_event_times_s` / `content_event_gaps_s` | token-budget 脚本中的客户端 SSE 内容事件时间 | 一个事件可能包含多个 token；不是严格 token ITL |
| `histogram_deltas` | 批次前后 `_sum`、`_count`、bucket 的差值 | `mean_s=sum/count`；`p95_bucket_upper_s` 是桶上界，不是精确 P95 |
| `num_preemptions_delta` | 服务端抢占 counter 批次差值 | before/after 均存在才返回数值，否则为 `null` |
| `max_running` / `max_waiting` / `max_kv_cache_usage` | 轮询采样捕获的峰值 | 短于采样周期的尖峰可能漏掉；缺失 gauge 的处理尚未完全统一 |
| `slo_goodput_req_s` | 成功且同时满足 TTFT、E2E 阈值的请求数 / 整批时间 | 依赖请求分布、阈值、计时窗口和客户端位置 |
| `metric_poll_errors` | 采样异常次数 | 读取结果前需检查，不能只看成功请求数 |

流式脚本的 `ok` 检查内容块、usage、DONE 和 `finish_reason=length`。固定输出 token 数也应对照实际 `usage` 检查；`ok` 并不独立验证目标 token 数完全相等。

Histogram 和抢占差值在缺失时可返回 `null`，但通用 `total()` 对缺失序列仍返回 0，因此部分 gauge 与 prefix 汇总不能单独证明指标存在。复核时结合原始 `metrics_before/after`、实际 exposition 与服务日志。未来可统一为显式的可用性字段。

## 数据归档

Token-budget 与负载曲线 JSON 使用相似结构：

```text
run_id / timestamp / warmup / model / 负载参数
├── summary          # 吞吐、延迟、SLO 或形状拆分等批次汇总
├── results[]        # 逐请求时间、usage、结束状态与错误
├── samples[]        # 引擎状态时间序列、采样错误
├── metrics_before   # 选定指标的批次前快照
└── metrics_after    # 选定指标的批次后快照
```

- `run_id` 包含时间戳与纳秒标识；token-budget 结果另存提示词字符数及 SHA-256。字符数不当作 token 数。
- `warmup` 明确标记预热，离线展示脚本只选择正式运行。
- KV 受控组使用通用 token-budget 采集器，因此文件名仍有 `token_budget` 前缀。展示脚本按 [KV 报告](../phase3/results/kv_pressure_comparison_20260923.md#原始数据) 的明确文件列表分组，不按观察到的性能结果倒推分组。
- [展示汇总](assets/evidence-summary.json) 为每个点记录原始文件路径；[`tools/build_showcase.py`](../tools/build_showcase.py) 在重复数或关键负载参数不符时拒绝生成。
- 环境、模型、服务参数和组别映射仍有一部分保存在报告与服务日志中，尚未统一成可自动重放的实验清单。

## 已验证的诊断路径

| 现象 | 交叉检查 | 当前证据 |
| --- | --- | --- |
| Running 到达序列名额上限、Waiting 增长 | 检查 KV 与抢占，比较不同 `max-num-seqs` | [名额对照](../phase3/results/scheduler_capacity_comparison.md) |
| 长 Prefill 排队，但请求名额和 KV 尚有余量 | 固定形状，只改变每轮 token budget | [Token Budget 对照](../phase3/results/token_budget_comparison_20260923.md) |
| Decode 中出现少数请求长停顿 | 联合 KV 接近上限、Preemption 增量、队列与逐请求事件 | [KV 对照](../phase3/results/kv_pressure_comparison_20260923.md) |
| 全部请求成功但尾延迟增大 | 比较 offered/completed RPS、Waiting、SLO 达标率和 Goodput | [容量曲线](../phase3/results/benchmark_capacity_curve_20260923.md) |
| 客户端取消或超时 | 观察连接关闭后 Running/Waiting、健康检查与后续请求 | [恢复报告](../phase3/results/overload_recovery_20260923.md) |

## 设计取舍与下一步

- **标准库客户端：** 测量脚本易于在 Mac 重跑，避免为采集引入完整服务框架；HTTP/SSE 与 Prometheus 解析只覆盖本项目实际使用的形式。
- **远程端到端测量：** SSH 转发让服务只监听回环地址，同时保留真实远程链路体验；解释引擎效率时需单独看服务端指标。
- **受控 KV 压力：** 缩小逻辑 block 池复现抢占，不通过追求 OOM 来观察机制；结果不能作为正常 48GB 容量结论。
- **短批次与重复：** 便于逐项解释与检查证据；要得到可靠容量结论，需增加持续时间、样本量和同机房复测。
- **恢复探针：** 已验证单实例请求清理和服务响应，尚未构建故障注入平台、自动重启、入口治理或多实例故障转移。
