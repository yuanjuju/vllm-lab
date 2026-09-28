# 源码精读笔记：请求前半段与 metrics 曝写层——abort counter 悬案告破

记录日期：2026-09-28。源码版本：vLLM v0.29.0（`../vllm-src`）。

## 版本考古：api_server.py 已废弃

`vllm/entrypoints/openai/api_server.py` 在 v0.29.0 只剩 59 行转发壳（文件自带 DeprecationWarning，`:24-30`），真实入口迁移到：

- `vllm/entrypoints/launchers/api_server/`（`routers.py`、`entry.py`——`vllm server` CLI）
- 聊天接口实现在 `vllm/entrypoints/openai/chat_completion/serving.py`

**教训：读源码必须对着自己的版本。** 网上教程多基于旧路径 `entrypoints/openai/api_server.py`。

## 请求前半段（API 进程内）

```
HTTP POST /v1/chat/completions
→ launchers 路由 → chat_completion/serving.py
→ InputProcessor（v1/engine/input_processor.py:38）：参数校验、聊天模板、分词
   （tokenizer 由 renderer 持有，:78-82 —— 确认分词在 API 进程完成）
→ AsyncLLM.generate()（第三课）→ ZMQ → EngineCore
```

## metrics 曝写层（`vllm/v1/metrics/loggers.py`）

phase3 轮询过的每个指标的定义位置：

| 指标 | 行号 | 类型/语义 |
| --- | --- | --- |
| `vllm:num_requests_running` / `waiting` | :494 / :504 | gauge，来自 SchedulerStats |
| `vllm:kv_cache_usage_perc` | :562 | gauge |
| `vllm:prefix_cache_queries` / `hits` | :585 / :596 | counter（第一课作业的落点） |
| `vllm:num_preemptions` | :662 | counter，源=IterationStats.num_preempted_reqs |
| `vllm:time_to_first_token_seconds` 等 | :797+ | histogram |
| `vllm:request_success{finished_reason}` | :713-725 | counter，**按 FinishReason 建 label** |

preemption 链路：`scheduler.py:1440`（Request.num_preemptions+1）→ event PREEMPTED → `stats.py:524-525`（IterationStats）→ `loggers.py:1191-1192`（counter.inc）。

## 悬案：abort counter 为何"存在但无效"

phase3 overload 实验结论"abort counter 未提供有效证据"的源码级答案（三条路径）：

1. **label 预建**：`loggers.py:719` `for reason in FinishReason`——abort 与 stop/length/error/repetition（`engine/__init__.py:47` 枚举）一样预建了 `vllm:request_success{finished_reason="abort"}` 时间序列，所以 `/metrics` 里**看得到这个序列（值恒 0）**。
2. **本地 abort 绕过统计**：客户端断开 → `output_processor.py:526-547` `abort_requests`——直接 pop 请求状态、给客户端推最后一个 ABORT 输出，**不调用** `_update_stats_from_finished`（计数只发生在正常 finish 分支 `:730-744` → `:863`）。
3. **迟到输出被丢弃**：EngineCore 稍后产生的 FINISHED_ABORTED 输出回到 API 进程时，请求状态已删，`:655` "Ignore output for already-aborted request" 直接 continue。

**结论：客户端取消的请求在 vLLM 0.29.0 的 Prometheus 体系中不可见**（无专门 counter、success counter 也不计）。可行的替代观测：发送数 − `request_success` 总增量的差值、访问日志、或客户端侧计数。这一发现应回填 phase3 的 [overload 报告](../../phase3/results/overload_recovery_20260923.md)结论。

## 与实验数据的对应

- phase3 `metrics_before/after` 快照里读的所有序列名，现在都有定义行号与更新路径。
- "abort counter 观测缺口"从现象观察升级为机制结论（上述三条路径）。
- gauge 更新点在 API 进程 output_handler（第三课）+ 本课曝写层——phase3 的指标语义表全部完成源码对应。

## 未解问题 / 下一步

- chat_completion/serving.py 的流式响应组装细节（SSE 分块与 keep-alive）未逐行读。
- `launchers/api_server/routers.py` 的路由注册细节未读。
- Phase 4 主干完成；剩余为可选深读项。→ 进入 Phase 5 特性对照实验（首选 `--enforce-eager` vs CUDA graph）。
