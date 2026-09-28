# Phase 4：源码对齐

把 Phase 3 的"现象解释"笔记升级为"源码验证"笔记：以 vLLM **v0.29.0** 源码为准，逐一定位每条实测现象背后的代码位置。本阶段只读代码和日志，不需要 GPU 实例。

## 源码副本

- 完整仓库（含 docs / tests / csrc）：`../vllm-src`，克隆自 tag `v0.29.0`，与云端实验环境版本一致。
- 本地 Metal 安装包：`../.venv/lib/python3.12/site-packages/vllm/`（同为 0.29.0，可用于快速 grep，但不含 docs 与测试）。

两者如有差异，以 `../vllm-src` 为准；笔记中的行号引用统一指向 `vllm-src`。

## 读码顺序（按请求流向）

| 步 | 文件（相对 `vllm/`） | 回答的问题 | 对应实验 / 笔记 |
| --- | --- | --- | --- |
| 1 | `entrypoints/openai/api_server.py` | HTTP 请求从哪进来，路由到哪 | [01-request-lifecycle](../phase3/learning_notes/01-request-lifecycle.md) |
| 2 | `v1/engine/async_llm.py` → `v1/engine/core.py` | EngineCore 主循环一次迭代做什么 | 同上 |
| 3 | `v1/engine/input_processor.py` | 聊天模板、token 化在哪发生 | 同上 |
| 4 | `v1/core/sched/scheduler.py` | budget 检查、chunked prefill、preemption 分支在哪 | [02-token-budget](../phase3/learning_notes/02-token-budget-performance.md) · [03-kv-cache-preemption](../phase3/learning_notes/03-kv-cache-preemption.md) |
| 5 | `v1/core/sched/request_queue.py` | Running / Waiting 队列的实现 | [scheduler_capacity 对照](../phase3/results/scheduler_capacity_comparison.md) |
| 6 | `v1/core/kv_cache_manager.py` | 块申请、复用与调度如何协作 | [KV pressure 对照](../phase3/results/kv_pressure_comparison_20260923.md) |
| 7 | `v1/core/block_pool.py` + `kv_cache_utils.py` | block hash、前缀命中、淘汰策略 | [prefix_partial_reuse](../phase3/results/prefix_partial_reuse_20260921-184943.json) |
| 8 | `v1/worker/gpu_model_runner.py` | 一次 forward 如何拼 batch、CUDA graph 捕获/重放 | 01-request-lifecycle · 后续特性实验的基础 |
| 9 | `v1/engine/output_processor.py` + `detokenizer.py` | token 如何流式回到客户端 | 01-request-lifecycle |

发现观测开关的最快方式：`grep -n "VLLM_" ../vllm-src/vllm/envs.py`。日志级别（`VLLM_LOGGING_LEVEL`）、V1 Perfetto trace 等开关都在这个文件里，使用前先在此确认名字。

## 产出约定

- [source-map.md](source-map.md)：**现象 → 代码位置**对照表。主干已全部验证（2026-09-28）；仅剩 serving.py 流式细节等可选项。
- [note-template.md](note-template.md)：源码精读笔记模板；每读完一个模块，按模板在 `notes/` 下建一篇。
- 已完成笔记（**Phase 4 主干收官**）：
  [01 Scheduler](notes/01-scheduler.md) · [02 GPU 执行路径](notes/02-gpu-model-runner.md) · [03 EngineCore](notes/03-engine-core.md) · [04 请求前半段与 metrics](notes/04-api-and-metrics.md)（含 abort counter 悬案的源码答案）。
- 下一步：Phase 5 特性对照实验（`--enforce-eager`、prefix cache 开关、KV dtype、量化、spec decode）。
- 每条结论必须带 `文件路径 + 函数名`（建议带行号）；行号以 v0.29.0 为准，引用格式如 `vllm/v1/core/sched/scheduler.py:120`。

## 本阶段不做什么

- 不修改源码、不重新编译、不跑新负载；性能实验属于 Phase 5。
- 不追求逐行读完；以"能解释已有实验数据"为完成标准。
