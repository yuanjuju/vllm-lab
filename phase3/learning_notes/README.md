# 引擎机制笔记

[请求从 API 进入到 GPU 执行的路径](01-request-lifecycle.md)是这部分的主线：用已经记录的 6 请求调度实验，把 API Server、Scheduler、KV Cache 和 GPU Worker 的职责与 `Running/Waiting` 指标对应起来。它是对实测现象的架构解释，不是对 vLLM 源码的完整实现分析。

继续阅读：

- [Token Budget 与 Chunked Prefill](02-token-budget-performance.md)
- [KV Cache 与 Preemption](03-kv-cache-preemption.md)
- [Benchmark 与 Goodput](04-benchmark-methodology.md)
- [过载与单实例运维边界](05-overload-operations.md)

配套证据在[结果目录](../results/)中。量化与 Speculative Decoding 已在 [Phase 5](../../phase5/README.md) 做独立实验并归档报告；本组五篇笔记聚焦 Phase 3 的调度、缓存、测量与运维机制。LoRA/SFT 尚未在本仓库做独立验证。
