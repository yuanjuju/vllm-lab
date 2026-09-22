# vLLM 核心学习笔记

这组笔记将已经完成的 API、性能指标和云端实验串成 vLLM 的请求处理主线。后续总结与上传以这里的笔记和 `phase3/results/` 中的原始数据为依据；目前只保存在本地，尚未上传。

| 顺序 | 主题 | 状态 | 材料 |
| --- | --- | --- | --- |
| 1 | 一条请求如何穿过 vLLM | 已讲解，理论课 | [01-request-lifecycle.md](01-request-lifecycle.md) |
| 2 | PagedAttention 与 KV 块分配 | 待讲解 | — |
| 3 | Prefill、Decode、分块 Prefill 的调度取舍 | 待实验 | — |
| 4 | 量化、LoRA 与服务基准测试 | 待讲解 | — |

已完成的 4/8 序列上限对照及局限见 [scheduler_capacity_comparison.md](../results/scheduler_capacity_comparison.md)。
