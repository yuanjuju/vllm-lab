# KV Cache 容量与 Preemption：从容量公式到实测证据链

## 容量关系

对这次 Qwen3-8B/BF16，模型配置为 36 层、8 个 KV heads、head dimension 128。每 token KV 大致为 `36 × 2 × 8 × 128 × 2 = 147456 bytes`，即 144 KiB。使用 16-token block 时，每个 block 约保存 2.25 MiB KV。

对均匀全注意力模型，可以用 `ceil(当前上下文 tokens / block_size)` 近似一条请求需要的 blocks；多请求的已分配上下文之和不能超过 KV pool。实际容量应以 vLLM 启动日志的 `GPU KV cache size` 为准，因为 vLLM 0.29.0 对复杂模型使用 group-aware 计算。

输入完成 Prefill 后占用 KV；Decode 每生成一个 token，又会增加这个请求的上下文。`max_tokens` 不是启动时一次性预分配的 KV，所以多个短输入可以同时被接纳，随后随着长输出共同增长才触及容量边界。

## Waiting 不等于 Preemption

- 请求因为 `max-num-seqs` 上限没有进入 Running：会 Waiting，但没有 Preemption。
- 等待请求完整输入暂时无法放入 KV：可以继续 Waiting，仍不必 Preemption。
- 运行中请求需要新 KV block 而分配失败：Scheduler 才可能释放某个运行请求的 KV，将其标记为 PREEMPTED，并放回 Waiting。

因此必须联合观察 `vllm:num_preemptions_total` 增量、Running/Waiting、KV 使用率、服务日志和请求级延迟。KV 使用率高本身也不是错误；高使用率但 counter 不增加、请求稳定完成，只能说明利用率高。

## Recompute 怎样影响延迟

vLLM 0.29.0 的 `_preempt_request()` 会释放被抢占请求的 blocks、把 `num_computed_tokens` 重置为 0 并增加 `num_preemptions`。请求稍后恢复时，需要重新建立缺失的 KV；仍可命中的缓存块可能减少实际重算量。

如果抢占发生在首 token 前，主要恶化 TTFT；如果发生在已经输出之后，TTFT 已经确定，更明显的表现是中途停顿、总完成时间和尾延迟上升。重复计算消耗 GPU，却不产生新的有效输出，因此整批有效吞吐量会下降。

vLLM 0.29.0 的 `request_queue_time_seconds` 只计算首次 `QUEUED` 到首次 `SCHEDULED`，抢占后的再次调度会被忽略。因而一次中途停顿通常同时包含重新等待和恢复 KV 状态的时间，不能用 queue-time histogram 把二者直接分开。

## 本次验收

[2026-09-23 对照实验](../results/kv_pressure_comparison_20260923.md)在相同 8 请求、每条 173-token 输入和 2048-token 输出下得到：

- 正常 137360-token KV 池：3/3 批次无 Waiting、无 Preemption，KV 峰值约 12.8%。
- 受控 16384-token KV 池：3/3 批次各发生 1 次 Preemption，KV 峰值约 99.5%–99.9%，Waiting 峰值 1。
- 两组均 24/24 请求成功，没有 OOM；受控组每批一条请求出现约 2.86–3.17 秒流式停顿，整批吞吐量中位数下降约 6.1%。

所以本模块的判读标准是：低 KV + Waiting + counter 不变属于排队；高 KV 仍不能单独证明不足；只有 counter 增量并得到状态变化与请求级延迟的交叉验证，才能确认发生了 KV 抢占和重算。
