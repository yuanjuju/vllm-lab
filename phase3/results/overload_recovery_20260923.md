# 过载、超时、取消与恢复实验

记录日期：2026-09-23（2026-10-07 回填 abort counter 的源码级机制）。

## 短时过载结果

在与容量曲线相同的 Qwen3-8B 服务上，以 open-loop 10 req/s 释放 40 个请求，每个输入约 198–199 tokens、固定输出 128 tokens。结果保存在[过载原始 JSON](load_curve_20260923-193937-1790163577005928000_open_r10p0_formal.json)。

| 指标 | 结果 |
| --- | ---: |
| 成功率 | 40/40 |
| 整批完成吞吐量 | 3.213 req/s、411.210 output tok/s |
| Running 峰值 | 8 |
| Waiting/capacity Waiting 峰值 | 21/21 |
| KV 峰值 | 1.55% |
| Preemption | 0 |
| 客户端 TTFT P50/P95 | 3.2218 / 6.2794 s |
| 客户端 E2E P50/P95 | 5.5122 / 8.5419 s |
| 服务端 queue time 均值 | 2.8934 s |
| 教学 SLO 达标率 | 20% |

这次过载没有报错，也没有 KV 抢占，但到达速度明显高于完成速度，队列峰值达到 21。停止注入新请求后积压最终排空。它证明：**成功率 100% 不能代表服务健康；若没有限流与背压，延迟会先失控。**

## 客户端取消与超时

取消探针在长请求收到第一个非空内容块后关闭流式连接，不重试。首块在 0.2402 秒到达，随后服务端 Running/Waiting 回到 0，短恢复请求在 0.2085 秒内成功并以 `stop` 结束。原始记录见[取消与恢复 JSON](overload_recovery_20260923-194100.json)。

超时探针给长请求设置 0.05 秒客户端 socket timeout，得到真实 `TimeoutError`，不重试；0.1749 秒采样时 Running/Waiting 已为 0，随后 `/health` 返回 HTTP 200。原始记录见[超时 JSON](client_timeout_20260923-194248.json)。

两个探针的 `request_success_total{finished_reason="abort"}` 都没有增加。因为取消探针已收到首个内容块，可以确认请求到过服务端；但当前 Chat Completions 流式断连路径没有给出可依赖的 abort counter 证据。正确结论是：

- 工作能够被清理，未观察到残留 Running/Waiting；
- 健康检查与新请求能够恢复；
- 不能把 abort counter 不变解释为没有发生客户端取消；这是当前可观测性缺口，需要结合连接日志、Running/Waiting 和恢复探针。

### 回填：abort counter 恒为 0 的源码级机制（2026-10-07 补充）

Phase 4 精读 vLLM v0.29.0 metrics 曝写层后，上述"观测缺口"从现象升级为机制结论：**客户端取消的请求在 vLLM 0.29.0 的 Prometheus 体系中完全不可见——不是采样漏掉，是设计如此。** 完整证据链（行号基于 v0.29.0）：

| 环节 | 代码位置 | 行为 |
| --- | --- | --- |
| 序列预建 | `vllm/v1/metrics/loggers.py:719-725` | 为 `FinishReason` 每个取值（含 `abort`）预建 counter label，所以 `/metrics` 里**看得到**这条序列，值恒 0 |
| 唯一自增点 | `loggers.py:1221-1224` | `for finished_request in iteration_stats.finished_requests: ...inc()`——只统计进入 IterationStats 的完成请求 |
| 唯一填充点 | `vllm/v1/engine/output_processor.py:852`（`_update_stats_from_finished`）→ `:863` | 只有它向 IterationStats 登记完成请求 |
| 调用条件 | `output_processor.py:741` | 该函数只在**正常 finish 分支**被调用 |
| 取消路径绕过 | `output_processor.py:526-547` | 客户端断开 → `abort_requests` 直接 pop 请求状态、推最后一个 ABORT 输出即返回，**不触碰统计** |
| 迟到输出丢弃 | `output_processor.py:654-656` | EngineCore 稍后产生的 `FINISHED_ABORTED` 回来时请求状态已删，直接 `continue` |

**修正后的建议**：不要用 abort label 判断取消是否发生（它永远是 0）。可靠的替代口径是 **发送数 − `request_success_total` 总增量的差值**，配合客户端侧计数交叉验证。源码推导详见 [Phase 4 笔记 04](../../phase4/notes/04-api-and-metrics.md)。

## 限流、背压和重试原则

- 在 API 入口限制“正在处理 + 排队”的总量；超过阈值应快速返回 429/503，而不是无限累积 Waiting。
- 客户端设置连接、首 token 和总请求三类超时，并在用户取消时关闭上游流。
- 不要对生成请求立即无限重试。只有在明确未被服务端接收，或业务有幂等键/去重策略时，才使用有限次数、指数退避和随机抖动重试。
- 过载时重试会形成 retry storm，使有效到达率进一步高于处理率。
- 本仓库尚未实现入口限流、幂等键或代理层背压；本页是测量证据和设计边界，不虚称生产能力。

## 启动验收与监控清单

启动前/后至少联合检查：

1. `nvidia-smi`：驱动、GPU 型号、显存和异常进程。
2. Python 探针：PyTorch/vLLM 版本、`torch.cuda.is_available()`。
3. 服务日志：模型、dtype、max model len、token budget、max seqs、KV tokens。
4. `/health`、`/v1/models` 和一条真实生成请求；只有 health 200 不等于模型生成可用。
5. Prometheus：Running、Waiting 及 reason、KV usage、Preemption、queue time、TTFT、ITL、成功/abort/error counter。**注意：abort counter 恒为 0，客户端取消不可见（见上文回填），告警不要依赖它。**
6. 主机侧 GPU/CPU/内存/磁盘与 API 层 HTTP 状态、超时、客户端取消共同观察。

可用于 Grafana 的基础 PromQL 示例：

```promql
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
rate(vllm:num_preemptions_total[5m])
rate(vllm:request_success_total[5m])
histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
histogram_quantile(0.95, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))
```

本次没有部署长期 Prometheus/Grafana，也没有实现告警、自动重启、限流、TLS、鉴权、多实例或容灾；项目仍是单实例推理服务原型。
