# 过载、超时、取消与恢复实验

记录日期：2026-09-23。

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
5. Prometheus：Running、Waiting 及 reason、KV usage、Preemption、queue time、TTFT、ITL、成功/abort/error counter。
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
