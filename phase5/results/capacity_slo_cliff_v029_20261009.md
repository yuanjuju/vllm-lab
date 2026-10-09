# Qwen3-8B baseline 的开环 SLO 悬崖补测（2026-10-09）

## 结论

在**本次节点、eager BF16 单实例、50:50 合成输入、固定 256-token 输出、60 秒窗口与示例 SLO**下，0.6、1.0、1.2 req/s 的三次正式轮均逐请求达标；1.4 req/s 的三轮均出现少量 TTFT >1 秒，1.6 req/s 则持续排队，TTFT P95 升至约 9 秒。故按[预注册协议](capacity_slo_cliff_protocol_20261009.md)的严格逐请求判据，**1.2 是最高通过的已测试速率，失效区间位于 (1.2, 1.4] req/s**。它不是可向真实业务承诺的长期最大 QPS。

这次过载更符合**并行序列名额/服务能力约束下的排队**：1.4 与 1.6 档 Running 达到配置上限 8，Waiting 峰值分别为 2、13，服务端队列时间随之增长；KV 使用率峰值仅约 2.63%、2.69%，`num_preemptions` 增量为 0。不能把 Waiting 当成 KV 抢占，也不能仅凭此实验断言单轮 token budget 是瓶颈。

## 实验条件与可复现性

- 新节点主机名 `positive-cedar-1105-cd5d67b6-fjch9`，RTX 4090 49140 MiB，GPU UUID `GPU-5faea9f4-6d37-d847-e664-e9d2d4ab4fe3`，驱动 590.44.01；项目独立 `.venv` 使用 vLLM 0.29.0、PyTorch 2.13.0+cu132。主机名与[上一轮容量网格](capacity_open_loop_v029_20261009.md)不同，因此不将其数据视为严格配对重复。
- 使用 [`phase3/run_cloud_vllm029_features.sh`](../../phase3/run_cloud_vllm029_features.sh) 的 `base` 变体。启动日志确认 Qwen3-8B BF16、`--enforce-eager`（同时关闭 torch.compile 与 CUDAGraphs）、Prefix Cache 关闭、`max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`；仅监听云端 `127.0.0.1:8000`。KV 池 19.74 GiB / 143,728 tokens。
- 云端 loopback 客户端运行 [`phase3/spec_ngram_open_loop_exploratory.py`](../../phase3/spec_ngram_open_loop_exploratory.py)：repeat/count 交替 50:50，真实 prompt token 数分别为 564–568 / 65–69；每条生成恰好 256 tokens，`temperature=0`，关闭 thinking。到达为确定性等间隔，不是 Poisson；不设客户端并发上限。每档 10 秒预热（不入正式统计）+ 3 × 60 秒正式窗口；从 0.6 升到 1.6 req/s，每档排空后进入下一轮。
- 示例 SLO：请求成功、客户端 TTFT ≤1 秒、E2E ≤10 秒；另要求该轮整体 TTFT P95 ≤1 秒。三轮正式都满足上述条件才标为该测试档通过。客户端指标含 loopback HTTP/SSE 开销；服务端 histogram 是另一口径。

云端执行的客户端命令（服务按上述 `base` 变体启动并通过 `/health` 与真实请求验收后）：

```bash
cd /root/fsas/vllm-lab/phase3/v029_clients
/usr/bin/python3 spec_ngram_open_loop_exploratory.py \
  --condition baseline --experiment-id v029cliff_20261009 \
  --rates 0.6,1.0,1.2,1.4,1.6 \
  --warmup-duration-s 10 --formal-duration-s 60 --formal-repetitions 3 \
  --server-profile v029-r590-eager-base
```

原始证据：[20 轮 JSON](../../phase3/results/spec_ngram_open_loop_v029cliff_20261009_baseline.json)（5 档 × 1 预热 + 3 正式）、[客户端日志](../../phase3/results/phase5_client_capacity_cliff_base_20261009.log)、[服务启动及运行日志](../../phase3/results/phase5_v029_cliff_base_20261009.log)。三份文件从云端复制后与云端 SHA-256 逐一吻合；正式轮共 1,044/1,044 请求成功且每条输出 256 tokens。没有触发安全停止，也无指标轮询错误。

## 正式结果

下表为**三次正式轮的中位数**；吞吐是 256 × 完成数 /（60 秒到达窗 + 排空时间），不是稳态服务率。KV 为池使用比例峰值；队列时间和 ITL 是服务端 histogram 的均值，并非客户端分位数。

| 到达率 req/s | TTFT P95 s | E2E P95 s | 逐请求 SLO 达标 | SLO goodput req/s | Running / Waiting 峰值 | 队列均值 s | KV 峰值 | ITL 均值 ms | 输出吞吐 tok/s | 排空 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | 0.0902 | 5.8042 | 36/36 (100%) | 0.5627 | 4 / 0 | 0.0000 | 1.31% | 22.4 | 144.045 | 3.9801 |
| 1.0 | 0.0918 | 5.8098 | 60/60 (100%) | 0.9281 | 6 / 0 | 0.0000 | 1.97% | 22.4 | 237.591 | 4.6488 |
| 1.2 | 0.0914 | 5.8545 | 72/72 (100%) | 1.1105 | 7 / 0 | 0.0000 | 2.46% | 22.5 | 284.297 | 4.8336 |
| 1.4 | 0.9370 | 6.6788 | 82/84 (97.62%) | 1.2437 | 8 / 2 | 0.4431 | 2.63% | 22.6 | 326.152 | 5.9324 |
| 1.6 | 9.0132 | 14.7341 | 16/96 (16.67%) | 0.2161 | 8 / 13 | 4.4793 | 2.68% | 22.6 | 331.981 | 14.0283 |

复现性与判据细节：

| 到达率 | 三次 TTFT P95 s | 三次逐请求 SLO 达标数 | 三次 Waiting 峰值 | 三次 preemption 增量 | 本协议判定 |
| ---: | --- | --- | --- | --- | --- |
| 0.6 | 0.0908 / 0.0902 / 0.0895 | 36 / 36 / 36（每轮 36） | 0 / 0 / 0 | 0 / 0 / 0 | 通过 |
| 1.0 | 0.0909 / 0.0924 / 0.0918 | 60 / 60 / 60（每轮 60） | 0 / 0 / 0 | 0 / 0 / 0 | 通过 |
| 1.2 | 0.0924 / 0.0907 / 0.0914 | 72 / 72 / 72（每轮 72） | 0 / 0 / 0 | 0 / 0 / 0 | 通过 |
| 1.4 | 0.9930 / 0.9348 / 0.9370 | 80 / 82 / 82（每轮 84） | 2 / 2 / 2 | 0 / 0 / 0 | 未通过逐请求判据 |
| 1.6 | 8.9645 / 9.0132 / 9.0340 | 16 / 16 / 16（每轮 96） | 13 / 13 / 13 | 0 / 0 / 0 | 未通过 |

1.4 档的关键细节是：三轮整体 TTFT P95 **都低于 1 秒**，E2E >10 秒的请求为 0，但分别有 4、2、2 条请求 TTFT >1 秒。因此它在只看“TTFT P95 ≤1 秒”的宽松定义下通过，在预注册的“所有请求均达标”定义下未通过。两种 SLO 不可互换。

1.6 档的所有请求最终成功，**不等于负载可稳定承受**：每轮 96 条中有 80 条 TTFT >1 秒、48 条 E2E >10 秒；到达窗口末仍有 13 条 Waiting，随后还要约 14 秒排空。该档输出吞吐仅比 1.4 档略高（约 332 vs 326 tok/s），但 SLO goodput 由约 1.244 降至 0.216 req/s。客户端派发偏差 P95 均不超过 0.6 ms，说明主要延迟不是客户端未按时发请求；服务端队列均值约 4.48 秒与高 TTFT 方向一致。

## 学到什么、仍不能推出什么

排队要结合多项证据判断：Running=8 达到 `max-num-seqs=8`、Waiting 上升、队列时间与 TTFT 同时上升、到达结束后还有排空尾巴，而 KV 峰值仅约 2.7%、抢占计数始终不增。这里的 Waiting 是**服务排队**，不是“KV 不足所以重算”。这也说明增加输出 token 吞吐并不必然增加符合时延要求的有效请求数。

但“Running 达 8”只支持本负载下序列名额参与约束；单轮 token budget 也可能影响调度，不能凭这次单配置实验单独量化贡献。要区分二者，需固定负载仅改变 `max-num-seqs` 或 `max-num-batched-tokens`，再比较相同到达率的 Running/Waiting、TTFT、ITL 和吞吐。本次数据也未测试 KV 容量边界或 preemption/recompute 的代价。

## 局限与收尾

- 这只是一个节点、一种模型、单实例、eager 配置、确定性 50:50 合成流量、60 秒窗口、三次重复。更长运行、非等间隔到达、真实 prompt/输出分布、其他 SLO 和默认 CUDA Graph 模式都可能改变达标区间。1.2 与 1.4 之间没有更细速率点；“最高通过”只相对本次网格。
- 预热仅 10 秒；1.6 档短预热曾暂时达标，但 60 秒正式轮连续严重不达标，说明短窗口可能掩盖队列积累。Goodput 的分母含排空时间，只适用于本协议口径。
- 服务已停止，云端 `/health` 不可达，`nvidia-smi` 显示 GPU 0 MiB / 0%；这**不代表云平台实例已经关机或停止计费**。须在 Vylai 控制台核实。
