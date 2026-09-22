# vLLM `max-num-seqs` 对照实验

环境：Qwen3-8B、单卡 RTX 4090 48GB、vLLM 0.29.0。除 `max-num-seqs` 外，服务配置相同。Mac 经 SSH 隧道访问云端 API。每组一次测试：同时提交 6 个请求，每个请求生成 384 tokens。

运行命令：

```bash
cd "/Users/jinian/Documents/llm learning/vllm-lab"
python3 phase3/scheduler_capacity.py --requests 6 --tokens 384
```

| 指标 | `max-num-seqs=4` | `max-num-seqs=8` |
| --- | ---: | ---: |
| 成功请求 | 6/6 | 6/6 |
| 最大 Running | 4 | 6 |
| 最大 Waiting | 2 | 0 |
| 容量原因导致的最大 Waiting | 2 | 0 |
| 总输出 tokens | 2304 | 2304 |
| 总耗时 | 13.672 s | 8.185 s |
| 聚合输出吞吐量（总 tokens / 总耗时） | 168.52 tokens/s | 281.49 tokens/s |
| 最大 KV cache 使用率 | 1.216% | 1.907% |
| 单请求耗时范围 | 6.806–13.507 s | 8.038–8.062 s |

这次单次实验中，把并行序列上限从 4 提至 8，允许 6 个请求同时运行，消除了因序列上限造成的等待；总耗时下降约 40.1%，聚合输出吞吐量提高约 67.0%。但前 4 个请求在 8 配置下各自耗时约 8.06 s，比 4 配置下先完成的请求约 6.84 s 更长，体现单请求延迟与整体吞吐量的取舍。

这些数值只代表此模型、硬件和固定输出长度的一次运行；没有重复测量，不能据此断言所有负载下 8 都优于 4。`finish_reason=length` 是实验固定生成 384 tokens 的预期结果，不是请求失败。最大 KV cache 使用率虽有所增加，但两组都远未用满，因此本次排队主要来自 `max-num-seqs` 上限。

原始数据：

- `scheduler_capacity_20260921-192356.json`：上限 4，运行 ID `20260921-192356-1789989836896097000`
- `scheduler_capacity_20260921-193949.json`：上限 8，运行 ID `20260921-193949-1789990789434227000`
