# Phase 5：特性对照实验

Phase 4 读完源码后，回到 GPU 实例做"一次只动一个特性开关"的对照实验，把源码结论变成实测数字。客户端协议、SSH 隧道与服务启动约定沿用 [Phase 3](../phase3/README.md)；每个实验必须保留两组启动日志（云端 `logs/`）作为配置证据，预热与正式分离、3 次重复取中位数。

## 实验清单

| # | 特性 | 状态 | 报告 |
| --- | --- | --- | --- |
| 1 | `--enforce-eager` vs CUDA graph + torch.compile | ✅ 2026-09-29 | [eager_vs_cudagraph](results/eager_vs_cudagraph_20260929.md) |
| 2 | prefix cache 开/关（`--no-enable-prefix-caching`） | v0.29.0 正式实验待做；v0.18.0 探索对照已做 | [R570 探索性报告](results/prefix_cache_exploratory_r570_v018_20261008.md) |
| 3 | KV cache dtype fp8（`--kv-cache-dtype fp8`） | v0.29.0 正式实验待做；v0.18.0 探索对照已做 | [R570 KV dtype 报告](results/kv_dtype_exploratory_r570_v018_20261008.md) |
| 4 | 量化权重（Qwen3-8B BF16 vs 官方 AWQ 4-bit） | v0.29.0 正式实验待做；v0.18.0/R570 探索对照已做 | [R570 权重量化报告](results/weight_awq_exploratory_r570_v018_20261009.md) |
| 5 | N-gram 投机解码 | v0.29.0 正式实验待做；v0.18.0/R570 探索对照及混合负载扩展已做 | [同质负载报告](results/spec_ngram_exploratory_r570_v018_20261009.md) · [混合负载报告](results/spec_ngram_mixed_exploratory_r570_v018_20261009.md) |

## 已完成实验的一句话结论

1. **eager 每步 decode 多付 ~4.5 ms 的 kernel 启动开销**（batch 1→8 近似常数），吞吐 −16%~18%；TTFT 不受影响（prefill 摊薄启动成本），代价是引擎初始化 132.7 s vs 20.6 s。注意口径：`--enforce-eager` 同时关掉 torch.compile（`vllm/config/vllm.py:1370-1375`），差异是两项之和。
2. **R570/v0.18.0 探索性 AWQ 对照**：在同一实例与相同 BF16 KV、长输入/固定输出负载下，模型加载显存 15.27→5.71 GiB，自动 KV 池 19.76→29.32 GiB；3 次正式轮输出吞吐中位数 189.550→204.312 tok/s（+7.8%）。质量 6/6 小题仅是冒烟检查；正式 v0.29.0 结论仍待补。
3. **R570/v0.18.0 探索性 N-gram 对照**：同一 Qwen3-8B BF16 目标模型、256-token 固定输出下，高重复抄写接受率 100%，并发 1 / 8 的吞吐为 baseline 的 4.11× / 3.34×；数字连续输出接受率约 5%–6%，吞吐为 1.16× / 1.07×。54 对正式输出文本哈希一致；服务端 ITL 直方图计数在投机解码下不再逐 token，详见报告。正式 v0.29.0 结论仍待补。
4. **R570/v0.18.0 混合负载扩展**：同时提交 4 条重复抄写 + 4 条数字请求时，N-gram 让重复请求 E2E P50 从 5.0824 降至 1.4036 s，但整批吞吐仅 400.532→424.591 tok/s（+6.0%），因为数字请求仍决定整批终点。24 对正式输出哈希一致；这是合成的同步批次，不是 open-loop 服务容量。

## 运行注意事项

- 每组实验结束立即 `pkill -f "[v]llm serve"`（方括号防自匹配）并核对 `nvidia-smi` 归零。
- 实例按量计费，全部实验结束后在平台控制台确认关机。
