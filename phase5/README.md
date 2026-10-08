# Phase 5：特性对照实验

Phase 4 读完源码后，回到 GPU 实例做"一次只动一个特性开关"的对照实验，把源码结论变成实测数字。客户端协议、SSH 隧道与服务启动约定沿用 [Phase 3](../phase3/README.md)；每个实验必须保留两组启动日志（云端 `logs/`）作为配置证据，预热与正式分离、3 次重复取中位数。

## 实验清单

| # | 特性 | 状态 | 报告 |
| --- | --- | --- | --- |
| 1 | `--enforce-eager` vs CUDA graph + torch.compile | ✅ 2026-09-29 | [eager_vs_cudagraph](results/eager_vs_cudagraph_20260929.md) |
| 2 | prefix cache 开/关（`--no-enable-prefix-caching`） | v0.29.0 正式实验待做；v0.18.0 探索对照已做 | [R570 探索性报告](results/prefix_cache_exploratory_r570_v018_20261008.md) |
| 3 | KV cache dtype fp8（`--kv-cache-dtype fp8`） | v0.29.0 正式实验待做；v0.18.0 探索对照已做 | [R570 KV dtype 报告](results/kv_dtype_exploratory_r570_v018_20261008.md) |
| 4 | 量化权重（如 GPTQ/AWQ 版模型） | 待做 | — |
| 5 | 投机解码 | 待做 | — |

## 已完成实验的一句话结论

1. **eager 每步 decode 多付 ~4.5 ms 的 kernel 启动开销**（batch 1→8 近似常数），吞吐 −16%~18%；TTFT 不受影响（prefill 摊薄启动成本），代价是引擎初始化 132.7 s vs 20.6 s。注意口径：`--enforce-eager` 同时关掉 torch.compile（`vllm/config/vllm.py:1370-1375`），差异是两项之和。

## 运行注意事项

- 每组实验结束立即 `pkill -f "[v]llm serve"`（方括号防自匹配）并核对 `nvidia-smi` 归零。
- 实例按量计费，全部实验结束后在平台控制台确认关机。
