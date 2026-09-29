# 特性对照 #1：`--enforce-eager` vs CUDA Graph（默认）

日期：2026-09-29。环境：casdao 单卡 RTX 4090（49140 MiB），vLLM 0.29.0，torch 2.13.0+cu132，Qwen3-8B bf16。客户端在 Mac 上经 SSH 隧道（127.0.0.1:18000 → 云端 127.0.0.1:8000）访问，两组走完全相同的网络路径。

## 实验设计

单变量对照：服务端配置逐项相同（`--max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-seqs 8 --enable-chunked-prefill --max-num-batched-tokens 4096`），唯一差别是 B 组追加 `--enforce-eager`。

| 组 | 启动参数 | 引擎初始化 | 启动日志证据（实例 `logs/` 下） |
| --- | --- | --- | --- |
| A：CUDA graph 开 | 默认 | **132.71 s** | `phase5_graphA_20260929.log`：`enforce_eager=False`；`Capturing CUDA graphs (PIECEWISE): 5/5` + `(FULL)`，档位 `[1,2,4,8,16]`；torch.compile 3.07 s（命中 `~/.cache/vllm/torch_compile_cache` 的 AOT 缓存） |
| B：eager | `--enforce-eager` | **20.56 s** | `phase5_eagerB_20260929.log`：`enforce_eager=True`；`Capturing` 行数为 0；日志原话 `Enforce eager set, disabling torch.compile and CUDAGraphs` |

**口径说明（源码）**：`vllm/config/vllm.py:1370-1375` —— `enforce_eager` 同时置 `CompilationMode.NONE`（关 torch.compile 算子融合）与 `CUDAGraphMode.NONE`（关 graph 捕获/重放）。因此下述差异是**两者合计**的效果，不能全部记在 CUDA graph 头上。

负载协议沿用 [phase3 曲线实验](../../phase3/results/benchmark_capacity_curve_20260923.md)：closed-loop，12 请求 × 固定 64 输出 token，每组先 1 次预热（不入统计）再 3 次正式，取中位数。客户端脚本 `phase3/load_curve_benchmark.py`，原始 JSON 见文末清单（写入 `phase3/results/`）。全部 192/192 请求成功，两组均 0 抢占，c8 时 `max_running=8`。

## 结果（3 次正式的中位数）

| 指标 | A：graph+compile | B：eager | 差异 |
| --- | --- | --- | --- |
| c1 输出吞吐 | 51.98 tok/s | 43.60 tok/s | **−16.1%** |
| c1 E2E P50 | 1.165 s | 1.464 s | +25.7% |
| c1 TTFT P50 | 0.111 s | 0.119 s | ≈持平（噪声内） |
| c8 输出吞吐 | 289.76 tok/s | 236.92 tok/s | **−18.2%** |
| c8 E2E P50 | 1.349 s | 1.630 s | +20.9% |
| c8 TTFT P50 | 0.237 s | 0.245 s | ≈持平（噪声内） |

单次吞吐明细（tok/s）：A/c1 = 45.8, 52.0, 55.3；A/c8 = 295.4, 283.5, 289.8；B/c1 = 43.7, 43.5, 43.6；B/c8 = 239.1, 235.3, 236.9。B 组重复间波动 <1%，A/c1 首轮（45.8）偏慢——预热后仍有残留升温，中位数可抑制。

## 每步 decode 耗时推算

E2E − TTFT 即 63 个 decode step 的耗时（首个 token 计入 TTFT）：

| 场景 | A：每步 | B：每步 | 差值 |
| --- | --- | --- | --- |
| c1（batch=1） | 16.7 ms | 21.3 ms | **+4.6 ms** |
| c8（batch=8） | 17.6 ms | 22.0 ms | **+4.3 ms** |

## 结论

1. **eager 的代价集中在 decode，且近似每步常数 ~4.5 ms**：batch 从 1 到 8，每步劣化几乎不变（4.6 → 4.3 ms）。这说明省掉的是 **CPU 端 kernel 启动开销**，不是 GPU 算力——36 层模型的 kernel 个数不随 batch 变，graph 重放把整串启动压成一次；batch 越大、单步计算越重，这 ~4.5 ms 占比越低，但本实验 1→8 区间内吞吐损失（16%→18%）基本持平。
2. **TTFT 不受影响**：prefill 一次前向处理整段 prompt（~173 token），少量大 kernel 的启动开销被真实计算摊薄，graph 与否无感。**graph 是 decode 的优化，不是 prefill 的**。
3. **代价在启动**：A 引擎初始化 132.71 s vs B 20.56 s（A 还命中了编译缓存，彻底冷启动首建缓存更慢）。生产常驻服务摊得平；短命进程、调试场景 eager 更合适。
4. **尾延迟视角**：两组 c8 的 P95≈P50（A: 1.336/1.335；B: 1.679/1.678）——8 请求整批准入、每步 batch 恒定，延迟高度整齐。本负载太"整齐"，不足以区分 graph 对尾延迟的贡献。

## 与源码/笔记的对应

- `vllm/config/vllm.py:1370-1375`：enforce_eager 的两连关（本报告口径）。
- 捕获档位 `[1,2,4,8,16]` 即 c8 命中的 batch=8 图（`phase4/notes/02-gpu-model-runner.md` 的 `_is_uniform_decode` 判定）。
- 每步一次 forward = 36 层循环（`qwen2.py:397-434`）：decode 时每 token 都要跑完这 36 层的小 kernel 串，是 launch 开销的来源。

## 局限

- A/B 共享一台实例先后执行（非同时），负载独占、间隔数分钟，硬件状态漂移风险低但非零。
- 端到端含 Mac、SSH 隧道开销；两组同路径，相对比较有效，绝对值高于纯服务端时延。
- 64 token 输出偏短，c1 的 TTFT 占 E2E 约 8%~10%，放大了 E2E 对 TTFT 噪声的敏感度。
- 编译缓存命中使 A 的启动成本被低估（首次部署需额外付一次编译时间）。

## 数据文件

`phase3/results/load_curve_20260929-16*`（A 组 16:40–16:42 的 c1/c8、B 组 16:48–16:49 的 c1/c8；各含 1 warmup + 3 formal）。
