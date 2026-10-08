# KV Cache dtype：BF16/auto vs FP8（R570 / vLLM 0.18.0 探索实验，2026-10-08）

> **版本边界**：这是 R570 宿主机上临时 vLLM 0.18.0 + PyTorch 2.10.0/cu128 环境的成对实验，不计入原计划的 vLLM 0.29.0 Phase 5 B2 正式完成状态。服务端只监听云端 `127.0.0.1:8000`；没有变动 Mac、系统 Python、原云端 `.venv` 或模型权重。

## 问题与机制

KV Cache 保存每个已处理 token 的 K/V；权重保持 BF16，不等于 KV 也必须 BF16。Qwen3-8B 配置下，按 36 层、8 个 KV heads、head dimension 128 估算，BF16 KV 每 token 是 `2 × 36 × 8 × 128 × 2 = 147456 bytes`（144 KiB）；FP8 按 1 byte/元素约减半。4 GiB KV 池的理论上限约 29,127/58,254 tokens，vLLM 按 16-token block 取整后，这次启动日志实际给出 **29,120/58,240 tokens**（1,820/3,640 blocks）。这些是 *KV 容量*，不是模型权重显存，也不是实际可承受请求数。

FP8 的数值范围需要缩放。本实验两组都传入 `--calculate-kv-scales`：在 `auto`/BF16 组不产生 FP8 缩放效果，在 FP8 组按 vLLM 0.18.0 的行为于预热时估算 K/V scales。这避免了无 checkpoint scales 时直接使用默认 `1.0`，但**不是使用代表性数据集校准，更不是质量保证**。对照该版本的[官方 CLI 说明](https://docs.vllm.ai/en/v0.18.0/cli/serve/)与[量化 KV Cache 文档](https://docs.vllm.ai/en/v0.18.0/features/quantization/quantized_kvcache/)。

## 实验协议与运行步骤

| 固定项 | 值 |
| --- | --- |
| 硬件/环境 | 单卡 RTX 4090 48GB，驱动 570.124.06；项目内 `.venv-cu128-vllm018`，vLLM 0.18.0，PyTorch 2.10.0+cu128 |
| 模型/服务 | 本地 Qwen3-8B，BF16 权重；`max-model-len=4096`，`max-num-seqs=8`，`max-num-batched-tokens=4096`，`--enforce-eager`，Prefix Cache 关闭 |
| KV 内存 | 两组均固定 `--kv-cache-memory-bytes 4G`，即 4 GiB；vLLM 日志明确此参数会跳过内存 profiling、取代 `gpu-memory-utilization` 对 KV 池的自动预算 |
| 唯一变化 | `--kv-cache-dtype auto`（本模型对应 BF16）vs `--kv-cache-dtype fp8`；两组其他启动参数由日志确认一致 |
| 负载 | 每轮 8 个同时释放的独立长请求，实际输入约 3101–3105 tokens/条、固定输出 128 tokens/条；`temperature=0`、关闭 thinking |
| 测量 | 两组各 1 轮预热 + 3 轮正式；另有相同的 6 个确定性质量探针，含 2 个长文本验证码回忆；云端本机 loopback 客户端，不含 Mac↔云端网络 |

在云端 `/root/fsas/vllm-lab`，先确认 GPU/驱动与该项目环境可用，且无其他 vLLM 服务。两个终端分别运行服务和客户端；待 `/health` 就绪后测量。第一组：

```bash
cd /root/fsas/vllm-lab
bash phase3/run_cloud_r570_vllm018_kv_dtype.sh auto
```

```bash
cd /root/fsas/vllm-lab
python3 phase3/kv_dtype_ab_exploratory.py --condition auto --experiment-id r570kv_20261008
```

停止第一组服务，确认 8000 端口关闭、GPU 显存回落，再运行 FP8 组：

```bash
cd /root/fsas/vllm-lab
bash phase3/run_cloud_r570_vllm018_kv_dtype.sh fp8
```

```bash
cd /root/fsas/vllm-lab
python3 phase3/kv_dtype_ab_exploratory.py --condition fp8 --experiment-id r570kv_20261008
```

脚本/证据：[服务脚本](../../phase3/run_cloud_r570_vllm018_kv_dtype.sh)、[测量脚本](../../phase3/kv_dtype_ab_exploratory.py)、[auto 原始 JSON](../../phase3/results/kv_dtype_ab_r570kv_20261008_auto.json)、[FP8 原始 JSON](../../phase3/results/kv_dtype_ab_r570kv_20261008_fp8.json)、[auto 启动日志](../../phase3/results/kv_dtype_auto_r570_v018_20261008.log)、[FP8 启动日志](../../phase3/results/kv_dtype_fp8_r570_v018_20261008.log)。相同 `experiment-id` 使两组对应请求的 prompt SHA-256 完全一致；重复运行须换 ID，以免覆盖 JSON。脚本依赖同目录已有的 `prefix_cache_ab_exploratory.py` 指标函数，均只用 Python 标准库。

## 实测结果

每个正式轮的吞吐口径为 `8 × 128 / 整批完成时间`，包含等待、Prefill 和 Decode；客户端 TTFT 是首个有内容的 SSE 事件，不等于纯 GPU Prefill 时间。服务端 ITL 从 `/metrics` 直方图增量求均值，不把 SSE 内容事件间隔冒充逐 token ITL。

| KV dtype | 正式轮 | TTFT P50 / P95 (s) | 整批耗时 (s) | 输出吞吐 (tok/s) | KV 峰值 | Waiting 峰值 | 抢占增量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto/BF16 | 1 | 1.7887 / 2.4909 | 5.4047 | 189.465 | 88.84% | 5 | 0 |
| auto/BF16 | 2 | 1.7131 / 2.5652 | 5.3817 | 190.275 | 88.84% | 5 | 0 |
| auto/BF16 | 3 | 1.7856 / 2.4903 | 5.3783 | 190.396 | 88.84% | 5 | 0 |
| FP8 | 1 | 1.8199 / 2.5197 | 5.1854 | 197.477 | 44.41% | 5 | 0 |
| FP8 | 2 | 1.6808 / 2.4445 | 5.1333 | 199.482 | 44.41% | 4 | 0 |
| FP8 | 3 | 1.7785 / 2.4814 | 5.1387 | 199.271 | 44.41% | 5 | 0 |

三轮中位数：TTFT P50 为 **1.7856 vs 1.7785 s**，差异很小；输出吞吐为 **190.275 vs 199.271 tok/s**，FP8 在此负载下约高 **4.7%**，不能叫作“速度翻倍”。服务端 ITL 均值的三轮中位数为 **0.0288 vs 0.0268 s**。两组正式请求均 **24/24 成功**、每条固定 128 输出 tokens；6 个简短/长文本质量探针两组均 **6/6 精确匹配**。这只覆盖少量可核对题目，不代表一般生成质量无损。

两组采样 Running 峰值均为 8；Waiting 约 4–5，队列时间直方图也有正值，但 `num_preemptions` 增量均为 0。**Waiting 不是 KV 抢占**：FP8 在 KV 使用率约 44% 时仍有 Waiting，说明本负载的分批输入/单轮 token budget 等调度约束也在起作用；不能把排队全归因于 BF16 的 KV 空间。采样峰值并不保证捕获所有瞬态。

单次启动日志中，引擎初始化（含 warmup）auto 为 1.45 s，FP8 为 31.42 s；FP8 日志另外记录 FlashInfer attention warmup。这个一次性启动差异可能与 FP8 缩放/内核预热有关，但并未做多次启动对照，**不作为稳定冷启动结论**。

## 怎样理解容量、速度与局限

1. **容量的证据强，速度的证据有限**：同样 4 GiB KV 字节预算，FP8 实际可放的 token 数正好翻倍，运行时同负载的 KV 使用率约减半；三次正式轮吞吐方向也一致，但优势只有约 4.7%，且固定顺序 auto→FP8、样本量小，不能推断普适加速比。
2. **池容量翻倍 ≠ 本次最大并发翻倍**：`max-num-seqs=8` 仍是请求名额上限，而且这组 8 条约 3.1k-token 输入加输出，在 BF16 29,120-token 池内也能完成、未发生抢占。要测极限并发须另做安全递增的容量曲线，不能从池大小直接宣称服务容量翻倍。
3. **同样 4 GiB KV 预算不等于整卡显存减半**：权重始终 BF16，KV 池两组都保留 4 GiB；本对照把省下的每-token 字节用于增加可容纳 token 数，而不是减少这 4 GiB 的预留内存。原先自动预算 19.76 GiB KV 池的结果与本次手动 4 GiB 池不可直接拼表。
4. **质量只做了冒烟检验**：FP8 使用启动时动态 scales，没有代表性数据集校准。6/6 的小题结果不足以证明长文、领域任务或整体生成质量等价。后续若考虑实际部署，需要更大、事先固定的质量集。
5. **版本与负载边界**：本实验为 vLLM 0.18.0 / R570 / eager / 关闭 Prefix Cache / 长输入固定输出，客户端在云端 loopback；不同驱动、版本、CUDA Graph、前缀复用或到达率下可能改变方向。每轮仅 8 个请求，P95 只是描述性分位值。

收尾时两组服务均已停止，API 8000 端口不可达，`nvidia-smi` 显存回到约 2 MiB。**云平台实例关机与计费停止须由用户在控制台确认；停止 vLLM 不等于停止实例。**
