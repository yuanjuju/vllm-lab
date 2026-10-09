# N-gram 投机解码：高/低可预测性请求混合批次（R570 / vLLM 0.18.0 探索实验，2026-10-09）

> **版本边界**：这仍是平台分配 NVIDIA 570.124.06 驱动的单卡 RTX 4090 48GB、项目内 vLLM 0.18.0 / PyTorch 2.10.0+cu128 上的探索实验，不是 vLLM 0.29.0 的正式 Phase 5 结果。它是[上一轮同质负载 N-gram 对照](spec_ngram_exploratory_r570_v018_20261009.md)的**混合负载扩展**，不重新安装环境或下载模型。客户端与服务在云端 loopback，未测 Mac↔云链路。

## 为什么做、预期是什么

上一轮分别测试“全是容易预测的重复抄写”和“全是较难预测的连续数字”，但真实服务不会让所有请求都具有相同草稿接受率。混在同一个 Decode 批次里时，容易预测的请求可能很快结束，整批时间却仍被较慢请求决定。因此需要同时看**整体输出吞吐**和**按请求类别分开的端到端延迟**；仅看其中一个会错过另一半故事。

[vLLM 0.18.0 官方 N-gram 文档](https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/n_gram/)展示了从已有 token 匹配 N-gram 生成草稿的配置；[投机解码总览](https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/)也提醒收益依赖流量形态和采样设置。本轮只切换 N-gram，目标模型、服务预算、请求参数和混合比例均保持不变。

| 项目 | 两组共同条件 |
| --- | --- |
| 模型与版本 | 同一 Qwen3-8B BF16；vLLM 0.18.0 / Torch 2.10.0+cu128；单卡 RTX 4090，570.124.06 驱动 |
| 服务参数 | `--kv-cache-dtype auto`、`max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`、`--no-enable-prefix-caching`、`--enforce-eager`、`127.0.0.1:8000` |
| 唯一开关 | baseline `speculative_config=None` vs N-gram `method=ngram`、最多 4 个草稿 token、`prompt_lookup_min=1`、`prompt_lookup_max=4` |
| 每轮负载 | 同时释放 4 条重复抄写（约 565–568 输入 tokens）+ 4 条连续数字（约 66–69 输入 tokens）；全部 `temperature=0`、关闭 thinking、固定 256 输出 tokens |
| 实验协议 | 每组 1 轮预热 + 3 轮正式；相同 experiment ID 生成成对提示词；跨组比较只使用正式轮，保留每请求原始内容/哈希、SSE 时间、`/metrics` 快照和周期采样 |

两份启动日志直接确认了引擎实际配置、相同 **19.76 GiB KV 池 / 143,872 tokens** 和唯一投机开关：[baseline 日志](../../phase3/results/spec_ngram_mixed_baseline_r570_v018_20261009.log) · [N-gram 日志](../../phase3/results/spec_ngram_mixed_ngram_r570_v018_20261009.log)。日志归档仅规范化回车与行尾空白，云端 `logs/` 仍保留原始文件。

## 可复制步骤与验收

云端项目路径 `/root/fsas/vllm-lab`。启动前先检查实际驱动、CUDA 与服务端口，不能因上次是 R570 就假定本次仍兼容；本轮实测为 570.124.06，CUDA 可用，开始时 GPU 显存 2 MiB、端口空闲。服务与客户端分别在两个云端终端执行。先 baseline，停服务并核对显存回落，再把两处 `baseline` 改为 `ngram`；**两组 experiment ID 必须相同**。新一轮实验应改用新 ID，避免覆盖已有 JSON。

```bash
cd /root/fsas/vllm-lab
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
bash phase3/run_cloud_r570_vllm018_ngram_spec.sh baseline
```

```bash
cd /root/fsas/vllm-lab
python3 phase3/spec_ngram_mixed_exploratory.py \
  --condition baseline --experiment-id r570mix_20261009
```

服务就绪需同时满足日志配置行与 `/health`；验收要求每组 3 轮正式 × 8 条全部完整成功、每条 256 completion tokens 且 `finish_reason=length`、成对提示词 SHA-256 相同，并保存两组启动日志。脚本：[复用的服务启动脚本](../../phase3/run_cloud_r570_vllm018_ngram_spec.sh) · [新混合负载客户端](../../phase3/spec_ngram_mixed_exploratory.py)。原始数据：[baseline JSON](../../phase3/results/spec_ngram_mixed_r570mix_20261009_baseline.json) · [N-gram JSON](../../phase3/results/spec_ngram_mixed_r570mix_20261009_ngram.json)。客户端仅使用标准库及本仓库已有指标函数。

## 原始结果与解读

以下中位数都只来自每组 **3 次正式轮**，方括号为三轮最小–最大值。每轮输出吞吐 = 2048 个成功输出 tokens / 从同时释放请求到最后一条完成的整批墙钟时间；各类 TTFT/E2E 是轮内 4 条请求的 P50/P95，再跨 3 轮取中位数。TTFT 从客户端发请求到首个非空内容 SSE，E2E 到完整结束。

| 指标 | baseline | N-gram | 这组负载的含义 |
| --- | ---: | ---: | --- |
| 整批输出吞吐 (tok/s) | **400.532** [399.611–401.828] | **424.591** [420.715–425.574] | +6.0%；不是上一轮全重复负载的 3 倍以上 |
| 整批完成时间 (s) | 5.1132 [5.0967–5.1250] | 4.8235 [4.8123–4.8679] | 最慢的数字请求决定整批终点 |
| 重复请求 TTFT P50 / P95 (s) | 0.2672 / 0.2676 | 0.2730 / 0.2735 | 未观察到 TTFT 改善；差异只有数毫秒 |
| 重复请求 E2E P50 / P95 (s) | 5.0824 / 5.0828 | **1.4036 / 1.4037** | 容易预测的 4 条请求明显提前完成 |
| 数字请求 TTFT P50 / P95 (s) | 0.2676 / 0.2680 | 0.2731 / 0.2739 | 同样无明显 TTFT 收益 |
| 数字请求 E2E P50 / P95 (s) | 5.0820 / 5.0827 | **4.4587 / 4.7029** | 数字请求也略快，但仍是尾部；轮内 P95 仅 4 个样本 |
| 聚合草稿接受率 | — | **24.68%** [24.39%–24.68%] | `/metrics` 是整轮合计，**不能拆给两类请求** |

两组各 **24/24 正式请求成功**；对应的 **24 对提示词哈希完全相同、24 对输出文本哈希也完全相同**，每条均完整返回 256 tokens。两组正式轮 Running 峰值均为 8、采样 Waiting 峰值为 0、抢占 counter 增量为 0；KV Cache 峰值 baseline 为 3.16%–3.20%，N-gram 为 2.58%–2.71%。服务端队列时间直方图合计约 0.0001 s/轮，不能把这组差异解释为 KV 空间压力或明显排队。

一个可核验的机制证据是：正式第 2、3 轮 N-gram 的草稿接受 token 数均为 **1006**，服务端 `vllm:inter_token_latency_seconds` 观察次数均为 **1034**；baseline 同指标每轮 **2040** 次。对这组每条 256-token、共 8 条的输出，baseline 排除每条首 token 后为 `8×255=2040` 次，观测上满足 `2040−1006=1034`。这体现一次验证事件推进多个 token；也再次说明**不能把两组 ITL 直方图的均值直接当作同口径的逐 token 延迟**。客户端 `(E2E−TTFT)/255` 只是摊销的端到端代理量，含调度与传输，也不是硬件逐 token ITL。

## 局限和下一步

1. 这是刻意合成的 **4:4 同时释放批次**，没有请求到达率控制，也不是持续补充请求的标准 closed-loop 测试，更不是真实聊天/代码分布或生产 Goodput。官方文档称实际收益依赖模型、流量和采样设置；本轮的 +6.0% 仅对当前配置和负载成立。
2. 提示词类型长度不同，但**同一类请求在 baseline 与 N-gram 之间完全相同**；不能跨类别把延迟差直接解释为预测能力。聚合接受率也不能拆出两类各自数值；上一轮同质实验只是解释背景，不是本轮的分组计数。
3. 顺序固定为 baseline→N-gram，且每条件仅 3 轮。N-gram 预热轮 TTFT P50 为 1.4908 s，正式轮降至约 0.27 s；来源未单独剖析，预热未计入正式结论。单轮每类只有 4 条，P95 只是描述性尾部数。
4. `temperature=0` 的 24/24 文本精确一致不能替代复杂任务质量评估；未测试不同输出长度、到达率、混合比例、CUDA Graph 或其他驱动/版本，也不能宣称投机解码必然提升真实服务容量。下一步可用**固定到达率的 open-loop 混合负载和 TTFT/E2E SLO**测容量曲线，但应作为另一项实验，不能与这次批次吞吐混为一谈。

本轮两组服务均已停止，`127.0.0.1:8000/health` 不可达，`nvidia-smi` 显存回到 **2 MiB**。停止 vLLM 不等于云实例关机；须在平台控制台另行确认关机与停止计费，并轮换本次使用的一次性 SSH 密码。
