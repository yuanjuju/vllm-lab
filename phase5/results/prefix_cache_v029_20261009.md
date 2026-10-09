# Prefix Cache 开/关：vLLM 0.29.0 / RTX 4090 正式对照（2026-10-09）

> **版本边界**：这是 Phase 5 B1 的 **vLLM 0.29.0 正式实验**（驱动 590.44.01），已补齐与 10-09 控制组的配对。它与 [R570 / v0.18.0 探索性对照](prefix_cache_exploratory_r570_v018_20261008.md)是不同 vLLM 版本、不同驱动的两组数据，**不得拼接或互相引用数字**。

## 为什么测

Phase 3 已经证明共享前缀会产生命中（实测 1792/1813），Phase 4 从源码解释了命中规则；但"命中"与"性能收益"是两件事。这里在相同模型、相同提示词、相同输出长度下**只切换一个开关**，回答：Phase 3 测到的命中值多少 TTFT 与吞吐。

## 实验条件与复现

| 项目 | 固定条件 |
| --- | --- |
| 实例 | 云端单卡 RTX 4090（49140 MiB）；驱动 590.44.01 |
| 环境 | 项目内 `.venv`：vLLM 0.29.0、PyTorch 2.13.0+cu132、`torch.cuda.is_available()=True` |
| 模型 | 本地 Qwen3-8B，BF16；`max-model-len=4096` |
| 服务参数 | `gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`、`--enforce-eager`、仅监听云端 `127.0.0.1:8000` |
| 唯一开关 | 开启组不加参数（默认 `enable_prefix_caching=True`）；关闭组加 `--no-enable-prefix-caching`。两组均以启动日志 `non-default args` 行核对 |
| 负载 | 每轮先发 1 条 prime 建立前缀；接着 4 条并发探针共享同一前缀（各 1421 tokens 输入）；之后发 1 条**改前缀**负对照（1417 tokens）。`temperature=0`、关闭 thinking、每条固定输出 64 tokens |
| 重复与链路 | 各组先独立预热一次，再正式重复 3 次；客户端与服务同在云端，走 loopback |
| 证据 | [关闭 JSON](../../phase3/results/prefix_cache_ab_v029pfx_20261009_off.json) · [开启 JSON](../../phase3/results/prefix_cache_ab_v029pfx_20261009_on.json) · [关闭服务日志](../../phase3/results/phase5_v029_base_20261009.log) · [开启服务日志](../../phase3/results/phase5_v029_prefix_on_20261009.log) · [客户端日志](../../phase3/results/phase5_client_prefix_on_20261009.log) |

同一 `experiment-id`（`v029pfx_20261009`）使两组对应轮次的提示词完全一致；提示词构造与 v0.18.0 那组相同，**但 tokenizer 模板版本不同，本组输入是 1421/1417 tokens，不是旧组的 1423**。

复现（云端，服务就绪后）：

```bash
cd /root/fsas/vllm-lab/phase3/v029_clients
python3 prefix_cache_ab_exploratory.py --condition on --experiment-id v029pfx_20261009 \
  --server-profile v029-r590-eager-prefix-on
```

切换服务前先 `pkill -f "[v]llm serve"` 并核对显存回落；关闭组把 `--condition` 换成 `off`。

## 正式结果

下表每项是**同轮 4 条并发探针**的统计，取 3 次正式轮的中位数；TTFT 是客户端首个非空 SSE 内容事件，吞吐是 `4 × 64 / 整批完成秒数`，不是纯 Decode 速度。预热轮不计入。

| Prefix Cache | 客户端 TTFT P50 (s) | TTFT P95 (s) | 整批耗时 (s) | 输出吞吐 (tok/s) | 命中 / 查询 tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| **关闭**（控制组） | 0.5628 | 0.5888 | 2.0540 | 124.633 | 0 / 0（计数未启用） |
| **开启**（实验组） | **0.0946** | **0.0949** | **1.4853** | **172.358** | 5568 / 5684 |
| 变化 | −83.2% | −83.9% | −27.7% | **+38.3%** | 命中率 **97.96%** |

三次正式轮的离散度极小（开启组 TTFT P50：0.0924 / 0.0946 / 0.0956；吞吐 172.149 / 172.358 / 172.556），关闭组 TTFT P50：0.5600 / 0.5628 / 0.5643。

负对照（改前缀，1417 tokens）：命中 **16** / 查询 1417，即 **1.1%**——缓存复用基本被切断。

**逐 token 核对（用真实 tokenizer 独立验证）。** 探针 1421 tokens，prime 1416 tokens；两者**前 1392 个 token 完全相同**，即 87 个完整 16-token 块。实测每次探针命中 5568/4 = **1392**，查询 5684/4 = **1421**，与预测逐数字吻合。

改前缀请求的 16 tokens 也完全解释得通：相邻两轮的改前缀提示词**只差 `{label}` 字符串**（`warmup` vs `formal-0`），tokenizer 显示首个差异落在第 21（warmup→formal-0）／23（formal-0→formal-1）个 token——都在**块 1 内部，不在块 0**。因此块 0 的 16 个 token 相同可复用，块 1 起被链式哈希全部污染。预热轮该值为 0，因为当时缓存里还没有任何"上一轮的改前缀请求"。

## 学到什么，怎样判读

1. **命中率 97.96% 不是一个数字，而是 87 个完整块的必然结果。** vLLM 只对**完整块**计算哈希（`vllm/v1/core/kv_cache_utils.py:829`），并对命中长度施加 `max_cache_hit_length = num_tokens − 1`（`vllm/v1/core/kv_cache_manager.py:252-258`，注释明确说明这是为了"最后一个 token 必须重算取 logits"，代价是可能整整退回一个块）。1392 = 87 × 16 正是这条规则 + 块对齐的乘积。

2. **收益全部落在 Prefill 上，Decode 不受影响。** TTFT 降 83%，而整批耗时只降 27.7%：探针仍各要生成 64 tokens，Decode 部分不因前缀命中而变快。吞吐 +38.3% 是"省下的 Prefill 时间被用来多跑请求"的结果，不是 Decode 提速。

3. **一个字符的差异，代价是整个后续前缀。** 负对照把开头的 `PrefixAB` 改成 `ChangedPrefixAB`，命中立刻从 97.96% 崩到 1.1%。链式块哈希（`kv_cache_utils.py:621,647`：`hash_function((parent_block_hash, block_tokens, extra_keys))`）意味着**差异点之后的每一块都失效**，无论内容是否相同。工程含义很直接：把变化的部分尽量放到提示词**末尾**，不要放在开头。

4. **命中不改变 KV 占用方向上的直觉但改变了占比。** 开启组 `max_kv_usage` 峰值 0.0124，反而低于关闭组的 0.0414——因为 Prefill 被跳过，同一时刻驻留在批里的 token 更少。缓存块本身仍占空间，只是摊销到了更多请求上。

## 验收与局限

- 两组各 4 条探针 × 3 次正式轮 = 12/12 成功且均生成 64 tokens；prime 与改前缀对照同样成功。两组配置证据分别取自启动日志 `non-default args` 行：关闭组显式 `'enable_prefix_caching': False`，开启组**该键不出现**（即默认值；默认 `True` 见 `vllm/config/cache.py:138`）。
- **局限一：全部实验在 `--enforce-eager` 下完成**，即关闭了 torch.compile 与 CUDA Graph。这个口径与 Phase 5 其他 v0.29.0 组一致，组内单变量对照有效，但**不代表 vLLM 默认配置下的绝对数字**；eager 的代价已在 [Phase 5 #1](eager_vs_cudagraph_20260929.md) 量化（decode 每步约 +4.5 ms）。
- **局限二：负载是合成的固定形状**（1421 tokens 输入 / 64 tokens 输出 / 4 并发），且提示词前缀由脚本确定性构造以保证逐轮可复现。它演示机制，**不构成生产 QPS 或成本结论**；真实流量下前缀复用率取决于业务提示词的共享结构。
- **局限三：客户端 TTFT 含排队与 SSE 事件粒度**，SSE 内容事件不等于单 token；服务端直方图只能给 bucket 上界。
- 本组只回答 Prefix Cache 开关；它与 `max-num-seqs`、`max-num-batched-tokens` 的交互未测。
