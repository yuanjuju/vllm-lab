# N-gram 投机解码：同一 Qwen3-8B 的开/关对照（R570 / vLLM 0.18.0 探索实验，2026-10-09）

> **版本边界**：本实验在平台分配的 NVIDIA 570.124.06 驱动、单卡 RTX 4090 48GB、项目内 vLLM 0.18.0 / PyTorch 2.10.0+cu128 上完成；**不计作 vLLM 0.29.0 的 Phase 5 正式实验**。客户端与服务均在云端，走 `127.0.0.1:8000` loopback；未测 Mac↔云端网络。没有安装新环境或下载新模型。

## 为什么做、机制是什么

普通 Decode 一轮通常为每条请求产出一个 token。N-gram 投机解码从当前序列的历史 token 中寻找匹配的后缀，取其后继 token 作草稿，再让同一个目标模型验证；草稿被接受时，一轮能推进多个 token。这里用目标模型 Qwen3-8B 自身的上下文作草稿，**没有额外 draft 模型**。[vLLM 0.18.0 投机解码说明](https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/)将它定位于低/中 QPS 等可能受 Decode 显存带宽制约的场景；[该版本 N-gram 配置示例](https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/n_gram/)展示了相同方法的服务端开关。收益取决于可预测性、验证开销和负载，并非开关一开就必然更快。

本实验提出两个相反负载：重复中文短句的“抄写”任务，预期历史匹配容易、接受率高；从 1 起连续输出自然数的任务，预期 token 后继较难从历史照抄、接受率低。两组条件之间**唯一主动改变的服务配置**是投机解码开关；每个负载只与自己的 baseline 比，不拿两个不同提示词长度互比。

| 固定项 | 实际条件 |
| --- | --- |
| 硬件和版本 | 同一 RTX 4090 48GB，驱动 570.124.06；`/root/fsas/vllm-lab/.venv-cu128-vllm018`，vLLM 0.18.0 / PyTorch 2.10.0+cu128 |
| 模型和缓存 | 同一个本地 Qwen3-8B BF16 目标模型；`--kv-cache-dtype auto`，Prefix Cache 关闭；两组启动日志均为 19.76 GiB KV 池、143,872 token 容量 |
| 其余服务参数 | `max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`、`--enforce-eager`，仅监听 `127.0.0.1:8000` |
| 唯一开关 | baseline 的 `speculative_config=None` vs N-gram 的 `method=ngram`、每次最多 4 个草稿 token、`prompt_lookup_min=1`、`prompt_lookup_max=4` |
| 请求协议 | `temperature=0`、thinking 关闭、每条 `min_tokens=max_tokens=256`；重复抄写输入约 565–568 tokens，数字连续输出约 66–69 tokens；每个条件 × 负载 × 并发度（1 或 8）各 1 轮预热 + 3 轮正式；对应请求提示词 SHA-256 完全相同 |

启动配置以两组日志中的 V1 引擎配置行为准，不能只凭命令行声明。日志显示 baseline `speculative_config=None`、N-gram `SpeculativeConfig(method='ngram', ..., num_spec_tokens=4)`，两组均为 `enable_prefix_caching=False`、`enforce_eager=True` 和同一个 KV 容量。[baseline 启动日志](../../phase3/results/spec_ngram_baseline_r570_v018_20261009.log) · [N-gram 启动日志](../../phase3/results/spec_ngram_ngram_r570_v018_20261009.log)。日志归档仅规范化了回车与行尾空白；云端 `logs/` 保留原始文件。

## 可复现步骤和验收

服务启动前检查驱动、GPU 和项目内环境；不要把服务暴露在公网，也不要将此前 v0.29.0 结果与本实验合并。云端项目路径为 `/root/fsas/vllm-lab`。终端 A 启动服务，待日志与 `/health` 显示就绪；终端 B 执行客户端。先 baseline，停服务并核对显存释放，再把两处 `baseline` 改为 `ngram`；两组保持**相同** experiment ID。新实验需换 ID，以免覆盖原始 JSON。

```bash
cd /root/fsas/vllm-lab
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
bash phase3/run_cloud_r570_vllm018_ngram_spec.sh baseline
```

```bash
cd /root/fsas/vllm-lab
python3 phase3/spec_ngram_ab_exploratory.py \
  --condition baseline --experiment-id r570spec_20261009
```

只改服务和客户端命令最后的 `baseline` 为 `ngram`。验收条件：每个条件各 4 个负载单元 × 3 正式轮，全部请求返回 256 completion tokens、`finish_reason=length`、完整 SSE；对应两组提示词哈希和输出文本哈希逐项核对；服务日志显示唯一开关；`/metrics` 的草稿/接受计数与实际吞吐相互支持。[服务启动脚本](../../phase3/run_cloud_r570_vllm018_ngram_spec.sh) · [客户端测量脚本](../../phase3/spec_ngram_ab_exploratory.py) · [baseline 原始 JSON](../../phase3/results/spec_ngram_ab_r570spec_20261009_baseline.json) · [N-gram 原始 JSON](../../phase3/results/spec_ngram_ab_r570spec_20261009_ngram.json)。脚本只依赖标准库及同目录已有的标准库指标函数。

服务结束后停服务并核对 API 不可达、GPU 显存回落。停止 vLLM **不等于云实例关机**；需在云平台控制台另行确认停止计费。

## 原始结果：三次正式轮，不混入预热

表中“吞吐范围”是三轮正式运行的最小–最大值；括号外是三轮中位数。输出吞吐 = 每轮成功输出 token 总数 / 从同时释放请求到整轮完成的墙钟时间，包含 Prefill 与可能的排队。TTFT 是客户端从发送到首个非空内容 SSE 的时间；同一轮 8 条请求的 P50/P95 再跨三轮取中位数。接受率 = `Δaccepted_tokens / Δdraft_tokens`，并非请求成功率。

| 负载 | 并发 | 条件 | 输出吞吐中位数 [范围] (tok/s) | TTFT P50 / P95 (s) | E2E P50 (s) | 草稿接受率 | 客户端 Decode 尾段/输出 token (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 重复抄写 | 1 | baseline | 54.662 [53.807–54.770] | 0.0769 / 0.0769 | 4.6740 | — | 18.0 |
| 重复抄写 | 1 | N-gram | 224.714 [222.681–231.631] | 0.0745 / 0.0745 | 1.0658 | 100% | 3.9 |
| 重复抄写 | 8 | baseline | 380.208 [379.050–381.674] | 0.4664 / 0.4668 | 5.3176 | — | 19.0 |
| 重复抄写 | 8 | N-gram | 1267.948 [1265.839–1280.376] | 0.4643 / 0.4647 | 1.5849 | 100% | 4.4 |
| 数字连续输出 | 1 | baseline | 53.374 [53.348–54.603] | 0.0534 / 0.0534 | 4.6849 | — | 18.1 |
| 数字连续输出 | 1 | N-gram | 62.051 [62.039–63.903] | 0.0374 / 0.0374 | 4.0297 | 5.27% | 15.7 |
| 数字连续输出 | 8 | baseline | 405.741 [405.302–406.533] | 0.1046 / 0.1049 | 4.9547 | — | 19.0 |
| 数字连续输出 | 8 | N-gram | 434.770 [434.357–444.862] | 0.0971 / 0.0976 | 4.4218 | 5.83% | 17.0 |

三轮中位数对照：重复抄写吞吐在并发 1 / 8 下分别为 **4.11× / 3.34×** baseline；数字连续输出为 **1.16× / 1.07×**。两组正式请求各 **54/54 成功**，对应的 **54 对提示词哈希相同、54 对输出文本哈希也相同**，每条均为 256 completion tokens 且以 `length` 结束。这证明本次测试样本的输出精确一致，不是对所有采样策略或题目质量的普遍保证。

高重复场景每请求 51 个草稿组、204 个草稿 token 全被接受，平均每组推进长度 `1 + 204/51 = 5` 个 token（含目标模型原本那一个）；这解释了为什么每条 256-token 输出的服务端 ITL 观察次数从 baseline 的 **255** 变成 N-gram 的 **51**。数字场景中，并发 1 正式轮的接受率为 5.17%–6.23%，并发 8 为 5.75%–5.97%，吞吐仍略高；这是**当前版本、配置和负载的观察**，不能推导为低接受率场景一律正收益。

两组正式轮 `num_preemptions` 增量全为 **0**，采样到的 Waiting 峰值全为 **0**，服务端请求队列时间直方图增量均值为 **0**；Running 峰值在并发 1 / 8 下分别为 1 / 8。KV Cache 峰值在重复负载并发 8 时 baseline / N-gram 约 **4.63% / 4.53%**，数字负载并发 8 时约 **1.87% / 1.82%**。这里没有 KV 压力或抢占，吞吐差异主要应从草稿被接受后减少 Decode 轮次来解释。

## 重要的指标口径与局限

1. **服务端 ITL 直方图不是跨条件可直接比较的“每 token 时间”**。vLLM 0.18.0 安装源码的 `vllm/v1/metrics/stats.py` 在非 Prefill 的每次 engine-core 输出事件记一次 `engine_core_timestamp - last_token_ts`，`loggers.py` 将这些事件逐次 observe 到 `vllm:inter_token_latency_seconds`。投机解码一次事件可含多个 token。因此高重复并发 1 的 histogram count 是 baseline 255、N-gram 51；N-gram 直方图均值约 19.5 ms 与 baseline 18.0 ms 虽接近，却分别对应不同的 token 数，不能据此说“投机解码 ITL 变慢”。表中的“Decode 尾段/输出 token”是客户端 `(E2E − TTFT)/(256−1)` 的**摊销代理量**，含流式传输和调度，不是真实单 token ITL；SSE 内容事件间隔同样不能冒充逐 token ITL。
2. **预热不可混入**。N-gram 第一条预热请求的 TTFT 为 4.9532 s，后续正式轮为 0.0732–0.0766 s；这提示首次使用可能存在额外初始化成本，但本实验未单独剖析其来源，不能断言一定是 JIT 编译。报告所有数字均不含预热。
3. **负载刻意合成、顺序固定**。重复抄写是高命中上界，不代表聊天、代码或真实业务；数字任务也不是“完全不命中”。先跑 baseline、后跑 N-gram，只有每单元 3 次重复，且未做交错运行或多日复测。并发 1 的 P95=P50 只是单请求的描述值，并发 8 的轮内 P95 样本也少。短输入数字任务与长输入抄写任务不应互相当成模型跑分。
4. **不声称成本/质量/生产收益**。本次未测试不同输出长度、不同 `num_speculative_tokens`、高 QPS/open-loop、CUDA Graph、其他版本或人类质量评估；`temperature=0` 下本样本 54/54 精确一致也不能证明复杂任务质量等价。投机解码需要额外验证计算；真实请求若很难预测，收益可能缩小甚至为负。vLLM 官方说明中的算法性质不等于任何硬件与数值内核下所有输出逐字不变。

本次两组服务均已停止；实测 `nvidia-smi` 显存回落为 **2 MiB**，`127.0.0.1:8000/health` 连接失败。云实例仍可能按开机计费，用户需在平台控制台确认关机并轮换这次使用的一次性 SSH 密码。
