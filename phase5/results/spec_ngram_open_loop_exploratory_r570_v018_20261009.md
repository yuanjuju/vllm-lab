# 固定到达率下的混合负载与 N-gram SLO 边界（R570 / vLLM 0.18.0 探索实验，2026-10-09）

> **版本边界**：这是单卡 RTX 4090 48GB、驱动 570.124.06、项目内 vLLM 0.18.0 / PyTorch 2.10.0+cu128 的探索对照，**不是 vLLM 0.29.0 的正式 Phase 5 结果**。客户端与服务同在云端，使用 `127.0.0.1:8000` loopback。没有安装新环境或下载模型。本实验延续[同质负载](spec_ngram_exploratory_r570_v018_20261009.md)与[同步混合批次](spec_ngram_mixed_exploratory_r570_v018_20261009.md)，但三者不能合并为同一次测量。

## 为什么要从“同时 8 条”改成固定到达率

上轮同步提交 8 条请求，回答的是“一批何时全部结束”。真实服务面对的是新请求不断到来：即使旧请求还在 Decode，新请求仍会进入系统。若到达率高于处理能力，Waiting 与队列时间会在窗口内积累，TTFT 的尾部首先恶化；一批全部成功并不代表满足延迟目标。本轮用独立于请求完成状态的**绝对时间表**发请求，找在给定负载和示例 SLO 下观察到的边界。

[vLLM 0.18.0 `bench serve` 文档](https://docs.vllm.ai/en/v0.18.0/cli/bench/serve/)同样区分 `--request-rate` 与 `--max-concurrency`，并提醒并发上限可能使实际发起速率低于设定值。本轮为固定 50:50 提示词混合和逐类延迟记录，使用仓库内标准库客户端，不设客户端并发上限；它采用**确定性等间隔到达**，不是 Poisson 流量。每条记录同时保存计划时间、实际 dispatch 和 worker 启动时间，以验证客户端确实按计划发送。

## 预先固定的实验协议

| 项目 | 两组共同条件 |
| --- | --- |
| 实例/模型 | 同一 RTX 4090 48GB、570.124.06 驱动；同一 Qwen3-8B BF16 目标模型、BF16 KV（`--kv-cache-dtype auto`） |
| 服务预算 | `max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`、Prefix Cache 关闭、`--enforce-eager`、只监听 loopback |
| 唯一服务开关 | baseline 的 `speculative_config=None` vs `method=ngram`、`num_spec_tokens=4`、`prompt_lookup_min=1`、`prompt_lookup_max=4` |
| 到达过程 | 0.5、1.0、1.5、2.0 请求/秒；等间隔，不等待前一个请求完成；每个正式窗口 20 秒，分别计划 10、20、30、40 条；交替发送重复抄写/连续数字，比例 50:50 |
| 请求 | 对应两组使用同一提示词 SHA-256；`temperature=0`、关闭 thinking、每条 `min_tokens=max_tokens=256`；输入长短不同，但同一负载档位的 A/B 完全一致 |
| 重复 | 每档先跑 8 秒预热，再做 3 次 20 秒正式轮；速率从低到高；结果 JSON 在每轮后原子保存，预热不参与正式统计 |
| 教学用 SLO | 成功、TTFT ≤ **1 秒**且 E2E ≤ **10 秒**；同时关注该轮整体 TTFT P95 ≤ 1 秒。阈值是本次预先声明的**演示目标**，不是现有业务承诺 |
| 安全边界 | 每正式轮最多 40 条；监测到 KV 使用率 ≥75%、Waiting >16 或 `num_preemptions` 增量 >0 就停止后续派发；任一轮请求失败或 drain 超时则不升下一档 |

服务启动日志确认了实际 V1 引擎配置：baseline `speculative_config=None`，N-gram `SpeculativeConfig(method='ngram', ..., num_spec_tokens=4)`；两组 `enable_prefix_caching=False`、`enforce_eager=True`，KV 池同为 **19.76 GiB / 143,872 tokens**。[baseline 启动日志](../../phase3/results/spec_ngram_open_loop_baseline_r570_v018_20261009.log) · [N-gram 启动日志](../../phase3/results/spec_ngram_open_loop_ngram_r570_v018_20261009.log)。云端 `logs/` 保留原始日志；归档副本只规范化回车与行尾空白。

## 复现与验收

先确认真实驱动、项目内 vLLM/CUDA、GPU 空闲及 8000 端口；不要盲目按旧实例版本安装。云端项目目录 `/root/fsas/vllm-lab`。终端 A 启动服务并检查日志配置与 `/health`，终端 B 运行客户端。先 baseline，确认结束后停服务、显存回落，再把两条命令的 `baseline` 改为 `ngram`。同一对照必须使用**相同 experiment ID**；新的实验用新 ID，避免覆盖 JSON。

```bash
cd /root/fsas/vllm-lab
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
bash phase3/run_cloud_r570_vllm018_ngram_spec.sh baseline
```

```bash
cd /root/fsas/vllm-lab
python3 phase3/spec_ngram_open_loop_exploratory.py \
  --condition baseline --experiment-id r570open_20261009
```

[固定到达率客户端](../../phase3/spec_ngram_open_loop_exploratory.py)复用[既有服务脚本](../../phase3/run_cloud_r570_vllm018_ngram_spec.sh)。原始数据：[baseline JSON](../../phase3/results/spec_ngram_open_loop_r570open_20261009_baseline.json) · [N-gram JSON](../../phase3/results/spec_ngram_open_loop_r570open_20261009_ngram.json)；客户端进度日志：[baseline](../../phase3/results/spec_ngram_open_loop_client_baseline_r570_v018_20261009.log) · [N-gram](../../phase3/results/spec_ngram_open_loop_client_ngram_r570_v018_20261009.log)。

验收先看测量有没有失真：两组各 16 轮（4 档 × 1 预热 + 3 正式），所有正式窗口均派发计划数；两组各 **300/300 正式请求成功**，每条均完整返回 256 completion tokens，**300 对提示词哈希、300 对输出文本哈希完全相同**。各正式轮实际 worker 启动时间偏差 P95 不超过约 **0.5 ms**，所以标称到达率确实由客户端执行；没有因为客户端并发门控而偷偷变成 closed-loop。

## 原始结果：三轮正式中位数

下表每个数值是三轮正式结果的中位数，不含预热。TTFT/E2E P95 是每轮全部 10/20/30/40 条请求的客户端分位值，再跨三轮取中位数；低速率 P95 尤其是描述性指标。输出吞吐 = 本轮成功输出 tokens / **到达窗口加最后请求 drain 的整段墙钟时间**，所以低速档不应直接拿它当“饱和服务能力”。“SLO 达标”是同时满足 TTFT≤1 s、E2E≤10 s 的请求比例；**本轮 SLO goodput** = 达标请求数 / 同一整段墙钟时间，也不是无限期持续 goodput。

| 到达率 (req/s) | 条件 | TTFT P95 (s) | E2E P95 (s) | SLO 达标 | SLO goodput (req/s) | 输出吞吐 (tok/s) | Waiting 峰值 / 窗口末 | 额外 drain (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | baseline | 0.0875 | 4.9466 | 100% | 0.439 | 112.3 | 0 / 0 | 2.793 |
| 0.5 | N-gram | 0.0861 | 4.1640 | 100% | 0.456 | 116.8 | 0 / 0 | 1.923 |
| 1.0 | baseline | 0.0860 | 4.9382 | 100% | 0.841 | 215.3 | 0 / 0 | 3.778 |
| 1.0 | N-gram | 0.0870 | 4.2613 | 100% | 0.871 | 223.0 | 0 / 0 | 2.960 |
| 1.5 | baseline | 0.0886 | 4.9708 | 100% | 1.242 | 318.1 | 0 / 0 | 4.144 |
| 1.5 | N-gram | 0.0905 | 4.6348 | 100% | 1.268 | 324.7 | 0 / 0 | 3.650 |
| 2.0 | baseline | **4.1167** | 9.0215 | **20%** | **0.282** | 360.5 | **8 / 8** | **8.402** |
| 2.0 | N-gram | **0.0938** | 4.7207 | **100%** | **1.702** | 435.6 | **0 / 0** | **3.508** |

2.0 请求/秒是本轮的分界证据：baseline 三轮 TTFT P95 分别 **4.1167 / 4.1221 / 4.0652 秒**，Waiting 峰值与到达窗口末均为 **8**，服务端请求队列时间直方图每轮均值约 **2.01 秒**，同时 40/40 请求都最终成功。N-gram 三轮 TTFT P95 为 **0.0909 / 0.0938 / 0.0941 秒**，Waiting 均为 0，40/40 请求均达标。baseline 同档在到达窗口结束前只完成约 24–25/40 条，客户端未完成请求峰值为 16；N-gram 完成约 34–35/40 条，峰值仅 7。**成功率 100% 与延迟 SLO 达标是不同判断**。

2.0 档按请求类别看：baseline 重复/数字请求 TTFT P95 的三轮中位数分别为 **4.1175 / 4.0996 s**；N-gram 分别为 **0.0948 / 0.0519 s**。N-gram 的重复请求 E2E P50 为 **1.1919 s**，数字请求为 **4.4914 s**；并非所有请求都以同一速度结束。N-gram 的聚合草稿接受率三轮中位数约 **24.91%**，这是所有请求合计，不能拆成“重复”和“数字”各自接受率。

服务端指标进一步排除错误归因：2.0 档 baseline / N-gram 的 Running 峰值为 **8 / 7**，KV 使用率峰值约 **2.72% / 1.78%**，两组 `num_preemptions` 增量始终 **0**。这次瓶颈是 baseline 的服务速率跟不上到达率，导致请求名额用满与 Waiting 积累；不是 KV 容量不足或重算抢占。N-gram 能让易预测请求更快离开 Running，从而在此混合负载下释放名额，但不意味着 `max-num-seqs=8` 变大。

## 怎么判断边界、又不能宣称什么

按预设的“所有正式轮成功、TTFT P95≤1 s、E2E P95≤10 s”标准，**本次测试网格**中 baseline 的最高通过档为 **1.5 req/s**，2.0 失败；N-gram 通过最高测试档 **2.0 req/s**。因此只能说：在这张卡、这套版本/配置、50:50 合成负载和每轮 20 秒窗口下，N-gram 把**观测到的 SLO 通过档位**从 1.5 推至至少 2.0 req/s。baseline 真正临界点位于本次粗网格的 1.5 与 2.0 之间；N-gram 的最大长期稳定到达率**尚未测出**，不能把 2.0 当作生产容量或保证持续数小时可承受。

局限还包括：到达是等间隔而非真实随机流量；只有两个刻意构造的提示词类型、固定 256 输出 tokens、关闭 Prefix Cache、eager 模式；每档仅 3 次 20 秒窗口，未做交错顺序、跨时段、长时稳态、取消/重试或不同混合比例。低速档总请求数只有 10–30，P95 波动解释应谨慎。客户端基于本机 loopback，不含 SSH 隧道或公网链路。服务端 ITL histogram 在投机解码下每次输出事件可对应多个 token，不能把两组 `Δsum/Δcount` 直接当逐 token ITL 比较。若要进一步界定容量，应在**不触发高 KV 压力**的前提下，把 1.5–2.0 间加密网格，并在已通过速率做更长时间、不同到达分布的复测。

两组服务均已停止，`/health` 不可达，GPU 显存回到 **2 MiB**。云实例是否仍按开机计费需在平台控制台确认；停止 vLLM 不等于关闭云实例。
