# AGENTS.md：项目交接与工作约定

> 面向接手的 AI agent 或协作者。最后更新：2026-10-09。

## 一、项目是什么

**目标**：系统掌握 LLM 推理引擎（vLLM）的工作机制，并产出可复现的工程证据。这不是"读源码"项目，而是三层闭环：

```
① 黑盒观察（跑实验、看现象） → ② 源码验证（定位代码） → ③ 对照实验（量化机制价值）
```

**铁律**：每条结论必须能追溯到 `phase3/results/` 的原始数据或 `../vllm-src`（v0.29.0）的源码行号。**行号禁止凭记忆写，发布前逐条核验**（历史上出现过笔记行号与实际偏差）。

## 二、环境

| 位置 | 内容 |
| --- | --- |
| 本仓库 | 客户端脚本（仅标准库）、实验数据、报告、笔记 |
| `../vllm-src` | vLLM v0.29.0 完整源码克隆（**行号一律以此为准**） |
| 云端实例 | 按量计费单卡 RTX 4090。接入方式（SSH 地址/端口/一次性密码）由用户现场提供，**绝不写入仓库**。启动协议见 `phase3/README.md`。实例内 `/root/fsas/vllm-lab/` 含模型 Qwen3-8B 与 `.venv`；服务日志写该目录的 `logs/` |
| 本地 Mac | Apple Silicon；Metal 版 vllm 0.29.0 在 `../.venv/lib/python3.12/site-packages/vllm/`（可 grep 快速定位，不含 docs/tests） |

## 三、当前状态（2026-10-07）

| 阶段 | 状态 | 证据 / 产出 |
| --- | --- | --- |
| Phase 2 Mac/Metal 基线 | ✅ 完成 | `phase2/`（生成参数、流式、并发初探） |
| Phase 3 云端黑盒刻画 | ✅ 完成（8 组实验） | `phase3/results/`（127 文件）+ `phase3/learning_notes/`（5 篇） |
| Phase 4 源码对齐 v0.29.0 | ✅ 主干收官 | `phase4/notes/`（4 篇）+ `phase4/source-map.md`（现象→代码位置对照表） |
| Phase 5 特性对照实验 | ✅ **v0.29.0 正式实验 5/5 + 开环容量网格**（另有 4 项 R570/v0.18.0 探索对照） | `phase5/results/`；完成记录与未竟事项见 `v029_pause_checkpoint_20261009.md`，版本边界见 `phase5/README.md` |

## 四、已确立的核心结论（勿重复推导，可直接引用）

1. **调度器无 prefill/decode 阶段之分**：请求的 `num_computed_tokens` 追赶 `num_tokens_with_spec`；双预算 `token_budget`（max_num_scheduled_tokens）与 `input_budget`（max_num_batched_tokens）。`vllm/v1/core/sched/scheduler.py:503-512`、`:521`、`:524`。
2. **前缀缓存命中规则**："必须重算最后一个 token 取 logits" → 解释实测 1792/1813（`vllm/v1/core/kv_cache_manager.py:258`）；链式块哈希，改首块级联失效（`kv_cache_utils.py:621-648`）。
3. **抢占 = 释放全部块 + `num_computed_tokens=0` 重算（不 swap）**；FCFS 受害者是最新准入者（`scheduler.py:705`）；抢占步不准入 Waiting（`:775`）；`_preempt_request`（`:1405-1446`）。
4. **abort counter 恒为 0 是设计如此**：客户端取消绕过统计（六环证据链见 `phase3/results/overload_recovery_20260923.md` 回填节 + `phase4/notes/04-api-and-metrics.md`）。
5. **CUDA graph 省的是 CPU 端 kernel 启动**：eager 每步 decode +4.5ms 常数（batch 1→8 不变）、吞吐 −16~18%、TTFT 持平、引擎初始化 132.7s vs 20.6s；`--enforce-eager` 同时关 torch.compile（`vllm/config/vllm.py:1370-1375`）。详见 phase5 #1 报告。
6. **每 token KV 字节数可直接算，且与实测逐数吻合**：字节 = 2(K,V) × 层数 × KV 头数 × head_dim × dtype 字节。Qwen3-8B 为 36 层 / 8 头 / 128 dim → BF16 **147,456 B**、FP8 **73,728 B**。`--kv-cache-memory-bytes 4G` 下实测 29,120 / 58,240 tokens，正好是「4 GiB ÷ 每 token 字节数后向下取整到 16-token 块」。**注意 `kv_cache_memory_bytes` 会跳过显存 profiling 且不遵守 `gpu_memory_utilization`**（`vllm/v1/worker/gpu_worker.py:544` 的日志明示），故该组对照中 `gpu-memory-utilization=0.75` 对 KV 池不起作用。
7. **投机解码几乎不改每步成本，只改每步吐多少 token**：0.29.0 实测各配置每步 21.3–23.8 ms（最大偏离 +7.5%），而每步 token 数从 1.00 升到 4.92（repeat）或 1.24（count）；**解码加速比 ≈ 平均接受长度**。另：投机下 **SSE 内容事件 = 引擎步**（事件数 = 输出 token 数 − 接受 token 数，已逐数核对），故事件时间戳可直接当步级计时器。详见 phase5 #5 报告。
8. **容量判断必须用 SLO goodput，不能只看吞吐**：1.6 req/s 下 baseline 原始吞吐是 ngram 的 86%（330 vs 385 tok/s），但 SLO goodput 只有 **1/7**（0.215 vs 1.505 req/s）。饱和有三个可独立观测的指纹：排队顶到 `max_num_seqs` 上限、到达结束后留下长排空（14.4 s vs 3.7 s）、达成吞吐与到达率脱钩。闭环测试自带背压，**永远不会暴露容量边界**。

## 五、下一步（按优先级）

**B1–B5 与容量网格已于 2026-10-09 全部完成**（vLLM 0.29.0 / 驱动 590.44.01）。结论见 `phase5/README.md`，未竟事项见 `phase5/results/v029_pause_checkpoint_20261009.md`。当前优先级最高的遗留项：

**A1：定位 baseline 的 SLO 悬崖（容量网格的最低档设得太高）**

- 已知 baseline 容量 ≈1.29 req/s，而网格最低档是 1.6 req/s，因此只知"< 1.6"。需补跑 **0.6 / 1.0 / 1.4 req/s** 才能画出 SLO 达标率随到达率的曲线
- 用同一 `spec_ngram_open_loop_exploratory.py` 与同一 50:50 形状，换 experiment-id；环境须先复核（0.29.0 / R590 / 同一张卡），否则重跑控制组

**A2：AWQ 在默认模式（CUDA Graph + torch.compile）下是否反超**

- 现有 AWQ 对照全部在 `--enforce-eager` 下完成，而 eager 正是最可能压制/放大反量化 kernel 差异的口径；本组结论（−6.0%）与 R570/v0.18.0（+7.8%）方向相反，跨版本不可互推
- 去掉 `--enforce-eager` 重跑 `base` 与 `awq` 两组即可；注意默认模式下引擎初始化更慢（见 Phase 5 #1）

**A3：ngram 的容量上限与调参**

- 2.0 req/s 仍零排队，上限未测到；且服务端警告 `max_num_batched_tokens` 与投机槽位冲突，收益很可能被低估
- 可补更高档位（2.5 / 3.0）与 `max_num_batched_tokens` 调大后的对照

**容量可信度补测**：R570/v0.18.0 的 N-gram 固定到达率探索只测到 2.0 req/s、每档 20 秒；这不是长期最大容量。若继续使用同一版本与硬件，应在临界负载附近细化档位、延长稳态窗口并按延迟 SLO 判断，独立记录环境，不与 v0.29.0 结果合并。

**C 方向（进阶备选）**：tensor parallel 多卡 · prefill/decode 分离部署 · torch profiler 迭代级剖析（`--profiler torch --torch-profiler-dir`）。

## 六、工作约定（必须遵守）

1. **实验协议**：单变量对照；预热与正式分离；正式 ≥3 次取中位数；配置证据取自服务启动日志（不信命令行自述）；保留两组启动日志；报告必须含"局限"一节。
2. **版本纪律**：行号基于 `../vllm-src` v0.29.0，发布前核验。
3. **安全**：密码/访问令牌一律不入库；云端接入信息现场向用户索取。
4. **沟通偏好（用户是中文学习者）**：每次实验必须展示 ①做了什么 ②原始结果 ③学到什么 ④重点是什么；解释配具体例子，避免干巴巴逐行念代码；发现自己出错要显式更正（"老师的提示也会错，源码说了算"）。
5. **提交规范**：一类变更一个提交（数据 / 报告 / 文档分开）；新实验原始 JSON 放 `phase3/results/`（沿用命名），报告放 `phase5/results/`；提交信息英文、说明动机。
6. **收尾清单**：实验结束立即停服务（`pkill -f "[v]llm serve"`，方括号防自匹配），核对 `nvidia-smi` 归零；提醒用户在平台控制台确认实例关机（按量计费）。

## 七、已知坑

- `git push` 偶发失败（网络对 `github.com` 间歇不通）：后台重试循环即可恢复；`api.github.com` 通常可用（可用 `gh api` 的 Git Data API 完成提交）。
- SSH 后台启动服务会挂住会话：用 `nohup ... < /dev/null &`；服务就绪以日志 + `/health` 为准，冷启动导入 torch 可能超过 100 秒。
- 用户 `~/.ssh` 中的 deploy key 只对特定仓库有权限，不能用于本仓库推送（本仓库走 https + `gh` 凭据）。
