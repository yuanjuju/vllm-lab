# 源码精读笔记：Scheduler——一次 schedule() 的完整决策路径

记录日期：2026-09-28。源码版本：vLLM v0.29.0（`../vllm-src`）。

## 阅读范围

- `vllm/v1/core/sched/scheduler.py:501-1230`（`schedule()` 主体：Running 循环、Waiting 准入循环）
- `vllm/v1/core/sched/scheduler.py:1405-1446`（`_preempt_request`）、`:2676-2712`（`make_stats`）、`:120`（配置绑定）
- `vllm/v1/core/sched/request_queue.py` 全文（208 行）
- `vllm/v1/core/kv_cache_manager.py:228-294`（`get_computed_blocks`）、`:343-534`（`allocate_slots` 关键分支）
- `vllm/v1/core/kv_cache_utils.py:621-648`（`hash_block_tokens`）
- `vllm/v1/core/block_pool.py:723-747`（`free_blocks`）

## 模块职责

Scheduler 是 EngineCore 每次迭代的决策者：上游从 EngineCore 收到请求与上一步输出，下游产出 `SchedulerOutput`（本轮执行哪些请求、各多少 token、用哪些 KV 块）交给 GPU Worker。KV 的实际分配 / 释放委托给 `kv_cache_manager`，队列本身只是 `deque`（FCFS）或堆（PRIORITY）。

核心心智模型（`scheduler.py:503-512` 作者注释）：**没有"prefill 阶段"和"decode 阶段"**。每条请求只有 `num_computed_tokens`（已算到哪）和 `num_tokens_with_spec`（总共要算多少，含 draft）；调度就是给请求分配 token 让前者追上后者。这个统一模型天然覆盖 chunked prefill、prefix caching 和 speculative decoding。

## schedule() 的三段结构

```
schedule()
├── 初始化：token_budget = max_num_scheduled_tokens (:521)
│            input_budget = max_num_batched_tokens  (:524)
├── ① Running 循环 (:552-763)：为已运行请求续分配 token
│     └── allocate_slots 失败 → 抢占分支 (:657-715)
├── ② Waiting 准入循环 (:775-1215)：仅当本步未发生抢占 (:775)
│     ├── 名额检查 (:784)、前缀命中查询 (:838)
│     ├── chunked prefill 切分 (:984-998)
│     └── allocate_slots 失败 → break，不抢占 (:1091-1098)
└── ③ 收尾：prefill_capacity_bound (:1215)、asserts (:1219-1223)
```

## 关键函数与分支

| 函数（`文件:行号`） | 做什么 | 触发条件 / 备注 |
| --- | --- | --- |
| `scheduler.py:120` | `max_num_running_reqs = max_num_seqs` | 即 `--max-num-seqs` 的绑定点 |
| `scheduler.py:552` | Running 循环：`while req_index < len(running) and token_budget > 0` | 续算 prefill 块与 decode 混在同一循环 |
| `scheduler.py:585-594` | `num_new_tokens = min(剩余, token_budget, input_budget)` | 单请求本轮 token 上限的确定处 |
| `scheduler.py:657-715` | 抢占分支 | `allocate_slots` 返回 `None` 时；FCFS 牺牲者 = `running.pop()`（最新准入者，:705）；把自己抢掉则放弃本轮 |
| `scheduler.py:775` | `if not preempted_reqs:` 才进入 Waiting 准入 | 发生抢占的步完全不准入新请求 |
| `scheduler.py:784` | `num_running >= max_num_running_reqs → break` | 名额上限，Waiting 堆积的直接原因 |
| `scheduler.py:838-845` | 准入时查前缀缓存 | 仅 `num_computed_tokens == 0` 时查一次 |
| `scheduler.py:954, 998` | `min(num_new_tokens, request_token_budget)` | chunked prefill 切分点；禁用时超额直接 break (:990-996) |
| `scheduler.py:1126-1183` | 出队、`running.append`、置 RUNNING、扣预算 | WAITING→new / PREEMPTED→resumed 两类 (:1167-1170) |
| `scheduler.py:1405-1446` | `_preempt_request` | 释放全部块 (:1421)、`num_computed_tokens=0` (:1425)、计数 +1 (:1440)、**prepend 到 Waiting 队头** (:1445) |
| `kv_cache_manager.py:524-526` | `required > available → None` | 含 watermark 余量 (:520-523)，抢占略早于物理满 |
| `kv_cache_manager.py:258` | 命中上限 `num_tokens - 1` | 全命中也须重算末 token 取 logits |
| `kv_cache_utils.py:621-648` | 链式块哈希 | 父块哈希参与子块哈希 → 改首块级联失效 |
| `block_pool.py:734-747` | `free_blocks` | 有哈希块留缓存（FIFO/LRU 淘汰）、无哈希块 LIFO 先复用 |
| `scheduler.py:2676-2712` | `make_stats` | `num_running_reqs` / `num_waiting_reqs` / `kv_cache_usage` 即 `/metrics` 轮询的 gauge 来源 |

## 与实验数据的对应

1. **scheduler_capacity（max-num-seqs=4 → 4/2）**：784 行名额检查 break，6 请求只进 4 条；前 4 条完成释放名额后，下一轮 Waiting 准入补进剩余 2 条。观测的 `max_running=4 / max_waiting=2` 即 `make_stats:2701-2702` 读的两个队列长度。改为 8 后名额不再触发，全部进入 Running——与整批 13.672 → 8.185 s 一致。
2. **token budget（long/512 capacity Waiting）**：512 budget 下长输入在 954/998 行被切小，单轮只能容纳部分请求的 token；778 行 `token_budget > 0` 耗尽后循环终止，剩余请求留在 Waiting——即实验里看到的 capacity Waiting，无需任何特殊标志位。
3. **TTFT/ITL 取舍**：budget 大 → 长 prefill 一轮更多 token → TTFT 低，但同轮 decode 请求等待久（ITL 升高）。源码层面这发生在同一 Running 循环的预算竞争（585-594），与实测"长输入 TTFT 0.5648→0.4495 s、混合负载 ITL +4.1%"方向一致。
4. **KV pressure（1024-block 池每批 Preemption +1）**：8 条并发增长中，最新准入请求申请块失败（`kv_cache_manager.py:524-526`），抢占分支弹出 Running 队尾——FCFS 下恰好是第 8 条（最新）请求。3 秒停顿 = 等待其他请求释放块 + 775 行"抢占步不准入" + 重新 prefill（`num_computed_tokens=0` 重算，1445 行队头优先恢复）。
5. **prefix_partial_reuse（1792/1813 命中、改词归零）**：1813 token 按 16-block 对齐至 1792（113 整块），加上"末 token 必须重算"（:258）——命中数与源码规则吻合。改首词 → 首块哈希变 → 链式哈希使后续全部失效（`hash_block_tokens:642-648`），命中归零。

## 与笔记结论的修正

- [03-kv-cache-preemption](../../phase3/learning_notes/03-kv-cache-preemption.md)："释放**某个**运行请求"→ 精确为：FCFS 策略下牺牲者是 **Running 队尾（最新准入者）**，非随机也非最早者（:705）。
- 同笔记："仍可命中的缓存块可能减少实际重算量" → **证实**：`free_blocks` 中带哈希的块留在缓存按 LRU 淘汰（`block_pool.py:737-742`），抢占者重入时可部分前缀命中。
- 新增事实（原笔记未写）：① 发生抢占的调度步完全不准入 Waiting 请求（:775）；② 被抢占请求 prepend 到队头、恢复时按 `resumed` 分类（:1169-1170），`prefill_stats` 只统计首次（:926）。
- [02-token-budget-performance](../../phase3/learning_notes/02-token-budget-performance.md)："受 `max_num_batched_tokens`（以及其他可配置的 scheduled-token 限制）约束" → 精确为两个独立预算：`token_budget`（`max_num_scheduled_tokens`，:521）与 `input_budget`（`max_num_batched_tokens`，:524），每请求对两者同时扣减（:727-728）。
- README"三类约束"（名额 / budget / KV）成立，补充第四个隐藏量：KV 准入含 **watermark 余量**（`kv_cache_manager.py:520-523`），抢占会略早于"物理满池"发生——受控组 KV 峰值 99.5-99.9% 而非 100% 与此相符。

## 未解问题 / 下一步

- `max_num_scheduled_tokens` 的默认值如何从 `max_num_seqs` 等推导（`vllm/config/scheduler.py` 相关段未读）。
- `reset_preempted_req_ids` 从 `SchedulerOutput:1344` 到 Prometheus counter 的完整曝写路径（`vllm/v1/metrics/` 未读）——也关联 phase3 发现的 abort counter 观测盲区。
- watermark_blocks 的具体取值逻辑（`kv_cache_manager.py` 分配主路径未逐行读）。
- 0.29 新增的 ubatching（`v1/worker/ubatch_utils.py`）与本循环的关系。
- 混合批（prefill 块 + decode）在 GPU 侧如何组织执行 → 下一篇：`gpu_model_runner.py`。
