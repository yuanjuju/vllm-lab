# 现象 → 代码位置对照表

读码进度跟踪。状态：`未读` / `进行中` / `已验证`（已验证 = 已在源码中找到与实验数据对应的具体分支并记录行号）。行号基于 vLLM v0.29.0（`../vllm-src`）。

## 请求路径主干

| 实测现象（Phase 3 证据） | 已定位的代码 | 文件 / 函数 | 行号 | 状态 |
| --- | --- | --- | --- | --- |
| SSE 请求进入服务，`/v1/models`、`/health` 可用 | 路由注册（**0.29 已迁移**：`openai/api_server.py` 仅剩废弃转发壳） | `vllm/entrypoints/launchers/api_server/`（routers.py、entry.py）；聊天实现在 `openai/chat_completion/serving.py` | api_server.py:24-30（DeprecationWarning） | 部分验证 |
| 聊天消息变成模型输入 | 聊天模板与预处理（API 进程内完成分词） | `vllm/v1/engine/input_processor.py` | :38（InputProcessor）、:78-82（tokenizer 由 renderer 持有） | 部分验证 |
| 引擎按迭代推进而非按请求推进 | EngineCore busy loop 与 step | `core.py` | :1411-1422（busy loop）、:597-627（step：schedule → execute → update_from_output）、:1437-1466（input queue） | 已验证 |
| 日志中 `(EngineCore pid=…)` 前缀 = 独立进程 | 进程边界与 ZMQ 传输 | `core.py:1291-1360`（`run_engine_core` 后台进程）、`core_client.py:21-57`（ZMQ） | 见左 | 已验证 |
| token 变成流式文本块 | output handler → 每请求 asyncio.Queue | `async_llm.py:740-822`（output_handler）、`output_processor.py:533-547`（`req_state.queue.put`；detokenizer/stop 检查在其上游） | 见左 | 已验证 |
| `/metrics` gauge 的更新点 | output_handler 内 record | `async_llm.py:806-817`（`update_scheduler_stats` + `logger_manager.record`）——gauge 可见延迟 = step 节奏 + ZMQ + 轮询 | 见左 | 已验证 |

## 调度（对应 scheduler_capacity / token_budget 实验）

| 实测现象 | 已定位的代码 | 文件 / 函数 | 行号 | 状态 |
| --- | --- | --- | --- | --- |
| `max-num-seqs=4` 时 `max_running=4`、`max_waiting=2` | 配置绑定与 Waiting 准入名额检查 | `scheduler.py` `schedule()` | :120（`max_num_running_reqs = max_num_seqs`）、:784（名额满则 break）、:1126–1183（出队、`self.running.append`、置 RUNNING） | 已验证 |
| `max-num-batched-tokens` 触发 capacity Waiting | 双预算初始化、逐请求扣减、耗尽即止 | `scheduler.py` `schedule()` | :521（`token_budget`）、:524（`input_budget = max_num_batched_tokens`）、:778 / :954（循环条件与 `min`）、:1180–1181（扣减） | 已验证 |
| 长 prefill 被切块执行 | chunked prefill 切分与后续轮续算 | `scheduler.py` | :984–998（threshold 截断 + `min(request_token_budget)`）、:990–996（禁用 chunked 时整批 break）、:1190–1191（未完 prefill 入 `_inflight_prefills`）、:585–594（Running 循环续算） | 已验证 |
| Running / Waiting 队列状态迁移 | 队列实现 | `request_queue.py` | :75（FCFS 即 `deque`）、:131（PRIORITY 为 `(priority, arrival_time)` 堆）；入队点 `scheduler.py:2389` | 已验证 |

## KV Cache 与抢占（对应 kv_pressure / prefix 实验）

| 实测现象 | 已定位的代码 | 文件 / 函数 | 行号 | 状态 |
| --- | --- | --- | --- | --- |
| 1024-block 池下每批恰好 Preemption +1 | 分配失败 → 抢占分支 | `kv_cache_manager.py:524-526`（`required_blocks > available_blocks` → `None`，含 watermark 余量）→ `scheduler.py:657-715`（FCFS 下牺牲者为 `running.pop()` 即最新准入的请求，:705） | 已验证 |
| 抢占后请求停顿约 3 s 再恢复 | 释放全部块、计数清零、队头重入、恢复准入 | `scheduler.py` `_preempt_request()` | :1405–1446（释放块 :1421、`num_computed_tokens = 0` :1425、prepend 队头 :1445）；:775（发生抢占的步完全不准入 Waiting）；:1169–1170（恢复时按 `resumed` 而非 new 分类） | 已验证 |
| 共享前缀命中 1792/1813 查询 token | 准入时前缀查询 | `kv_cache_manager.py:228-294`（`get_computed_blocks`，:258 命中上限 `num_tokens - 1` 须重算末 token 取 logits）；调用点 `scheduler.py:838-845`、统计记录 :1121-1124 | 已验证 |
| 改变开头一个词后命中归零 | 链式块哈希 | `kv_cache_utils.py:621-648`（`hash_block_tokens`：父块哈希参与本块哈希，改首块级联失效后续所有块） | 已验证 |
| 块的分配、释放与淘汰 | block 池 free 队列 | `block_pool.py:723-747`（`free_blocks`：有哈希块留在缓存按 FIFO/LRU 淘汰、无哈希块 LIFO 先复用——证实被抢占请求的块可被前缀重命中） | 部分验证（分配主路径未逐行读） |
| 调度器与 KV 管理的协作接口 | `allocate_slots` 两个调用点 + 每步入口 | `scheduler.py` | :542（`new_step_starts`）、:658（Running 路径）、:1077（Waiting 准入路径，传入命中块） | 已验证 |

## GPU 执行（后续特性实验的基础）

| 实测现象 | 已定位的代码 | 文件 / 函数 | 行号 | 状态 |
| --- | --- | --- | --- | --- |
| prefill 与 decode 拼成同一批次执行 | execute_model 编排 + 模型侧只看 token 流 | `gpu_model_runner.py:4249-4360`（`execute_model`）、`:3917-3947`（`_model_forward`）；`qwen2.py:397-434`（embed → 36 层循环 → norm） | 见左 | 已验证 |
| 每层注意力结构（GQA 4:1 的代码来源） | Attention.forward | `qwen3.py:153-170`（qkv split：q 4096 维、k/v 各 1024 维） | 见左 | 已验证 |
| KV 块的物理组织 | block table 与 slot mapping | `block_table.py:157-177`（每请求一行块号）、`:201-229`（token → 块号+偏移） | 见左 | 已验证 |
| phase2 的 temperature / top_p 生效位置 | Sampler | `sampler.py:228-242`（temperature=除法、greedy=argmax）、`:274-303`（主流程） | 见左 | 已验证 |
| 服务启动日志出现 CUDA graph 捕获 | 捕获档位与 uniform decode 判定 | 归档日志 `cudagraph_capture_sizes: [1,2,4,8,16]`；`gpu_model_runner.py:3950-3968`（`_is_uniform_decode`）；捕获本体在 `v1/cudagraph_dispatcher.py`（未逐行读） | 见左；实验对照见 [phase5 #1](../phase5/results/eager_vs_cudagraph_20260929.md) | 已验证（A/B 实测） |
| `num-gpu-blocks-override` 决定逻辑池大小 | profile 与块数确定 | `vllm/v1/worker/gpu_worker.py` | — | 未读 |

## 观测开关

| 开关 | 用途 | 定义位置 | 状态 |
| --- | --- | --- | --- |
| `VLLM_LOGGING_LEVEL` | 日志级别 | `vllm/envs.py:41`（默认 `INFO`） | 已确认 |
| `VLLM_TRACE_FUNCTION` | 函数级 trace 深度 | `vllm/envs.py:48`（默认 `0`） | 已确认，未试用 |
| torch profiler | 引擎迭代级性能剖析 | `vllm/config/profiler.py:49`，经服务参数 `--profiler torch` + `--torch-profiler-dir` 配置 | 已确认定义，未试用 |
| Prometheus 指标注册 | `/metrics` 各指标来源 | 调度侧 `scheduler.py:2676-2712`；曝写层 `vllm/v1/metrics/loggers.py`（全部指标定义 :494-:1066；preemption 链 stats.py:524-525 → loggers.py:1191-1192） | 已验证 |
| phase3 悬案：abort counter 无效 | abort 的三条路径 | `loggers.py:713-725`（abort label 预建但恒 0）、`:1221-1224`（唯一自增点）；`output_processor.py:526-547`（本地 abort 绕过统计）、`:654-656`（迟到输出丢弃）、`:741`（正常 finish 才统计） | 已验证——详见 [notes/04](notes/04-api-and-metrics.md)；2026-10-07 已回填 phase3 过载报告 |

注：早期版本的 `VLLM_TRACE_ENABLED`（Perfetto 引擎 trace）在 v0.29.0 中已不存在（全仓库 grep 无结果）；迭代级剖析走 `vllm/config/profiler.py` 定义的 profiler 配置。
