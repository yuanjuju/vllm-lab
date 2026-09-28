# 源码精读笔记：EngineCore——进程边界、busy loop 与输出回程

记录日期：2026-09-28。源码版本：vLLM v0.29.0（`../vllm-src`）。

## 阅读范围

- `vllm/v1/engine/core.py:597-627`（`step`）、`:1411-1466`（`run_busy_loop`、`_process_input_queue`）、`:1027,1291-1360`（`EngineCoreProc` / `run_engine_core`）
- `vllm/v1/engine/async_llm.py:740-822`（`_run_output_handler`）、`:1002`（API 进程持有 tokenizer）
- `vllm/v1/engine/output_processor.py:533-547,597-600`（推入每请求 `asyncio.Queue`）
- `vllm/v1/engine/core_client.py:21-57`（ZMQ 传输）
- `vllm/v1/core/kv_cache_manager.py:201-224`（`make_prefix_cache_stats` / `record_prefix_cache_stats`，第一课作业）
- `vllm/v1/sample/ops/topk_topp_sampler.py:349-403`（`apply_top_k_top_p`，第二课作业）
- phase3 归档日志：`(EngineCore pid=1530)` 前缀行 vs 无前缀行（1424 行）——进程边界的直接证据

## 架构：两个进程，一条 ZMQ

| 进程 | 拥有 | 职责 |
| --- | --- | --- |
| API Server（uvicorn/FastAPI） | `AsyncLLM`、tokenizer、output_handler 后台任务 | HTTP/SSE、聊天模板与分词、detokenize、推流、metrics 曝写 |
| EngineCore（独立 pid，`core.py:1309` 置进程名） | Scheduler、KV manager、GPU executor | busy loop：收请求 → schedule → execute → 回传 outputs |

通信：input_queue / output_queue，多进程模式走 ZMQ（`core_client.py:21-35`）。设计动机：API 进程的 asyncio/解析/推流与引擎的紧循环互不抢 GIL。

## EngineCore 主循环（`core.py:1411-1422`）

```python
while self._handle_shutdown():
    self._process_input_queue()     # 取新请求/abort，直到有活干
    self._process_engine_step()     # = step()
```

`step()`（`:597-627`）是三节课的交汇点：

```python
scheduler_output = self.scheduler.schedule(...)          # ← 第一课
future = self.model_executor.execute_model(..., non_block=True)  # ← 第二课
grammar_output = self.scheduler.get_grammar_bitmask(...)  # GPU 忙时 CPU 做别的（重叠）
model_output = future.result()                            # 等 GPU
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

`update_from_output`（`:1802`）：把采样出的 token 喂回 Request，判 stop / finish、释放名额与块（衔接第一课 `_free_request_blocks`），产出 `EngineCoreOutputs`。

## 输出回程（API 进程，`async_llm.py:761-817`）

```
EngineCore output_queue ──ZMQ──▶ output_handler（后台 asyncio 任务）
  → output_processor.process_outputs()   # detokenizer 增量解码 + stop string 检查
  → req_state.queue.put(request_output)   # output_processor.py:547，每请求一个 asyncio.Queue
  → generate() 从该 queue yield → SSE 推给客户端
  → update_scheduler_stats + logger_manager.record  # ← /metrics gauge 的更新点（:806-817）
```

**重要推论**：`/metrics` 里的 Running/Waiting/KV gauge 是 API 进程在 output_handler 里更新的，数值来自 EngineCore 每次 step 后跨进程运来的 `SchedulerStats`。因此 gauge 的可见延迟 = engine step 节奏 + 队列传输 + 客户端轮询周期，三者叠加——phase3 "短于采样周期的尖峰可能漏掉"的机制根源。

## 作业答案

1. **prefix hit/query delta 的累加点**：`record_prefix_cache_stats`（`kv_cache_manager.py:217-224`，调用点 `scheduler.py:1121-1124`）——queries += `request.num_tokens`、hits += `num_new_local_computed_tokens`，**只在准入时记录**（未准入的查询不计）；`make_prefix_cache_stats`（:201-211）取走并清零，所以是周期差值语义。
2. **top_p 的实现**：`apply_top_k_top_p`（`topk_topp_sampler.py:349-364`）按 batch 大小选 Triton（≥8）或 PyTorch 路径。PyTorch 路径（:367-403）**升序排序**后 `cumsum`，掩码条件 `probs_sum <= 1 - p`——从最小概率端累计，恰好保留最短 top 前缀；batch≥8 用 Triton kernel 避免全量排序。（修正第二课作业提示：并非"不排序"，小 batch 就是老实排序。）

## 与实验数据的对应

- 归档日志 `(EngineCore pid=1530)` 前缀 = EngineCore 进程输出；无前缀 1424 行 = API 进程。两进程结构在日志中直接可见。
- phase3 的指标轮询语义（上文推论）；overload 实验中 abort counter 的观测盲区亦与跨进程队列的时序相关（待 metrics 曝写层精读后下结论）。
- 一条 173-token 请求的 TTFT 全链路：Mac → API 进程（模板/分词/add_request）→ ZMQ → EngineCore busy loop 取出 → scheduler.waiting → 下一步 schedule 进批 → GPU prefill → 采样 → update_from_output → ZMQ 回 → detokenizer → req_state.queue → SSE → 客户端首内容块计时。

## 未解问题 / 下一步

- `api_server.py` 的路由与 serving_chat 模板渲染细节（请求路径主干表第 1 行仍未读）。
- `input_processor.py`：模板→token 的确切位置与 API 进程内调用点。
- `vllm/v1/metrics/`（loggers.py）：counter/histogram 的曝写层，abort counter 盲区的最终答案。
- async scheduling 的完整重叠细节（`async_scheduler.py`）。
