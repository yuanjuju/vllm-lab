# A3 预注册协议：N-gram 开环容量边界（2026-10-10）

## 问题与范围

在 vLLM 0.29.0、RTX 4090、Qwen3-8B BF16、N-gram 投机解码的固定合成负载下，找出本次示例逐请求 SLO 的最高**已测通过档**与首个**已测失败档**。这是 60 秒确定性到达窗口的实验边界，不是生产级最大 QPS。沿用[先前容量网格](capacity_open_loop_v029_20261009.md)的 eager 配置与 50:50 repeat/count 负载；此前 2.0 req/s 已通过，但本次容器重启后须独立复测，旧结果不作严格配对。

## 固定环境与负载

- 启动前重新记录主机名、GPU UUID、驱动、PyTorch/CUDA、项目 `.venv` 中的 vLLM、模型目录、服务和 GPU 状态。客户端及服务同在云端，只走 `127.0.0.1:8000` loopback。
- 服务用 `phase3/run_cloud_vllm029_features.sh ngram eager`：Qwen3-8B BF16，`num_speculative_tokens=4`、`prompt_lookup_min=1`、`prompt_lookup_max=4`、Prefix Cache 关闭、`max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`。启动日志是配置证据，保留投机槽位警告；本次不调整 token budget。
- 客户端复用 `phase3/spec_ngram_open_loop_exploratory.py`：repeat/count 交替 50:50，每请求固定生成 256 tokens，`temperature=0`、关闭 thinking；确定性等间隔开环到达，无客户端并发上限。每个速率档单独保存 JSON，先 10 秒预热，后 3 × 60 秒正式轮，每轮等全部请求排空后再开始下一轮。
- 示例 SLO：每条请求成功、TTFT ≤1 秒、E2E ≤10 秒，且该轮 TTFT P95 ≤1 秒；3 次正式轮都满足才判该档通过。预热不参与判定。另报告每轮逐请求达标数、Waiting/Running/KV、preemption、队列时间、排空、完成速率和 SLO goodput。

## 选档与停止

1. 本容器从 2.0 req/s 复测锚点。通过后依次试 2.4、2.8、3.2 req/s；若 3.2 仍通过，以 0.4 req/s 继续上探，直到出现失败或安全停止。遇首个完整失败档后停止升档，在最近通过档与失败档之间以 0.2 req/s 细化一次。若 2.0 不通过，则从 1.8 req/s 向下寻找通过档。所有档位均满足 `rate × 10 s` 和 `rate × 60 s` 是偶数整数。
2. 每档单独调用客户端，检查 3 次正式轮后才决定下一档；不能仅凭吞吐或单轮 TTFT P95 升档。若某轮被安全停止截断、派发不足、请求失败或 drain 超时，保留原始数据，把该轮标为不可用的完整容量读数，不继续升档。
3. 沿用客户端安全停止：KV 使用率 ≥75%、Waiting >16 或 preemption 增量 >0 即停止投递；每轮最长排空 60 秒。若客户端实际派发偏差显著（P95 >50 ms）或指标轮询失败，先排查测量有效性，再解释容量。
4. 若最高通过档在窗口末仍有 Waiting，或排空尾巴随重复增长，额外以同档 3 × 120 秒窗口核查稳定性，结果单独标注，不与 60 秒轮混算。

## 验收与收尾

保存每档原始 JSON、客户端完整日志、服务启动/运行日志和环境记录；复制到本地后与云端 SHA-256 比对。报告清楚区分到达率、含排空的完成速率、逐请求 SLO 达标率与 SLO goodput，并列出局限。结束后停止 vLLM、核对 GPU 显存归零，在云平台控制台确认关机与不计费状态；密码不入库。
