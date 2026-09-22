# CUDA 服务：观察排队、缓存与延迟

这一部分使用单卡 RTX 4090 48GB、Qwen3-8B 和 vLLM 0.29.0。服务端只监听 `127.0.0.1:8000`；Mac 把本机 `18000` 端口通过 SSH 转发过去，再调用 OpenAI 兼容 API 和 `/metrics`。这样既能从真实客户端测端到端延迟，也能看到 vLLM 引擎当时在运行或等待的请求数。

## 已完成的观察

- [首次远程流式请求](results/cloud_baseline.md)：验证 `/v1/models`、聊天生成和 SSE 结束标记；单次 TTFT 为 0.9299 s，包含首次请求可能的预热。
- [20 次远程流式请求](results/cloud_20run_summary_20260921-141120.json)：预热 2 次后，20/20 成功；TTFT P50 为 0.136 s，总耗时 P50 为 1.214 s。
- [共享前缀与改变前缀](results/prefix_partial_reuse_20260921-184943.json)：共享前缀的请求命中 1792/1813 个查询 token，改变开头的对照请求命中 0；单次耗时差异仅作观察，不当作稳态加速比。
- [调度容量对照](results/scheduler_capacity_comparison.md)：6 个并发请求下，`max-num-seqs=4` 出现最多 2 个 Waiting；设为 8 后未观察到排队，整批耗时缩短。

## 重跑客户端实验

仓库提供的是 Mac 端脚本，不包含云端镜像、权重或服务启动脚本。需要先在云端提供模型 ID 为 `Qwen/Qwen3-8B` 的 vLLM 服务，并建立 `127.0.0.1:18000 → 云端 127.0.0.1:8000` 的 SSH 转发。SSH 地址和端口以当前实例为准，不应把密码或访问令牌写进仓库。

在 Mac 上先确认服务已就绪：

```bash
curl -fsS http://127.0.0.1:18000/health
curl -fsS http://127.0.0.1:18000/v1/models
```

然后从仓库根目录运行客户端脚本；它们只依赖 Python 标准库，结果会写入 `phase3/results/`：

```bash
python3 phase3/metrics_counter_walkthrough.py
python3 phase3/prefix_partial_reuse.py
python3 phase3/scheduler_capacity.py --requests 6 --tokens 384
```

`scheduler_capacity.py` 支持 `--base-url`；其余两个脚本当前固定访问 `127.0.0.1:18000`。对照 `max-num-seqs=4/8` 时，需要在云端分别以相应配置启动服务，每次确认 `/health` 后再运行同一脚本。

数据和解释分别在 [results/](results/) 与 [请求链路笔记](learning_notes/01-request-lifecycle.md)。端到端 TTFT 包含 Mac、SSH 隧道和服务端开销，不等于 GPU 的纯计算时间。云端实例按量计费，实验结束后需在平台确认关机状态。
