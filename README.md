# vLLM 推理服务实验室

基于 Apple Silicon Metal 与单卡 NVIDIA CUDA 环境，围绕 vLLM 的推理服务、可观测性和调度行为开展可复核的实验。项目保存运行脚本、原始结果与分析笔记，将 API 层看到的延迟和吞吐量，关联到引擎层的队列、KV Cache 与前缀缓存指标。

## 工程与实验重点

- **跨平台部署与隔离**：在 Mac M5 上使用 vLLM-Metal 服务 Qwen3-0.6B，在云端 RTX 4090 48GB 上使用 vLLM 0.29.0 服务 Qwen3-8B；通过相同的 OpenAI 兼容 API 验证模型加载和生成，并让本地环境、缓存与结果留在项目目录内。
- **从请求到引擎的观测**：记录流式 TTFT、响应时间、输出速度与成功率，同时读取 `/metrics` 中的 Running、Waiting、KV Cache 使用率和 Prefix Cache 命中计数，理解 Prefill、Decode、连续批处理与缓存复用对服务行为的影响。
- **单变量配置对照**：在 6 个并发请求、每个请求固定生成 384 tokens 的一次实验中，将 `max-num-seqs` 从 4 调整为 8。最大 Waiting 从 2 降至 0，总耗时从 13.672 s 降至 8.185 s；但先完成请求的单请求延迟反而上升，体现排队、延迟与吞吐量之间的取舍。详见[配置对照及原始数据](phase3/results/scheduler_capacity_comparison.md)。
- **证据边界**：上述对照只在特定模型、硬件和负载下各运行一次，并非通用跑分或生产容量规划依据；仓库不声称实现了 vLLM 内核优化。

## 内容

| 阶段 | 环境与重点 | 入口 |
| --- | --- | --- |
| 第一阶段 | Mac M5、vLLM-Metal、Qwen3-0.6B；启动与 OpenAI 兼容 API | 下方本地启动步骤 |
| 第二阶段 | 生成参数、流式输出、TTFT、连续请求与并发基线 | [第二阶段说明](phase2/README.md)、[Mac 基线](phase2/results/mac_baseline.md) |
| 第三阶段 | RTX 4090 上的 Qwen3-8B；云端 API、KV、前缀缓存与调度实验 | [第三阶段说明](phase3/README.md)、[学习笔记](phase3/learning_notes/README.md) |

vLLM 用于**推理和服务**，不负责训练。后续如进行 LoRA 微调，训练与 vLLM 部署会分别记录。

## Mac 本地服务

本地 Python 环境、模型和缓存均在此目录，不安装进系统 Python：

```bash
cd /path/to/vllm-lab
source env.sh
./run_server.sh
```

`run_server.sh` 启动 `Qwen/Qwen3-0.6B`，API 监听 `127.0.0.1:8000`，最大上下文 2048 tokens。首次运行可能下载模型到 `.cache/huggingface`。在另一个终端检查：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
./test_api.sh
```

`test_api.sh` 关闭 Qwen3 思考模式以测试直接回答。停止服务时在运行 vLLM 的终端按 `Ctrl+C`。

## 仓库边界

仓库保留脚本、学习笔记及小型原始结果；不提交 `.venv/`、`.cache/`、模型权重、日志、密钥或云端实例数据。`.cache/pip/` 是可清理的安装缓存；清理 `.cache/huggingface/` 则需要以后重新下载模型。运行 `./storage_report.sh` 可查看本地占用。

云端实例可能产生费用；第三阶段的脚本是 Mac 客户端，运行前应按 [第三阶段说明](phase3/README.md) 确认服务与 SSH 隧道。仓库未包含可直接复现云端安装的完整脚本，因此结果须连同环境说明阅读。
