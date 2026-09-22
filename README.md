# vLLM 学习实验室

从 Apple Silicon 本地推理，逐步学习 vLLM 的生成参数、性能指标、云端服务、调度与 KV Cache。实验数字是特定模型、硬件、配置和时间下的记录，不是通用跑分。

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
