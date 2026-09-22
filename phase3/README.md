# 第三阶段：云端 vLLM 服务与引擎机制

本目录保存 Mac 客户端实验脚本、实测结果和学习笔记。云端使用 Qwen3-8B 与单卡 RTX 4090 48GB；云端虚拟环境、模型权重和服务脚本**不在本仓库中**。记录对应 vLLM 0.29.0 的一次学习环境，不保证未来机器或版本得到相同数字。

## 使用前提

1. 云端已启动 vLLM，模型 ID 为 `Qwen/Qwen3-8B`，API 仅监听云端 `127.0.0.1:8000`。
2. Mac 已通过 SSH 将本机 `127.0.0.1:18000` 转发到云端 API。SSH 主机和端口由云平台当前实例提供，不要把密码或访问令牌提交进仓库。
3. `curl -fsS http://127.0.0.1:18000/health` 成功后，才运行以下客户端实验。云端实例关机时，本目录数据仍可离线阅读。

客户端脚本只使用 Python 标准库，在 Mac 上运行，不依赖云端 Python 环境：

```bash
cd /path/to/vllm-lab
python3 phase3/metrics_counter_walkthrough.py
python3 phase3/prefix_partial_reuse.py
python3 phase3/scheduler_capacity.py --requests 6 --tokens 384
```

这些脚本会将本次结果写入 `results/`。`scheduler_capacity.py` 可用 `--base-url` 指向其他兼容服务；其余两个脚本默认访问 `127.0.0.1:18000`。

## 阅读顺序

- [云端链路与首次流式请求](results/cloud_baseline.md)
- [调度序列上限 4/8 的对照](results/scheduler_capacity_comparison.md)
- [学习笔记索引](learning_notes/README.md)

`results/` 中的 JSON、CSV、TSV 和 TXT 是小规模实验的原始记录，用于核对汇总结论。TTFT 与请求耗时包含 Mac 客户端和 SSH 隧道开销，不能当作纯 GPU 执行时间。停止学习时请确认云平台实例计费状态。
