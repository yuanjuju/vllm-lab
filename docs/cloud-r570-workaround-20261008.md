# R570 节点上的临时 CUDA 12.8 服务恢复（2026-10-08）

这是一条**临时可用性路线**，不是 Phase 5 的 vLLM 0.29.0 性能实验。新实例的 RTX 4090 宿主机驱动为 `570.124.06`；共享盘 `/root/fsas/vllm-lab` 保留了 Qwen3-8B 模型和原 `.venv`。原环境为 vLLM `0.29.0`、PyTorch `2.13.0+cu132`，在该节点上 `torch.cuda.is_available()` 为 `False`，并报告驱动过旧。原 `.venv`、模型文件与本地 Mac 环境均未修改。

## 独立环境

在共享盘中创建 `/root/fsas/vllm-lab/.venv-cu128-vllm018`；项目内缓存位于 `/root/fsas/vllm-lab/cache/uv-cu128-vllm018`。没有安装到系统 Python。安装命令：

```bash
UV_CACHE_DIR=/root/fsas/vllm-lab/cache/uv-cu128-vllm018 \
  uv venv --python /usr/bin/python3.12 /root/fsas/vllm-lab/.venv-cu128-vllm018
UV_CACHE_DIR=/root/fsas/vllm-lab/cache/uv-cu128-vllm018 \
  uv pip install \
  --python /root/fsas/vllm-lab/.venv-cu128-vllm018/bin/python \
  --torch-backend=cu128 'vllm==0.18.0'
```

选择 0.18.0 是因为[该版本官方安装文档](https://docs.vllm.ai/en/v0.18.0/getting_started/installation/gpu/)说明其预编译 CUDA 二进制对应 12.8；解析出的 PyTorch 是 `2.10.0+cu128`。安装后实测：CUDA 可用，GPU 上 `[2.0, 3.0]` 求和返回 `5.0`。这只证明该环境能在当前节点运行，不证明其与原 0.29.0 具有相同调度或性能行为。

## 服务验收

用 [临时启动脚本](../phase3/run_cloud_r570_vllm018.sh)运行（云端放在同一相对路径）。它仅监听 `127.0.0.1:8000`，使用 BF16、4096 最大上下文、0.75 GPU 内存利用率、`max-num-seqs=8`、`max-num-batched-tokens=4096`。为优先验证可用性，加了 `--enforce-eager`，因此关闭了 torch.compile 和 CUDA Graph；**不能拿本次延迟与旧基线直接比较**。不要 `source` 云端旧 `env.sh`，它会激活不兼容的原 `.venv`。

后续需要临时服务时，在云端项目目录前台运行 `bash phase3/run_cloud_r570_vllm018.sh`；验收结束用 `Ctrl-C` 停止，并核对 `nvidia-smi`。脚本沿用本次已验收的启动参数；随后还用相同脚本完成了 [Prefix Cache 开/关探索性对照](../phase5/results/prefix_cache_exploratory_r570_v018_20261008.md)，关闭组只额外传入 `--no-enable-prefix-caching`。该结果仍不能替代 v0.29.0 的 Phase 5 B1 正式实验。

原始[服务启动日志](r570_vllm018_smoke_20261008.log)记录了实际配置、模型加载、KV Cache 初始化和 API 访问。验收观察：

| 检查 | 实测 |
| --- | --- |
| `/health` | HTTP 200 |
| `/v1/models` | `Qwen/Qwen3-8B` |
| 真实聊天请求 | `15+27` 返回 `42`，`finish_reason=stop`，prompt/completion/total 为 24/3/27 tokens |
| 服务启动时 GPU 显存 | `36434 MiB`（单次采样，不代表稳态需求） |
| 收尾 | 向服务主进程发送 `SIGTERM` 后，API 端口关闭、GPU 显存回到 `2 MiB` |

本次仅做一次冒烟验收，没有预热/正式重复、TTFT、吞吐量或质量评估。vLLM 版本、PyTorch/CUDA 和执行模式都与 Phase 5 的 0.29.0 原基线不同，因此不得把这个环境的结果混入已有性能对照。按量计费实例是否关机，只能在云平台控制台确认；停止 vLLM 服务不等于停止实例计费。
