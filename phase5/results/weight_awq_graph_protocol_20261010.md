# A2 预注册协议：默认执行模式下 BF16 与 AWQ 权重对照（2026-10-10）

## 问题与判据

在同一节点、同一 vLLM 0.29.0 环境中，去掉 `--enforce-eager` 后，AWQ 相对 BF16 的输出吞吐方向是否仍与 [eager 对照](weight_awq_v029_20261009.md)一致？主要指标是 3 次正式轮的输出吞吐中位数及逐轮值；同时报告 ITL、TTFT、整批耗时、模型加载占用与自动 KV 池。默认模式同时启用适用的 torch.compile 与 CUDA Graph；本次不单独归因两者。

## 固定条件

- 单张 RTX 4090；开始前现场记录主机名、GPU UUID、驱动、CUDA、项目 `.venv` 中的 vLLM/PyTorch、模型文件、GPU 空闲与服务状态。客户端与服务同在云端，走 `127.0.0.1:8000` loopback。
- BF16 组使用 `models/Qwen3-8B`；AWQ 组使用官方 `models/Qwen3-8B-AWQ` 和 `--quantization awq_marlin`。两组均为 `--dtype bfloat16 --kv-cache-dtype auto --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-seqs 8 --max-num-batched-tokens 4096 --no-enable-prefix-caching`；均不传 `--enforce-eager`。模型目录及量化方式是唯一组间变量。
- 使用 `phase3/weight_awq_ab_exploratory.py` 的同一 `experiment-id` 和默认负载：8 条同时到达、约 3100-token 输入、每条固定 128-token 输出、`temperature=0`、关闭 thinking。两组分别完成 6 题质量冒烟、1 次预热及至少 3 次正式轮；预热与质量题不计入性能中位数。
- 启动日志须证实模型目录、量化方式、`enforce_eager=False`、编译及 CUDA Graph 捕获情况；每组通过 `/health`、`/v1/models` 和真实请求后才运行正式实验。服务切换前停止上一组并确认 GPU 显存释放。

## 验收与停止

- 两组每轮均 8/8 成功，输出各 128 token，输入 prompt 哈希逐组相同；原始 JSON、客户端日志、两组完整服务日志和环境核对记录保存到项目目录。
- 若任一组无法启动或未使用预期执行模式，则记录失败日志，不把该组数字当作配对结果。若正式轮请求失败或输出长度异常，先定位原因再决定是否重跑，两组保持同一协议。
- 报告分别列出逐轮值、中位数、相对变化与局限。与历史 eager 结果因节点或环境状态可能不同，只比较方向，不合并绝对吞吐为严格四组配对。
- 结束时停 vLLM、核对 GPU 归零，并在云平台控制台核实实例是否停止计费；SSH 上的服务停止不等于实例关机。
