# vLLM 第二阶段归档

第二阶段已于 2026-09-20 完成。实验脚本在数据收集完成后已从项目目录移除，模型、虚拟环境、原始结果和性能基线均保留。

## 完成内容

- `temperature` 对随机性的影响
- `top_p` 对候选范围的影响
- `max_tokens` 与 `finish_reason`
- Qwen3 思考模式开关
- 普通输出与流式输出
- TTFT、总响应时间和输出 tokens/s
- 20 次连续请求稳定性
- 2、4、8 并发请求
- `Running` 与 `Waiting` 指标采样

## 数据来源

- 用户亲自完成：生成参数、思考模式、流式输出、单次性能和 20 次连续请求。
- 助手收尾采集：缺失的 2、4、8 并发测试，以及 `/metrics` 中的 `Running`、`Waiting`。

## 保留内容

- `results/mac_baseline.md`：最终验收结论和 Mac 性能基线。
- `results/*.jsonl`：非流式实验的原始 API 响应。
- `results/*.tsv`：生成参数实验的表格结果。
- `results/*.csv`：流式、连续请求和并发测试的逐请求数据。
- `results/*_summary_*.json`：性能实验的机器可读汇总。

## 脚本清理

第二阶段的临时 `.sh`、`.py` 实验脚本及 `__pycache__` 已从项目目录移除，不随仓库发布；原始结果和汇总数据保留在 `results/`。项目根目录中的 `env.sh`、`run_server.sh`、`test_api.sh` 和 `storage_report.sh` 未改动、未移除，仍可用于后续阶段。

## 最终入口

请以 `results/mac_baseline.md` 为第二阶段的最终记录。并发数据只代表当前 `Qwen/Qwen3-0.6B`、短提示词、2048 上下文和 Metal 配置，不应直接外推到更大的云端 GPU 模型。
