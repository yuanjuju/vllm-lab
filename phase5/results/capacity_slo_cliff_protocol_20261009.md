# Baseline SLO 悬崖补测：预注册协议（2026-10-09）

目标：补足[上一轮开环容量报告](capacity_open_loop_v029_20261009.md)在 1.6 req/s 以下的缺口，定位 Qwen3-8B BF16 baseline 在**这套合成负载和示例 SLO 下**从达标转为不达标的速率区间。不是寻找可普遍承诺的生产 QPS。

## 环境与负载

- 新接入容器主机名与上一轮不同；本轮重新记录 GPU UUID、驱动、vLLM/PyTorch/CUDA、启动日志和模型验收。即使版本相同，也不把上一节点的 1.6 req/s 数据当作严格配对点。
- 服务使用[同一启动脚本](../../phase3/run_cloud_vllm029_features.sh)的 `base` 变体：Qwen3-8B BF16、eager、Prefix Cache 关闭、`max-model-len=4096`、`gpu-memory-utilization=0.75`、`max-num-seqs=8`、`max-num-batched-tokens=4096`，只监听云端 `127.0.0.1:8000`。
- 客户端在云端 loopback，复用[固定到达率脚本](../../phase3/spec_ngram_open_loop_exploratory.py)；repeat/count 交替 50:50、每条固定输出 256 tokens、`temperature=0`、关闭 thinking，无客户端并发上限。
- 速率从低到高：**0.6 / 1.0 / 1.2 / 1.4 / 1.6 req/s**；每档先 10 秒预热，再 3 次 60 秒正式窗口。加入 1.2 是为缩小可能位于 1.0 与 1.4 之间的边界；1.6 只在前档安全完成后作为本节点锚点。
- 示例 SLO：成功、客户端 TTFT ≤1 秒且 E2E ≤10 秒；另外要求该轮整体 TTFT P95 ≤1 秒。三次正式轮均通过才把该速率记为“本次测试网格通过”；脚本遇请求失败、未完成或触发安全停止会停止升级。若指标缺失，保留原始数据并在分析时将该指标标为不可判定，不把缺失误写成 0。
- 安全停止沿用客户端：KV 使用率 ≥75%、Waiting >16 或 `num_preemptions` 增量 >0 时停止后续派发；每轮最长 drain 60 秒。停止后的截断轮**不能作为完整速率档的容量读数**。

## 必须保存和解释

每个正式轮保存计划/实际派发数及时间偏差、成功率、固定输出 token 验证、客户端 TTFT/E2E P50/P95、SLO goodput、输出吞吐、Running/Waiting/KV 峰值、窗口末 Waiting、额外 drain、preemption、服务端队列时间和 TTFT/ITL histogram。原始 JSON、客户端日志与服务启动日志留在仓库；分析时区分“最终完成速率（含 drain）”和“到达率”，不把前者直接称为稳态最大容量。

若云端入口或实例关闭，只有本协议和静态校验算已完成，不能写入实测数字。实验结束停 vLLM 服务、核对显存归零；云平台是否停止计费仍需平台控制台确认。
