# Mac Metal：先建立测量基线

云端服务之前，我先用 Mac M5 和 Qwen3-0.6B 把请求、流式计时与指标采集跑通。小模型的好处是能快速重复请求；它的速度不代表 Qwen3-8B，也不能用来推断 4090 的硬件优势。

## 这部分回答了什么

| 问题 | 观察 | 数据 |
| --- | --- | --- |
| 生成参数改变了什么？ | 对比 `temperature`、`top_p`、`max_tokens` 和 Qwen3 思考模式；16/64 token 上限的请求以 `length` 结束，256 token 上限的同一任务自然停止。 | [temperature](results/01_temperature_20260920-131622.tsv) · [top_p](results/02_top_p_20260920-132326.tsv) · [max_tokens](results/03_max_tokens_20260920-132332.tsv) · [思考模式](results/04_thinking_mode_20260920-132344.tsv) |
| 流式请求有多快？ | 单次请求记录了首个内容块、总时间和 `[DONE]`；补采的 20 次请求全部成功，TTFT P50 为 0.0301 s。 | [连续请求逐条数据](results/06_sequential_stability_20260920-132447.csv)、[机器汇总](results/06_sequential_stability_summary_20260920-132447.json) |
| 增加并发会怎样？ | 2、4、8 并发档位各发送 8 个短请求；本次最高 Running 为 8，Waiting 为 0。负载太轻，没有制造出排队。 | [并发汇总](results/07_concurrency_summary_20260920-135629.json) |

`temperature`、`top_p`、思考模式、流式请求和连续请求由我在终端操作；2/4/8 并发与引擎指标由助手补采。临时采集脚本按当时的要求在收尾后移除，因此这一阶段保留了原始响应与汇总，**没有保留完整的重跑入口**。仓库根目录的 `env.sh`、`run_server.sh` 和 `test_api.sh` 仍可启动并检查本地服务。

完整环境、手动终端记录及补采数据的区别见 [Mac 性能基线](results/mac_baseline.md)。连续请求复用了相同提示词，预热与缓存收益可能同时存在，不能仅凭低 TTFT 就断言某个缓存机制起了作用。
