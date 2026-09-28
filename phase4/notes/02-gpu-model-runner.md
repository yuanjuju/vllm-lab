# 源码精读笔记：GPU 执行路径——模型本体、block table 与 sampling

记录日期：2026-09-28。源码版本：vLLM v0.29.0（`../vllm-src`）。

## 阅读范围

- `vllm/v1/worker/gpu_model_runner.py:3917-3968`（`_model_forward`、`_is_uniform_decode`）、`:4249-4360`（`execute_model` 编排）
- `vllm/v1/worker/block_table.py:157-232`（`BlockTable.append_row / add_row / compute_slot_mapping`）
- `vllm/v1/sample/sampler.py:228-242, 274-303`（`apply_temperature`、`greedy_sample`、`sample` 主流程）
- `vllm/model_executor/models/qwen3.py:140-246, 318-335`（Attention、DecoderLayer、ForCausalLM）
- `vllm/model_executor/models/qwen2.py:397-434`（`Qwen2Model.forward`——Qwen3 继承的层循环）
- phase3 归档日志 `phase3/results/token_budget_server_4096_20260923.log:12,43-52`（CUDA graph 捕获的真实记录）

## 模块职责

GPU Model Runner 拿到 Scheduler 的"算什么"（`SchedulerOutput`），组装成张量喂给模型，再对输出 logits 采样。**模型本体（Qwen3ForCausalLM）完全不感知"请求"**——它只看一个拼接后的一维 token 流；请求边界全部存在于元数据（positions、block table、seq_lens）中。

## 关键代码与尺寸（Qwen3-8B 实际数字）

模型主体就是一个 for 循环（`qwen2.py:408-429`，Qwen3 经 `qwen3.py:262` 继承）：

```python
hidden_states = self.embed_input_ids(input_ids)      # [N] int → [N, 4096]
for idx, layer in enumerate(islice(self.layers, ...)):   # 36 层
    hidden_states, residual = layer(positions, hidden_states, residual)
hidden_states, _ = self.norm(hidden_states, residual)
# 之后 ForCausalLM.compute_logits: lm_head → [N, 151936] logits
```

一层内部（`qwen3.py:153-170` Attention / `:226-246` Layer）：

```python
qkv, _ = self.qkv_proj(hidden_states)                # [N,4096] → [N,6144]
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
#  q_size = 32×128 = 4096；kv_size = 8×128 = 1024  ← GQA 4:1 在这里现形
q, k = self.rotary_emb(positions, q, k)              # 位置信息每层现算
attn_output = self.attn(q, k, v)                     # FlashAttention，读 block table
```

**GQA 与 KV 公式的对应**：每层每 token 写 K(1024) + V(1024) 个 bf16 = 4 KiB，×36 层 = **144 KiB/token**——正是 README 公式 `2×36×8×128×2B` 的代码级来源。

## Block table：PagedAttention 的"页表"

`block_table.py:157-177`：每条请求一行 numpy 数组，存它占用的块号序列；`compute_slot_mapping`（:201）把每个新 token 映射到"块号 + 块内偏移"，K/V 写入时按此定位。attention kernel 读逻辑位置 → 查表 → 物理块。**逻辑连续、物理散落**，与 OS 页表同构。第一课 `allocate_slots` 返回的 block_ids，最终就填进这张表。

## Sampling：phase2 参数的真身

`sampler.py`：

```python
def apply_temperature(logits, temp, ...):   # :228
    return logits.div_(temp.unsqueeze(dim=1))   # temperature 就是除法

def greedy_sample(logits):                  # :240
    return logits.argmax(dim=-1)                 # 贪心 = argmax
```

主流程 `sample()`（:274-303）：temperature → argmax-invariant processors → `topk_topp_sampler`（top_k/top_p 在 `v1/sample/ops/`）；greedy 与随机按每请求温度 `torch.where` 混合。phase2 的 temperature/top_p 实验改的就是这里的输入参数。

## CUDA graph：decode 为什么能"录像回放"

phase3 归档日志的真实记录（`token_budget_server_4096_20260923.log`）：

```text
cudagraph_capture_sizes: [1, 2, 4, 8, 16]，max_cudagraph_capture_size: 16
Capturing CUDA graphs (PIECEWISE): 5/5
Capturing CUDA graphs (FULL): 2/2
```

原理：decode 步形状规整（每请求恰好 1 token，`_is_uniform_decode` :3950-3968 判定），可按 batch-size 档位（1/2/4/8/16）预先"录像"再一键"回放"，省掉每步数百次 kernel launch 的 CPU 开销。实际 batch=7 → 回放 size-8 的图，多余行填 dummy。prefill token 数任意、形状多变 → 只能做 PIECEWISE（attention 之外的固定段捕获）。`--enforce-eager` 即全部禁用——Phase 5 第一个实验的对照对象。

## 与实验数据的对应

- **TTFT（服务端 0.45-0.56 s）**：一条 173-token 输入的 prefill = 上述 for 循环一次性算 173 行 + 排队；ITL = 每步 1 行的整套循环 + launch 开销。
- **phase2 temperature/top_p 差异**：`apply_temperature` 一行除法 + top_p 截断改变分布；16/64 上限以 `length` 结束是输出侧 stop 检查（output_processor），与采样独立。
- **KV pressure 的 1024-block 池**：每块 16 token × 144 KiB/token ≈ 2.25 MiB，块行长度即 `block_ids` 数量——分配失败（第一课 :524）就是这张表填不进新行。

## 与笔记结论的修正

- [01-request-lifecycle](../../phase3/learning_notes/01-request-lifecycle.md) 中"多个请求可以动态合到执行批次"→ 源码级精确：GPU 侧不存在"批"的张量维度，**批 = token 流拼接 + 每请求元数据**（`_prepare_inputs` 组装，模型无感知）。
- README 架构图里 "GPU Worker 执行模型前向"→ 具体化为 `execute_model`（编排）→ `_model_forward`（调用）→ `Qwen2Model.forward`（36 层循环）。

## 未解问题 / 下一步

- `topk_topp_sampler` 的 top_p 截断细节（`v1/sample/ops/`）——留作练习。
- attention 后端如何用 block table 索引（FlashAttention metadata：`_build_attention_metadata` :2318 未精读）。
- `num-gpu-blocks-override` 到逻辑池大小的确定路径（`gpu_worker.py` profile 段未读）。
- piecewise 编译的 splitting_ops 边界（日志中 `unified_attention_with_output` 等即切分点）。
