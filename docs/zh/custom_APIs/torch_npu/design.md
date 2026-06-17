# 词汇表并行融合交叉熵算子补充手册和用例

## 1. 概述

本次提交（`9b12f18`）新增3个 `torch_npu` 融合算子，用于**词汇表并行（Vocabulary Parallel）场景下融合矩阵乘与交叉熵损失计算**。功能等价于 Megatron 的 `fused_vocab_parallel_cross_entropy` 实现，支持 `vocabulary_size` 维度切卡融合 matmul 与 celoss，解决超大规模词汇表下的显存和计算效率问题。

三个算子形成完整的**前向-损失计算-反向梯度**流水线：

```
fused_linear_online_max_sum  →  全局通信(all-reduce)  →  fused_cross_entropy_loss_with_max_sum
                                                                  ↓ (反向)
                                         fused_linear_cross_entropy_loss_with_max_sum_grad
```

## 2. 新增算子接口

### 2.1 fused_linear_online_max_sum

**功能**：词汇表并行场景下融合矩阵乘与交叉熵前处理。单卡内完成 matmul、行最大值、exp 累加和、target mask 等计算。需与 `fused_cross_entropy_loss_with_max_sum` 配合使用。

**函数签名**：

```python
torch_npu.fused_linear_online_max_sum(
    input,                # Tensor [batch, hidden], bfloat16/float16
    weight,               # Tensor [vocab_size, hidden], 与input同类型
    target,               # Tensor [batch], int32/int64
    vocab_start_index,    # int, 本卡词汇表起始索引
    vocab_end_index,      # int, 本卡词汇表结束索引
    return_logits=False   # bool, True=高性能分支, False=省显存分支
) -> (logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits)
```

**计算流程**：

1. 矩阵乘：`vocabParallelLogits = input @ weight^T`
2. 行最大值：`logitsMax = max(vocabParallelLogits, dim=-1)`
3. 差值计算：`subRes = vocabParallelLogits - logitsMax`
4. exp 累加和：`sumExpLogits = sum(exp(subRes), dim=-1)`
5. target mask：判断 target 是否在 `[vocabStartIndex, vocabEndIndex]` 范围内
6. maskedTarget：范围内映射为 `target - vocabStartIndex`，范围外映射为 0
7. predictedLogits：范围内取 `subRes[maskedTarget]`，范围外取 0
8. targetMask 输出为 packed bit 表示（uint8，每 bit 代表 1 个布尔值）

**输出**：

| 输出 | 类型 | Shape |
|------|------|-------|
| logits_max | float32 | [batch] |
| sum_exp_logits | float32 | [batch] |
| predicted_logits | float32 | [batch] |
| target_mask | uint8 | [(batch+7)//8] |
| masked_target | int32/int64 | [batch] |
| vocab_parallel_logits | float16/bfloat16/None | [batch, vocab_size] |

**约束**：input 与 weight 数据类型一致；`vocab_start_index >= 0`；`vocab_end_index >= vocab_start_index`；确定性实现。

---

### 2.2 fused_cross_entropy_loss_with_max_sum

**功能**：词汇表并行场景下交叉熵计算的后半部分，计算 loss 与 softmax 结果。需在多卡场景下先对 `logits_max` 和 `sum_exp_logits` 执行全局通信（all-reduce）。

**函数签名**：

```python
torch_npu.fused_cross_entropy_loss_with_max_sum(
    logits_max,                     # Tensor [batch], float32 (全局通信后)
    sum_exp_logits,                 # Tensor [batch], float32 (全局通信后)
    predicted_logits,               # Tensor [batch], float32 (全局通信后)
    *,
    label_smoothing=0.0,            # float, 当前仅支持0
    input=None,                     # Tensor, 当前仅支持None
    weight=None,                    # Tensor, 当前仅支持None
    vocab_parallel_logits=None      # Tensor [batch, vocab_size], 传入时计算softmax
) -> (loss, softmax)
```

**计算公式**：

- `loss = log(sum_exp_logits) - predicted_logits`
- `softmax = exp(vocab_parallel_logits - logits_max.unsqueeze(-1)) / sum_exp_logits.unsqueeze(-1)`（仅 vocab_parallel_logits 非 None 时）

**输出**：

| 输出 | 类型 | Shape |
|------|------|-------|
| loss | float32 | [batch] |
| softmax | float32/None | [batch, vocab_size]（vocab_parallel_logits 为 None 时返回 None） |

**约束**：三个必选参数 shape 一致；`label_smoothing` 仅支持 0。

---

### 2.3 fused_linear_cross_entropy_loss_with_max_sum_grad

**功能**：词汇表并行场景下交叉熵损失的反向梯度算子，计算 input 和 weight 的梯度。支持两种模式。

**函数签名**：

```python
torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad(
    grad,               # Tensor [batch], float32
    input,              # Tensor [batch, hidden], float16/bfloat16
    weight,             # Tensor [vocab_size, hidden], 与input同类型, dim0 >= 128
    target_mask,        # Tensor [(batch+7)//8], uint8
    masked_target,      # Tensor [batch], int32/int64
    label_smoothing=0.0,
    logits_max=None,    # 省显存模式必选
    sum_exp_logits=None,# 省显存模式必选
    softmax=None        # 高性能模式必选
) -> (input_grad, weight_grad)
```

**两种计算模式**：

- **高性能模式**（传入 softmax）：直接使用 softmax 结果计算梯度
  - `softmax[arange, maskedTarget] -= (1 - targetMask)`
  - `softmax *= grad.unsqueeze(-1)`
  - `gradInput = softmax @ weight`
  - `gradWeight = softmax^T @ input`

- **省显存模式**（传入 logits_max + sum_exp_logits）：重计算 softmax
  - `vocabParallelLogits = input @ weight^T`
  - `expLogits = exp(vocabParallelLogits - logitsMax.unsqueeze(-1))`
  - `expLogits /= sumExpLogits.unsqueeze(-1)`
  - 后续同高性能模式

**输出**：

| 输出 | 类型 | Shape |
|------|------|-------|
| input_grad | float16/bfloat16 | [batch, hidden] |
| weight_grad | float16/bfloat16 | [vocab_size, hidden] |

**约束**：input 与 weight 类型一致；`softmax` 为 None 时 `logits_max` 和 `sum_exp_logits` 必须同时提供；`weight` 第 0 维 >= 128。

## 3. 数据类型支持

| 参数 | 支持的数据类型 |
|------|---------------|
| input / weight | float16, bfloat16 |
| target / masked_target | int32, int64 |
| logits_max / sum_exp_logits / predicted_logits / grad | float32 |
| target_mask | uint8 |
| loss / softmax (输出) | float32 |
| vocab_parallel_logits | float16, bfloat16 |
| input_grad / weight_grad (输出) | float16, bfloat16 |

## 4. 产品支持

- Atlas A3 训练/推理系列产品
- Atlas A2 训练/推理系列产品

## 5. 修改文件清单

| 文件路径 | 变更类型 | 说明 |
|---------|---------|------|
| `codegen/templates/_op_plugin_docs.py` | 修改 | 新增3个算子的文档字符串注册 |
| `docs/zh/custom_APIs/torch_npu/torch_npu-fused_linear_online_max_sum.md` | 新增 | fused_linear_online_max_sum 中文文档 |
| `docs/zh/custom_APIs/torch_npu/torch_npu-fused_cross_entropy_loss_with_max_sum.md` | 新增 | fused_cross_entropy_loss_with_max_sum 中文文档 |
| `docs/zh/custom_APIs/torch_npu/torch_npu-fused_linear_cross_entropy_loss_with_max_sum_grad.md` | 新增 | fused_linear_cross_entropy_loss_with_max_sum_grad 中文文档 |
| `test/test_base_ops/test_fused_linear_online_max_sum.py` | 新增 | fused_linear_online_max_sum 测试用例（4个） |
| `test/test_base_ops/test_fused_cross_entropy_loss_with_max_sum.py` | 新增 | fused_cross_entropy_loss_with_max_sum 测试用例（4个） |
| `test/test_base_ops/test_fused_linear_cross_entropy_loss_with_max_sum_grad.py` | 新增 | fused_linear_cross_entropy_loss_with_max_sum_grad 测试用例（4个） |

**统计**：新增 660 行，删除 3 行（行尾空白修正）。

## 6. 测试覆盖

共 12 个测试用例，分布在 3 个测试文件中。

### 6.1 test_fused_linear_online_max_sum.py（4个用例）

| 用例 | 配置 | 验证点 |
|------|------|--------|
| `test_fused_linear_online_max_sum_fp16` | batch=128, hidden=64, vocab=256, range=[0,64), fp16, return_logits=True | 全部6个输出的shape、dtype正确性：logits_max/sum_exp_logits/predicted_logits 为 float32[128]，target_mask 为 uint8[16]，masked_target 为 int32[128]，vocab_parallel_logits 为 float16[128,256] |
| `test_fused_linear_online_max_sum_bf16` | batch=64, hidden=32, vocab=128, range=[0,32), bf16, return_logits=True | bf16 下 logits_max/sum_exp_logits shape正确，vocab_parallel_logits dtype 为 bfloat16 |
| `test_fused_linear_online_max_sum_no_logits` | batch=32, hidden=16, vocab=64, range=[0,16), fp16, return_logits=False | 省显存模式下 logits_max/sum_exp_logits/predicted_logits 仍正常输出，vocab_parallel_logits 为 None |
| `test_fused_linear_online_max_sum_int64_target` | batch=16, hidden=8, vocab=32, range=[0,8), fp16, int64 target, return_logits=True | masked_target dtype 跟随 target 为 int64，logits_max shape 正确 |

### 6.2 test_fused_cross_entropy_loss_with_max_sum.py（4个用例）

| 用例 | 配置 | 验证点 |
|------|------|--------|
| `test_fused_cross_entropy_loss_with_max_sum_basic` | batch=128, sum_exp_logits保证>0, vocab_parallel_logits=None | loss 为 float32[128]，softmax 为 None（不传 logits 时不计算 softmax） |
| `test_fused_cross_entropy_loss_with_max_sum_with_logits` | batch=64, vocab=256, vocab_parallel_logits=float16[64,256] | loss 为 float32[64]，softmax 为 float32[64,256]（传入 logits 后正确计算并输出 softmax） |
| `test_fused_cross_entropy_loss_with_max_sum_bf16_logits` | batch=32, vocab=128, vocab_parallel_logits=bf16[32,128] | bf16 logits 输入下 loss shape[32]、softmax shape[32,128] 正确 |
| `test_fused_cross_entropy_loss_with_max_sum_default_smoothing` | batch=16, 仅传3个必选参数 | 使用 label_smoothing 默认值0.0，loss shape[16] 正确 |

### 6.3 test_fused_linear_cross_entropy_loss_with_max_sum_grad.py（4个用例）

| 用例 | 配置 | 验证点 |
|------|------|--------|
| `test_grad_high_perf_mode_fp16` | batch=128, hidden=64, vocab=256, fp16, softmax=float32[128,256] | 高性能模式：input_grad 为 fp16[128,64]，weight_grad 为 fp16[256,64]，dtype/shape 正确 |
| `test_grad_high_perf_mode_bf16` | batch=64, hidden=32, vocab=128, bf16, softmax=float32[64,128] | 高性能模式 bf16：input_grad/weight_grad dtype 为 bfloat16 |
| `test_grad_memory_save_mode_fp16` | batch=128, hidden=64, vocab=256, fp16, logits_max+sum_exp_logits, softmax=None | 省显存模式：不传 softmax，传入 logits_max 和 sum_exp_logits 重计算，input_grad/weight_grad shape 和 fp16 dtype 正确 |
| `test_grad_int64_target` | batch=64, hidden=32, vocab=128, fp16, masked_target=int64 | int64 target 下梯度 shape 正确 |

### 6.4 覆盖维度总结

| 维度 | 覆盖情况 |
|------|---------|
| 数据类型 | fp16, bf16, int32, int64 |
| 运行模式 | 高性能模式(return_logits=True/softmax非None), 省显存模式(return_logits=False/softmax=None) |
| 输出完整性 | 6输出/2输出的 shape 和 dtype 逐一验证 |
| 省略参数默认值 | label_smoothing 默认、可选参数 None |
| vocab范围 | vocab_start=0, vocab_end < vocab_size（模拟切卡场景） |

## 7. 设计要点

1. **两阶段前向拆分**：将 matmul + softmax 前处理拆为 `fused_linear_online_max_sum`，在单卡内完成；loss 计算拆为 `fused_cross_entropy_loss_with_max_sum`，在全局通信后完成。中间只需通信两个 1 维向量（`logits_max` 和 `sum_exp_logits`），而非整个 matmul 结果，显著降低通信量。

2. **高性能/省显存双模式**：通过 `return_logits` 和 `softmax` 参数控制是否保留中间 matmul 结果。高性能模式保存完整 logits/softmax 避免反向重计算；省显存模式丢弃中间结果，反向时重计算。

3. **packed bit mask**：`target_mask` 采用 uint8 packed bit 表示，每 bit 代表一个布尔值，节省 8 倍存储。
