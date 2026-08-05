# torch_npu.npu_cross_entropy_sumexp_and_index_logit

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 训练系列产品</term>  | √   |
|<term>Atlas A3 训练系列产品</term>  | √   |
|<term>Atlas A2 训练系列产品</term>  | √   |

## 功能说明

- API功能：面向 vocab-parallel CrossEntropy 场景，融合本地 vocab shard 的 exp 计算、reduce sum、以及 target 对应 logit 的 gather 操作。用于大模型训练中 all_reduce(MAX) 之后、all_reduce(SUM) 之前的本地计算段。

- 计算逻辑：
  1. 计算 `target_mask` 和 `target_offset`：判断 target 是否在当前 rank 的 vocab 范围 `[vocab_start_index, vocab_end_index)` 内
  2. 计算 `predicted_logits`：从 `vocab_parallel_logits` 中按 target 索引取出对应 logit，减去 `global_logits_max`；若 target 不在当前 rank，则置 0
  3. 计算 `exp_logits`：`exp(vocab_parallel_logits - global_logits_max)`
  4. 计算 `sum_exp_logits`：对 `exp_logits` 沿最后一维（vocab 维度）求和

## 函数原型

```python
torch_npu.npu_cross_entropy_sumexp_and_index_logit(
    vocab_parallel_logits,
    target,
    global_logits_max,
    vocab_start_index,
    vocab_end_index
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **vocab_parallel_logits** (`Tensor`)：必选参数，当前 TP rank 本地 vocab shard 的 logits。数据类型支持 `float32`、`bfloat16`，数据格式为 ND。支持 2 维 `[N, V_local]` 或 3 维 `[S, B, V_local]`，其中 `V_local` 为当前 rank 的 vocab 分片大小。支持非连续 Tensor（框架自动转连续）。
- **target** (`Tensor`)：必选参数，全局 vocab 索引。数据类型为 `int32`，数据格式为 ND。shape 与 `vocab_parallel_logits.shape[:-1]` 一致（即 2 维输入时为 `[N]`，3 维输入时为 `[S, B]`）。支持非连续 Tensor。
- **global_logits_max** (`Tensor`)：必选参数，all_reduce(MAX) 后的全局最大 logit。数据类型与 `vocab_parallel_logits` 一致（`float32` 或 `bfloat16`），数据格式为 ND。shape 与 `target` 相同。支持非连续 Tensor。
- **vocab_start_index** (`int`)：必选参数，当前 rank vocab 起始索引（包含），类型为 `int64`。
- **vocab_end_index** (`int`)：必选参数，当前 rank vocab 结束索引（不包含），类型为 `int64`。需满足 `vocab_end_index > vocab_start_index` 且 `vocab_end_index - vocab_start_index == V_local`。

## 返回值说明

- **predicted_logits** (`Tensor`)：target 对应的 logit 减去 global_max，shape 同 `target`，数据类型固定为 `float32`。若 target 不在当前 rank 的 vocab 范围内，对应位置为 0。
- **sum_exp_logits** (`Tensor`)：本地 `exp(logits - global_max)` 沿最后一维求和，shape 同 `target`，数据类型固定为 `float32`。
- **exp_logits** (`Tensor`)：`exp(vocab_parallel_logits - global_logits_max)`，shape 同 `vocab_parallel_logits`，数据类型固定为 `float32`。
- **target_offset** (`Tensor`)：`target - vocab_start_index`，shape 同 `target`，数据类型固定为 `int32`。若 target 不在当前 rank，对应位置为 0。
- **target_mask** (`Tensor`)：标识 target 是否不在当前 rank（1 表示不在，0 表示在），shape 同 `target`，数据类型固定为 `int32`。

## 约束说明

1. **输入 shape 限制**：
   - `vocab_parallel_logits` 仅支持 2 维 `[N, V_local]` 或 3 维 `[S, B, V_local]`
   - `vocab_parallel_logits.shape[:-1]` 必须与 `target.shape` 完全一致
   - `global_logits_max.shape` 必须与 `target.shape` 完全一致
   - `vocab_end_index - vocab_start_index` 必须等于 `vocab_parallel_logits.size(-1)`

2. **数据类型约束**：
   - `vocab_parallel_logits` 和 `global_logits_max` 的数据类型必须一致（均为 `float32` 或均为 `bfloat16`）
   - `target` 必须为 `int32` 类型，值域为非负整数（全局 vocab 索引）
   - 所有浮点输出固定为 `float32`，所有整型输出固定为 `int32`

3. **对齐约束**：
   - `bfloat16` 输入时，`V_local` 必须为 16 的倍数（UB 32 字节对齐）
   - `float32` 输入时，`V_local` 必须为 8 的倍数

4. **范围约束**：
   - N（`prod(target.shape)`）范围：`[1, 32768]`
   - `V_local` 范围：`[16, 204800]`

5. **其他**：
   - 中间计算强制使用 `float32` 精度（`bfloat16` 输入自动升精度）
   - 非连续输入 Tensor 由框架自动转为连续 ND 布局，kernel 侧无需感知

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu

    # 假设 4 个 TP rank，当前为 rank 1，vocab 总大小 8192，每个 rank 分配 2048
    N = 128
    V_local = 2048
    vocab_start = 2048  # rank 1 的 vocab 起始索引
    vocab_end = 4096    # rank 1 的 vocab 结束索引

    # 构造输入
    vocab_parallel_logits = torch.randn(N, V_local, dtype=torch.float32).npu()
    target = torch.randint(0, 8192, (N,), dtype=torch.int32).npu()
    global_logits_max = torch.randn(N, dtype=torch.float32).npu()

    # 调用算子
    predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask = \
        torch_npu.npu_cross_entropy_sumexp_and_index_logit(
            vocab_parallel_logits, target, global_logits_max,
            vocab_start, vocab_end)

    print(f"predicted_logits shape: {predicted_logits.shape}")  # [N]
    print(f"sum_exp_logits shape: {sum_exp_logits.shape}")      # [N]
    print(f"exp_logits shape: {exp_logits.shape}")              # [N, V_local]
    print(f"target_offset shape: {target_offset.shape}")        # [N]
    print(f"target_mask shape: {target_mask.shape}")            # [N]
    ```

- 3 维输入调用示例

    ```python
    import torch
    import torch_npu

    S, B = 32, 16
    V_local = 4096
    vocab_start = 0
    vocab_end = 4096

    vocab_parallel_logits = torch.randn(S, B, V_local, dtype=torch.bfloat16).npu()
    target = torch.randint(0, 20000, (S, B), dtype=torch.int32).npu()
    global_logits_max = torch.randn(S, B, dtype=torch.bfloat16).npu()

    predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask = \
        torch_npu.npu_cross_entropy_sumexp_and_index_logit(
            vocab_parallel_logits, target, global_logits_max,
            vocab_start, vocab_end)

    print(f"predicted_logits dtype: {predicted_logits.dtype}")  # torch.float32
    print(f"sum_exp_logits dtype: {sum_exp_logits.dtype}")      # torch.float32
    print(f"exp_logits dtype: {exp_logits.dtype}")              # torch.float32
    ```
