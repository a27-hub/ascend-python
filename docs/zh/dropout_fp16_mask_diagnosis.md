# `fp16` dropout mask 语义差异说明

## 问题描述

当前 `fp16` 场景下，NPU 上两条 dropout 相关路径生成的 keep mask 不一致：

- `bernoulli_` / `dropout_` 路径
- `native_dropout` / `aclnnDropoutGenMaskV2` 路径

表现为：

- `fp32` 下两条路径一致
- `fp16` 下两条路径不一致

这会导致：

- `dropout(inplace=True)` 的结果
- 与 `native_dropout` 相关路径推导出的结果

在相同 seed、相同概率下出现分叉。

## 当前代码路径

## `torch.nn.Dropout` 链路说明

`torch.nn.Dropout` 在 Python 前端就分成两条调用：

- `inplace=False` -> `torch.nn.functional.dropout(...)` -> `_VF.dropout(...)`
- `inplace=True` -> `torch.nn.functional.dropout(...)` -> `_VF.dropout_(...)`

原生 torch 后端也保持这两条链路分离。

### 原生 torch / CUDA 的链路

在原生 torch 中：

- `dropout(input, p, train)`
  - 对 CUDA 这类 fused kernel 可用设备，优先走 `native_dropout`
  - CUDA 后端对应 `native_dropout_cuda`
- `dropout_(input, p, train)`
  - 走通用 inplace 实现
  - 逻辑是 `bernoulli_(1 - p) + div_(1 - p) + mul_`

这意味着：

> `inplace=False` 和 `inplace=True` 本来就是两条不同的实现链路。

这一点不是 NPU 特有行为，CUDA 也是一样。

### 当前 NPU 的链路

当前 NPU 实际运行时：

- `inplace=False`
  - `aten::dropout`
  - `aten::native_dropout`
  - `aclnnDropoutGenMaskV2`
  - `aclnnDropoutDoMask`
- `inplace=True`
  - `aten::dropout_`
  - `aten::bernoulli_`
  - `aclnnInplaceBernoulli`
  - `aclnnInplaceDivs`
  - `aclnnInplaceMul`

这和原生 torch 的“非原地走 `native_dropout`、原地走 `bernoulli_ + div_ + mul_`”设计是一致的。

因此：

> 当前问题不在于 `inplace=False` 和 `inplace=True` 走了不同链路。

因为这件事本身就是 upstream/CUDA 的既有设计。

当前真正的问题是：

> 在 NPU 上，这两条链路在 `fp16` 下没有得到一致的 keep mask 语义。

### `dropout_` 路径

真实 `dropout_` 运行时走的是：

- `aten::bernoulli_`
- `aclnnInplaceBernoulli`
- `aclnnInplaceDivs`
- `aclnnInplaceMul`

因此 `dropout_` 的 mask 语义实际由 `bernoulli_` 路径决定。

### `native_dropout` 路径

`native_dropout` 走的是：

- `aten::native_dropout`
- `aclnnDropoutGenMaskV2`
- `aclnnDropoutDoMask`

因此 `native_dropout` 的 mask 语义由 `DropoutGenMaskV2` 路径决定。

## 代码差异点

### `aclnnInplaceBernoulli`

在 `CANN/ops-math/random/dsa_gen_bit_mask/op_host/op_api/aclnn_bernoulli.cpp` 的 DSA 路径中，概率标量按 `float32` 构造：

```cpp
auto probScalar = executor->AllocScalar(static_cast<float>(1 - prob->ToDouble()));
auto mask = l0op::DSAGenBitMask(..., probScalar, executor);
```

### `aclnnDropoutGenMaskV2`

在 `CANN/ops-math/random/dsa_gen_bit_mask/op_host/op_api/aclnn_dropout_gen_mask.cpp` 中，`DropoutGenMaskV2` 按传入的 `probDataType` 生成概率标量：

```cpp
switch (dtype) {
    case op::DataType::DT_FLOAT16:
        return executor->AllocScalar(&ratioF16.val, op::DataType::DT_FLOAT16);
    case op::DataType::DT_BF16:
        return executor->AllocScalar(&ratioBf16.value, op::DataType::DT_BF16);
    case op::DataType::DT_FLOAT:
        return executor->AllocScalar(prob);
}
```

调用 `DropoutGenMaskV2` 时：

- `fp16 input -> probDataType = ACL_FLOAT16`
- `fp32 input -> probDataType = ACL_FLOAT`

因此 `fp16` 下，`DropoutGenMaskV2` 会用 `float16` 概率标量生成 mask，而 `bernoulli_` 路径使用的是 `float32` 概率标量。

## 历史结论

`CANN/ops-math` 历史显示：

- `aclnnInplaceBernoulli` 当前 scalar-path 的 `float32` 概率语义不是近期引入的
- `aclnnDropoutGenMaskV2` 的 `fp16/bf16/float` 概率 dtype 支持也不是近期引入的

因此当前问题更像是：

> 两条路径长期存在概率 dtype 语义差异，在 `fp16` 场景下被暴露出来。

## 最小复现脚本

脚本：

- `op-plugin/test/test_base_ops/dropout_gen_mask_vs_bernoulli_diag.py`

运行方式：

```bash
python op-plugin/test/test_base_ops/dropout_gen_mask_vs_bernoulli_diag.py
```

脚本行为：

1. 调用 `torch_npu._npu_dropout_gen_mask(...)` 生成 mask
2. 对同 shape、同 seed、同 `keep_prob` 执行 `bernoulli_`
3. 直接比较两份 keep mask

固定参数：

- `shape=(12, 12)`
- `p=0.5`
- `seed=2`
- 比较 `float32` 与 `float16`

预期现象：

- `float32`: `gen_mask keep mask == bernoulli keep mask`
- `float16`: `gen_mask keep mask != bernoulli keep mask`

脚本末尾会输出：

```text
Conclusion: fp16 gen_mask keep mask diverges from bernoulli keep mask.
```

## 当前结论

当前应将问题定性为：

> `fp16` 下，`bernoulli_` 路径与 `aclnnDropoutGenMaskV2` 路径的 keep mask 语义不一致。

后续修复应围绕两条路径的概率 dtype 统一展开，而不是继续从 `copy_` 或上层包装逻辑入手。
