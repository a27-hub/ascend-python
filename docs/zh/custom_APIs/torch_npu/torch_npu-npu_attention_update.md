# torch_npu.npu_attention_update

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |


## 功能说明

- API功能：将各SP（Sequence Parallelism）域PA算子输出的中间结果lse和attention out两个局部结果更新成全局结果。

- 计算公式：

  $$
  lsemax = \max(lse_i)
  $$

  $$
  lse\_sum = \sum \exp(lse_i - lsemax)
  $$

  $$
  lsem = lsemax + \log(lse\_sum)
  $$

  $$
  O = \sum O_i \cdot \exp(lse_i - lsem)
  $$

## 函数原型
```
torch_npu.npu_attention_update(lse, local_out, update_type) -> (Tensor, Tensor)
```

## 参数说明

- **lse**(`Tensor[]`)：必选参数，Device侧的TensorList，TensorList长度为SP数，每个Tensor形状为$(batch \times seqLen \times headNum)$，表示各SP域计算的lse。数据类型支持`float32`，数据格式支持$ND$。
- **local_out**(`Tensor[]`)：必选参数，Device侧的TensorList，TensorList长度为SP数，每个Tensor形状为$(batch \times seqLen \times headNum, head\_dim)$，表示各SP域计算的output。数据类型支持`float32`、`float16`、`bfloat16`，数据格式支持$ND$。
- **update_type**(`int`)：必选参数，指定执行的操作类型。取值为`0`时，仅输出合并后的out；取值为`1`时，同时输出合并后的out和lse_out。

## 返回值说明

- **out**(`Tensor`)：合并后的全局attention输出。形状为$(batch \times seqLen \times headNum, head\_dim)$，数据类型与输入`local_out`中的Tensor数据类型一致，数据格式为$ND$。
- **lse_out**(`Tensor`)：合并后的全局lse输出。形状为$(batch \times seqLen \times headNum)$，数据类型为`float32`，数据格式为$ND$。仅当`update_type=1`时有效。

## 约束说明

- `lse`和`local_out`的TensorList长度必须一致。
- `lse`中每个Tensor的数据类型必须为`float32`。
- 所有输入输出Tensor的数据格式推荐使用ND格式。

## 调用示例

```python
import torch
import torch_npu

dtype = torch.float32
N = 4
head_dim = 32

lse = [
    torch.randn(N, dtype=dtype, device='npu'),
    torch.randn(N, dtype=dtype, device='npu'),
]

local_out = [
    torch.randn(N, head_dim, dtype=dtype, device='npu'),
    torch.randn(N, head_dim, dtype=dtype, device='npu'),
]

# update_type=0：仅输出合并后的out
out, lse_out = torch_npu.npu_attention_update(lse, local_out, 0)
print("out:", out)
print("out.shape:", out.shape)

# update_type=1：同时输出合并后的out和lse_out
out, lse_out = torch_npu.npu_attention_update(lse, local_out, 1)
print("out:", out)
print("lse_out:", lse_out)
```
