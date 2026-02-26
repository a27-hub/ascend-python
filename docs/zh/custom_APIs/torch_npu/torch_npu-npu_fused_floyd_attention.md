# torch_npu.npu_fused_floyd_attention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>            |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>               | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

- API功能：训练场景下，npu_fused_floyd_attention相较于传统FA(npu_fusion_attention)主要是计算qk/pv注意力时会额外将seq作为batch轴从而转换为batchMatmul。
- 计算公式：

$$
P = \text{Softmax}\left(\text{Mask}\left(\text{scale} \times \left(\text{query} \times \text{key}_1^T + \text{query} \times \text{key}_2^T\right), \text{atten\_mask}\right)\right)
$$
$$
Y = \left(P \times \text{value}_1 + P \times \text{value}_2\right)
$$


## 函数原型

```
npu_fused_floyd_attention(query_ik, key_ij, value_ij, key_jk, value_jk, *, atten_mask=None, scale_value=1.) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **query_ik** (`Tensor`)：必选参数，输入张量，数据类型支持BFLOAT16、FLOAT16。数据类型与key_ij/value_ij/key_jk/value_jk的数据类型一致，数据格式支持ND，输入shape支持[BHMND]。
- **key_ij** (`Tensor`)：必选参数，输入张量，代表从节点i到其直接邻居j的关系或特征，数据类型支持BFLOAT16、FLOAT16。数据类型与query_ik/value_ij/key_jk/value_jk的数据类型一致，数据格式支持ND，输入shape支持[BHMKD]。
- **value_ij** (`Tensor`)：必选参数，输入张量，代表从节点i到其直接邻居j的信息内容，数据类型支持BFLOAT16、FLOAT16。数据类型与query_ik/key_ij/key_jk/value_jk的数据类型一致，数据格式支持ND，输入shape支持[BHKND]。
- **key_jk** (`Tensor`)：必选参数，输入张量，代表从直接邻居j到支点k的关系或特征，数据类型支持FLOAT16、BFLOAT16，数据类型与query_ik/key_ij/value_ij/value_jk的数据类型一致，数据格式支持ND，输入shape支持[BHKND]。
- **value_jk** (`Tensor`)：必选参数，输入张量，代表从节点j到其直接邻居 k的信息内容，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据类型与query_ik/key_ij/value_ij/key_jk的数据类型一致，数据格式支持ND，输入shape支持[BHMKD]。
- **atten_mask** (`Tensor`)：可选张量，数据类型支持BOOL、UINT8，数据格式支持ND，输入shape类型需为[BNK]，默认值为None。
- **scale_value** (`float`)：代表缩放系数，数据类型支持float。一般设置为D^-0.5。

## 返回值说明
- **softmax_max_out** (`Tensor`)：输出张量，Softmax计算的Max中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[BHMN8]。数据格式支持ND。
- **softmax_sum_out** (`Tensor`)：输出张量，Softmax计算的Sum中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[BHMN8]。数据格式支持ND。
- **attention_out** (`Tensor`)：输出张量，计算公式的最终输出。数据类型支持FLOAT16、BFLOAT16。数据类型和shape类型与query_ik保持一致，数据格式支持ND，输入shape支持[BHMND]。

## 约束说明

- 关于数据shape的约束，其中：
B：取值范围为1~2K。
H：取值范围为1~256。
M：取值范围为1~1M。
N：取值范围为1~1M。
K：取值范围为1~1M。
D：取值范围为32~256。

## 支持版本
PyTorch 2.6及更高版本

## 调用示例
```python
import torch
import torch_npu
import math

def truncated_normal(mean, std, min, max, size):
    x = torch.normal(mean, std, size)
    x = torch.where((x < min) | (x > max), torch.tensor(0.0), x)
    return x
B, N, S1, S2, S3, D = 1, 1, 16, 256, 256, 64
Q = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S2, D)).to(torch.bfloat16).npu()
K1 = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
K2 = truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
V1 = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
V2 = truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
atten_mask = torch.randint(0, 2, [B, 1, S1, 1, S3]).to(torch.bool).npu()
scale = 1.0/math.sqrt(D)
x_max_npu, x_sum_npu, output_npu = torch_npu.npu_fused_floyd_attention(
    Q,
    K1,
    V1,
    K2,
    V2,
    atten_mask = atten_mask,
    scale_value = scale
)
```
