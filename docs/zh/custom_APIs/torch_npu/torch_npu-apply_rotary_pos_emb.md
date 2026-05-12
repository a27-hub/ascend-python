# npu_apply_rotary_pos_emb

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    √    |
| <term>Atlas 训练系列产品</term>                              |    x    |

## 功能说明

- 接口功能：推理网络为了提升性能，将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。

## 函数原型

```python
torch_npu.npu_apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, *, str layout='BSND', str rotary_mode='half') -> (Tensor, Tensor)
```

## 参数说明

- **query**（`Tensor`）：必选参数，待执行旋转位置编码的第一个张量。要求为连续的Tensor，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据格式支持ND。
- **key**（`Tensor`）：必选参数，待执行旋转位置编码的第二个张量。要求为连续的Tensor，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据格式支持ND。
- **cos**（`Tensor`）：必选参数，旋转位置编码余弦值张量。要求为连续的Tensor，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据格式支持ND。
- **sin**（`Tensor`）：必选参数，旋转位置编码正弦值张量。要求为连续的Tensor，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据格式支持ND。
- **layout**（`str`）：可选参数，张量布局格式，支持"BSND"、"SBND"、"BNSD"、"TND"，默认值为"BSND"。
- **rotary_mode**（`str`）：可选参数，旋转编码模式，支持"half"、"quarter"、"interleave"，默认值为"half"。

## 返回值说明

- **query_out**（`Tensor`）：表示原地更新后的query张量。
- **key_out**（`Tensor`）：表示原地更新后的key张量。

## 调用示例

```python
import torch
import torch_npu

def test_npu_apply_rotary_pos_emb():

    # 固定参数
    batch = 1
    seq_len = 64
    num_heads = 8
    head_dim = 64

    # 创建输入数据
    query = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
    cos = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()
    sin = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()

    # 调用 npu_apply_rotary_pos_emb API
    q_out, k_out = torch_npu.npu_apply_rotary_pos_emb(
        query, key, cos, sin,
        layout="BSND",
        rotary_mode="half"
    )

    print("API: npu_apply_rotary_pos_emb test passed!")
    print(f"Output query: {q_out}")
    print(f"Output key: {k_out}")

if __name__ == "__main__":
    test_npu_apply_rotary_pos_emb()
```
