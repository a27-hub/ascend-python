# npu_apply_rotary_pos_emb

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 推理系列产品</term>                             |    √    |


## 功能说明

推理网络为了提升性能，将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。

## 函数原型

```python
torch_npu.npu_apply_rotary_pos_emb(query, key,  cos, sin, *, layout='BSND', rotary_mode='half') -> (Tensor, Tensor)
```

## 参数说明

- **query**（`Tensor`）：必选参数，待执行旋转位置编码的第一个张量。要求为连续的Tensor，数据类型支持bfloat16、float16、float，数据格式支持 $ND$。
- **key**（`Tensor`）：必选参数，待执行旋转位置编码的第二个张量。要求为连续的Tensor，数据类型支持bfloat16、float16、float，数据格式支持$ND$。
- **cos**（`Tensor`）：必选参数，旋转位置编码余弦值张量。要求为连续的Tensor，数据类型支持bfloat16、float16、float，数据格式支持$ND$。
- **sin**（`Tensor`）：必选参数，旋转位置编码正弦值张量。要求为连续的Tensor，数据类型支持bfloat16、float16、float，数据格式支持$ND$。
- **layout**（`str`）：可选参数，张量布局格式，支持"BSND"、"SBND"、"BNSD"、"TND"，默认值为"BSND"。
- **rotary_mode**（`str`）：可选参数，旋转编码模式，支持"half"、"quarter"、"interleave"，默认值为"half"。

## 返回值说明

- **query_out**（`Tensor`）：表示原地更新后的query张量。
- **key_out**（`Tensor`）：表示原地更新后的key张量。

## 约束说明

- <term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - layout为"BSND"或"SBND"时，query、key、cos、sin输入shape的前2维（B、S）必须相等；layout为"TND"时，第1维（T）必须相等。
  - query、key输入shape的最后一维（D）必须相等，cos、sin输入shape的最后一维（D）必须相等。
  - 输入张量query、key、cos、sin的dtype必须相同。
  - layout为1时，输入query_out的shape用（q_b, q_s, q_n, q_d）表示，keyout shape用（q_b, q_s, k_n, q_d）表示，cos和sin shape用（q_b, q_s, 1, cos_d）表示。其中，b表示batch_size，s表示seq_length，n表示head_num，d表示head_dim。layout为4时，输入query的shape用（q_t, q_n, q_d）表示，key shape用（q_t, k_n, q_d）表示，cos和sin shape用（q_t, 1, cos_d）表示。其中，t表示b和s合轴，n表示head_num，d表示head_dim

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 对于任意layout，query与key除N维度外其他维度必须相同；query、key输入shape的最后一维（D）必须相等，cos、sin输入shape的最后一维（D）必须相等，且小于等于query、key输入shape的最后一维（D）。
  - 输入张量query、key、cos、sin的dtype必须相同。
  - rotary_mode为"half"和"interleave"时，输入shape最后一维必须被2整除；rotary_mode为"quarter"时，输入shape最后一维必须被4整除。

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
