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

## 约束说明

- 确定性计算：
  - aclnnApplyRotaryPosEmbV2默认确定性实现。

- <term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - layout为1时，queryRef、keyRef、cos、sin输入shape的前2维（B、S）必须相等；layout为4时，第1维（T）必须相等。
  - queryRef、keyRef输入shape的最后一维（D）必须相等，cos、sin输入shape的最后一维（D）必须相等。
  - 输入张量queryRef、keyRef、cos、sin的dtype必须相同。
  - layout为1时，输入queryRef的shape用（q_b, q_s, q_n, q_d）表示，keyRef shape用（q_b, q_s, k_n, q_d）表示，cos和sin shape用（q_b, q_s, 1, cos_d）表示。其中，b表示batch_size，s表示seq_length，n表示head_num，d表示head_dim。layout为4时，输入queryRef的shape用（q_t, q_n, q_d）表示，keyRef shape用（q_t, k_n, q_d）表示，cos和sin shape用（q_t, 1, cos_d）表示。其中，t表示b和s合轴，n表示head_num，d表示head_dim

    - 当输入是BFLOAT16时，cast表示为1，castSize为4，DtypeSize为2
    - 当输入是FLOAT16或FLOAT32时，cast表示为0，castSize = DtypeSize（FLOAT16时为2，FLOAT32时为4）

    使用lastDim表示输入shape需要旋转位置编码的维度，计算需要使用的UB空间大小：
    `ub_required = (q_n + k_n) * lastDim * castSize * 2 + lastDim * DtypeSize * 4 + (q_n + k_n) * lastDim * castSize + (q_n + k_n) * lastDim * castSize * 2 + cast * (lastDim * 4 * 2)`，
    当计算出`ub_required`的大小超过当前AI处理器的UB空间总大小时，不支持使用该融合算子。

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 对于任意layout，queryRef与keyRef除N维度外其他维度必须相同；queryRef、keyRef输入shape的最后一维（D）必须相等，cos、sin输入shape的最后一维（D）必须相等，且小于等于queryRef、keyRef输入shape的最后一维（D）。
  - 输入张量queryRef、keyRef、cos、sin的dtype必须相同。
  - rotaryMode为"half"和"interleave"时，输入shape最后一维必须被2整除；rotaryMode为"quarter"时，输入shape最后一维必须被4整除。

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
