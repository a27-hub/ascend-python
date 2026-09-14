# torch\_npu.npu\_quant\_fusion\_attention

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## 功能说明

- **API功能**：实现“Transformer Attention Score”的融合FP8量化计算。

- **计算公式**：

    ![](../../figures/zh-cn_formulaimage_0000002524894668.png)

    其中：

    ![](../../figures/zh-cn_formulaimage_0000002525053994.png)

## 函数原型

```python
torch_npu.npu_quant_fusion_attention(query, key, value, head_num, input_layout, *, d_scale_q, d_scale_k, d_scale_v, p_scale, scale=1., query_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
```

## 参数说明

- **query**（`Tensor`）：必选参数，数据类型支持`hifloat8`，数据格式支持ND。torch没有hifloat8类型，用uint8的数据表示。
- **key**（`Tensor`）：必选参数，数据类型支持`hifloat8`，数据格式支持ND。torch没有hifloat8类型，用uint8的数据表示。
- **value**（`Tensor`）：必选参数，数据类型支持`hifloat8`，数据格式支持ND。torch没有hifloat8类型，用uint8的数据表示。
- **head\_num**（`int`）：必选参数，代表head个数，数据类型支持`int64`。
- **input\_layout**（`str`）：必选参数，代表输入query、key、value的数据排布格式，支持BSND；后续章节如无特殊说明，S1表示query的S，S2表示key、value的S。N1表示query的N，N2表示key，value的N。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **d\_scale\_q**（`Tensor`）：必选参数（当前必须不为None），代表query的scale。当前版本要求非空，输入shape为\[B, N1, S1/128, 1\]，数据类型支持`float`，数据格式支持ND。
- **d\_scale\_k**（`Tensor`）：必选参数（当前必须不为None），代表key的scale。当前版本要求非空，输入shape为\[B, N2, S2/256, 1\]，数据类型支持`float`，数据格式支持ND。
- **d\_scale\_v**（`Tensor`）：必选参数（当前必须不为None），代表value的scale。当前版本要求非空，输入shape为\[B, N2, S2/512, 1\]，数据类型支持`float`，数据格式支持ND。
- **p\_scale**（`Tensor`）：可选参数（当前必须不为None），代表量化参数scale。当前版本要求非空，输入shape为\[1\]，数据类型支持`float`，数据格式支持ND。
- **scale**（`float`）：可选参数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持`float`，默认值为1.。
- **query\_dtype**（`int`）：可选参数，表示query的类型。跑hifloat8的时候需要进行配置，值为torch\_npu.hifloat8，其他场景可忽略。由于当前仅支持hifloat8场景，因此该参数必须手动配置为`torch_npu.hifloat8`。

## 返回值说明

共7个输出，类型依次为Tensor、Tensor、Tensor、Tensor、int、int、int。

- 第1个输出为Tensor，计算公式的最终输出attention\_out，数据类型支持`bfloat16`。
- 第2个输出为Tensor，Softmax计算的Max中间结果，用于反向计算，数据类型支持`float32`。
- 第3个输出为Tensor，Softmax计算的Sum中间结果，用于反向计算，数据类型支持`float32`。
- 第4个输出为Tensor，预留输出，暂未使用。
- 第5个输出为int，预留输出，暂未使用。
- 第6个输出为int，预留输出，暂未使用。
- 第7个输出为int，预留输出，暂未使用。

## 约束说明

- 该接口仅在训练场景下使用。
- 该接口仅支持单算子模式。
- 输入`query`、`key`、`value`的数据类型和`input_layout`必须一致；`query`、`key`、`value`输入dtype为hifloat8，不支持TND数据排布。
- 输入`query`、`key`、`value`的shape说明，当前支持以下场景：
  - 不支持任何可选参数。

    **表 1**  支持shape列表

    | input_layout | queryShape | keyShape | valueShape |
    | --- | --- | --- | --- |
    | BSND | [1, 57600, 5, 128] | [1, 57600, 5, 128] | [1, 57600, 5, 128] |
    | BSND | [1, 7200, 40, 128] | [1, 512, 40, 128] | [1, 512, 40, 128] |

## 调用示例

单算子模式调用

```python
import math
import unittest
import numpy as np
import torch
import torch_npu
import en_dtypes #en_types下载链接：https://pypi.org/project/en-dtypes/#files
from en_dtypes import hifloat8
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
class TestNPUQuantFlashAttentionV2(TestCase):
    def custom_op_exec_quant(self, query, key, value, d_scale_q, d_scale_k, d_scale_v, p_scale):
        scale = 0.08838
        return torch_npu.npu_quant_fusion_attention(
            query, key, value, head_num=5, input_layout="BSND", scale=scale,
            d_scale_q=d_scale_q, d_scale_k=d_scale_k, d_scale_v=d_scale_v, p_scale=p_scale,
            query_dtype = torch_npu.hifloat8)
    def test_npu_quant_flash_attention_with_fp8(self, device="npu"):
        query_fp8 = torch.from_numpy(np.random.randn(1, 57600, 5, 128).astype(hifloat8).astype(np.float32)).to(torch.uint8)
        key_fp8 = torch.from_numpy(np.random.randn(1, 57600, 5, 128).astype(hifloat8).astype(np.float32)).to(torch.uint8)
        value_fp8 = torch.from_numpy(np.random.randn(1, 57600, 5, 128).astype(hifloat8).astype(np.float32)).to(torch.uint8)
        p_scale = torch.ones(1, dtype=torch.float32) * 15
        d_scale_q = torch.ones(1, 5, 450, 1, dtype=torch.float32) * 6e-4
        d_scale_k = torch.ones(1, 5, 225, 1, dtype=torch.float32) * 6e-4
        d_scale_v = torch.ones(1, 5, 113, 1, dtype=torch.float32) * 6e-4
        fa_result = self.custom_op_exec_quant(query_fp8.npu(), key_fp8.npu(), value_fp8.npu(), d_scale_q.npu(), d_scale_k.npu(),
                                              d_scale_v.npu(), p_scale.npu())
if __name__ == "__main__":
    run_tests()
```
