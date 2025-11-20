# torch_npu.npu_recurrent_gated_delta_rule

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |    ×     |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |


## 功能说明

- 算子功能：完成变步长的Recurrent Gated Delta Rule计算。

- 计算公式：

  Recurrent Gated Delta Rule（循环门控Delta规则，RGDR）是一种应用于循环神经网络的算子，也被应用于一种线性注意力机制中。
  在每个时间步 $t$，网络根据当前的输入 $q_t$、$k_t$、$v_t$ 和上一个隐藏状态 $S_{t-1}$，计算当前的输出 $o_t$ 和新的隐藏状态 $S_t$。
  在这个过程中，门控单元会决定有多少新信息存入隐藏状态，以及有多少旧信息需要被遗忘。
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := \frac{S_t q_t}{\sqrt{d_k}}
  $$

  其中，$S_{t-1},S_t \in R^{d_v \times d_k}$，$q_t, k_t \in R^{d_k}$，$v_t \in R^{d_v}$，$\alpha_t \in R$，$\beta_t \in R$，$o \in R^{d_v}$。


## 函数原型

```
torch_npu.npu_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    *,
    beta=None,
    scale=None,
    actual_seq_lengths=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    g=None,
    gk=None) -> Tensor
```


## 参数说明

令 $B$ 表示batch size，$L_i$ 表示第i个序列的长度，$T=\sum_i^B L_i$ 表示累积序列长度。$N_k$ 表示key的头数，$N_v$ 表示value的头数，$D_k$ 表示key向量的维度，$D_v$ 表示value向量的维度。
- **query** (`Tensor`)：必选输入，对应公式中的q，数据类型支持`bfloat16`，数据格式支持ND，shape为（T, Nk, Dk）。

- **key** (`Tensor`)：必选输入，对应公式中的k，数据类型支持`bfloat16`，数据格式支持ND，shape为（T, Nk, Dk）。

- **value** (`Tensor`)：必选输入，对应公式中的v，数据类型支持`bfloat16`，数据格式支持ND，shape为（T, Nv, Dv）。

- **state** (`Tensor`)：必选输入&输出，对应公式中的状态矩阵S，数据类型支持`bfloat16`，数据格式支持ND，shape为（BlockNum, Nv, Dv, Dk）。

- **beta** (`Tensor`)：必选关键字输入，对应公式中的β，数据类型支持`bfloat16`，数据格式支持ND，shape为（T, Nv）。

- **scale** (`Scalar`)：必选关键字输入，query的缩放因子，对应公式中的 $1/\sqrt{d_k}$。数据类型支持`float32`。

- **actual_seq_lengths** (`Tensor`)：必选关键字输入，各batch的输入序列长度。数据类型支持`int32`，数据格式支持ND，shape为（B,）。

- **ssm_state_indices** (`Tensor`)：必选关键字输入，输入序列到状态矩阵的映射索引。`state[ssm_state_indices[i]]`表示第i个token的状态矩阵。数据类型支持`int32`，数据格式支持ND，shape为（T,）。

- **num_accepted_tokens** (`Tensor`)：可选关键字输入，投机推理每个batch接受的token数量。默认为None，表示每个batch接受的token数为1。数据类型支持`int32`，数据格式支持ND，shape为（B,）。

- **g** (`Tensor`)：可选关键字输入，衰减系数，对应公式中的α=e^g。默认为None，表示全0。数据类型支持`float32`，数据格式支持ND，shape为（T, Nv）。

- **gk** (`Tensor`)：可选关键字输入，衰减系数，当前版本暂不支持，传None即可。


## 返回值说明

`Tensor`

公式中的$o$，注意力计算结果。输出的数据类型为`bfloat16`，数据格式为ND，形状为(T, Nv, Dv)。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持静态图模式。
- 输入shape大小需满足约束：$L_i \le 8$，$N_k \le 256$，$N_v \le 256$，$D_k \le 256$，$D_v \le 256$。


## 调用示例

- 单算子调用
    ```python
    import torch
    import torch_npu

    # 构造输入
    bs, mtp, nv, dv, dk, nk = (2, 3, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(bs) * mtp).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((T, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    ssm_state_indices = (torch.arange(T).npu()).to(torch.int32)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5
    num_accepted_tokens = (torch.randint(1,, mtp + 1, (bs,)).npu()).to(torch.int32)

    # 调用算子
    o = torch_npu.npu_recurrent_gated_delta_rule(
        query, key, value, state, bata=beta, scale=scale,
        actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens, g=g, gk=None)
    print(o)
    ```

- 图模式调用
    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger

    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np

    # "ENABLE_ACLNN"是否使能走aclnn, true: 回调走aclnn, false: 在线编译
    os.environ["ENABLE_ACLNN"] = "false"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
 
        def forward(self, query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g):
            return torch_npu.npu_recurrent_gated_delta_rule(
                query, key, value, state, bata=beta, scale=scale,
                actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens, g=g, gk=None)


    # 构造输入
    bs, mtp, nv, dv, dk, nk = (2, 3, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(bs) * mtp).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((T, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    ssm_state_indices = (torch.arange(T).npu()).to(torch.int32)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5
    num_accepted_tokens = (torch.randint(1,, mtp + 1, (bs,)).npu()).to(torch.int32)

    #调用
    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)
    o = model(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, num_accepted_tokens, g)
    print(o)
    ```
