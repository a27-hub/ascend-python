# torch_npu.npu_chunk_gated_delta_rule

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- API功能：该接口实现了chunk版本的 Gated Delta Rule（GDR）的计算逻辑，是Transformer线性注意力机制的关键算子之一。

- 计算公式：

  Gated Delta Rule（门控Delta规则，GDR）是一种应用于循环神经网络的算子，也被应用于一种线性注意力机制中。在每个时间步 $t$，GDR根据当前的输入 $q_t$、$k_t$、$v_t$、上一个隐藏状态 $S_{t-1}$、衰减系数 $\alpha_t$ 以及更新强度 $\beta_t$，计算当前的注意力输出 $o_t$ 和新的隐藏状态 $S_t$，其计算公式如下：
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  其中，$S_{t-1},S_t \in R^{D_v \times D_k}$，$q_t, k_t \in R^{D_k}$，$v_t \in R^{D_v}$，$\alpha_t \in R$，$\beta_t \in R$，$o \in R^{D_v}$。
  
  Chunked Gated Delta Rule (CGDR)是GDR的chunk版实现([参考论文](https://arxiv.org/abs/2412.06464))，它通过将输入序列切块，实现了一定的并行效果，在长上下文场景其计算效率相对Recurrent Gated Delta Rule更高，适用于prefill阶段。输入一个长度为L的序列，CGDR算子可以计算出每一步的输出 $o_t, t \in \{1, .., L\}$ 以及最终的状态矩阵 $S_L$。

## 函数原型

```python
torch_npu.npu_chunk_gated_delta_rule(query, key, value, *, beta=None, initial_state=None, actual_seq_lengths=None, scale=None, g=None) -> tuple(Tensor, Tensor)
```

## 参数说明

$T=\sum_i^B L_i$ 表示累积序列长度，$B$ 表示batch size。$L_i$ 表示第i个序列的长度。
$N_k$ 表示key的头数，$N_v$ 表示value的头数。
$D_k$ 表示key的hidden size，$D_v$ 表示value的hidden size。

- **query** (`Tensor`)：必选输入，对应公式中的$q$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_k$, $D_k$）。

- **key** (`Tensor`)：必选输入，对应公式中的$k$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_k$, $D_k$）。

- **value** (`Tensor`)：必选输入，对应公式中的$v$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_v$, $D_v$）。

- **beta** (`Tensor`)：必选输入。对应公式中的$β$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$T$, $N_v$）。

- **initial_state** (`Tensor`)：必选输入，对应公式中的状态矩阵$S_0$，数据类型支持`bfloat16`，数据格式支持ND，shape为（$B$, $N_v$, $D_v$, $D_k$）。

- **actual_seq_lengths** (`Tensor`)：必选输入。表示各batch的输入序列长度。数据类型支持`int32`，数据格式支持ND，shape为（$B$,）。

- **scale** (`float`)：必选输入。表示query的缩放因子，对应公式中的 $scale$。数据类型支持`float32`。默认值None表示为1.0。实际场景一般设为 $1/\sqrt{D_k}$

- **g** (`Tensor`)：必选输入，衰减系数，对应公式中的$α=e^g$。默认为None，表示全0。数据类型支持`float32`，数据格式支持ND，shape为（$T$, $N_v$）。

## 返回值说明

- **out** (`Tensor`)：公式中的$o_t$，注意力计算结果。数据类型为`bfloat16`，数据格式为ND，shape为($T$, $N_v$, $D_v$)。

- **final_state** (`Tensor`)：最终的状态矩阵$S_L$，数据类型为`bfloat16`，数据格式为ND，shape为（$B$, $N_v$, $D_v$, $D_k$）。

## 约束说明

- 该接口仅支持推理场景下使用。
- 输入shape大小需满足约束：
  - $Nv \le 64$，$Nk \le 64$，且Nv是Nk的整数倍。
  - $Dv = Dk = 128$。

## 调用示例

- 单算子调用

    ```python
    import torch
    import torch_npu

    # 构造输入
    B, seqlen, nk, nv, dk, dv = (2, 100, 4, 8, 128, 128)
    actual_seq_lengths = (torch.ones(B) * seqlen).npu().to(torch.int32)
    T = int(torch.sum(actual_seq_lengths))

    state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16).npu()
    query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
    value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
    g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
    beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()

    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    scale = dk ** -0.5

    # 调用算子
    o, final_state = torch_npu.npu_chunk_gated_delta_rule(
        query, key, value, beta=beta, initial_state=state, actual_seq_lengths=actual_seq_lengths, scale=scale, g=g)
    print(o.shape, final_state.shape)
    ```
