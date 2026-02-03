# torch_npu.npu_add_rms_norm

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_rms_norm)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。AddRmsNorm算子将RmsNorm前的Add算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## 函数原型
```Cpp
torch_npu.npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor)
```

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>表示用于Add计算的第一个输入。对应公式中的`x1`。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示用于Add计算的第二个输入。对应公式中的`x2`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape和数据类型需要与`x1`的shape和数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示RmsNorm的缩放因子（权重）。对应公式中的`gamma`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与`x1`的数据类型保持一致。</li><li>shape需要与`x1`后几维保持一致，后几维为`x1`需要norm的维度。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>表示添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</td>
      <td>建议值为1e-6。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>表示最后的输出。对应公式中的`RmsNorm(x)`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape、数据类型与输入`x1`的shape、数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>表示归一化后的标准差的倒数。对应公式中`Rms(x)`的倒数。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape与`x1`前几维保持一致，前几维表示不需要norm的维度。rstdOut shape与x1 shape，gamma shape关系举例：若x1 shape:(2，3，4，8)，gamma shape:(8)，rstdOut shape(2，3，4，1)；若x1 shape:(2，3，4，8)，gamma shape:(4，8)，rstdOut shape(2，3，1，1)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas 推理系列产品</term>：
    - 参数`x1`、`x2`、`gamma`、`yOut`、`xOut`的数据类型不支持BFLOAT16。
    - 参数`rstdOut`在当前产品使用场景下无效。


## 约束说明

- 边界值场景说明
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。
- 确定性计算：
  - aclnnAddRmsNorm默认确定性实现。

## 调用示例

```python
    >>> import torch
    >>> import torch_npu
    >>> x1 = torch.rand(512, 1024, dtype=torch.float16, device='npu') * 100
    >>> x2 = torch.rand(512, 1024, dtype=torch.float16, device='npu') * 100
    >>> gamma = torch.rand(1024, dtype=torch.float16, device='npu') * 100
    >>> epsilon = 1e-6
    >>> y, rstd, x = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)
    >>> y
    >>> y.shape
    >>> y.dtype
    >>> rstd
    >>> rstd.dtype
    >>> x
```