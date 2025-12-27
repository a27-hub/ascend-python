# torch_npu.npu_matmul_all_to_all

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |
## 功能说明

- API功能：完成Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**。
- 计算公式:
  假设x1的shape为(BS, H1), x2的shape为(H1, H2)
  $$
  computeOut = x1 @ x2 + bias \\
  permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
  output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
  $$

## 函数原型

```
torch_npu.npu_matmul_all_to_all(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, int[]? all2all_axes=None) -> Tensor
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
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
   <th>非连续tensor</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>x1</td>
   <td>输入</td>
   <td>融合算子的左矩阵输入，对应公式中的x1</td>
   <td>该输入作为MatMul计算的左矩阵输入</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维, shape为(BS, H1)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x2</td>
   <td>输入</td>
   <td>融合算子的右矩阵输入，也是MatMul计算的右矩阵</td>
   <td>直接作为MatMul计算的右矩阵输入</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
   <td>2维，shape为(H1, H2)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>hcom</td>
   <td>输入</td>
   <td>通信域名</td>
   <td>字符串长度要求(0, 128)</td>
   <td>STRING</td>
   <td>-</td>
   <td>1维</td>
   <td>x</td>
  </tr>
  <tr>
   <td>world_size</td>
   <td>输入</td>
   <td>通信域大小</td>
   <td>对应公式中的rankSize，支持2、4、8、16</td>
   <td>INT</td>
   <td>-</td>
   <td></td>
   <td></td>
  </tr>
  <tr>
   <td>bias</td>
   <td>可选输入</td>
   <td>阵乘运算后累加的偏置，对应公式中的bias。</td>
   <td></td>
   <td>FLOAT16、BFLOAT16、FLOAT32</td>
   <td>ND</td>
   <td>1维，shape为(H2)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>all2all_axes</td>
   <td>输入</td>
   <td>可选输入，AlltoAll和Pemute数据交换的方向</td>
   <td>支持配置空或者[-1,-2]，传入空时默认按[-1,-2]处理，表示将输入由(BS, H2)转为(BS * rankSize, H2 / rankSize)</td>
   <td>List[int]</td>
   <td>ND</td>
   <td>1维，shape为(2)</td>
   <td>x</td>
  </tr>
 </tbody></table>

## 返回值说明
`Tensor`

代表`npu_matmul_all_to_all`的计算结果，对应公式中的output，假设x1的shape为(BS,H1)，x2的shape为(H1,H2)，则output的shape为(BS*rankSize,H2/rankSize)。输出output的数据类型和x1一致。

## 约束说明
* 默认支持确定性计算
* 右矩阵和输出矩阵的H2必须整除rankSize
* 仅支持BS为0的空tensor
* H1范围仅支持[1, 65535]
* ranSize仅支持2,4,8,16
* x1、x2的数据类型必须一致
* bias的数据类型可以为x1 x2的数据类型，也可以为float32
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用示例
- 单算子模式调用
  ```python
  import torch
  import torch_npu
  
  # 初始化输入
  x1 = torch.randint(-1, 2, (16, 32), dtype=torch.float16).npu()
  x2 = torch.randint(-1, 2, (32, 32), dtype=torch.float16).npu()
  bias = torch.randint(-1, 2, (32,), dtype=torch.float16).npu()
  # 其他参数
  hcom = "fake group info"
  world_size = 2
  # 调用MatmulAlltoAll算子
  res = torch_npu.npu_matmul_all_to_all(x1, x2, hcom, world_size, bias, [-1,-2])
  # 输出res的形状类似于
  tensor([[ 0.7483, -0.2262, -0.5252,  ..., -0.6810,  1.5576, -0.3997],
          [ 0.3010,  0.9226,  0.7723,  ..., -1.0785,  0.8963, -0.4670],
          [-1.6897, -0.6455,  0.0328,  ..., -0.6632, -0.0950, -0.0679],
          ...,
          [ 0.2493, -0.2285,  1.1562,  ..., -0.2299,  0.1272,  0.4843],
          [-0.2058, -0.5962, -0.2101,  ...,  1.2606,  0.8451,  1.1243],
          [-0.8396,  1.3138, -0.4066,  ..., -0.7683,  1.3133,  0.7471]])
  ```

