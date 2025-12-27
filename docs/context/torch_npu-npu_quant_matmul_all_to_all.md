# torch_npu.npu_quant_matmul_all_to_all

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

- API功能：完成量化的Matmul计算、Permute(保证通信后地址连续)和AlltoAll通信的融合，**先计算后通信**。
    支持K-C量化模式
- 计算公式:
  假设x1的shape为(BS, H1), x2的shape为(H1, H2)
    - K-C量化模式：
      $$
      computeOut = (x1 @ x2 + bias) * x1Scale * x2Scale \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$
    - K-C量化模式带bias：
      $$
      computeOut = (x1 @ x2) * x1Scale * x2Scale  + bias \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

## 函数原型

```
torch_npu.npu_quant_matmul_all_to_all(Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, Tensor? common_scale=None, Tensor? x1_offset=None, Tensor? x2_offset=None, int x1_quant_mode=3, int x2_quant_mode=2, int common_quant_mode=0, int[]? group_sizes=None, int[]? all2all_axes=None, int comm_quant_dtype=0, int? x1_dtype=None, int? x2_dtype=None, int? x1_scale_dtype=None, int? x2_scale_dtype=None, int? output_scale_dtype=None, int? comm_scale_dtype=None, int? y_dtype=None) -> Tensor
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
   <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
   <td>ND</td>
   <td>2维, shape为(BS, H1)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x2</td>
   <td>输入</td>
   <td>融合算子的右矩阵输入，对应公式中的x2</td>
   <td>直接作为MatMul计算的右矩阵输入</td>
   <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
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
   <td>输入</td>
   <td>可选输入, 阵乘运算后累加的偏置，对应公式中的bias</td>
   <td>传入非空时生效</td>
   <td>K-C量化模式且x1/x2为FLOAT8_E4M3FN/FLOAT8_E5M2时，该参数类型为FLOAT32</td>
   <td>ND</td>
   <td>1维，shape为(H2,)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x1_scale</td>
   <td>输入</td>
   <td>左矩阵的量化系数</td>
   <td>对应公式中的x1Scale</td>
   <td>K-C量化模式且x1为FLOAT8_E4M3FN/FLOAT8_E5M2时，该参数类型为FLOAT32</td>
   <td>ND</td>
   <td>K-C量化模式下是1维, shape为(BS,)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>x2_scale</td>
   <td>输入</td>
   <td>右矩阵的量化系数</td>
   <td>对应公式中的x2Scale</td>
   <td>K-C量化模式且x2为FLOAT8_E4M3FN/FLOAT8_E5M2时，该参数类型为FLOAT32</td>
   <td>ND</td>
   <td>K-C量化模式下是1维, shape为(H2,)</td>
   <td>x</td>
  </tr>
  <tr>
   <td>comm_scale</td>
   <td>输入</td>
   <td>可选输入, 低比特通信的量化系数</td>
   <td>预留参数，暂不支持低比特通信</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x1_offset</td>
   <td>输入</td>
   <td>可选输入，左矩阵的量化偏置</td>
   <td>预留参数，暂不支持</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x2_offset</td>
   <td>输入</td>
   <td>可选输入，右矩阵的量化偏置</td>
   <td>预留参数，暂不支持</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x1_quant_mode</td>
   <td>输入</td>
   <td>左矩阵的量化方式</td>
   <td>当前仅支持配置为3，表示PerToken</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x2_quant_mode</td>
   <td>输入</td>
   <td>左矩阵的量化方式</td>
   <td>当前仅支持配置为2，表示PerChannel</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>comm_quant_mode</td>
   <td>输入</td>
   <td>低比特通信的量化方式</td>
   <td>预留参数，当前仅支持配置为0，表示不量化</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>group_sizes</td>
   <td>输入</td>
   <td>用于Matmul计算三个方向上的量化分组大小</td>
   <td>预留参数，K-C量化模式下仅支持配置为0，取值不生效。groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
   <td>List[int]</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
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
  <tr>
   <td>comm_quant_dtype</td>
   <td>输入</td>
   <td>低比特通信的量化类型</td>
   <td>预留参数，暂不支持</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x1_dtype</td>
   <td>输入</td>
   <td>x1的数据类型</td>
   <td>用于在pytorch侧标识海思特有的数据类型</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x2_dtype</td>
   <td>输入</td>
   <td>x2的数据类型</td>
   <td>用于在pytorch侧标识海思特有的数据类型</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x1_scale_dtype</td>
   <td>输入</td>
   <td>x1_scale的数据类型</td>
   <td>用于在pytorch侧标识海思特有的数据类型</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>x2_scale_dtype</td>
   <td>输入</td>
   <td>x2_scale的数据类型</td>
   <td>用于在pytorch侧标识海思特有的数据类型</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>output_scale_dtype</td>
   <td>输入</td>
   <td>对输出tensor进行量化scale的数据类型，暂不支持</td>
   <td>用于在pytorch侧标识海思特有的数据类型</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>comm_scale_dtype</td>
   <td>输入</td>
   <td>低比特通信的量化scale数据类型，暂不支持</td>
   <td>用于在pytorch侧标识海思特有的数据类型</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
  <tr>
   <td>y_dtype</td>
   <td>输入</td>
   <td>输出output的数据类型</td>
   <td>用于在pytorch侧标识海思特有的数据类型</td>
   <td>INT</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
  </tr>
 </tbody></table>

x1_quant_mode、x2_quant_mode、comm_quant_mode的枚举值跟量化模式关系如下:
* 0: 不量化
* 1: pertensor
* 2: perchanenl
* 3: pertoken
* 4: pergroup
* 5: perblock
* 6: mx量化
* 7: pertoken动态量化

## 返回值说明
`Tensor`

代表`npu_quant_matmul_all_to_all`的计算结果，对应公式中的output，假设x1的shape为(BS,H1)，x2的shape为(H1,H2)，则output的shape为(BS*rankSize,H2/rankSize)。

## 约束说明
* 默认支持确定性计算
* 右矩阵和输出矩阵的H2必须整除rankSize
* 仅支持左矩阵perToken量化，x1QuantMode=3，右矩阵perChannel量化,x2QuantMode=2
* <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：传入的x1、x2、biasOptional、x1Scale、x2Scale或者output不为空指针
* <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：x1、x2计算输入的数据类型必须为INT8，output计算输出的数据类型为BFLOAT16时，biasOptional的数据类型为FLOAT或BFLOAT16，output的数据类型为FLOAT16时，biasOptional的数据类型为FLOAT16
* H1范围仅支持[1, 65535]
* rankSize仅支持2,4,8,16
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：支持2、4、8卡
    - <term>昇腾910_95 AI处理器</term>：支持2,4,8,16卡
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。

## 调用示例
- 单算子模式调用
  ```python
    import torch
    import torch_npu
    
    # 初始化输入
    x1 = torch.randint(-1, 2, (16, 32), dtype=torch.float8_e4m3fn).npu()
    x2 = torch.randint(-1, 2, (32, 32), dtype=torch.float8_e4m3fn).npu()
    bias = torch.randint(-1, 2, (32,), dtype=torch.float32).npu()
    x1Scale = torch.randint(-1, 2, (32,), dtype=torch.float32).npu()
    x2Scale = torch.randint(-1, 2, (32,), dtype=torch.float32).npu()
    # 其他参数
    hcom = "fake group info"
    world_size = 2
    common_scale = None
    x1_offset = None
    x2_offset = None
    x1_quant_mode = 3
    x2_quant_mode = 2
    common_quant_mode = 0
    group_size = [0]
    all2all_axes=[-1,-2]
    comm_quant_dtype = 0
    x1_dtype = None
    x2_dtype = None
    x1_scale_dtype = None
    x2_scale_dtype = None
    output_scale_dtype = None
    comm_scale_dtype = None
    y_dtype = None
    # 调用QuantMatmulAlltoAll算子
    res = torch_npu.npu_quant_matmul_all_to_all(x1, x2, hcom, world_size, bias, x1Scale, x2Scale, common_scale, x1_offset, x2_offset,
        x1_quant_mode, x2_quant_mode, common_quant_mode, group_size, all2all_axes, comm_quant_dtype, x1_dtype, x2_dtype, x1_scale_dtype,
        x2_scale_dtype, output_scale_dtype, comm_scale_dtype, y_dtype)
    # 输出res的形状类似于
    tensor([[-1.3330,  0.3899,  0.3205,  ...,  0.3482, -0.4815,  0.4637],
            [ 0.3511,  1.7768, -0.2588,  ..., -0.3826,  0.7349, -0.0110],
            [ 1.2357, -1.0084, -1.4964,  ..., -0.4084, -0.9424, -0.5332],
            ...,
            [ 1.2147, -1.7824,  1.6442,  ...,  0.6114, -1.6988,  1.1249],
            [ 0.5168, -0.7395, -0.3615,  ...,  1.6038,  0.0826,  0.8413],
            [ 1.0905, -0.1828,  0.7823,  ...,  0.2952, -0.7067,  0.9949]])
  ```

