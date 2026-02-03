# （beta）torch_npu.npu_indexing

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>                             |    √     |
|<term>Atlas A2 训练系列产品</term>                             |    √     |
|<term>Atlas 推理系列产品</term>                                |    √     |
|<term>Atlas 训练系列产品</term>                                |    √     |
|<term>Ascend 950PR/Ascend 950DT</term>                        |    √     |

## 功能说明

- 接口功能：按照指定的起始、结束位置和步长，从输入张量中提取一个子张量。
- 计算公式：在指定维度$dim$上，按照指定的起始位置$begin$、结束位置$end$和步长$strides$，从输入张量$self$中提取子张量$out$。
  $begin$和$end$可以取$[0, self.shape[dim]]$以外的值，取值后根据以下公式转换为合法值，假设self.shape[dim] = N：

  $$
  begin = \begin{cases}
  &0, & if\;begin < -N \\
  &N, & if\;begin >= N\\
  &(begin+N) \% N, & otherwise
  \end{cases}
  $$

  $$
  end = \begin{cases}
  &N, & if\; end >= N\\
  &begin, & if\;  (end+N)\%N < begin \\
  &(end+N)\%N, & otherwise\\
  \end{cases}
  $$

  $out.shape$与$self.shape$只有dim轴上不一致，其他轴一致:

  $$
  out.shape[dim] = ⌊\frac{end - begin + strides - 1}{strides}⌋
  $$

  若存在mask参数，则按照以下规则对outshape进行进一步计算，
  $beginMask$指定$bit$位为1对应的索引维度的$begin$被忽略，
  $endMask$指定$bit$位为1对应的索引维度的$end$被忽略，
  $ellipsisMask$从$bit$位为1对应的索引维度开始全选后续维度，直到遇到指定$begin$才退出，
  $newAxisMask$指定$bit$位为1对应的索引维度增加维度为1的$shape$，
  $shrinkAxisMask$指定$bit$位为1对应的索引维度强制降为1 。

## 函数原型

```
torch_npu.npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor
```
## 参数说明

- **self** (`Tensor`)：必选参数，输入张量。数据格式支持ND，数据类型支持int8、uint8、int16、uint16、int32、uint32、int64、uint64、float、float16、bfloat16、bool、complex32、complex64、hifloat8、float8_e5m2、float8_e4m3fn。
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品</term>、<term>Atlas A3 训练系列产品</term>：不支持hifloat8、float8_e5m2、float8_e4m3fn数据类型。
- **begin** (`List[int]`)：必选参数，每个维度的起始索引（包含）。
- **end** (`List[int]`)：必选参数，每个维度的结束索引（不包含）。
- **strides** (`List[int]`)：必选参数，每个维度的步长，表示索引的增量。可以为负数，实现反向切片。步长值不能为零。
- **begin_mask** (`int`)：可选参数，位掩码。其中第 `i` 位为 1 表示忽略 `begin[i]` 的值，此时该维度将从有效范围的起始位置开始取值（即索引 0）。默认值为 0。
- **end_mask** (`int`)：可选参数，位掩码，类似于`begin_mask`。第 `i` 位为 1 表示忽略 `end[i]` 的值，此时该维度将一直取值到有效范围的末尾（即最后一个元素）。默认值为 0。
- **ellipsis_mask** (`int`)：可选参数，位掩码。其中第 `i` 位为 1 表示在该位置插入省略号（`...`），从而在对应位置引入所需数量的未指定维度（相当于“填充”所有中间维度）。整个掩码中最多只能有一个非零位。默认值为 0。
- **new_axis_mask** (`int`)：可选参数，位掩码。其中第 `i` 位为 1 表示在该位置新增一个长度为 1 的维度，此时对应的 `begin[i]`、`end[i]` 和 `strides[i]` 值将被忽略。默认值为 0。
- **shrink_axis_mask** (`int`)：可选参数，位掩码。其中第 `i` 位为 1 表示将对应维度缩减（移除），仅保留索引为 `begin[i]` 的单个元素，此时 `end[i]` 和 `strides[i]` 被忽略。默认值为 0。

## 返回值说明

- **out** (`Tensor`)：输出张量，公式中的out，数据类型与计算输入self的类型一致，不支持非连续的Tensor。数据格式支持ND。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口仅支持单算子模式调用。
- begin、end和strides的长度必须一致
- shrink_axis_mask中bit位为1的索引，对应的strides必须大于0.
- 超出begin长度的mask为会被自动忽略。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8]], dtype=torch.int32).to("npu")
>>> input
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]], device='npu:0', dtype=torch.int32)
>>> output = torch_npu.npu_indexing(input, [0, 0], [2, 2], [1, 1])
>>> output
tensor([[1, 2],
        [5, 6]], device='npu:0', dtype=torch.int32)
```
