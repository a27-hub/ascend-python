# torch_npu.npu_grouped_matmul_finalize_routing

## 产品支持情况


| 产品                              | 是否支持 |
| ------------------------------- | ---- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |


## 功能说明

- **API功能**
  GroupedMatMul和MoeFinalizeRouting的融合算子，GroupedMatMul计算后的输出按照索引做combine动作。
- **术语说明**
  - **MoE**：Mixture of Experts，混合专家模型。每个token按路由结果分配到一个或多个专家进行计算。
  - **MoeFinalizeRouting**：MoE路由最终化过程。将各专家计算结果按路由索引回填到token原始顺序，并对同一token的多个专家结果进行聚合，得到最终输出。
- **流程说明**
  GroupedMatMul负责分专家计算，MoeFinalizeRouting负责按路由关系回填与聚合，两者串联形成完整的MoE输出路径。详细步骤如下：
  - 按`group_list`完成GroupedMatMul，得到各专家分组的计算结果；
  - 若配置`logit`，对专家结果进行加权；
  - 按`row_index`将专家结果回填到对应token位置，并进行累加；
  - 若配置`shared_input`，再按`shared_input_weight`和`shared_input_offset`与共享专家结果融合。  

## 函数原型

```python
torch_npu.npu_grouped_matmul_finalize_routing(x, w, group_list, *, scale=None, bias=None, offset=None, pertoken_scale=None, shared_input=None, logit=None, row_index=None, dtype=None, shared_input_weight=1.0, shared_input_offset=0, output_bs=0, group_list_type=1) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数。矩阵计算的左矩阵，不支持非连续的Tensor。数据类型支持`int8`，数据格式支持$ND$，维度为m, k。m取值范围为1, 1610248。
- **w** (`Tensor`)：必选参数。矩阵计算的右矩阵，不支持非连续的Tensor。数据类型支持`int8`、`int4`。
  - A8W8量化场景下，数据格式支持$NZ$，维度为e, n1, k1, k0, n0，其中k0=16、n0=32，`x` shape中的k和`w` shape中的k1需要满足以下关系：ceilDivk, 16 = k1，e取值范围1, 256，k取值为16整倍数，n取值为32整倍数，且n大于等于256。
  - A8W4量化场景下数据格式支持$ND$，维度为e, k, n，k支持2048，n只支持7168。
- **grouplist** (`Tensor`)：必选参数。GroupedMatMul的各分组大小。不支持非连续的Tensor。数据类型支持`int64`，数据格式支持$ND$，维度为e,，e与`w`的e一致。`group_list`的值总和要求≤m。
- *****：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **scale** (`Tensor`)：可选参数。矩阵计算反量化参数，对应weight矩阵。A8W8场景下支持perchannel量化方式，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，维度e, n，这里的n=n1n0，A8W4量化场景下，数据类型支持`int64`，维度为e, 1, n。
- **bias** (`Tensor`)：可选参数。矩阵计算的bias参数，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，维度为e, n，只支持A8W4量化场景。
- **offset** (`Tensor`)：可选参数。矩阵计算量化参数的偏移量，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，输入维度只支持3维。只支持A8W4量化场景。
- **pertokenscale** (`Tensor`)：可选参数。矩阵计算的反量化参数，对应`x`矩阵，pertoken量化方式，不支持非连续的Tensor。维度为m,，m与`x`的m一致。数据类型支持`float32`，数据格式支持$ND$。
- **sharedinput** (`Tensor`)：可选参数。MoE计算中共享专家的输出，需要与MoE专家的输出进行combine操作，不支持非连续的Tensor。数据类型支持`bfloat16`，数据格式支持$ND$，维度batch/dp, n，n与`scale`的n一致，batch/dp取值范围1, 21024，batch取值范围1, 161024。
- **logit** (`Tensor`)：可选参数。MoE专家对各个token的logit大小，矩阵乘的计算输出与该logit做乘法，然后索引进行combine，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，维度m,，m与`x`的m一致。
- **rowindex** (`Tensor`)：可选参数。MoE专家输出按照该row_index进行combine，其中的值即为combine做scatter add的索引，不支持非连续的Tensor。数据类型支持`int32`、`int64`，数据格式支持$ND$，维度为m,，m与`x`的m一致。
- **dtype** (`ScalarType`)：可选参数。指定GroupedMatMul计算的输出类型。0表示`float32`，1表示`float16`，2表示`bfloat16`。默认值为0。
- **sharedinputweight** (`float`)：可选参数。指共享专家与MoE专家进行combine的系数，`shared_input`先与该参数乘，然后再和MoE专家结果累加。默认为1.0。
- **sharedinputoffset** (`int`)：可选参数。共享专家与MoE专家累加的行偏移量。默认值为0，表示没有偏移。
- **outputbs** (`int`)：可选参数。输出的最高维大小。默认值为0。
- **grouplisttype** (`List[int]`)：可选参数。GroupedMatMul的分组模式。默认为1，表示count模式；若配置为0，表示cumsum模式，即为前缀和。

## 返回值说明

`Tensor`

返回值，不支持非连续的Tensor，输出的数据类型固定为`float32`，维度为batch, n。

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持图模式。
- 输入和输出Tensor支持的数据类型组合如下：

  | x      | w      | group_list | scale     | bias      | offset    | pertoken_scale | shared_input | logit     | row_index | y         |
  | ------ | ------ | ---------- | --------- | --------- | --------- | -------------- | ------------ | --------- | --------- | --------- |
  | `int8` | `int8` | `int64`    | `float32` | None      | None      | `float32`      | `bfloat16`   | `float32` | `int64`   | `float32` |
  | `int8` | `int8` | `int64`    | `float32` | None      | None      | `float32`      | None         | None      | `int64`   | `float32` |
  | `int8` | `int4` | `int64`    | `int64`   | `float32` | None      | `float32`      | `bfloat16`   | `float32` | `int64`   | `float32` |
  | `int8` | `int4` | `int64`    | `int64`   | `float32` | `float32` | `float32`      | `bfloat16`   | `float32` | `int64`   | `float32` |


## 调用示例

- 单算子模式调用
- 图模式调用：

