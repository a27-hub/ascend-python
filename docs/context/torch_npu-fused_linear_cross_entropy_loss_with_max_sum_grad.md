# torch_npu.npu_quant_matmul

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |


## 功能说明

- API功能：并行场景下交叉熵计算的一部分，解决超大规模词汇表下的显存和计算效率问题，功能等价Megatron的matmul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切卡融合matmul与cross\_entropy\_loss，中间根据通信拆分为torch_npu.fused_linear_online_max_sum、torch_npu.fused_cross_entropy_loss_with_max_sum，需要依次调用实现完整功能。torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad为梯度计算实现，用于计算`input`和`weight`的梯度。以及`logits`相关的全局通信结果作为本接口输入。
- 计算公式：

&emsp;&emsp;高性能模式，softmax非None：

$$
\text{softmax} \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = \mathbf{1} - \text{target\_mask}.view(-1) \in \mathbb{R}^{BT}
$$

$$
\text{softmax}[\text{arange\_1d}, \text{masked\_target}] \leftarrow \text{softmax}[\text{arange\_1d}, \text{masked\_target}] - \text{softmax\_update}
$$

$$
\text{softmax} \leftarrow \text{softmax} \odot \text{grad}.unsqueeze(-1) \in \mathbb{R}^{BT \times V}
$$

$$
\text{input\_grad} = \text{softmax} \cdot \text{weight}^T \in \mathbb{R}^{BT \times H}
$$

$$
\text{weight\_grad} = \text{softmax}^T \cdot \text{input} \in \mathbb{R}^{V \times H}
$$

</br>
&emsp;&emsp;省显存模式，即softmax为None时：

$$
\text{vocab\_parallel\_logits} = \text{input} \cdot \text{weight}^T \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{logits\_sub} = \text{vocab\_parallel\_logits} - \text{logits\_max}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} = \exp(\text{logits\_sub}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} \gets \frac{\text{exp\_logits}}{\text{sum\_exp\_logits}.unsqueeze(-1)} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_logits} = \text{exp\_logits} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_2d} = \text{grad\_logits}.view(-1, \text{partition\_vocab\_size}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \quad \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = 1 - \text{target\_mask}.view(-1) \quad \in \mathbb{R}^{BT}
$$

$$
\text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] \gets \text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] - \text{softmax\_update}
$$

$$
\text{grad\_logits} \gets \text{grad\_logits} \odot \text{grad}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{input\_grad} = \text{grad\_logits} \cdot \text{weight} \quad \in \mathbb{R}^{BT \times H}
$$

$$
\text{weight\_grad} = \text{grad\_logits}^T \cdot \text{input} \quad \in \mathbb{R}^{V \times H}
$$


## 函数原型

```
torch_npu.fused_linear_cross_entropy_loss_with_max_sum_grad(grad, input, weight, target_mask, masked_target, label_smoothing=0.0, logits_max=None, sum_exp_logits=None, softmax=None) -> (input_grad, weight_grad)
```

## 参数说明


  - **grad**(`Tensor`)：表示输入的梯度，公式中的`grad`，Device侧的Tensor。数据类型支持FLOAT32，shape支持1维，支持非连续Tensor，支持空tensor，支持ND。
  - **input**(`Tensor`)：表示矩阵乘的输入矩阵，公式中的`input`，Device侧的Tensor。数据类型支持FLOAT16、BFLOAT16，shape支持2维，其中第1维长度与输入grad的长度相同，支持非连续Tensor，支持空tensor，支持ND。
  - **weight**(`Tensor`)：表示矩阵乘的权重矩阵，公式中的`weight`，Device侧的Tensor。数据类型支持FLOAT16、BFLOAT16，数据类型与输入input相同，shape支持2维，其中第1维长度不支持小于128，第2维长度与输入input的第2维长度相同，支持非连续Tensor，支持空tensor，支持ND。
  - **target_mask**(`Tensor`)：表示中间变量，代表对应词ID是否在目标范围内，公式中的`target_mask`，Device侧的Tensor。数据类型支持UINT8，其中每1bit数据代表1个布尔值，0代表false，1代表true。shape支持1维，长度乘以8须不小于输入grad的长度，支持非连续Tensor，支持空tensor，支持ND。
  - **masked_target**(`Tensor`)：表示中间变量，代表对应词ID映射到当前设备词汇表分片的局部索引，无效目标会被掩码target_mask处理，公式中的`masked_target`，Device侧的Tensor。数据类型支持INT64、INT32，shape支持1维，长度与输入grad相同，支持非连续Tensor，支持空tensor，支持ND。
  - **label_smoothing**(float)：表示标签平滑系数，用于缓解过拟合，目前仅支持取值为0，属性参数。
  - **logits_max**(`Tensor`)：表示中间变量，全局logits的最大值，公式中的`logits_max`，Device侧的Tensor。可选输入，支持输入None，输入None的场景需要提供有效的softmax输入。数据类型支持FLOAT32，shape支持1维，长度与输入grad相同，支持非连续Tensor，支持空tensor，支持ND。
  - **sum_exp_logits**(`Tensor`)：表示中间变量，处理后的logits，公式中的`sum_exp_logits`，Device侧的Tensor。可选输入，支持输入None，输入None的场景需要提供有效的softmax输入。数据类型支持FLOAT32，shape支持1维，长度与输入grad相同，支持非连续Tensor，支持空tensor，支持ND。
  - **softmax**(`Tensor`)：表示中间变量，矩阵乘的结果，公式中的`softmax`，Device侧的Tensor。可选输入，支持输入None，输入None时须提供有效的logits_max、sum_exp_logits输入；输入非None时，logits_max、sum_exp_logits输入无效。数据类型支持FLOAT32，shape支持2维，其中第1维长度与输入grad长度相同，第二维长度与输入weight的第1维长度相同，支持非连续Tensor，支持空tensor，支持ND。



## 返回值说明

  - **input_grad**(`Tensor`)：表示对应叶子节点input的梯度，公式中的`input_grad`，Device侧的Tensor。数据类型支持FLOAT16、BFLOAT16，数据类型与输入input相同，shape支持2维，第1维长度与输入grad长度相同，第2维长度与输入weight的第2维长度相同，格式为ND。
  - **weight_grad**(`Tensor`)：表示对应叶子节点weight的梯度，公式中的`weight_grad`，Device侧的Tensor。数据类型支持FLOAT16、BFLOAT16，数据类型与输入input相同，shape支持2维，第1维长度与输入weight的第1维长度相同，第2维长度与输入grad长度相同，格式为ND。


## 约束说明
