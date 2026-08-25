# （beta）torch_npu.npu.config.allow_internal_format

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

控制是否使用私有格式，设置为True时允许使用私有格式，设置为False时，不允许申请任何私有格式的tensor，避免了适配层出现私有格式流通。

## 函数原型

```python
torch_npu.npu.config.allow_internal_format = bool
```

## 参数说明

输入`bool`值。

- <term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>默认值为`False`。
- <term>Atlas 推理系列产品</term>/<term>Atlas 训练系列产品</term>默认值为`True`。

## 约束说明

设置后，仅后续创建的Tensor会遵循该配置，已创建的Tensor不会受到影响。

该开关直接影响以下API的行为：

- `torch_npu.npu_format_cast`/`torch_npu.npu_format_cast_`：开关为`False`时不支持转换为私有格式，目标私有格式（FRACTAL_NZ、FRACTAL_Z、NC1HWC0等）会自动降级为对应的base格式（FRACTAL_NZ降级为ND，FRACTAL_Z/NC1HWC0降级为NCHW）并打印告警，数据与shape保持不变；开关为`True`时真实转换为私有格式。
- 卷积类算子（如`torch.nn.Conv1d`、`torch.nn.Conv2d`、`torch.nn.ConvTranspose2d`等，`Conv3d`除外）：开关为`False`且JIT编译关闭时走aclnn路径，输入保持base格式（NCHW/ND）；开关为`True`或开启JIT编译时走aclop路径，输入会被转为私有格式（如NC1HWC0）参与计算，可能触发在线编译。

> [!NOTE]
> 该开关仅控制"是否允许产生私有格式的tensor"。base格式（ND/NCHW/NHWC等）之间的转换与计算不受该开关影响；向**已存在**的私有格式tensor做`copy_`等写入操作时，tensor自身的私有格式保持不变。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.config.allow_internal_format = False
```
