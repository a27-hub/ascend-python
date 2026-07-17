# torch_npu.npu.use_compatible_impl

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term>                        |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

该接口用于控制算子API的实现是否与PyTorch社区完全对齐。该接口仅用于切换算子API所调用的底层算子。
该功能是否开启，可通过`torch_npu.npu.are_compatible_impl_enabled`进行查询。

## 函数原型

```python
torch_npu.npu.use_compatible_impl(is_enable)
```

## 参数说明

**is_enable**(`bool`)：控制是否开启兼容性实现。

 - True: 开启兼容性算子替换。
 - False: 关闭兼容性算子替换。

## 返回值说明

无

## 约束说明

目前支持`torch.nn.functional.gelu`、`torch.nn.LayerNorm`、`torch.distributed.all_to_all`、`torch.distributed.scatter`、`torch.distributed.gather`、`torch.nn.MaxPool2d`、`torch.nn.functional.max_pool2d`、`torch.nn.functional.scaled_dot_product_attention`。

对于`torch.nn.functional.scaled_dot_product_attention`，开启该功能后，支持按SDPA后端选择上下文指定MATH后端。

可通过环境变量`TORCH_NPU_LEGACY_IMPL_LIST`指定部分compatible key使用legacy实现。该环境变量用于API级别覆盖compatible impl策略，优先级高于全局compatible impl开关。若某个key命中该列表，则该key使用legacy实现；未命中的key仍按`TORCH_NPU_USE_COMPATIBLE_IMPL`或`torch_npu.npu.use_compatible_impl()`的配置判断。

当前支持的配置项如下：

| 配置项 | 说明 |
| ------ | ---- |
| `randomness` | 随机数相关API使用legacy随机数实现。 |

`randomness`用于控制随机数相关API的实现，例如`torch.randn`、`torch.nn.init.normal_`等。

配置示例：`export TORCH_NPU_LEGACY_IMPL_LIST=randomness`。多个配置项使用英文逗号分隔。配置未知key会报错，并提示当前支持的key。

建议在启动Python进程前配置该环境变量。进程运行过程中修改该环境变量不保证生效。

## 调用示例

```python
import torch
import torch_npu

torch_npu.npu.use_compatible_impl(True)
shape = [100, 400]
mode = "none"
input = torch.rand(shape, dtype=torch.float16).npu()
output = torch.nn.functional.gelu(input, approximate=mode)
```
