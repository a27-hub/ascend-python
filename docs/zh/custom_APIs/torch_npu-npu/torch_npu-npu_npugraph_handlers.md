# torch_npu.npu.NpuGraphOpHandler

## 产品支持情况

| 产品 | 是否支持 |
|------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 支持 |
| Atlas 推理系列加速卡产品 | 支持 |

## 功能说明

本接口用于注册自定义算子处理器，使自定义算子支持 NPU Graph 的动态 Shape 更新与重放功能。

在 NPU Graph 模式下，用户调用 `g.update()` 传入新的参数数据时，框架需要知道如何将这些数据映射到算子的具体输入位置。通过继承 `NpuGraphOpHandler` 并注册，用户可定义算子在 Capture 阶段的预处理逻辑，以及在 Update 阶段的参数更新逻辑。

主要能力包括：

1. 动态参数更新：通过覆写 `update_args` 方法，将 `g.update()` 传入的数据映射到算子位置参数。
2. 输出预分配：在 Capture 阶段预分配输出 Tensor，支持动态 Shape。
3. 框架解耦：无需修改 torch_npu 源码即可扩展新算子的 Graph 支持。

## 接口原型

### 1. NpuGraphOpHandler 基类

```python
class NpuGraphOpHandler:
    @classmethod
    def prepare_capture(cls, func, args, kwargs): ...
    @classmethod
    def postprocess_result(cls, result, kwargs): ...
    @classmethod
    def update_args(cls, dispatch_record, update_input): ...
    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names): ...
```

### 2. register_npu_graph_handler 装饰器

```python
def register_npu_graph_handler(op_names: str | list[str]): ...
```

## 参数与返回值说明

### NpuGraphOpHandler 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| prepare_capture | func, args, kwargs | func, args, kwargs | Capture 阶段调用。用于 output 预分配或算子变体替换。默认透传。 |
| postprocess_result | result, kwargs | Any | Capture 阶段调用。用于调整返回值格式。默认返回 result。 |
| update_args | dispatch_record, update_input | None | Update 阶段调用。用于修改位置参数以响应动态更新。默认空实现。 |
| record_wrap_kwarg | key, value, tensor_param_names | Any | Capture 阶段调用。用于自定义 kwargs 存储方式。默认处理 Tensor 弱引用。 |

注意：所有方法必须声明为 classmethod，禁止使用 self 存储状态。

### register_npu_graph_handler 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| op_names | str 或 list[str] | 算子名称，对应 func.__name__。建议同时注册 op_name 和 op_name.default。 |

返回值：被装饰的原始类。

## 约束说明

1. 无状态要求：Handler 方法必须为 classmethod，禁止依赖实例状态。
2. 更新数据长度：g.update 传入的 cpu_update_input 列表长度必须与 Graph 内录制的算子数量一致。
3. 变体一致性：若 prepare_capture 将 default 变体替换为 out 变体，两者需注册到同一 Handler 类。
4. 重复注册：重复注册同一算子名称会触发警告，不会覆盖现有注册。
5. 未注册算子：未注册 Handler 的算子支持 Graph 录制，但不支持动态参数更新。

## 调用示例

### 示例 1：动态位置参数更新

```python
import torch
import torch_npu
from torch_npu.npu.npugraph_handlers import NpuGraphOpHandler, register_npu_graph_handler

@register_npu_graph_handler(["my_custom_op.default"])
class MyCustomOpHandler(NpuGraphOpHandler):

    @classmethod
    def update_args(cls, dispatch_record, update_input):
        if "seq_len" in update_input and len(dispatch_record.args) >= 3:
            args_list = list(dispatch_record.args)
            args_list[2] = update_input["seq_len"]
            dispatch_record.args = tuple(args_list)

g = torch.npu.NPUGraph()
with torch.npu.graph(g, auto_dispatch_capture=True):
    output = torch.ops.my_namespace.my_custom_op(x, weight, seq_len)

g.update(cpu_update_input=[{"seq_len": new_seq_len}])
g.replay()
```

### 示例 2：输出预分配与 TensorList 返回

```python
@register_npu_graph_handler(["my_tensorlist_op.default"])
class TensorListOpHandler(NpuGraphOpHandler):

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        if "out" not in kwargs:
            kwargs["out"] = [torch.empty_like(args[0]) for _ in range(2)]
        return torch.ops.my_namespace.my_tensorlist_op.out, args, kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        return kwargs["out"]

g = torch.npu.NPUGraph()
with torch.npu.graph(g, auto_dispatch_capture=True):
    out_list = torch.ops.my_namespace.my_tensorlist_op(input)

g.update(cpu_update_input=[{"dynamic_dim": new_value}])
g.replay()
```

### 示例 3：结合 torch.compile 使用

```python
@register_npu_graph_handler(["custom_relu_ext.default"])
class CustomReluHandler(NpuGraphOpHandler):
    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        kwargs["_custom_flag"] = True
        return func, args, kwargs

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.ops.my_ops.custom_relu_ext(x, alpha=0.1)

model = Model().npu()
compiled_model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
output = compiled_model(x)
```

## 常见问题

Q1: 为什么钩子方法必须是 classmethod？
框架通过注册表调用类对象而非实例。使用 classmethod 可避免实例状态污染，保证线程安全。

Q2: 未覆写 update_args 时，动态更新还能生效吗？
能。框架会执行通用的 kwargs 更新。仅当算子有需要特殊映射的位置参数时，才需覆写 update_args。

Q3: 同一个算子的 default 和 out 变体需要注册两次吗？
建议注册到同一个 Handler 类，在 prepare_capture 内根据 func 名称分支处理，确保 update_args 语义一致。

Q4: 内置支持哪些算子的动态更新？
已内置支持 npu_fused_infer_attention_score 系列、_npu_paged_attention 等算子，无需手动注册。

