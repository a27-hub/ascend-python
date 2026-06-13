# torch_npu.npu.NpuGraphOpHandler

> [!NOTICE]  
> 此接口为本版本新增功能，具体依赖要求请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √   |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √   |
| Atlas 推理系列加速卡产品 | √   |

## 功能说明

本接口用于注册自定义算子处理器，使自定义算子支持NPU Graph的动态参数更新与重放功能。在NPU Graph模式下，用户调用`g.update()`传入新的参数时，Ascend Extension for PyTorch框架通过注册的处理器将数据映射到算子输入位置。

核心机制：

1. Capture预处理：定义算子捕获时的输入数据预处理逻辑，例如workspace申请、输出预分配、从`.default`重定向到`.out`等。
2. Update动态更新：在Graph Replay（回放）阶段，无需重新Capture图结构，即可动态修改算子输入参数（如序列长度、Batch Size等）。具体流程如下：
    1. 用户在Replay前调用`g.update(cpu_update_input=[...])`传入新参数。
    2. 框架遍历Graph中的算子记录，查找注册的`NpuGraphOpHandler`。
    3. 框架根据Handler声明的`UPDATE_SPECS`，将`update_input`中的字段写回算子的`args`或`kwargs`。
    4. 框架自动调用底层图任务更新接口并重放对应算子。

推荐通过`UPDATE_SPECS`声明更新映射关系。仅在`UPDATE_SPECS`无法表达复杂更新逻辑时，才需要重载`update_args`。

## 定义文件

torch_npu/npu/_npugraph_handlers/npugraph_handler.py

## 函数原型

### NpuGraphOpHandler基类

```python
class NpuGraphOpHandler:
    UPDATE_SPECS = {}
    @classmethod
    def prepare_capture(cls, func, args, kwargs): ...
    @classmethod
    def postprocess_result(cls, result, kwargs): ...
    @classmethod
    def get_update_specs(cls, op_name): ...
    @classmethod
    def update_args(cls, dispatch_record, update_input): ...
    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names): ...
```

### register_npu_graph_handler装饰器

```python
def register_npu_graph_handler(op_names: str | list[str]): ...
```

## 参数说明

### NpuGraphOpHandler基类

#### 1. prepare_capture

- **func**（`Callable`）：原始算子函数对象。
- **args**（`Tuple`）：位置参数元组。
- **kwargs**（`Dict`）：关键字参数字典。

#### 2. postprocess_result

- **result**（`Any`）：算子执行的原始结果。
- **kwargs**（`Dict`）：关键字参数字典。

#### 3. update_args

- **dispatch_record**（`DispatchRecord`）：包含算子运行时信息的记录对象，可通过`dispatch_record.args`或`dispatch_record.kwargs`修改参数。
- **update_input**（`Dict`）：用户传入的更新参数字典。

#### 4. get_update_specs

- **op_name**（`str`）：算子名称，对应`func.__name__`。

#### 5. record_wrap_kwarg

- **key**（`str`）：关键字参数的键名。
- **value**（`Any`）：关键字参数的值。
- **tensor_param_names**（`List[str]`）：张量参数名称列表。

> [!CAUTION]<br>
> 所有方法必须声明为 `classmethod`，禁止使用 `self` 存储状态（无状态设计）。

### register_npu_graph_handler装饰器

**op_names**（`str` 或 `list[str]`）：算子名称，对应 `func.__name__`。建议同时注册 `op_name` 和 `op_name.default`。

### UPDATE_SPECS

`UPDATE_SPECS`用于声明算子更新字段与原始参数位置的映射关系。格式如下：

```python
UPDATE_SPECS = {
    "op_name.default": [
        ("arg", 2, "seq_len"),
        ("kwarg", "actual_seq_lengths", "actual_seq_lengths"),
    ],
}
```

每个三元组的含义如下：

- 第1项：参数位置类型，支持`"arg"`和`"kwarg"`。
- 第2项：参数位置。`"arg"`对应位置参数下标，`"kwarg"`对应关键字参数名。
- 第3项：`g.update(cpu_update_input=[...])`中传入的更新字段名。

## 返回值说明

### NpuGraphOpHandler基类

#### 1. prepare_capture

返回处理后的 `func, args, kwargs`。用于output预分配或算子变体替换。默认透传。源码中常用于切换`.default`到`.out`并预分配workspace。

#### 2. postprocess_result

返回处理后的结果。用于调整返回值格式。默认返回result。源码中常用于将C++ TensorList转为 Python list。

#### 3. update_args

无返回值。用于响应动态更新。默认实现会读取`get_update_specs`返回的映射关系，并将`update_input`中的值写入`dispatch_record.args`或`dispatch_record.kwargs`。通常无需重载。

#### 4. get_update_specs

返回当前算子的更新映射关系列表。默认从`UPDATE_SPECS`中按`op_name`查询。

#### 5. record_wrap_kwarg

返回处理后的值。用于自定义kwargs存储方式。默认行为以当前版本实现为准。通常使用`UPDATE_SPECS`声明更新映射时无需重载；仅在算子需要自定义kwargs保存方式时重载。

### register_npu_graph_handler装饰器

返回被装饰的原始类。 

## 调用示例

本示例展示如何自定义一个Handler，同时实现输出预分配（`prepare_capture`）、返回值格式调整（`postprocess_result`）以及通过`UPDATE_SPECS`声明动态参数更新映射关系。

```python
import torch
import torch_npu
from torch_npu.npu import (
    NpuGraphOpHandler,
    register_npu_graph_handler,
)

@register_npu_graph_handler(["my_custom_op.default", "my_custom_op.out"])
class ComprehensiveOpHandler(NpuGraphOpHandler):
    # 定义参数映射：第3个位置参数(index 2)对应update_input中的"seq_len"。
    UPDATE_SPECS = {
        "my_custom_op.default": [("arg", 2, "seq_len")],
        "my_custom_op.out": [("arg", 2, "seq_len")],
    }

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        # 1. 切换到 .out 算子并预分配 output
        func_out = torch.ops.my_namespace.my_custom_op.out
        if "out" not in kwargs:
            # 预分配逻辑，例如根据输入形状分配
            kwargs["out"] = [torch.empty_like(args[0]) for _ in range(2)]
        return func_out, args, kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        # 2. 调整返回值格式，将底层 TensorList 转为 Python list
        return kwargs["out"]

# 使用流程
g = torch.npu.NPUGraph()
with torch.npu.graph(g, auto_dispatch_capture=True):
    # Capture 阶段：调用算子，触发 prepare_capture，并记录待更新任务。
    out_list = torch.ops.my_namespace.my_custom_op(x, weight, seq_len)

# Replay 前：传入新参数。框架会根据 UPDATE_SPECS 更新算子参数。
g.update(cpu_update_input=[{"seq_len": new_seq_len}])
g.replay()
```

如果一次图捕获中记录了多个支持更新的算子，`cpu_update_input`需要传入与记录算子数量相同的列表，列表顺序与捕获时算子实际执行顺序一致。若列表中只有一个字典，框架会将该字典复制给所有记录算子使用。
