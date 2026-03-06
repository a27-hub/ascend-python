# torch_npu.npu.NpuGraphOpHandler

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √   |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √   |
| Atlas 推理系列加速卡产品 | √   |

## 功能说明

本接口用于注册自定义算子处理器，使自定义算子支持 NPU Graph 的动态 Shape 更新与重放功能。

在 NPU Graph 模式下，用户调用 `g.update()` 传入新的参数数据时，框架需要知道如何将这些数据映射到算子的具体输入位置。通过继承 `NpuGraphOpHandler` 并注册，用户可定义算子在 Capture 阶段的预处理逻辑，以及在 Update 阶段的参数更新逻辑。

### Update 操作说明

**Update 操作** 是指在 Graph Replay（回放）阶段，无需重新 Capture 图结构，即可动态修改算子输入参数（如序列长度、Batch Size 等）的机制。

- **核心目的**：支持动态 Shape 场景（如变长序列推理），避免频繁重编译图，提升性能。
- **工作流程**：
    1.  用户在 Replay 前调用 `g.update(cpu_update_input=[...])` 传入新参数。
    2.  框架遍历 Graph 中的算子，查找注册的 `NpuGraphOpHandler`。
    3.  框架调用 Handler 的 `update_args` 方法，传入 `dispatch_record` 和 `update_input`。
    4.  **源码实现逻辑**：在 `update_args` 中，用户直接修改 `dispatch_record.args` 指定索引的值。

### 内置支持的典型场景

框架已内置注册了以下典型大模型推理算子的 Handler，用户可直接使用 Graph 模式并调用 `update()` 更新动态参数，无需手动注册：

| 场景分类 | 算子名称 | 支持动态更新参数 | 对应 Handler 类 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **IFA v1** | `npu_fused_infer_attention_score`<br>`npu_fused_infer_attention_score.default`<br>`npu_fused_infer_attention_score.out` | `actual_seq_lengths_kv` | `_IFAv1DefaultHandler` | 用于标准 Attention 推理，支持动态序列长度更新及 Workspace 预分配。 |
| **IFA v2** | `npu_fused_infer_attention_score_v2`<br>`npu_fused_infer_attention_score_v2.default`<br>`npu_fused_infer_attention_score_v2.out` | `actual_seq_kvlen` | `_IFAv2DefaultHandler` | IFA 升级版，支持更多 dtype 及布局配置，支持动态序列长度更新。 |
| **Paged Attention** | `_npu_paged_attention.default` | `context_lens` | `_SimpleGraphHandler` | 用于 LLM KV Cache 管理，支持动态上下文长度更新。 |
| **MLA** | `npu_multi_head_latent_attention.out` | `context_lens` | `_SimpleGraphHandler` | 用于多头潜在注意力机制（如 DeepSeek MLA），支持动态上下文长度更新。 |

## 定义文件

torch_npu/npu/graphs/npugraph_handler.py

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
| prepare_capture | func, args, kwargs | func, args, kwargs | Capture 阶段调用。用于 output 预分配或算子变体替换。默认透传。源码中常用于切换 `.default` 到 `.out` 并预分配 workspace。 |
| postprocess_result | result, kwargs | Any | Capture 阶段调用。用于调整返回值格式。默认返回 result。源码中常用于将 C++ TensorList 转为 Python list。 |
| update_args | dispatch_record, update_input | None | Update 阶段调用。用于修改位置参数以响应动态更新。默认空实现。**核心更新逻辑在此实现**。 |
| record_wrap_kwarg | key, value, tensor_param_names | Any | Capture 阶段调用。用于自定义 kwargs 存储方式。默认处理 Tensor 弱引用。 |

**注意**：所有方法必须声明为 `classmethod`，禁止使用 `self` 存储状态（无状态设计）。

### register_npu_graph_handler 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| op_names | str 或 list[str] | 算子名称，对应 `func.__name__`。建议同时注册 `op_name` 和 `op_name.default`。 |

返回值：被装饰的原始类。

## 成员函数详解与源码结合

### torch_npu.npu.NpuGraphOpHandler.update_args

**功能**：应用算子特定的参数更新。

**源码实现解析**：
框架级别的 kwargs 更新由调度骨架处理，仅当需要按索引更新位置参数（args）时重写此钩子。
- **输入**：`dispatch_record` 包含录制时的算子参数列表 `args`。
- **操作**：直接修改 `dispatch_record.args[index]`。
- **典型实现**：参考 `_SimpleGraphHandler`，通过 `_OP_ARG_SPECS` 字典管理不同算子的参数索引映射，避免硬编码。
  ```python
  # _SimpleGraphHandler 源码片段
  _OP_ARG_SPECS = {"_npu_paged_attention.default": (7, "context_lens")}
  @classmethod
  def update_args(cls, record, update_input):
      spec = cls._OP_ARG_SPECS.get(record.op_cache_entry.__name__)
      if spec:
          arg_index, key = spec
          if key in update_input and len(record.args) >= (arg_index + 1):
              record.args[arg_index] = update_input[key]
  ```

### torch_npu.npu.NpuGraphOpHandler.prepare_capture

**功能**：在 `graph_task_group_begin` 之前准备算子调用。

**源码实现解析**：
- **用途**：可用于算子特定的预处理，如 workspace 分配、输出预分配或从 `.default` 切换到 `.out` 重载。
- **典型实现**：参考 `_IFAv1DefaultHandler`，在 `prepare_capture` 中调用 `_npu_fused_infer_attention_score_infer_output` 预分配 `output` 和 `softmax_lse`，并将其放入 `kwargs["out"]`，然后返回 `.out` 算子函数。这确保了 Graph 录制时使用预分配的内存。

### torch_npu.npu.register_npu_graph_handler

**功能**：通过类装饰器注册算子 handler。

**源码实现解析**：
- **注册表**：全局字典 `_NPU_GRAPH_OP_HANDLERS` 存储类对象（非实例）。
- **覆盖保护**：若重复注册同一算子名称，源码会触发 `logger.warning` 警告，但不会静默覆盖，确保调试可见性。

## 调用示例

### 示例 1：动态位置参数更新（参考 _SimpleGraphHandler 逻辑）

```python
import torch
import torch_npu
from torch_npu.npu.npugraph_handlers import NpuGraphOpHandler, register_npu_graph_handler

@register_npu_graph_handler(["my_custom_op.default"])
class MyCustomOpHandler(NpuGraphOpHandler):
    # 定义参数映射：第 3 个参数 (index 2) 对应 update_input 中的 "seq_len"
    _OP_ARG_SPECS = {
        "my_custom_op.default": (2, "seq_len"),
    }

    @classmethod
    def update_args(cls, dispatch_record, update_input):
        spec = cls._OP_ARG_SPECS.get(dispatch_record.op_cache_entry.__name__)
        if spec:
            arg_index, key = spec
            # 结合源码逻辑：检查 key 是否存在并更新 args
            if key in update_input and len(dispatch_record.args) >= (arg_index + 1):
                dispatch_record.args[arg_index] = update_input[key]

g = torch.npu.NPUGraph()
with torch.npu.graph(g, auto_dispatch_capture=True):
    output = torch.ops.my_namespace.my_custom_op(x, weight, seq_len)

# 回放阶段传入新参数
g.update(cpu_update_input=[{"seq_len": new_seq_len}])
g.replay()
```

### 示例 2：输出预分配与 TensorList 返回（参考 _IFAv1DefaultHandler 逻辑）

```python
@register_npu_graph_handler(["my_tensorlist_op.default"])
class TensorListOpHandler(NpuGraphOpHandler):

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        # 类似 IFA  handler，预分配 output 并切换到 .out 算子
        func_out = torch.ops.my_namespace.my_tensorlist_op.out
        if "out" not in kwargs:
            # 预分配逻辑
            kwargs["out"] = [torch.empty_like(args[0]) for _ in range(2)]
        return func_out, args, kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        # 返回 Python list 而非底层 TensorList
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

