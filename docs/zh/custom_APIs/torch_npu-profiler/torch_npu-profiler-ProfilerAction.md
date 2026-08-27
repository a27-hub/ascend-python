# torch_npu.profiler.ProfilerAction

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

用于控制Profiler行为状态，如性能数据采集、性能数据采集预热、性能数据采集并保存，属于Enum类型。

## 枚举定义

```python
class torch_npu.profiler.ProfilerAction(Enum):
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3
```

## 枚举成员说明

- **torch_npu.profiler.ProfilerAction.NONE**：枚举成员，表示无任何行为。
- **torch_npu.profiler.ProfilerAction.WARMUP**：枚举成员，表示性能数据采集预热。
- **torch_npu.profiler.ProfilerAction.RECORD**：枚举成员，表示性能数据采集。
- **torch_npu.profiler.ProfilerAction.RECORD_AND_SAVE**：枚举成员，表示性能数据采集并保存。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
import torch
import torch_npu

...
with torch_npu.profiler.profile(
    schedule=torch_npu.profiler.ProfilerAction.RECORD,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```
