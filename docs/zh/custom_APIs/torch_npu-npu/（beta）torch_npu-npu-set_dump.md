# （beta）torch_npu.npu.set_dump

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas 350 加速卡</term>            |    √     |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

设置 Dump 配置文件的路径。调用后不会立即生效，需配合 `torch_npu.npu.init_dump()` 启用 Dump 功能。

Dump 功能的完整使用流程为：

1. 设置环境变量 `export NPU_DUMP_ENABLE=1`
2. 调用 `torch_npu.npu.set_dump(path_to_json)` 指定配置文件路径
3. 调用 `torch_npu.npu.init_dump()` 启用 Dump
4. 执行模型推理或训练
5. 调用 `torch_npu.npu.finalize_dump()` 结束 Dump

## 函数原型

```python
torch_npu.npu.set_dump(path_to_json)
```

## 参数说明

 **path_to_json**：配置文件所在的路径，包含文件名，用户需根据实际情况配置。具体配置请参考《CANN Runtime运行时 API》中“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclpythondevg_01_0155.html">函数：set_dump</a>”章节。

## 调用示例

```python
>>> import os
>>> import torch
>>> import torch_npu
>>>
>>> # 1. 设置环境变量（需在启动脚本中设置，或代码中提前设置）
>>> os.environ["NPU_DUMP_ENABLE"] = "1"
>>>
>>> # 2. 指定 Dump 配置文件路径
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
>>>
>>> # 3. 启用 Dump
>>> torch_npu.npu.init_dump()
>>>
>>> # 4. 执行模型推理（示例）
>>> # output = model(input_data)
>>>
>>> # 5. 结束 Dump
>>> torch_npu.npu.finalize_dump()
```
