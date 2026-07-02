# torch_npu.npu.set_deterministic_level

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

该接口用于控制CANN侧确定性计算等级。具体为重新配置CANN侧参数，参数详细说明可见：[aclSysParamOpt](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_1393.html)。

实际level对应配置如下表所示：

| torch_npu.npu.set_deterministic_level配置 | 调用aclrtSetSysParamOpt配置 | 实际功能 | 与原生接口torch.use_deterministic_algorithms的对应关系 |
| :---------------------------------------- | :-------------------------------------------------------------------------------------------- | :--------- | :------------------------|
| 0                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | 关闭确定性 | torch.use_deterministic_algorithms(False) |
| 1                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | 仅开启确定性 | torch.use_deterministic_algorithms(True) |
| 2                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1) | 开启强一致性 | 用于解决部分场景下，某些算子因为累加顺序不一致而导致结果不一致的问题 |
| 3                        | aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3) | 开启batch一致性 | 需要CANN Runtime和算子包版本均支持batch一致性 |

开启强一致性计算后，会导致算子执行速度变慢，影响性能。仅建议在需要严格保证不同位置上的相同数据计算结果一致，或进行精度调优时，才开启强一致性计算，以辅助模型调试和优化。

## 函数原型

```python
torch_npu.npu.set_deterministic_level(level)
```

## 参数说明

**level**(`int`)：所配置的确定性等级，目前可配置为0/1/2/3，默认值为0。

## 返回值说明

无

## 约束说明

- 该功能当前仅影响`torch.mm`计算结果。

- 该接口支持动态修改，后续新下发算子按新的确定性等级生效；已经进入task queue的算子按入队时的确定性快照执行。

- 调用该接口时，会根据所配置的`level`对齐`torch.use_deterministic_algorithms`的bool配置。

- `torch.use_deterministic_algorithms(False)`会使后续算子的实际生效等级为0，但不会清空本接口设置的requested level；再次开启后会恢复到之前的requested level。

- `level=3`仅在CANN Runtime和相关算子包版本均支持batch一致性时可用，否则会直接报错。

## 调用示例

```python
import torch
import torch_npu
torch_npu.npu.set_deterministic_level(2)
```
