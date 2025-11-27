# torch_npu.npu_sim_exponential_


## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        |    √     |


## 功能说明

- 生成服从参数lambd的指数分布随机数，并将其填充到Tensor张量中
- 计算公式：f(x) = -1/λ * ln(1-u), u ~ Uniform(0, 1]


## 函数原型

```
torch_npu.npu_sim_exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
```


## 参数说明

***self***(`Tensor`)：输入/输出，源数据张量。支持连续的Tensor，数据类型支持FLOAT16、BFLOAT6、FLOAT32，数据格式支持ND，维度小于等于8维
***lambd***(`double`)：可选参数，公式中的λ，默认值为1.0
***generator***(`Generator`):：可选参数，用于生成aclnnSimThreadExponential算子中seed和offset，默认为None


## 返回值说明

`Tensor`
代表输入张量被更新后的结果


## 调用示例

```python
>>> import torch
>>> import torch_npu

>>> shape = [100, 400]
>>> gen = torch.Generator(device="npu")
>>> gen.manual_seed(0)
>>> input = torch.zeros(shape, dtype=torch.float32).npu()
>>> torch_npu.npu_sim_exponential_(input, generator=gen)

```