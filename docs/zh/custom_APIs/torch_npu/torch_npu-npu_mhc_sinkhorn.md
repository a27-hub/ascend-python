# torch_npu.npu_mhc_sinkhorn

## 产品支持情况

| 产品                             | 是否支持 |
| ---------------------------------| :------: |
|<term>Atlas A5 训练系列产品</term> |    √     |
|<term>Atlas A5 训练系列产品</term>     |    √     |

## 功能说明

- API功能：在mhc架构中，对上一层输出h_res，将其约束为一个双随机矩阵，即矩阵的每一行每一列相加都等于1。

## 函数原型

```
torch_npu.npu_mhc_sinkhorn(Tensor x, *, float eps=1e-6, Symint num_iters=20, int out_flag=0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入张量，数据类型支持float32, 数据格式支持ND, shape是3维(T, n, n)或者4维(B, S, n, n), 其中n仅支持4, 6, 8. 待计算数据，表示mHC层的输入数据。
- **eps** (`float`)：可选参数，数据类型支持float32, 表示归一化防除零参数，默认值为1e-6。
- **num_iters** (`int`)：可选参数，数据类型支持int64, 表示迭代次数, 取值范围[1, 100], 默认值为20。
- **out_flag** (`int`)：可选参数. 数据类型支持int64, 表示是否输出中间结果标识, 默认值为0(仅输出最终变换结果)。

## 返回值说明

- **output** (`Tensor`)：一个Tensor类型的输出, 代表mhc_sinkhorn. 数据类型和张量形状与输入x保持一致。数据格式支持ND。

## 约束说明

- 该接口支持训练场景和推理场景下使用。
- 该接口支持图模式。
- Shape约束：
  - 3维或4维张量，即[T, n, n]或[B, S, n, n]
  - n的取值范围：{4, 6, 8}
-参数约束：
 -num_iters的取值范围：[1, 100]
 -out_flag：仅支持取值为0
## 调用示例

单算子模式调用：

```python
import torch
import torch_npu
 
#Example: (B, S, n, n)
x_shape = [1, 128, 4 , 4]
x = torch.rand(x_shape, dtype=torch.float32).npu()
eps = 1e-6
num_iters = 20
out_flag = 0

npu_out = torch_npu.npu_mhc_sinkhorn(x, eps=eps, num_iters=num_iters, out_flag=out_flag)
```

图模式调用：
```python
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.mode="reduce-overhead"
npu_backend = tng.get_npu_backend(compiler_config=config)

class MhcSinkhornModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, eps, num_iters, out_flag):
        y = torch_npu.npu_mhc_sinkhorn(x_npu, eps=eps, num_iters=num_iters, out_flag=out_flag)
        return y

def main():
    x_shape = [1, 128, 4 , 4]
    x = torch.randn(x_shape, dtype=torch.float32).npu()
    eps = 1e-6
    num_iters = 20
    out_flag = 0

    mhc_sinkhorn_model = MhcSinkhornModel().npu()
    mhc_sinkhorn_model = torch.compile(mhc_sinkhorn_model, backend=npu_backend)
    output = mhc_sinkhorn_model(x, eps, num_iters, out_flag)

if __name__ == '__main__':
    main()
```