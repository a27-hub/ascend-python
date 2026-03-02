# torch_npu.npu_mhc_post<a name="ZH-CN_TOPIC_0000002309015149"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 推理系列产品</term>                            |    √     |
|<term>Atlas A5 训练系列产品</term>                               |    √     |

## 功能说明<a name="zh-cn_topic_0000002271534921_section1650913464368"></a>

- API功能：mHC（multi-head context）架构中，对上一层输出$h_{l}^{out}$进行Post Mapping，对上一层的输入$x_l$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。
- 计算公式：
    
    $x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}$
    
    其中：
    
    - $(H_{l}^{res})^{T} \times x_l$ 表示对$x_l$使用转置后的组合矩阵进行变换
    - $h_{l}^{out} \otimes H_{t}^{post}$ 表示逐元素相乘后广播

## 函数原型<a name="zh-cn_topic_0000002271534921_section14509346133619"></a>

```
torch_npu.npu_mhc_post(x, h_res, h_out, h_post) -> (Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002271534921_section2050919466368"></a>

- **x** (`Tensor`)：必选参数，表示网络中mHC层的输入数据，要求为3维或4维张量，shape为[B,S,n,D]或[T,n,D]。数据类型支持`float16`、`bfloat16`，数据格式要求为ND。支持非连续Tensor。
- **h_res** (`Tensor`)：必选参数，表示mHC的h_res变换矩阵，是做完sinkhorn变换后的双随机矩阵，要求为3维或4维张量，shape为[B,S,n,n]或[T,n,n]。数据类型支持`float32`，数据格式要求为ND。支持非连续Tensor。
- **h_out** (`Tensor`)：必选参数，表示Atten/MLP层的输出，要求为2维或3维张量，shape为[B,S,D]或[T,D]。数据类型支持`float16`、`bfloat16`，数据格式要求为ND。支持非连续Tensor。
- **h_post** (`Tensor`)：必选参数，表示mHC的h_post变换矩阵，要求为2维或3维张量，shape为[B,S,n]或[T,n]。数据类型支持`float32`，数据格式要求为ND。支持非连续Tensor。

## 返回值说明<a name="zh-cn_topic_0000002271534921_section18510124618369"></a>

- **output** (`Tensor`)：网络中mHC层的输出数据，作为下一层的输入，要求为3维或4维张量，shape为[B,S,n,D]或[T,n,D]。数据类型同`x`，支持`float16`、`bfloat16`，数据格式要求为ND。支持非连续Tensor。

## 约束说明<a name="zh-cn_topic_0000002271534921_section75102046193619"></a>

- 该接口支持训练场景和推理场景下使用。
- 该接口支持图模式。
- Shape约束：
  - n 的取值范围：{4, 6, 8}
  - D 的取值范围：[512, 16384]
  - B*S 或 T的取值范围：[512, 65536]

## 调用示例<a name="zh-cn_topic_0000002271534921_section12510194643619"></a>

- 单算子模式调用
  
  ```python
  import torch
  import torch_npu
  
  # Example: BSND format (B, S, n, D)
  x_shape = (1, 1024, 4, 5120)   # BSND
  h_res_shape = (1, 1024, 4, 4)   # BSND
  h_out_shape = (1, 1024, 5120)   # BSD
  h_post_shape = (1, 1024, 4)     # BSn
  
  x = torch.randn(x_shape, dtype=torch.float16).npu()
  h_res = torch.randn(h_res_shape, dtype=torch.float32).npu()
  h_out = torch.randn(h_out_shape, dtype=torch.float16).npu()
  h_post = torch.randn(h_post_shape, dtype=torch.float32).npu()
  
  output = torch_npu.npu_mhc_post(x, h_res, h_out, h_post)
  ```
- 图模式调用
  
  ```python
  import torch
  import torch.nn as nn
  import torch_npu
  import torchair as tng
  from torchair.configs.compiler_config import CompilerConfig
  
  config = CompilerConfig()
  npu_backend = tng.get_npu_backend(compiler_config=config)
  
  class MhcPostModel(nn.Module):
      def __init__(self):
          super().__init__()
  
      def forward(self, x, h_res, h_out, h_post):
          return torch.ops.npu.npu_mhc_post(x, h_res, h_out, h_post)
  
  def main():
      x_shape = (1, 1024, 4, 5120)   # BSND
      h_res_shape = (1, 1024, 4, 4)   # BSND
      h_out_shape = (1, 1024, 5120)   # BSD
      h_post_shape = (1, 1024, 4)     # BSn
  
      x = torch.randn(x_shape, dtype=torch.float16).npu()
      h_res = torch.randn(h_res_shape, dtype=torch.float32).npu()
      h_out = torch.randn(h_out_shape, dtype=torch.float16).npu()
      h_post = torch.randn(h_post_shape, dtype=torch.float32).npu()
  
      mhc_post_model = MhcPostModel().npu()
      mhc_post_model = torch.compile(mhc_post_model, backend=npu_backend, dynamic=True)
      output = mhc_post_model(x, h_res, h_out, h_post)
  
  if __name__ == '__main__':
      main()
  ```