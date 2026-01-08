# torch_npu-npu_qkv_rms_norm_rope_cache

## 产品支持情况
| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 推理系列产品</term> |    √     |

## 功能说明

- API功能：输入qkv融合张量，通过SplitVD拆分q、k、v张量。q和k执行RmsNorm、ApplyRotaryPosEmb，k和v执行Quant(可选)、Scatter融合操作，输出q_out、k_cache、v_cache、q_out_before_quant(可选)、k_out_before_quant(可选)、v_out_before_quant(可选)。

- 计算公式：

  (1) SplitVD
  
  令 $N_q$、$N_k$、$N_v$ 分别表示 q、k、v 分量的注意力头数量，必须满足：
  $$
  \begin{cases}
  N_k = N_v \\
  N_{qkv} = N_k + N_v + N_q \\
  D_{qkv} = D_q = D_k = D_v
  \end{cases}
  $$
  

  $$
  \begin{aligned}
  q &= qkv[..., [:N_q] \cdot D_{qkv}] \\
  k &= qkv[..., [N_q:-N_V] \cdot D_{qkv}] \\
  v &= qkv[..., [-N_v:] \cdot D_{qkv}]
  \end{aligned}
  $$
  
  (2) RmsNorm
  
  此处 $x$ 和 $y$ 分别表示RmsNorm的输入张量和输出张量，归一化沿最后一维（feature dimension）进行，该计算规则通用于q、k分量。
  
  $$
  \begin{aligned}
  \text{squareX} &= x \cdot x \\
  \text{meanSquareX} &= \text{squareX}.mean(dim=-1, keepdim=\text{True}) \\
  \text{rms} &= \sqrt{\text{meanSquareX} + \epsilon} \\
  y &= \frac{x}{\text{rms}} \cdot \gamma
  \end{aligned}
  $$
  
  (3) RoPE (Half-and-Half)
  
  此处的 $y$ 指代完成RmsNorm计算的输出结果。
  
  $$
  \begin{aligned}
  y_1 &= y[..., :d/2] \\
  y_2 &= y[..., d/2:] \\
  y_{\text{RoPE}} &= \text{torch.cat}((-y_2, y_1), \text{dim}=-1) \\
  y_{\text{embed}} &= y \cdot \cos + y_{\text{RoPE}} \cdot \sin
  \end{aligned}
  $$

  (4) Quant
  
  - 无量化：
    $$
    k_{\text{Quant}} = k_{\text{RoPE}}, \quad v_{\text{Quant}} = v
    $$
  
  - 对称量化：
    $$
    k_{\text{Quant}} = k_{\text{RoPE}} \cdot k_{\text{scale}}, \quad v_{\text{Quant}} = v \cdot v_{\text{scale}}
    $$
  
  - 非对称量化：
    $$
    k_{\text{Quant}} = k_{\text{RoPE}} \cdot k_{\text{scale}} + k_{\text{offset}}, \quad v_{\text{Quant}} = v \cdot v_{\text{scale}} + v_{\text{offset}}
    $$
  

## 函数原型

```
torch_npu.npu_qkv_rms_norm_rope_cache(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, *, k_scale=None, v_scale=None, k_offset=None, v_offset=None, qkv_size=[], head_nums=[], epsilon=1e-6, cache_mode='PA_NZ', is_output_qkv=False) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **qkv**（`Tensor`）：必选参数，表示融合的QKV矩阵输入，要求是一个2D张量，数据格式为$ND$，数据类型支持`bfloat16`和`float16`。shape为$[B_{\text{qkv}} \times S_{\text{qkv}}, N_{\text{qkv}} \times D_{\text{qkv}}]$，其中$D_{\text{qkv}}$必须是128。

- **q_gamma**（`Tensor`）：必选参数，表示RMSNorm计算中Q分量使用的gamma参数，要求是一个1D张量，数据格式为$ND$，数据类型与`qkv`保持一致。shape为$[D_{\text{qkv}}]$，其中$D_{\text{qkv}}$必须是128。

- **k_gamma**（`Tensor`）：必选参数，表示RMSNorm计算中K分量使用的gamma参数，要求是一个1D张量，数据格式为$ND$，数据类型与`qkv`保持一致。shape为$[D_{\text{qkv}}]$，其中$D_{\text{qkv}}$必须是128。

- **cos**（`Tensor`）：必选参数，表示RoPE位置编码中的cos系数，要求是一个2D张量，数据格式为$ND$，数据类型与`qkv`保持一致。shape为$[B_{\text{qkv}} \times S_{\text{qkv}}, 1 \times D_{\text{rope}}]$，其中要求$D_{\text{rope}} = D_{\text{qkv}}$ = 128。目前仅支持$N_{\text{rope}}=1$，不支持每个头使用不同的RoPE系数。

- **sin**（`Tensor`）：必选参数，表示RoPE位置编码中的sin系数，要求是一个2D张量，数据格式为$ND$，数据类型与`qkv`保持一致。shape格式与`cos`保持一致。

- **index**（`Tensor`）：必选参数，表示更新cache时使用的索引，数据格式为$ND$，数据类型支持`int64`。要求是一个1D张量，shape为$[B_{\text{qkv}} \times S_{\text{qkv}}]$。要求索引值满足：$\text{value}$不能重复，且$\text{value} \in [-1, \text{BlockNum} \times \text{BlockSize})$。同时需满足约束：$\text{BlockNum} \geq \lceil S_{\text{qkv}} / \text{BlockSize} \rceil \times B_{\text{qkv}}$。

- **q_out**（`Tensor`）：必选参数，表示Q输出的预分配内存（输入输出同地址复用），数据格式为$ND$，数据类型与`qkv`相同。shape为$[B_{\text{qkv}} \times S_{\text{qkv}}, N_q \times D_{\text{qkv}}]$的2D矩阵，使用TH分形格式。

- **k_cache**（`Tensor`）：必选参数，表示K cache的预分配内存（输入输出同地址复用），要求是一个4D张量，数据格式为$ND$。在K不量化场景下，数据类型与`qkv`相同，shape为$[\text{BlockNum}, N_k \times D_{\text{qkv}} \div 16, \text{BlockSize}, 16]$；在K量化场景下，数据类型为`int8`，shape为$[\text{BlockNum}, N_k \times D_{\text{qkv}} \div 32, \text{BlockSize}, 32]$。

- **v_cache**（`Tensor`）：必选参数，表示V cache的预分配内存（输入输出同地址复用），要求是一个4D张量，数据格式为$ND$。在V不量化场景下，数据类型与`qkv`相同，shape为$[\text{BlockNum}, N_v \times D_{\text{qkv}} \div 16, \text{BlockSize}, 16]$；在V量化场景下，数据类型为`int8`，shape为$[\text{BlockNum}, N_v \times D_{\text{qkv}} \div 32, \text{BlockSize}, 32]$。注意：`k_cache`和`v_cache`中的BlockNum和BlockSize必须保持一致。

- <strong>*</strong>：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **k_scale**（`Tensor`）：可选参数，当`k_cache`为`int8`类型时需要提供，表示K分量的量化scale参数。数据格式为$ND$，数据类型支持`float32`。必须为2D张量，shape为$[N_k, D_{\text{qkv}}]$。

- **v_scale**（`Tensor`）：可选参数，当`v_cache`为`int8`类型时需要提供，表示V分量的量化scale参数。数据格式为$ND$，数据类型支持`float32`。必须为2D张量，shape为$[N_v, D_{\text{qkv}}]$。

- **k_offset**（`Tensor`）：可选参数，当K采用非对称量化时需要提供，表示K分量的量化offset参数。数据格式为$ND$，数据类型支持`float32`。必须为2D张量，shape为$[N_k, D_{\text{qkv}}]$。注意：此参数依赖`k_scale`输入，且仅在`k_cache`为`int8`数据类型时有效。

- **v_offset**（`Tensor`）：可选参数，当V采用非对称量化时需要提供，表示V分量的量化offset参数。数据格式为$ND$，数据类型支持`float32`。必须为2D张量，shape为$[N_v, D_{\text{qkv}}]$。注意：此参数依赖`v_scale`输入，且仅在`v_cache`为`int8`数据类型时有效。

- **qkv_size**（`List[int]`）：必选属性，按$[B_{\text{qkv}}, S_{\text{qkv}}, N_{\text{qkv}}, D_{\text{qkv}}]$顺序提供输入qkv矩阵的维度尺寸。元素数量必须为4，各元素必须为非零有效正整数。要求$N_{\text{qkv}} = N_k + N_v + N_q$。

- **head_nums**（`List[int]`）：必选属性，按$[N_q, N_k, N_v]$顺序提供QKV分量中N维度的具体尺寸。要求传入实际原始值，不可进行约化。需要满足：$N_q = T \times N_k$，$N_k = N_v$，且$T \geq 1$。

- **epsilon**（`float`）：可选属性，表示RMSNorm中防止除零的小常数，默认值为$10^{-6}$。

- **cache_mode**（`str`）：可选属性，表示cache的格式类型，目前支持值为`"PA_NZ"`。

- **is_output_qkv**（`bool`）：可选属性，表示是否需要输出各cache在未经量化和scatter前的原始值。


## 返回值说明

- **q_out** (`Tensor`)：原地更新，和输入`q_out`的数据类型、维度、数据格式完全一致（本质in-place更新）。

- **k_cache** (`Tensor`)：原地更新，和输入`k_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。

- **v_cache** (`Tensor`)：原地更新，和输入`v_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。

- **q_out_before_quant** (`Tensor`)：仅当`is_output_qkv`为True时有效，表示RoPE处理后，量化处理前的值。要求为2维张量，shape为$[B_{\text{qkv}} \times S_{\text{qkv}}, N_{\text{q}} \times D_{\text{qkv}}]$，数据类型和格式同输入`qkv`一致。

- **k_out_before_quant** (`Tensor`)：仅当`is_output_qkv`为True时有效，表示RoPE处理后，量化处理前的值。要求为2维张量，shape为$[B_{\text{qkv}} \times S_{\text{qkv}}, N_{\text{k}} \times D_{\text{qkv}}]$，数据类型和格式同输入`qkv`一致。

- **v_out_before_quant** (`Tensor`)：仅当`is_output_qkv`为True时有效，表示量化处理前的值。要求为2维张量，shape为$[B_{\text{qkv}} \times S_{\text{qkv}}, N_{\text{v}} \times D_{\text{qkv}}]$，数据类型和格式同输入`qkv`一致。


## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   `qkv`张量为二维张量，shape为$[B_{\text{qkv}} \times S_{\text{qkv}},\ N_q \times D_{\text{qkv}} + N_k \times D_{\text{qkv}} + N_v \times D_{\text{qkv}}]$。
-   $B_{\text{qkv}}$为输入`qkv`的batch size。
-   $S_{\text{qkv}}$为输入`qkv`的sequence length（大小由`qkv_size`属性决定）。
-   $N_{\text{qkv}}$为输入`qkv`的总head number，$N_q$为Q分量的head number，$N_k$为K分量的head number，$N_v$为V分量的head number。需要满足：$N_{\text{qkv}} = N_q + N_k + N_v$且$N_q = T \times N_k$，$N_k = N_v$，$T \geq 1$。
-   $D_{\text{qkv}}$为输入qkv的head dimension，要求$D_{\text{qkv}} = D_q = D_k = D_v$。当前仅支持$D_{\text{qkv}} = 128$。
-   当`cacheMode`为`"PA_NZ"`场景时，`k_cache`的shape中`BlockSize`必须能被32整除，`v_cache`的shape中`BlockSize`必须与k_cache保持一致。


## 调用示例

- 单算子模式调用

  ```python
  import torch
  import torch_npu
  import random
  
  torch_npu.npu.set_compile_mode(jit_compile=True)
  device = torch.device("npu:0")
  
  # 参数
  B, S = 16, 3
  Nq, Nk, Nv = 16, 1, 1
  D = 128
  N = Nq + Nk + Nv
  
  qkv_size = [B, S, N, D]
  head_nums = [Nq, Nk, Nv]
  
  block_size = 128
  block_num = (S + block_size - 1) // block_size * B
  
  dtype = torch.float16
  eps = 1e-6
  
  # 输入 tensor
  valueRange = [-100, 100]
  
  qkv = torch.randn(B * S, N * D, dtype=dtype, device=device) * \
      (valueRange[1] - valueRange[0]) + valueRange[0]
  q_gamma = torch.randn(D, dtype=dtype, device=device) * \
      (valueRange[1] - valueRange[0]) + valueRange[0]
  k_gamma = torch.randn(D, dtype=dtype, device=device) * \
      (valueRange[1] - valueRange[0]) + valueRange[0]
  
  cos = torch.randn(B * S, D, dtype=dtype, device=device)
  sin = torch.randn(B * S, D, dtype=dtype, device=device)
  
  data = list(range(-1, block_num * block_size))
  if B*S > block_num*block_size:
      sampled_data = random.sample(data,  block_num*block_size)
      for _ in range(B*S-block_num*block_size):
          sampled_data.append(-1)
  else:
      sampled_data = random.sample(data, B*S)
  index = torch.tensor(sampled_data, dtype=torch.int64, device=device)
  
  q_out = torch.zeros(B * S, Nq * D, dtype=dtype, device=device)
  # k对称量化
  k_cache = torch.ones(block_num, Nk * D // 32,
                       block_size, 32, dtype=torch.int8, device=device)
  k_scale = torch.randn(Nk, D, dtype=torch.float32, device=device)
  # v对称量化
  v_cache = torch.ones(block_num, Nv * D // 32,
                       block_size, 32, dtype=torch.int8, device=device)
  v_scale = torch.randn(Nv, D, dtype=torch.float32, device=device)
  
  # 单算子调用
  _, _, _ = torch_npu.npu_qkv_rms_norm_rope_cache(
      qkv=qkv,
      q_gamma=q_gamma,
      k_gamma=k_gamma,
      cos=cos,
      sin=sin,
      index=index,
      q_out=q_out,
      k_cache=k_cache,
      v_cache=v_cache,
      k_scale=k_scale,
      v_scale=v_scale,
      k_offset=None,
      v_offset=None,
      qkv_size=qkv_size,
      head_nums=head_nums,
      epsilon=eps,
      cache_mode="PA_NZ",
      is_output_qkv=False,
  )
  
  print(f"q_out:\n{q_out}")
  print(f"k_cache:\n{k_cache}")
  print(f"v_cache:\n{v_cache}")
  ```
  
- 图模式调用

  ```python
  import torch
  import torch_npu
  import torchair
  from torchair.configs.compiler_config import CompilerConfig
  import random
  
  torch_npu.npu.set_compile_mode(jit_compile=True)
  device = torch.device("npu:0")
  
  class Model(torch.nn.Module):
      def __init__(self):
          super().__init__()
  
      def forward(self, qkv, q_gamma, k_gamma, cos, sin, index,
                  q_out, k_cache, v_cache, k_scale, v_scale,
                  qkv_size, head_nums, eps):
          return torch_npu.npu_qkv_rms_norm_rope_cache(
              qkv=qkv,
              q_gamma=q_gamma,
              k_gamma=k_gamma,
              cos=cos,
              sin=sin,
              index=index,
              q_out=q_out,
              k_cache=k_cache,
              v_cache=v_cache,
              k_scale=k_scale,
              v_scale=v_scale,
              k_offset=None,
              v_offset=None,
              qkv_size=qkv_size,
              head_nums=head_nums,
              epsilon=eps,
              cache_mode="PA_NZ",
              is_output_qkv=False,
          )
  
  
  # 参数
  B, S = 16, 3
  Nq, Nk, Nv = 16, 1, 1
  D = 128
  N = Nq + Nk + Nv
  
  qkv_size = [B, S, N, D]
  head_nums = [Nq, Nk, Nv]
  
  block_size = 128
  block_num = (S + block_size - 1) // block_size * B
  
  dtype = torch.float16
  eps = 1e-6
  
  valueRange = [-100, 100]
  
  qkv = torch.randn(B * S, N * D, dtype=dtype, device=device) * \
      (valueRange[1] - valueRange[0]) + valueRange[0]
  q_gamma = torch.randn(D, dtype=dtype, device=device) * \
      (valueRange[1] - valueRange[0]) + valueRange[0]
  k_gamma = torch.randn(D, dtype=dtype, device=device) * \
      (valueRange[1] - valueRange[0]) + valueRange[0]
  
  cos = torch.randn(B * S, D, dtype=torch.float16, device=device)
  sin = torch.randn(B * S, D, dtype=torch.float16, device=device)
  
  data = list(range(-1, block_num * block_size))
  if B*S > block_num*block_size:
      sampled_data = random.sample(data,  block_num*block_size)
      for _ in range(B*S-block_num*block_size):
          sampled_data.append(-1)
  else:
      sampled_data = random.sample(data, B*S)
  index = torch.tensor(sampled_data, dtype=torch.int64, device=device)
  
  q_out = torch.zeros(B * S, Nq * D, dtype=torch.float16, device=device)
  # k对称量化
  k_cache = torch.ones(block_num, Nk * D // 32,
                       block_size, 32, dtype=torch.int8, device=device)
  k_scale = torch.randn(Nk, D, dtype=torch.float32, device=device)
  # v对称量化
  v_cache = torch.ones(block_num, Nv * D // 32,
                       block_size, 32, dtype=torch.int8, device=device)
  v_scale = torch.randn(Nv, D, dtype=torch.float32, device=device)
  
  # 实例化模型model
  model = Model().to(device)
  # 从TorchAir获取NPU提供的默认backend
  config = CompilerConfig()
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  # 使用TorchAir的backend去调用compile接口编译模型
  model = torch.compile(model, backend=npu_backend,
                        dynamic=False, fullgraph=True)
  
  # 静态图执行
  _, _, _ = model(
      qkv, q_gamma, k_gamma, cos, sin, index,
      q_out, k_cache, v_cache,
      k_scale, v_scale,
      qkv_size, head_nums, eps
  )
  
  print(f"q_out:\n{q_out}")
  print(f"k_cache:\n{k_cache}")
  print(f"v_cache:\n{v_cache}")
  ```
  
  
