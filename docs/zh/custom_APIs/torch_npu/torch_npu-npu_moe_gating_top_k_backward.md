# torch_npu.npu_moe_gating_top_k_backward

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- API 功能：`npu_moe_gating_top_k_backward` 是 `npu_moe_gating_top_k` 的反向算子，计算前向算子输入 `x` 的梯度。当前仅支持 sigmoid 模式（`norm_type=1`）。

- 计算公式：

  设上游梯度为 $\frac{\partial L}{\partial \text{weights}} \in \mathbb{R}^{M \times K}$，前向保存的归一化得分为 `x_norm`（即前向输出 `normOut`），前向选出的专家索引为 `expertIdx`（即前向输出 `expertIdxOut`）。

  **Step 1：应用缩放因子的反向**

  $$\frac{\partial L}{\partial w_{ip}} = \text{routed\_scaling\_factor} \cdot \frac{\partial L}{\partial \text{weights}_{ip}}$$

  **Step 2：归一化的反向（sigmoid 模式，`norm_type=1`）**

  令 $D_i = \sum_{p'=1}^{K} w'_{ip'} + \epsilon$，其中 $w'_{ip} = \text{x\_norm}_{i,\,\text{expertIdx}_{ip}}$：

  $$\frac{\partial L}{\partial w'_{ip}} = \frac{1}{D_i}\left(\frac{\partial L}{\partial w_{ip}} - \sum_{q=1}^{K} w_{iq} \cdot \frac{\partial L}{\partial w_{iq}}\right)$$

  **Step 3：gather 的反向（scatter_add）**

  $$\frac{\partial L}{\partial \hat{S}_{ij}} = \sum_{p:\,\text{expertIdx}_{ip}=j} \frac{\partial L}{\partial w'_{ip}}$$

  **Step 4：sigmoid 的反向**

  $$\text{grad\_x}_{ij} = \text{x\_norm}_{ij} \cdot (1 - \text{x\_norm}_{ij}) \cdot \frac{\partial L}{\partial \hat{S}_{ij}}$$

- 等价计算逻辑：

  ```python
  import torch

  def npu_moe_gating_top_k_backward_ref(
          x_norm, grad_y, expert_idx, *,
          renorm=0, norm_type=1,
          routed_scaling_factor=1.0, eps=1e-20):
      """CPU float64 参考实现（sigmoid 模式）。"""
      x_norm_f64 = x_norm.double()
      grad_y_f64 = grad_y.double()
      expert_idx_i64 = expert_idx.long()

      # Step 1: 缩放因子反向
      grad_w = grad_y_f64 * routed_scaling_factor

      # Step 2: 归一化反向（sigmoid 模式）
      w_prime = x_norm_f64.gather(1, expert_idx_i64)
      D = w_prime.sum(dim=-1, keepdim=True) + eps
      dot = (grad_w * w_prime).sum(dim=-1, keepdim=True)
      grad_w_prime = grad_w / D - dot / (D ** 2)

      # Step 3: gather 反向（scatter_add）
      grad_s_hat = torch.zeros_like(x_norm_f64)
      grad_s_hat.scatter_add_(1, expert_idx_i64, grad_w_prime)

      # Step 4: sigmoid 反向
      grad_x = x_norm_f64 * (1.0 - x_norm_f64) * grad_s_hat

      return grad_x.to(grad_y.dtype)
  ```

## 函数原型

```python
npu_moe_gating_top_k_backward(x_norm, grad_y, expertIdx, *, renorm=0, norm_type=0, routed_scaling_factor=1.0, eps=1e-20) -> Tensor
```

## 参数说明

- **x_norm**（`Tensor`）：必选参数，前向算子 sigmoid/softmax 归一化后的得分，对应前向输出 `normOut`。要求是一个 2D 的 Tensor，维度为 [M, N]，数据类型支持 `float32`，数据格式要求为 ND。支持非连续 Tensor。最后一维的大小（即专家数 N）要求不大于 `2048`。

- **grad_y**（`Tensor`）：必选参数，前向算子输出 `yOut` 的上游梯度。要求是一个 2D 的 Tensor，维度为 [M, K]，数据类型支持 `float16`、`bfloat16`、`float32`，数据格式要求为 ND。不支持非连续 Tensor。

- **expertIdx**（`Tensor`）：必选参数，前向算子输出 `expertIdxOut`，对应 top-K 专家的索引。shape 要求与 `grad_y` 一致，数据类型支持 `int32`，数据格式要求为 ND。不支持非连续 Tensor。

- **renorm**（`int`）：可选参数，renorm 标记，默认值为 `0`，表示先进行 norm 再进行 topk 计算。当前仅支持 `0`。

- **norm_type**（`int`）：可选参数，norm 函数类型，`1` 表示使用 Sigmoid 函数，`0` 表示 Softmax 函数。默认值为 `0`。当前仅支持 `1`（sigmoid 模式）。

- **routed_scaling_factor**（`float`）：可选参数，前向计算 `yOut` 使用的缩放系数，默认值为 `1.0`。

- **eps**（`float`）：可选参数，前向归一化使用的防除零系数，默认值为 `1e-20`。

## 返回值说明

- **grad_x**（`Tensor`）：前向算子输入 `x` 的梯度。要求是一个 2D 的 Tensor，维度为 [M, N]，数据类型与 `grad_y` 需要保持一致，shape 与 `x_norm` 需要一致，数据格式要求为 ND。不支持非连续 Tensor。

## 约束说明

- 该接口支持训练场景下使用。
- 当前仅支持 sigmoid 模式（`norm_type=1`）。
- `renorm` 当前仅支持 `0`。
- `x_norm` 最后一维（专家数 N）要求不大于 `2048`。
- `grad_y` 与 `expertIdx` 的 shape 必须一致。
- `grad_x` 的数据类型必须与 `grad_y` 一致。

## 调用示例

- 单算子模式调用

  ```python
  import torch
  import torch_npu

  # 示例参数：M=2048, N=192, K=10，sigmoid 模式，PanGuV3 典型 shape
  M, N, K = 2048, 192, 10
  norm_type = 1
  routed_scaling_factor = 1.0
  eps = 1e-20

  # 构造输入（模拟前向算子的保存值）
  x_norm = torch.rand(M, N, dtype=torch.float32).npu()          # 前向 normOut
  grad_y = torch.randn(M, K, dtype=torch.bfloat16).npu()        # 上游梯度
  expert_idx = torch.randint(0, N, (M, K), dtype=torch.int32).npu()  # 前向 expertIdxOut

  # 调用反向算子
  grad_x = torch_npu.npu_moe_gating_top_k_backward(
      x_norm, grad_y, expert_idx,
      norm_type=norm_type,
      routed_scaling_factor=routed_scaling_factor,
      eps=eps)

  print(f"grad_x shape: {grad_x.shape}")   # [2048, 192]
  print(f"grad_x dtype: {grad_x.dtype}")   # bfloat16（与 grad_y 一致）
  ```

- 通过正向算子自动反向调用

  ```python
  import torch
  import torch_npu

  M, N, K = 2048, 192, 10

  # 前向传播
  x = torch.randn(M, N, dtype=torch.bfloat16, requires_grad=False).npu().requires_grad_(True)
  y, expert_idx, x_norm = torch_npu.npu_moe_gating_top_k(
      x, K, norm_type=1, routed_scaling_factor=1.0, eps=1e-20, out_flag=True)

  # 模拟上游梯度
  grad_y = torch.randn_like(y)

  # 反向传播
  grad_x = y.backward(grad_y)

  print(f"grad_x shape: {grad_x.shape}")   # [2048, 192]
  ```