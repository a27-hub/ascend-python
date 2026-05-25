# torch_npu.npu_moe_gating_top_k_backward

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 接口功能：完成MoE（Mixture of Experts）门控Top-K选择的反向梯度计算。该算子是aclnnMoeGatingTopK的反向算子，根据前向算子输出的归一化得分（xNorm）、上游梯度（gradY）和专家索引（expertIdx），计算输入得分矩阵的梯度（gradX）。支持sigmoid模式（normType=1）。
- 计算公式（sigmoid模式，normType=1）：

    1. 缩放梯度
        $$
        gradYScaled_{ip} = routedScalingFactor \cdot gradY_{ip}
        $$

    2. 正向renorm的反向传播
        $$
        wPrime_{ip} = xNorm_{i,\ expertIdx_{ip}}
        $$

        $$
        D_i = \sum_{p} wPrime_{ip} + eps
        $$

        $$
        w_{ip} = \frac{wPrime_{ip}}{D_i}
        $$

        $$
        beta_i = \sum_{p} w_{ip} \cdot gradYScaled_{ip}
        $$

        $$
        gradWPrime_{ip} = \frac{gradYScaled_{ip} - beta_i}{D_i}
        $$

    3. 散射到完整维度
        $$
        gradNormX_{ij} = \sum_{p:\ expertIdx_{ip}=j} gradWPrime_{ip}
        $$

    4. Sigmoid反向传播
        $$
        gradX_{ij} = xNorm_{ij} \cdot (1 - xNorm_{ij}) \cdot gradNormX_{ij}
        $$

- 等价计算逻辑：

    ```python
    import torch

    def npu_moe_gating_top_k_backward_ref(
        x_norm, grad_y, expert_idx, *,
        renorm=0, norm_type=1,
        routed_scaling_factor=1.0, eps=1e-20
    ):
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

- **x_norm**（`Tensor`）：必选参数，前向算子 sigmoid/softmax 归一化后的得分，对应前向输出 `normOut`。要求是一个 2D 的 Tensor，维度为 [M, N]，数据类型支持 `float32`，数据格式要求为 ND。支持非连续 Tensor。最后一维的大小（即专家数 N）要求大于等于 `2`，并小于等于 `2048`。

- **grad_y**（`Tensor`）：必选参数，前向算子输出 `yOut` 的上游梯度。要求是一个 2D 的 Tensor，维度为 [M, K]，数据类型支持 `float16`、`bfloat16`、`float32`，数据格式要求为 ND。不支持非连续 Tensor。K的范围要求大于等于 `1`，并小于等于 N。

- **expertIdx**（`Tensor`）：必选参数，前向算子输出 `expertIdxOut`，对应 top-K 专家的索引。shape 要求与 `grad_y` 一致，数据类型支持 `int32`，数据格式要求为 ND。不支持非连续 Tensor。

- **renorm**（`int`）：可选参数，前向算子在 softmax 模式下的 renorm 标记。`0` 表示不做renorm，`1` 表示需要做 renorm。预留参数，当前仅支持 sigmoid 模式。

- **norm_type**（`int`）：可选参数，norm 函数类型，`1` 表示使用 Sigmoid 函数，`0` 表示 Softmax 函数。默认值为 `0`。当前仅支持 `1`（sigmoid 模式）。

- **routed_scaling_factor**（`float`）：可选参数，前向计算 `yOut` 使用的缩放系数，默认值为 `1.0`。

- **eps**（`float`）：可选参数，前向归一化使用的防除零常数，默认值为 `1e-20`。

## 返回值说明

- **grad_x**（`Tensor`）：前向算子输入 `x` 的梯度。要求是一个 2D 的 Tensor，维度为 [M, N]，数据类型与 `grad_y` 需要保持一致，shape 与 `x_norm` 需要一致，数据格式要求为 ND。不支持非连续 Tensor。

## 约束说明

- 该接口支持训练场景下使用。

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
    y.backward(grad_y)
    grad_x = x.grad.detach().clone()
    print(f"grad_x shape: {grad_x.shape}")   # [2048, 192]
    ```
