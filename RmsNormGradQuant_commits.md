# RmsNormGradQuant 分支前两次 Commit 变更记录

## Commit 1: `4783f87f` — add rms_norm_backward_quant

**Author:** guoxu7 <guoxu48@huawei.com>
**Date:** Thu May 21 19:08:56 2026 +0800

### 涉及文件（6个）

#### 1. `op_plugin/config/op_plugin_functions.yaml` — 新增函数声明

在 `fused_linear_cross_entropy_loss_with_max_sum_grad` 之后、`symint:` 之前，新增：

```yaml
  - func: _npu_rms_norm_backward_quant(Tensor dy, Tensor x, Tensor rstd, Tensor gamma, *, Tensor? scale_x=None, Tensor? offset_x=None, bool div_mode=True, str quant_mode="static", int dst_type=2) -> (Tensor, Tensor)
    op_api: all_version
```

#### 2. `op_plugin/ops/opapi/RmsNormGradQuantKernelNpuOpApi.cpp` — 新建文件

```cpp
// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _npu_rms_norm_backward_quant(
    const at::Tensor &dy,
    const at::Tensor &x,
    const at::Tensor &rstd,
    const at::Tensor &gamma,
    const c10::optional<at::Tensor> &scale_x,
    const c10::optional<at::Tensor> &offset_x,
    bool div_mode,
    c10::string_view quant_mode,
    int64_t dst_type)
{
    // Convert string parameter to char*
    char *quant_mode_ptr = const_cast<char *>(quant_mode.data());

    // Convert dst_type to aclDataType
    aclDataType dx_acltype = static_cast<aclDataType>(dst_type);

    // dgamma_out: always FLOAT32, shape = gamma.shape
    at::Tensor dgamma_out = npu_preparation::apply_tensor_without_format(
        gamma.sizes(), gamma.options().dtype(at::ScalarType::Float));

    // dx_out: depends on whether quant is enabled
    at::Tensor dx_out;
    if (scale_x.has_value() && scale_x->defined()) {
        // Quant mode: output type determined by dst_type
        at::ScalarType dx_scalar = npu_preparation::convert_to_scalar_type(dx_acltype);
        dx_out = npu_preparation::apply_tensor_without_format(
            dy.sizes(), dy.options().dtype(dx_scalar));
        TensorWrapper dx_wrapper = {dx_out, dx_acltype};
        EXEC_NPU_CMD(aclnnRmsNormGradQuant, dy, x, rstd, gamma,
                      scale_x, offset_x, div_mode, quant_mode_ptr,
                      dx_acltype, dx_wrapper, dgamma_out);
    } else {
        // Non-quant mode: output same dtype as dy
        dx_out = npu_preparation::apply_tensor_without_format(
            dy.sizes(), dy.options());
        EXEC_NPU_CMD(aclnnRmsNormGradQuant, dy, x, rstd, gamma,
                      scale_x, offset_x, div_mode, quant_mode_ptr,
                      dx_acltype, dx_out, dgamma_out);
    }

    return std::make_tuple(std::move(dx_out), std::move(dgamma_out));
}

} // namespace op_api
```

#### 3. `op_plugin/python/meta/_meta_registrations.py` — 文件末尾新增 meta 注册

在文件末尾 `return result` 之后追加：

```python


@impl(m, "_npu_rms_norm_backward_quant")
def _npu_rms_norm_backward_quant_meta(
    dy, x, rstd, gamma, *, scale_x=None, offset_x=None,
    div_mode=True, quant_mode="static", dst_type=2
):
    # dxOut: shape = dy.shape, dtype depends on scale_x and dst_type
    ACL_INT8 = 2
    ACL_HIFLOAT8 = 34
    if scale_x is not None:
        if dst_type == ACL_INT8:
            dx_out = torch.empty(dy.shape, dtype=torch.int8, device=dy.device)
        elif dst_type == ACL_HIFLOAT8:
            dx_out = torch.empty(dy.shape, dtype=torch.uint8, device=dy.device)
        else:
            dx_out = torch.empty(dy.shape, dtype=torch.int8, device=dy.device)
    else:
        dx_out = torch.empty(dy.shape, dtype=dy.dtype, device=dy.device)

    # dgammaOut: shape = gamma.shape, dtype = float32
    dgamma_out = torch.empty(gamma.shape, dtype=torch.float32, device=gamma.device)

    return (dx_out, dgamma_out)
```

#### 4. `test/core_tests/test_fake_tensor.py` — 新增测试类

在 `class TestMatmul` 之后、`if __name__ == "__main__":` 之前，新增：

```python
class TestNpuRmsNormBackwardQuant(TestCase):
    def test_npu_rms_norm_backward_quant_meta(self):
        # 3D, float16, quant mode (INT8, dst_type=2)
        dy = torch.randn(2, 4, 64, dtype=torch.float16).to("npu")
        x = torch.randn(2, 4, 64, dtype=torch.float16).to("npu")
        rstd = torch.randn(2, 4, dtype=torch.float32).to("npu")
        gamma = torch.randn(64, dtype=torch.float16).to("npu")
        scale_x = torch.randn(1, dtype=torch.float32).to("npu")
        offset_x = torch.zeros(1, dtype=torch.int32).to("npu")

        actual_dx, actual_dgamma = torch_npu._npu_rms_norm_backward_quant(
            dy, x, rstd, gamma, scale_x=scale_x, offset_x=offset_x, dst_type=2)

        with FakeTensorMode():
            fake_dy = torch.randn(2, 4, 64, dtype=torch.float16).to("npu")
            fake_x = torch.randn(2, 4, 64, dtype=torch.float16).to("npu")
            fake_rstd = torch.randn(2, 4, dtype=torch.float32).to("npu")
            fake_gamma = torch.randn(64, dtype=torch.float16).to("npu")
            fake_scale_x = torch.randn(1, dtype=torch.float32).to("npu")
            fake_offset_x = torch.zeros(1, dtype=torch.int32).to("npu")

            fake_dx, fake_dgamma = torch_npu._npu_rms_norm_backward_quant(
                fake_dy, fake_x, fake_rstd, fake_gamma,
                scale_x=fake_scale_x, offset_x=fake_offset_x, dst_type=2)

        self.assertEqual(actual_dx.shape, fake_dx.shape)
        self.assertEqual(actual_dgamma.shape, fake_dgamma.shape)

        # 3D, bfloat16, no quant mode (scale_x=None)
        dy = torch.randn(2, 4, 64, dtype=torch.bfloat16).to("npu")
        x = torch.randn(2, 4, 64, dtype=torch.bfloat16).to("npu")
        rstd = torch.randn(2, 4, dtype=torch.float32).to("npu")
        gamma = torch.randn(64, dtype=torch.bfloat16).to("npu")

        actual_dx, actual_dgamma = torch_npu._npu_rms_norm_backward_quant(
            dy, x, rstd, gamma)

        with FakeTensorMode():
            fake_dy = torch.randn(2, 4, 64, dtype=torch.bfloat16).to("npu")
            fake_x = torch.randn(2, 4, 64, dtype=torch.bfloat16).to("npu")
            fake_rstd = torch.randn(2, 4, dtype=torch.float32).to("npu")
            fake_gamma = torch.randn(64, dtype=torch.bfloat16).to("npu")

            fake_dx, fake_dgamma = torch_npu._npu_rms_norm_backward_quant(
                fake_dy, fake_x, fake_rstd, fake_gamma)

        self.assertEqual(actual_dx.shape, fake_dx.shape)
        self.assertEqual(actual_dgamma.shape, fake_dgamma.shape)
```

#### 5. `test/core_tests/torch_npu_OpApi_schema_all.json` — 末尾新增 schema

在最后一个 `}` 之前新增：

```json
  ,
  "func: _npu_rms_norm_backward_quant(Tensor dy, Tensor x, Tensor rstd, Tensor gamma, *, Tensor? scale_x=None, Tensor? offset_x=None, bool div_mode=True, str quant_mode=\"static\", int dst_type=2) -> (Tensor, Tensor)": {
    "version": ["all_version"]
  }
```

#### 6. `test/test_custom_ops/test_npu_rms_norm_backward_quant.py` — 新建文件

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch.testing import assert_close


class TestNpuRmsNormBackwardQuant(TestCase):
    def supported_op_exec(self, dy, x, rstd, gamma, scale_x=None, offset_x=None,
                          div_mode=True, dst_type=2):
        """Golden: splice npu_rms_norm_backward + manual quantization."""
        dx, dgamma = torch_npu.npu_rms_norm_backward(dy, x, gamma, rstd)
        if scale_x is not None:
            dx_fp = dx.float()
            if div_mode:
                dx_quant = torch.round(dx_fp / scale_x.float() + offset_x.float())
            else:
                dx_quant = torch.round(dx_fp * scale_x.float() + offset_x.float())
            dx_quant = dx_quant.clamp(-128, 127).to(torch.int8)
        else:
            dx_quant = dx
        return dx_quant, dgamma

    def custom_op_exec(self, dy, x, rstd, gamma, scale_x=None, offset_x=None,
                       div_mode=True, quant_mode="static", dst_type=2):
        return torch_npu._npu_rms_norm_backward_quant(
            dy, x, rstd, gamma,
            scale_x=scale_x, offset_x=offset_x,
            div_mode=div_mode, quant_mode=quant_mode, dst_type=dst_type)

    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_backward_quant_fp16(self, device="npu"):
        dy = torch.randn(2, 4, 64, dtype=torch.float16).to(device)
        x = torch.randn(2, 4, 64, dtype=torch.float16).to(device)
        rstd = torch.randn(2, 4, dtype=torch.float32).to(device)
        gamma = torch.randn(64, dtype=torch.float16).to(device)
        scale_x = torch.tensor([0.1], dtype=torch.float32).to(device)
        offset_x = torch.tensor([0], dtype=torch.int32).to(device)

        golden_dx, golden_dgamma = self.supported_op_exec(
            dy.clone(), x.clone(), rstd.clone(), gamma.clone(),
            scale_x=scale_x.clone(), offset_x=offset_x.clone())
        custom_dx, custom_dgamma = self.custom_op_exec(
            dy.clone(), x.clone(), rstd.clone(), gamma.clone(),
            scale_x=scale_x.clone(), offset_x=offset_x.clone())

        assert torch.all(custom_dx == golden_dx)
        assert_close(custom_dgamma.float(), golden_dgamma.float(), atol=1e-3, rtol=1e-3)

    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_backward_quant_bf16(self, device="npu"):
        dy = torch.randn(2, 4, 64, dtype=torch.bfloat16).to(device)
        x = torch.randn(2, 4, 64, dtype=torch.bfloat16).to(device)
        rstd = torch.randn(2, 4, dtype=torch.float32).to(device)
        gamma = torch.randn(64, dtype=torch.bfloat16).to(device)
        scale_x = torch.tensor([0.1], dtype=torch.float32).to(device)
        offset_x = torch.tensor([0], dtype=torch.int32).to(device)

        golden_dx, golden_dgamma = self.supported_op_exec(
            dy.clone(), x.clone(), rstd.clone(), gamma.clone(),
            scale_x=scale_x.clone(), offset_x=offset_x.clone())
        custom_dx, custom_dgamma = self.custom_op_exec(
            dy.clone(), x.clone(), rstd.clone(), gamma.clone(),
            scale_x=scale_x.clone(), offset_x=offset_x.clone())

        assert torch.all(custom_dx == golden_dx)
        assert_close(custom_dgamma.float(), golden_dgamma.float(), atol=1e-3, rtol=1e-3)

    @SupportedDevices(['Ascend950'])
    def test_npu_rms_norm_backward_quant_no_quant(self, device="npu"):
        dy = torch.randn(2, 4, 64, dtype=torch.float16).to(device)
        x = torch.randn(2, 4, 64, dtype=torch.float16).to(device)
        rstd = torch.randn(2, 4, dtype=torch.float32).to(device)
        gamma = torch.randn(64, dtype=torch.float16).to(device)

        golden_dx, golden_dgamma = self.supported_op_exec(
            dy.clone(), x.clone(), rstd.clone(), gamma.clone())
        custom_dx, custom_dgamma = self.custom_op_exec(
            dy.clone(), x.clone(), rstd.clone(), gamma.clone())

        assert_close(custom_dx.float(), golden_dx.float(), atol=1e-3, rtol=1e-3)
        assert_close(custom_dgamma.float(), golden_dgamma.float(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
```

---

## Commit 2: `96e8931d` — change fileName

**Author:** guoxu7 <guoxu48@huawei.com>
**Date:** Fri May 22 10:14:00 2026 +0800

### 涉及文件（1个）

#### 1. `op_plugin/ops/opapi/RmsNormGradQuantKernelNpuOpApi.cpp` → 重命名为 `op_plugin/ops/opapi/_RmsNormBackwardQuantKernelNpuOpApi.cpp`

同时修改内容：

**新增参数校验（函数开头）：**

```cpp
    TORCH_CHECK(dy.dim() >= 1, "dy must have at least 1 dimension, but got ", dy.dim(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension, but got ", x.dim(),
                OPS_ERROR(ErrCode::PARAM));
```

**修改 quant_mode 转换方式（替换原来的两行）：**

旧代码：
```cpp
    // Convert string parameter to char*
    char *quant_mode_ptr = const_cast<char *>(quant_mode.data());
```

新代码：
```cpp
    // Convert string parameter to null-terminated char*
    std::string quant_mode_str(quant_mode);
    char *quant_mode_ptr = const_cast<char *>(quant_mode_str.c_str());
```

**修改 EXEC_NPU_CMD 参数顺序（两处）：**

旧代码：
```cpp
        EXEC_NPU_CMD(aclnnRmsNormGradQuant, dy, x, rstd, gamma,
                      scale_x, offset_x, div_mode, quant_mode_ptr,
                      dx_acltype, ...);
```

新代码：
```cpp
        EXEC_NPU_CMD(aclnnRmsNormGradQuant, dy, x, rstd, gamma,
                      scale_x, offset_x, quant_mode_ptr, div_mode,
                      dx_acltype, ...);
```

---

## 合并后的最终文件状态（应用两个 commit 后）

| 文件 | 操作 |
|---|---|
| `op_plugin/config/op_plugin_functions.yaml` | 追加 `_npu_rms_norm_backward_quant` 函数声明 |
| `op_plugin/ops/opapi/_RmsNormBackwardQuantKernelNpuOpApi.cpp` | 新建（含参数校验 + 修正后的代码） |
| `op_plugin/python/meta/_meta_registrations.py` | 末尾追加 meta 注册函数 |
| `test/core_tests/test_fake_tensor.py` | 新增 `TestNpuRmsNormBackwardQuant` 测试类 |
| `test/core_tests/torch_npu_OpApi_schema_all.json` | 末尾追加 schema 条目 |
| `test/test_custom_ops/test_npu_rms_norm_backward_quant.py` | 新建测试文件 |
