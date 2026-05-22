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
