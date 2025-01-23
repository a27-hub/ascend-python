import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestDequantSwigluQuant(TestCase):
    def get_golden(self, input_self_tensor, quant_scale, quant_offset, activate_left, quant_mode):
        def swiglu_quant(input_self_tensor, quant_scale, quant_offset, activate_left, quant_mode):
            swiglu_res = torch_npu.npu_swiglu_quant(input_self_tensor, -1)
            res = torch_npu.npu_quantize(swiglu_res, quant_scale, quant_offset, torch.qint8, -1)
            return res

        output = swiglu_quant(input_self_tensor, quant_scale, quant_offset, activate_left, quant_mode)
        return output

    @SupportedDevices(['Ascend910B'])
    def test_swiglu_quant(self):
        shape = [384, 1, 7168]
        shape2 = [1, ]
        input_self_tensor = torch.rand(shape, device='cpu', dtype=torch.float16).npu()
        quant_scale = torch.rand(shape2, device='cpu', dtype=torch.float32).npu()
        quant_offset = torch.rand(shape2, device='cpu', dtype=torch.float32).npu()
        torch.npu.synchronize()
        output = torch_npu.npu_swiglu_quant(input_self_tensor, quant_scale, quant_offset, False, 0)
        torch.npu.synchronize()

        golden = self.get_golden(input_self_tensor, quant_scale, quant_offset, False, 0)
        self.assertRtolEqual(output, golden)


if __name__ == "__main__":
    run_tests()
