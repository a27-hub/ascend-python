import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantBatchMatmulInplaceAdd(TestCase):
    def _run_tt_case(self, transpose_x1, transpose_x2):
        m, n, k = 16, 32, 64
        y = torch.ones((m, n), dtype=torch.float32, device="npu")
        x1 = torch.zeros((m, k), dtype=torch.uint8, device="npu")
        x2 = torch.zeros((k, n), dtype=torch.uint8, device="npu")
        x1 = x1.transpose(-2, -1) if transpose_x1 else x1
        x2 = x2.transpose(-2, -1) if transpose_x2 else x2
        x1_scale = torch.ones((1,), dtype=torch.float32, device="npu")
        x2_scale = torch.ones((1,), dtype=torch.float32, device="npu")

        inplace_out = torch_npu.npu_add_quant_matmul_(
            y, x1, x2, x2_scale, x1_scale=x1_scale, group_sizes=[0, 0, 0],
            x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8,
            x1_scale_dtype=torch.float32, x2_scale_dtype=torch.float32)
        self.assertRtolEqual(inplace_out.cpu().numpy(), torch.ones((m, n), dtype=torch.float32).numpy())

        functional_y = torch.ones((m, n), dtype=torch.float32, device="npu")
        functional_out = torch_npu.npu_add_quant_matmul(
            functional_y, x1, x2, x2_scale, x1_scale=x1_scale, group_sizes=[0, 0, 0],
            x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8,
            x1_scale_dtype=torch.float32, x2_scale_dtype=torch.float32)
        self.assertRtolEqual(functional_out.cpu().numpy(), torch.ones((m, n), dtype=torch.float32).numpy())
        self.assertRtolEqual(functional_y.cpu().numpy(), torch.ones((m, n), dtype=torch.float32).numpy())

    @SupportedDevices(['Ascend950'])
    def test_npu_add_quant_matmul_hif8_tt(self):
        for transpose_x1 in (False, True):
            for transpose_x2 in (False, True):
                self._run_tt_case(transpose_x1, transpose_x2)

if __name__ == "__main__":
    run_tests()