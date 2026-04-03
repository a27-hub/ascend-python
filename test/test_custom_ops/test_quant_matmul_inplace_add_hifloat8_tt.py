import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestQuantMatmulInplaceAddHifloat8TT(TestCase):
    def _build_inputs(self):
        m = 16
        k = 32
        n = 8
        x1_fp32 = torch.full((m, k), 0.5, dtype=torch.float32)
        x2_fp32 = torch.full((k, n), 2.0, dtype=torch.float32)
        y_fp32 = torch.arange(m * n, dtype=torch.float32).reshape(m, n) / 100.0
        expected = torch.matmul(x1_fp32, x2_fp32) + y_fp32

        x1_hif8 = torch_npu.npu_dtype_cast(x1_fp32.npu(), torch_npu.hifloat8)
        x2_hif8 = torch_npu.npu_dtype_cast(x2_fp32.npu(), torch_npu.hifloat8)
        x1_scale = torch.ones((1,), dtype=torch.float32).npu()
        x2_scale = torch.ones((1,), dtype=torch.float32).npu()
        y_npu = y_fp32.npu()
        return y_fp32, y_npu, x1_hif8, x2_hif8, x1_scale, x2_scale, expected

    @SupportedDevices(['Ascend950'])
    def test_npu_add_quant_matmul_hifloat8_tt_outplace(self):
        y_fp32, y_npu, x1_hif8, x2_hif8, x1_scale, x2_scale, expected = self._build_inputs()
        result = torch_npu.npu_add_quant_matmul(
            y_npu,
            x1_hif8,
            x2_hif8,
            x2_scale,
            x1_scale=x1_scale,
            group_sizes=None,
            x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8)

        self.assertRtolEqual(expected.cpu().numpy(), result.float().cpu().numpy(), 0.001)
        self.assertRtolEqual(y_fp32.cpu().numpy(), y_npu.float().cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_npu_add_quant_matmul_hifloat8_tt_inplace(self):
        _, y_npu, x1_hif8, x2_hif8, x1_scale, x2_scale, expected = self._build_inputs()
        result = torch_npu.npu_add_quant_matmul_(
            y_npu,
            x1_hif8,
            x2_hif8,
            x2_scale,
            x1_scale=x1_scale,
            group_sizes=None,
            x1_dtype=torch_npu.hifloat8,
            x2_dtype=torch_npu.hifloat8)

        self.assertRtolEqual(expected.cpu().numpy(), result.float().cpu().numpy(), 0.001)
        self.assertRtolEqual(expected.cpu().numpy(), y_npu.float().cpu().numpy(), 0.001)


if __name__ == "__main__":
    run_tests()
