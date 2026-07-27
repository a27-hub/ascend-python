import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestNpuGeluMul(TestCase):
    """
    npu_gelu_mul: split input last dim into x1 (left half) and x2 (right half),
    out = GELU(x1) * x2. Output last dim = input last dim / 2, dtype == input dtype.
    """

    def supported_op_exec(self, input_tensor, approximate="none"):
        """CPU reference: PyTorch F.gelu on the left half, multiplied by right half."""
        last_dim = input_tensor.shape[-1]
        half = last_dim // 2
        x1 = input_tensor[..., :half]
        x2 = input_tensor[..., half:]
        output = torch.nn.functional.gelu(x1, approximate=approximate) * x2
        # numpy has no bfloat16 -> cast bf16 to float32 only (keep fp16/fp32 native)
        if output.dtype == torch.bfloat16:
            output = output.to(torch.float32)
        return output.numpy()

    def custom_op_exec(self, input_tensor, approximate="none"):
        """NPU op, returns CPU numpy result."""
        output = torch.ops.npu.npu_gelu_mul(input_tensor, approximate=approximate).cpu()
        if output.dtype == torch.bfloat16:
            output = output.to(torch.float32)
        return output.numpy()

    @SupportedDevices(['Ascend910B'])
    def test_basic_bf16(self, device="npu"):
        torch.manual_seed(0)
        input_tensor = torch.randn(128, 128, dtype=torch.bfloat16).npu()
        supported_output = self.supported_op_exec(input_tensor.cpu())
        custom_output = self.custom_op_exec(input_tensor)
        # bf16 has ~7 mantissa bits -> looser rtol than the float32 default 1e-4
        self.assertRtolEqual(supported_output, custom_output, prec=1e-2)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp16(self, device="npu"):
        torch.manual_seed(0)
        input_tensor = torch.randn(128, 128, dtype=torch.float16).npu()
        supported_output = self.supported_op_exec(input_tensor.cpu())
        custom_output = self.custom_op_exec(input_tensor)
        self.assertRtolEqual(supported_output, custom_output)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp32(self, device="npu"):
        torch.manual_seed(0)
        input_tensor = torch.randn(128, 128, dtype=torch.float32).npu()
        supported_output = self.supported_op_exec(input_tensor.cpu())
        custom_output = self.custom_op_exec(input_tensor)
        self.assertRtolEqual(supported_output, custom_output)

    @SupportedDevices(['Ascend910B'])
    def test_min_dimension(self, device="npu"):
        # boundary: minimal rank (2D) with a small last dim
        torch.manual_seed(0)
        input_tensor = torch.randn(2, 4, dtype=torch.float32).npu()
        supported_output = self.supported_op_exec(input_tensor.cpu())
        custom_output = self.custom_op_exec(input_tensor)
        self.assertRtolEqual(supported_output, custom_output)

    @SupportedDevices(['Ascend910B'])
    def test_3d_input(self, device="npu"):
        torch.manual_seed(0)
        input_tensor = torch.randn(4, 32, 128, dtype=torch.float32).npu()
        supported_output = self.supported_op_exec(input_tensor.cpu())
        custom_output = self.custom_op_exec(input_tensor)
        self.assertRtolEqual(supported_output, custom_output)

    @SupportedDevices(['Ascend910B'])
    def test_output_shape_dtype(self, device="npu"):
        # verify output shape (last dim halved) and dtype (== input)
        torch.manual_seed(0)
        input_tensor = torch.randn(4, 32, 128, dtype=torch.float16).npu()
        output = torch.ops.npu.npu_gelu_mul(input_tensor)
        self.assertEqual(list(output.shape), [4, 32, 64])
        self.assertEqual(output.dtype, torch.float16)
        supported_output = self.supported_op_exec(input_tensor.cpu())
        self.assertRtolEqual(supported_output, output.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_approximate_tanh(self, device="npu"):
        # covers the approximate='tanh' branch
        torch.manual_seed(0)
        input_tensor = torch.randn(128, 128, dtype=torch.float32).npu()
        supported_output = self.supported_op_exec(input_tensor.cpu(), approximate="tanh")
        custom_output = self.custom_op_exec(input_tensor, approximate="tanh")
        self.assertRtolEqual(supported_output, custom_output)

    @SupportedDevices(['Ascend910B'])
    def test_input_8d(self, device="npu"):
        # boundary: maximal rank (8D), last dim 16 (even)
        torch.manual_seed(0)
        input_tensor = torch.randn(2, 2, 2, 2, 2, 2, 2, 16, dtype=torch.float32).npu()
        supported_output = self.supported_op_exec(input_tensor.cpu())
        custom_output = self.custom_op_exec(input_tensor)
        self.assertRtolEqual(supported_output, custom_output)

    @SupportedDevices(['Ascend910B'])
    def test_neg_input_invalid_dtype(self, device="npu"):
        input_tensor = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch.ops.npu.npu_gelu_mul(input_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_neg_dim_below_min(self, device="npu"):
        input_tensor = torch.randn(4, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch.ops.npu.npu_gelu_mul(input_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_neg_dim_above_max(self, device="npu"):
        input_tensor = torch.randn(2, 2, 2, 2, 2, 2, 2, 2, 2, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch.ops.npu.npu_gelu_mul(input_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_neg_last_dim_odd(self, device="npu"):
        input_tensor = torch.randn(128, 127, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch.ops.npu.npu_gelu_mul(input_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_neg_approximate_invalid(self, device="npu"):
        input_tensor = torch.randn(128, 128, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch.ops.npu.npu_gelu_mul(input_tensor, approximate="relu")


if __name__ == "__main__":
    run_tests()
