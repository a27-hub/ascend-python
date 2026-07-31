import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuAddRmsNormQuant(TestCase):

    def _quantize(self, data, scale, zero_point, div_mode, dtype_info):
        """Quantize data with given scale and zero_point."""
        if div_mode:
            quant = np.round(data / scale + zero_point)
        else:
            quant = np.round(data * scale + zero_point)
        quant = np.clip(quant, dtype_info["min"], dtype_info["max"])
        return quant.astype(dtype_info["dtype"])

    def supported_op_exec(
        self, x1, x2, gamma, scales1,
        scales2=None, zero_points1=None, zero_points2=None, beta=None,
        axis=-1, epsilon=1e-6, div_mode=True, output_res=False, dst_type=None
    ):
        """CPU reference implementation of add_rms_norm_quant (all float32)."""
        if isinstance(x1, torch.Tensor):
            x1 = x1.numpy()
        if isinstance(x2, torch.Tensor):
            x2 = x2.numpy()
        if isinstance(gamma, torch.Tensor):
            gamma = gamma.numpy()
        if isinstance(scales1, torch.Tensor):
            scales1 = scales1.numpy()
        if scales2 is not None and isinstance(scales2, torch.Tensor):
            scales2 = scales2.numpy()
        if zero_points1 is not None and isinstance(zero_points1, torch.Tensor):
            zero_points1 = zero_points1.numpy()
        if zero_points2 is not None and isinstance(zero_points2, torch.Tensor):
            zero_points2 = zero_points2.numpy()
        if beta is not None and isinstance(beta, torch.Tensor):
            beta = beta.numpy()

        x = x1 + x2
        x_sq = x ** 2
        axis_dim = axis if axis >= 0 else x.ndim + axis
        rms = np.sqrt(np.mean(x_sq, axis=axis_dim, keepdims=True) + epsilon)
        y_normalized = x / rms * gamma
        rms_norm_out = y_normalized.copy()

        if beta is not None:
            y_normalized = y_normalized + beta

        dtype_info = {"dtype": np.int8, "min": -128, "max": 127}
        y1 = self._quantize(y_normalized, scales1,
                            zero_points1 if zero_points1 is not None else 0,
                            div_mode, dtype_info)
        if scales2 is not None:
            y2 = self._quantize(y_normalized, scales2,
                                zero_points2 if zero_points2 is not None else 0,
                                div_mode, dtype_info)
        else:
            y2 = y1.copy()

        if output_res:
            ret_rms = rms_norm_out.astype(np.float32)
        else:
            ret_rms = np.array([], dtype=np.float32)
        return (y1.astype(np.float32), y2.astype(np.float32),
                x.astype(np.float32), ret_rms)

    def custom_op_exec(
        self, x1, x2, gamma, scales1,
        scales2=None, zero_points1=None, zero_points2=None, beta=None,
        axis=-1, epsilon=1e-6, div_mode=True, output_res=False, dst_type=None
    ):
        """NPU op execution, returning CPU numpy arrays."""
        kwargs = {}
        if scales2 is not None:
            kwargs["scales2"] = scales2
        if zero_points1 is not None:
            kwargs["zero_points1"] = zero_points1
        if zero_points2 is not None:
            kwargs["zero_points2"] = zero_points2
        if beta is not None:
            kwargs["beta"] = beta
        if dst_type is not None:
            kwargs["dst_type"] = dst_type
        y1, y2, x_out, rms_norm_out = torch_npu.npu_add_rms_norm_quant(
            x1, x2, gamma, scales1,
            axis=axis, epsilon=epsilon, div_mode=div_mode,
            output_res=output_res, **kwargs)

        def to_float32(t):
            t_cpu = t.cpu()
            if t_cpu.dtype == torch.bfloat16:
                t_cpu = t_cpu.float()
            return t_cpu.numpy().astype(np.float32)

        return (
            to_float32(y1),
            to_float32(y2),
            to_float32(x_out),
            to_float32(rms_norm_out),
        )

    def _assert_quant_outputs(self, cpu_out, npu_out, prec_quant=1.0, prec_fp=0.001):
        """Assert with per-output tolerances: y1/y2 int8 ±1 LSB, x_out/rms_norm_out fp16."""
        tol = [prec_quant, prec_quant, prec_fp, prec_fp]
        for i in range(4):
            self.assertRtolEqual(cpu_out[i], npu_out[i], tol[i])

    # ============ Positive Tests ============

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp16(self, device="npu"):
        """Test basic functionality with float16."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float16) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float())
        npu_out = self.custom_op_exec(x1, x2, gamma, scales1)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_basic_bf16(self, device="npu"):
        """Test basic functionality with bfloat16."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.bfloat16).npu()
        x2 = torch.randn(shape, dtype=torch.bfloat16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.bfloat16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.bfloat16) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float())
        npu_out = self.custom_op_exec(x1, x2, gamma, scales1)
        self._assert_quant_outputs(cpu_out, npu_out, prec_fp=0.004)

    @SupportedDevices(['Ascend950'])
    def test_basic_fp32(self, device="npu"):
        """Test basic functionality with float32 (Ascend950 only)."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float32).npu()
        x2 = torch.randn(shape, dtype=torch.float32).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float32) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float32) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float())
        npu_out = self.custom_op_exec(x1, x2, gamma, scales1)
        self._assert_quant_outputs(cpu_out, npu_out, prec_fp=0.0001)

    @SupportedDevices(['Ascend910B'])
    def test_min_dimension(self, device="npu"):
        """Test 1D input (minimum dimension)."""
        torch.manual_seed(0)
        shape = (256,)
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(shape, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(shape, dtype=torch.float16) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float())
        npu_out = self.custom_op_exec(x1, x2, gamma, scales1)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_3d_input(self, device="npu"):
        """Test 3D input."""
        torch.manual_seed(0)
        shape = (2, 32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float16) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float())
        npu_out = self.custom_op_exec(x1, x2, gamma, scales1)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_with_optional_beta(self, device="npu"):
        """Test with optional beta parameter."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float16) * 0.01 + 0.01).npu()
        beta = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float(), beta=beta.cpu().float())
        npu_out = self.custom_op_exec(x1, x2, gamma, scales1, beta=beta)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_with_all_optional_params(self, device="npu"):
        """Test with all optional parameters provided."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float32) * 0.01 + 0.01).npu()
        scales2 = (torch.rand(last_dim, dtype=torch.float32) * 0.01 + 0.01).npu()
        zero_points1 = torch.zeros(last_dim, dtype=torch.int32).npu()
        zero_points2 = torch.zeros(last_dim, dtype=torch.int32).npu()
        beta = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float(), scales2=scales2.cpu().float(),
            zero_points1=zero_points1.cpu().float(),
            zero_points2=zero_points2.cpu().float(),
            beta=beta.cpu().float())
        npu_out = self.custom_op_exec(
            x1, x2, gamma, scales1,
            scales2=scales2, zero_points1=zero_points1,
            zero_points2=zero_points2, beta=beta)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_div_mode_false(self, device="npu"):
        """Test with div_mode=False (multiply mode)."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float16) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float(), div_mode=False)
        npu_out = self.custom_op_exec(
            x1, x2, gamma, scales1, div_mode=False)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend950'])
    def test_output_res_true(self, device="npu"):
        """Test with output_res=True (Ascend950 only; 910B computeV1 requires nullptr)."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float16) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float(), output_res=True)
        npu_out = self.custom_op_exec(
            x1, x2, gamma, scales1, output_res=True)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_large_input(self, device="npu"):
        """Test large input shape."""
        torch.manual_seed(0)
        shape = (3072, 7168)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float32) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float())
        npu_out = self.custom_op_exec(x1, x2, gamma, scales1)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_dst_type_int8(self, device="npu"):
        """Test dst_type=int8 explicitly."""
        torch.manual_seed(0)
        shape = (32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float32) * 0.01 + 0.01).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float(), dst_type=1)
        npu_out = self.custom_op_exec(
            x1, x2, gamma, scales1, dst_type=1)
        self._assert_quant_outputs(cpu_out, npu_out)

    @SupportedDevices(['Ascend910B'])
    def test_scale_value(self, device="npu"):
        """Test scale numerical values (compare golden, not just shape)."""
        torch.manual_seed(0)
        shape = (2, 32, 256)
        last_dim = shape[-1]
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = (torch.randn(last_dim, dtype=torch.float16) * 0.1).npu()
        scales1 = (torch.rand(last_dim, dtype=torch.float32) * 0.01 + 0.01).npu()
        zero_points1 = torch.zeros(last_dim, dtype=torch.int32).npu()

        cpu_out = self.supported_op_exec(
            x1.cpu().float(), x2.cpu().float(), gamma.cpu().float(),
            scales1.cpu().float(), zero_points1=zero_points1.cpu().float())
        npu_out = self.custom_op_exec(
            x1, x2, gamma, scales1, zero_points1=zero_points1)
        self._assert_quant_outputs(cpu_out, npu_out)

    # ============ Negative Tests ============

    def _neg_dtype_helper(self, bad_tensor, **kwargs):
        """Helper: call op with a bad-dtype tensor, expect RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(**kwargs)

    @SupportedDevices(['Ascend910B'])
    def test_neg_x1_invalid_dtype(self, device="npu"):
        """Negative test: x1 with invalid dtype."""
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        x1_bad = torch.randint(-10, 10, (32, 256), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(x1_bad, x2, gamma, scales1)

    @SupportedDevices(['Ascend910B'])
    def test_neg_x2_invalid_dtype(self, device="npu"):
        """Negative test: x2 with invalid dtype."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        x2_bad = torch.randint(-10, 10, (32, 256), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(x1, x2_bad, gamma, scales1)

    @SupportedDevices(['Ascend910B'])
    def test_neg_gamma_invalid_dtype(self, device="npu"):
        """Negative test: gamma with invalid dtype."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        gamma_bad = torch.randint(-10, 10, (256,), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(x1, x2, gamma_bad, scales1)

    @SupportedDevices(['Ascend910B'])
    def test_neg_scales1_invalid_dtype(self, device="npu"):
        """Negative test: scales1 with invalid dtype."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1_bad = torch.randint(0, 10, (256,), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1_bad)

    @SupportedDevices(['Ascend910B'])
    def test_neg_scales2_invalid_dtype(self, device="npu"):
        """Negative test: scales2 with invalid dtype."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        scales2_bad = torch.randint(0, 10, (256,), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(
                x1, x2, gamma, scales1, scales2=scales2_bad)

    @SupportedDevices(['Ascend910B'])
    def test_neg_zero_points1_invalid_dtype(self, device="npu"):
        """Negative test: zero_points1 with invalid dtype."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        zp1_bad = torch.randn(256, dtype=torch.float64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(
                x1, x2, gamma, scales1, zero_points1=zp1_bad)

    @SupportedDevices(['Ascend910B'])
    def test_neg_zero_points2_invalid_dtype(self, device="npu"):
        """Negative test: zero_points2 with invalid dtype."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        zp2_bad = torch.randn(256, dtype=torch.float64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(
                x1, x2, gamma, scales1, zero_points2=zp2_bad)

    @SupportedDevices(['Ascend910B'])
    def test_neg_beta_invalid_dtype(self, device="npu"):
        """Negative test: beta with invalid dtype."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        beta_bad = torch.randint(0, 10, (256,), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(
                x1, x2, gamma, scales1, beta=beta_bad)

    @SupportedDevices(['Ascend910B'])
    def test_neg_dim_above_max(self, device="npu"):
        """Negative test: input exceeds max dimension (9D)."""
        torch.manual_seed(0)
        shape = (2, 2, 2, 2, 2, 2, 2, 2, 2)
        x1 = torch.randn(shape, dtype=torch.float16).npu()
        x2 = torch.randn(shape, dtype=torch.float16).npu()
        gamma = torch.randn(2, dtype=torch.float16).npu()
        scales1 = (torch.rand(2, dtype=torch.float32) * 0.01 + 0.01).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1)

    @SupportedDevices(['Ascend910B'])
    def test_neg_dst_type_invalid(self, device="npu"):
        """Negative test: dst_type with invalid value."""
        x1 = torch.randn(32, 256, dtype=torch.float16).npu()
        x2 = torch.randn(32, 256, dtype=torch.float16).npu()
        gamma = torch.randn(256, dtype=torch.float16).npu()
        scales1 = (torch.rand(256, dtype=torch.float32) * 0.01 + 0.01).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_add_rms_norm_quant(
                x1, x2, gamma, scales1, dst_type=999)


if __name__ == "__main__":
    run_tests()