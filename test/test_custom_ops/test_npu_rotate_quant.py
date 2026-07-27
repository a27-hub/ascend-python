import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def conv_rot(input_matrix, rot_matrix):
    """CPU reference: rotation transformation."""
    stride = rot_matrix.shape[0]
    h, w = input_matrix.shape
    num_blocks = w // stride
    x_reshaped = input_matrix.view(h, num_blocks, stride).reshape(-1, stride)
    out = torch.matmul(x_reshaped.float(), rot_matrix.float())
    return out.reshape(h, w)


class TestNpuRotateQuant(TestCase):
    """Unit tests for npu_rotate_quant operator."""

    def _cpu_rotate_quant_int8(self, x_rot):
        """CPU reference: symmetric pertoken quant (INT8)."""
        max_abs = x_rot.abs().max(dim=-1).values
        scale = max_abs / 127.0
        y = torch.round(x_rot / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
        return y, scale.float()

    def _cpu_rotate_quant_int4(self, x_rot):
        """CPU reference: INT4 quant packed as int32."""
        max_abs = x_rot.abs().max(dim=-1).values
        scale = max_abs / 7.0
        y = torch.round(x_rot.float() / scale.float().unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
        return y, scale.float()

    def custom_op_exec(self, x, rot, alpha=None, y_dtype=None, axis=-1,
                       round_mode="rint", scale_alg=0, dst_type_max=0.0, trans=False):
        """CPU reference: rotate + quantize."""
        x_rot = conv_rot(x, rot)
        # clamp (only when scale_alg=2 and alpha in valid range)
        if alpha is not None and alpha.numel() > 0 and scale_alg == 2:
            av = alpha.item()
            if 0.0 < av < 1.0:
                abs_y = x_rot.abs()
                n = abs_y.shape[-1]
                if n >= 32:
                    groups = abs_y.reshape(-1, n // 32, 32)
                    group_max = groups.max(dim=-1).values
                    limit = av * group_max.unsqueeze(-1).expand_as(x_rot)
                    x_rot = torch.min(torch.max(x_rot, -limit), limit)
        # quantize
        if y_dtype == torch.quint4x2:
            y, scale = self._cpu_rotate_quant_int4(x_rot)
        else:
            y, scale = self._cpu_rotate_quant_int8(x_rot)
        return y.cpu().detach(), scale.cpu().detach()

    def npu_op_exec(self, x, rot, alpha=None, y_dtype=None, axis=-1,
                    round_mode="rint", scale_alg=0, dst_type_max=0.0, trans=False):
        """NPU operator call."""
        y, scale = torch_npu.npu_rotate_quant(
            x, rot, alpha=alpha, y_dtype=y_dtype, axis=axis,
            round_mode=round_mode, scale_alg=scale_alg,
            dst_type_max=dst_type_max, trans=trans)
        return y.cpu().detach(), scale.cpu().detach()

    def _run_positive(self, x_shape, dtype, rot_shape, alpha_val=None, y_dtype_val=None,
                       axis=-1, round_mode="rint", scale_alg=0, dst_type_max=0.0, trans=False):
        """Run NPU + CPU and compare for positive tests (INT8 output)."""
        device = "npu"
        K = rot_shape[-1]
        x = torch.randn(x_shape, dtype=dtype, device=device)
        rot = torch.eye(K, dtype=dtype, device=device)
        if len(rot_shape) == 3:
            rot = rot.unsqueeze(0).expand(rot_shape[0], K, K)
        alpha = torch.tensor([alpha_val], dtype=torch.bfloat16, device=device) if alpha_val is not None else None

        npu_y, npu_scale = self.npu_op_exec(
            x, rot, alpha=alpha, y_dtype=y_dtype_val, axis=axis,
            round_mode=round_mode, scale_alg=scale_alg,
            dst_type_max=dst_type_max, trans=trans)

        cpu_x = x.cpu().detach()
        cpu_rot = rot.cpu().detach()
        cpu_alpha = alpha.cpu().detach() if alpha is not None else None
        cpu_y, cpu_scale = self.custom_op_exec(
            cpu_x, cpu_rot, alpha=cpu_alpha, y_dtype=y_dtype_val, axis=axis,
            round_mode=round_mode, scale_alg=scale_alg,
            dst_type_max=dst_type_max, trans=trans)

        self.assertEqual(npu_y.shape, cpu_y.shape)
        if npu_scale.dtype == torch.float32 or npu_scale.dtype == torch.float64:
            self.assertRtolEqual(npu_scale, cpu_scale, 0.005)
        if npu_y.dtype in [torch.int8, torch.int32]:
            self.assertRtolEqual(npu_y, cpu_y, 1)

    # ========== Positive Tests ==========

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_basic_bf16(self, device="npu"):
        """Basic bf16, default INT8, A2/A3 constraints: N in [128,16000], K in [16,1024]."""
        self._run_positive((32, 256), torch.bfloat16, (64, 64))

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_basic_fp16(self, device="npu"):
        """Basic fp16."""
        self._run_positive((32, 256), torch.float16, (64, 64))

    @SupportedDevices(['Ascend950'])
    def test_alpha_with_scale_alg_2(self, device="npu"):
        """alpha clamp requires scale_alg=2, A5 only per CANN constraint."""
        self._run_positive((32, 256), torch.bfloat16, (64, 64),
                            alpha_val=0.5, scale_alg=2)

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_int8_output(self, device="npu"):
        """INT8 output, explicit y_dtype."""
        self._run_positive((32, 256), torch.bfloat16, (64, 64),
                            y_dtype_val=torch.int8)

    @SupportedDevices(['Ascend910B'])
    def test_int4_output_shape(self, device="npu"):
        """INT4 output: last_dim // 8, dtype=int32."""
        M, N, K = 16, 256, 64  # N=256 >= 128 OK for A2/A3
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        rot = torch.eye(K, dtype=torch.bfloat16, device=device)
        y, scale = torch_npu.npu_rotate_quant(x, rot, y_dtype=torch.quint4x2)
        self.assertEqual(y.shape[-1], N // 8)
        self.assertEqual(y.dtype, torch.int32)
        self.assertEqual(scale.dtype, torch.float32)

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_large_input(self, device="npu"):
        """Large input, N=1024 within [128,16000]."""
        self._run_positive((128, 1024), torch.bfloat16, (64, 64))

    @SupportedDevices(['Ascend950'])
    def test_f8e4m3_output(self, device="npu"):
        """FLOAT8_E4M3FN output (A5 only). Shape verification."""
        M, N, K = 32, 256, 64
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        rot = torch.eye(K, dtype=torch.bfloat16, device=device)
        y, scale = torch_npu.npu_rotate_quant(x, rot, y_dtype=torch.float8_e4m3fn)
        self.assertEqual(list(y.shape), [M, N])
        self.assertEqual(y.dtype, torch.float8_e4m3fn)

    @SupportedDevices(['Ascend950'])
    def test_f8e5_output(self, device="npu"):
        """FLOAT8_E5M2 output (A5 only). Shape verification."""
        M, N, K = 32, 256, 64
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        rot = torch.eye(K, dtype=torch.bfloat16, device=device)
        y, scale = torch_npu.npu_rotate_quant(x, rot, y_dtype=torch.float8_e5m2)
        self.assertEqual(list(y.shape), [M, N])
        self.assertEqual(y.dtype, torch.float8_e5m2)

    @SupportedDevices(['Ascend950'])
    def test_f4e2m1_output(self, device="npu"):
        """FLOAT4_E2M1 output (A5 only). Shape verification."""
        M, N, K = 32, 256, 64
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        rot = torch.eye(K, dtype=torch.bfloat16, device=device)
        y, scale = torch_npu.npu_rotate_quant(x, rot, y_dtype=torch_npu.float4_e2m1fn_x2)
        self.assertEqual(y.shape[-1], N // 2)  # FLOAT4 packs 2 values per uint8
        self.assertEqual(y.dtype, torch.uint8)
        self.assertEqual(scale.dtype, torch.uint8)  # FLOAT8_E8M0 stored as uint8

    # ========== Negative Tests ==========

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_neg_x_invalid_dtype(self, device="npu"):
        """x with invalid dtype should raise."""
        with self.assertRaises(RuntimeError):
            x = torch.randn(32, 256, dtype=torch.float32, device=device)
            rot = torch.eye(64, dtype=torch.bfloat16, device=device)
            torch_npu.npu_rotate_quant(x, rot)

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_neg_rot_invalid_dtype(self, device="npu"):
        """rot with invalid dtype should raise."""
        with self.assertRaises(RuntimeError):
            x = torch.randn(32, 256, dtype=torch.bfloat16, device=device)
            rot = torch.eye(64, dtype=torch.float32, device=device)
            torch_npu.npu_rotate_quant(x, rot)

    @SupportedDevices(['Ascend910B', 'Ascend950'])
    def test_neg_x_rot_dtype_mismatch(self, device="npu"):
        """x and rot with different dtype should raise."""
        with self.assertRaises(RuntimeError):
            x = torch.randn(32, 256, dtype=torch.bfloat16, device=device)
            rot = torch.eye(64, dtype=torch.float16, device=device)
            torch_npu.npu_rotate_quant(x, rot)


if __name__ == "__main__":
    run_tests()
