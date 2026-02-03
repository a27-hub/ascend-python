import math
import unittest
import numpy as np
import torch
import torch_npu
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantMatmulGelu(TestCase):

    def gelu_tanh(self, x):
        """GELU tanh approximation"""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def gelu_erf(self, x):
        """GELU erf"""
        m = nn.GELU()
        return m(x)

    def cpu_quant_matmul_gelu(self, x1, x2, x1_scale, x2_scale, fused_op_type, bias=None):
        """CPU reference implementation"""
        # Compute quantized matmul: x1 @ x2 * x1_scale * x2_scale
        result = torch.matmul(x1.to(torch.int32), x2.to(torch.int32)).float()
        result = result * x1_scale.view(-1, 1) * x2_scale.view(1, -1)
        
        # Add bias if provided
        if bias is not None:
            result = result + bias.view(1, -1)
        
        # Apply GELU
        if fused_op_type == "gelu_tanh":
            result = self.gelu_tanh(result)
        elif fused_op_type == "gelu_erf":
            result = self.gelu_erf(result)
        else:
            raise ValueError(f"Unsupported fused_op_type: {fused_op_type}")
        
        return result

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nd_gelu_tanh(self):
        """Test A8W8 with ND format and gelu_tanh"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, "gelu_tanh")
        
        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh")
        
        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nd_gelu_erf(self):
        """Test A8W8 with ND format and gelu_erf"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, "gelu_erf")
        
        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_erf")
        
        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nd_with_bias(self):
        """Test A8W8 with ND format, gelu_tanh and bias"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        bias = torch.randn(n, dtype=torch.float32) * 0.1
        
        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, "gelu_tanh", bias)
        
        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh", bias=bias.npu())
        
        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nz_gelu_tanh(self):
        """Test A8W8 with NZ format and gelu_tanh"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, "gelu_tanh")
        
        # Convert x2 to NZ format
        x2_nz = torch_npu.npu_format_cast(x2.npu().contiguous(), 29)  # 29 is ACL_FORMAT_FRACTAL_NZ
        
        # Debug: print shapes
        print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}, x2_nz shape: {x2_nz.shape}")
        print(f"x1_scale shape: {x1_scale.shape}, x2_scale shape: {x2_scale.shape}")
        
        # NPU custom op with NZ format
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2_nz, x1_scale.npu(), x2_scale.npu(), "gelu_tanh")
        
        print(f"cpu_result shape: {cpu_result.shape}, npu_result shape: {npu_result.shape}")
        print(f"cpu_result[0, :5]: {cpu_result[0, :5]}")
        print(f"npu_result[0, :5]: {npu_result.cpu().float()[0, :5]}")
        
        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nz_gelu_erf(self):
        """Test A8W8 with NZ format and gelu_erf"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, "gelu_erf")
        
        # Convert x2 to NZ format
        x2_nz = torch_npu.npu_format_cast(x2.npu().contiguous(), 29)  # 29 is ACL_FORMAT_FRACTAL_NZ
        
        # NPU custom op with NZ format
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2_nz, x1_scale.npu(), x2_scale.npu(), "gelu_erf")
        
        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a8w8_nz_simple(self):
        """Test A8W8 with NZ format - simple case with smaller dimensions"""
        torch.manual_seed(0)
        m, k, n = 32, 64, 128
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, "gelu_tanh")
        
        # Convert x2 to NZ format
        x2_nz = torch_npu.npu_format_cast(x2.npu().contiguous(), 29)  # 29 is ACL_FORMAT_FRACTAL_NZ
        
        # NPU custom op with NZ format
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2_nz, x1_scale.npu(), x2_scale.npu(), "gelu_tanh")
        
        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_a4w4_nd_gelu_tanh(self):
        """Test A4W4 with ND format and gelu_tanh"""
        torch.manual_seed(0)
        m, k, n = 128, 256, 512
        
        # Generate INT4 data in INT32 format (8 int4 values per int32)
        x1_fp = torch.randn(m, k, dtype=torch.float32)
        x2_fp = torch.randn(k, n, dtype=torch.float32)
        
        # Quantize to INT4
        scale_tmp = torch.ones(1, dtype=torch.float32).npu()
        x1 = torch_npu.npu_quantize(x1_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)
        x2 = torch_npu.npu_quantize(x2_fp.npu(), scale_tmp, None, torch.quint4x2, -1, False)
        
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1, x2, x1_scale.npu(), x2_scale.npu(), "gelu_tanh")
        
        # For A4W4, we mainly test that the op runs without errors
        # Full accuracy comparison requires proper INT4 CPU implementation
        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result).all())

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_bf16_output(self):
        """Test with BF16 output (x2_scale is BF16)"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.bfloat16).abs() * 0.01  # BF16 scale
        
        # NPU custom op
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh")
        
        # Check output dtype is BF16
        self.assertEqual(npu_result.dtype, torch.bfloat16)
        self.assertEqual(npu_result.shape, (m, n))
        self.assertTrue(torch.isfinite(npu_result.float()).all())

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_batch(self):
        """Test with batch dimensions"""
        torch.manual_seed(0)
        batch, m, k, n = 4, 64, 128, 256
        
        x1 = torch.randint(-5, 5, (batch, m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (batch, k, n), dtype=torch.int8)
        # x1_scale: per-token scale, shape should be (m,) for all batches or (batch*m,) for per-batch-per-token
        # For simplicity, use same scale for all batches: (m,)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # Process each batch separately for CPU reference
        cpu_results = []
        for i in range(batch):
            cpu_result = self.cpu_quant_matmul_gelu(
                x1[i], x2[i], x1_scale, x2_scale, "gelu_tanh")
            cpu_results.append(cpu_result)
        cpu_result = torch.stack(cpu_results, dim=0)
        
        # NPU custom op should handle batch automatically
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), 
            x1_scale.npu(),  # Per-token scale: (m,)
            x2_scale.npu(), 
            "gelu_tanh")
        
        self.assertEqual(npu_result.shape, (batch, m, n))
        self.assertTrue(torch.isfinite(npu_result.float()).all())

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_invalid_fused_type(self):
        """Test with invalid fused_op_type"""
        m, k, n = 64, 128, 256
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # Should raise error for invalid fused_op_type
        with self.assertRaises(RuntimeError):
            torch_npu.npu_quant_matmul_gelu(
                x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "invalid_type")

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_reserved_params_none(self):
        """Test that reserved parameters can be explicitly set to None"""
        torch.manual_seed(0)
        m, k, n = 64, 128, 256
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # CPU reference
        cpu_result = self.cpu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, "gelu_tanh")
        
        # NPU custom op with all reserved parameters explicitly set to None
        npu_result = torch_npu.npu_quant_matmul_gelu(
            x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh",
            bias=None, y_scale=None, x1_offset=None, x2_offset=None, 
            y_offset=None, x3=None, group_size=None)
        
        self.assertRtolEqual(cpu_result.numpy(), npu_result.cpu().float().numpy(), 0.01)

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_matmul_gelu_reserved_params_not_supported(self):
        """Test that reserved parameters raise error when provided (not None)"""
        m, k, n = 64, 128, 256
        
        x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_scale = torch.randn(m, dtype=torch.float32).abs() * 0.01
        x2_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        
        # Test y_scale (reserved parameter)
        y_scale = torch.randn(n, dtype=torch.float32).abs() * 0.01
        with self.assertRaises(RuntimeError):
            torch_npu.npu_quant_matmul_gelu(
                x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh",
                y_scale=y_scale.npu())
        
        # Test x1_offset (reserved parameter)
        x1_offset = torch.zeros(m, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_quant_matmul_gelu(
                x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh",
                x1_offset=x1_offset.npu())
        
        # Test x2_offset (reserved parameter)
        x2_offset = torch.zeros(n, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_quant_matmul_gelu(
                x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh",
                x2_offset=x2_offset.npu())
        
        # Test y_offset (reserved parameter)
        y_offset = torch.zeros(n, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_quant_matmul_gelu(
                x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh",
                y_offset=y_offset.npu())
        
        # Test x3 (reserved parameter)
        x3 = torch.randint(-5, 5, (m, n), dtype=torch.int8)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_quant_matmul_gelu(
                x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh",
                x3=x3.npu())
        
        # Test group_size (reserved parameter)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_quant_matmul_gelu(
                x1.npu(), x2.npu(), x1_scale.npu(), x2_scale.npu(), "gelu_tanh",
                group_size=128)


if __name__ == "__main__":
    run_tests()
