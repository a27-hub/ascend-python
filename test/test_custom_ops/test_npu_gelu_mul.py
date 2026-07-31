import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestNPUGeluMul(TestCase):

    def get_golden(self, input_self_tensor, approximate="none"):
        # Split input along last dim into x1 (first half) and x2 (second half)
        d = input_self_tensor.shape[-1] // 2
        x1 = input_self_tensor[..., :d]
        x2 = input_self_tensor[..., d:]
        golden = torch.nn.functional.gelu(x1, approximate=approximate) * x2
        return golden

    @SupportedDevices(['Ascend910B'])
    def test_basic_bf16(self):
        shape = [128, 128]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        input_tensor = cpu_input.to(torch.bfloat16).npu()
        output = torch_npu.npu_gelu_mul(input_tensor)
        golden = self.get_golden(cpu_input).to(torch.bfloat16)
        # bf16 has ~7 bits of mantissa; allow rtol/atol = 2e-2 (operator computes in bf16, golden in fp32)
        self.assertTrue(torch.allclose(output.cpu(), golden.cpu(), rtol=2e-2, atol=2e-2))

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp16(self):
        shape = [128, 128]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        input_tensor = cpu_input.to(torch.float16).npu()
        output = torch_npu.npu_gelu_mul(input_tensor)
        golden = self.get_golden(cpu_input).to(torch.float16)
        # fp16 has ~10 bits of mantissa; allow rtol/atol = 5e-3 (operator computes in fp16, golden in fp32)
        self.assertTrue(torch.allclose(output.cpu(), golden.cpu(), rtol=5e-3, atol=5e-3))

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp32(self):
        shape = [128, 128]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        input_tensor = cpu_input.npu()
        output = torch_npu.npu_gelu_mul(input_tensor)
        golden = self.get_golden(cpu_input)
        self.assertRtolEqual(output, golden)

    @SupportedDevices(['Ascend910B'])
    def test_min_dimension(self):
        # Minimum dimension: 2D with smallest last dim (2)
        shape = [2, 2]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        input_tensor = cpu_input.npu()
        output = torch_npu.npu_gelu_mul(input_tensor)
        golden = self.get_golden(cpu_input)
        self.assertRtolEqual(output, golden)
        self.assertEqual(output.shape, torch.Size([2, 1]))

    @SupportedDevices(['Ascend910B'])
    def test_3d_input(self):
        shape = [4, 32, 128]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        input_tensor = cpu_input.npu()
        output = torch_npu.npu_gelu_mul(input_tensor)
        golden = self.get_golden(cpu_input)
        self.assertRtolEqual(output, golden)
        self.assertEqual(output.shape, torch.Size([4, 32, 64]))

    @SupportedDevices(['Ascend910B'])
    def test_approximate_tanh(self):
        shape = [64, 64]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        input_tensor = cpu_input.npu()
        output = torch_npu.npu_gelu_mul(input_tensor, approximate='tanh')
        golden = self.get_golden(cpu_input, approximate='tanh')
        self.assertRtolEqual(output, golden)

    @SupportedDevices(['Ascend910B'])
    def test_empty_tensor(self):
        shape = [4, 0, 128]
        np.random.seed(0)
        input_tensor = torch.randn(shape, dtype=torch.float32).npu()
        output = torch_npu.npu_gelu_mul(input_tensor)
        self.assertEqual(output.shape, torch.Size([4, 0, 64]))

    @SupportedDevices(['Ascend910B'])
    def test_non_contiguous(self):
        shape = [128, 128]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        # Make non-contiguous by transposing a non-square view
        contiguous = cpu_input.npu()
        non_contiguous = contiguous.t()
        self.assertFalse(non_contiguous.is_contiguous())
        # npu_gelu_mul operates on last dim; transpose makes original rows into cols.
        # Compute golden on the transposed (non-contiguous) data directly.
        output = torch_npu.npu_gelu_mul(non_contiguous)
        golden = self.get_golden(cpu_input.t())
        self.assertRtolEqual(output, golden)

    @SupportedDevices(['Ascend910B'])
    def test_graph_mode(self):
        shape = [64, 64]
        np.random.seed(0)
        cpu_input = torch.from_numpy(np.random.randn(*shape).astype(np.float32))
        input_tensor = cpu_input.npu()

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch_npu.npu_gelu_mul(x, approximate='none')

        model = Model().npu()
        compiled_model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
        output = compiled_model(input_tensor)
        golden = self.get_golden(cpu_input)
        self.assertRtolEqual(output, golden)

    # --- Negative test cases ---

    @SupportedDevices(['Ascend910B'])
    def test_neg_input_invalid_dtype(self):
        # int64 is unsupported; use a valid-factory dtype then cast
        input_tensor = torch.randn(128, 128, dtype=torch.float32).npu().to(torch.int64)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_gelu_mul(input_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_neg_dim_below_min(self):
        # 1D input — below minimum 2D
        input_tensor = torch.randn(2, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_gelu_mul(input_tensor)

    @SupportedDevices(['Ascend910B'])
    def test_neg_dim_above_max(self):
        # 9D input — above maximum 8D
        shape = (2, 2, 2, 2, 2, 2, 2, 2, 2)
        input_tensor = torch.randn(shape, dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_gelu_mul(input_tensor)


if __name__ == "__main__":
    run_tests()
