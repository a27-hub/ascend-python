import copy
import os
import subprocess
import sys
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion


def _check_lazy_fusion_modes():
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    torch.npu.set_device(0)
    checked = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for transposed in (False, True):
            x = (torch.randn(17, 31) * 2).to(dtype)
            dy = torch.randn_like(x)
            if transposed:
                x, dy = x.t(), dy.t()
            for mode in ("none", "tanh"):
                xn = x.npu().detach().requires_grad_(True)
                yn = torch.nn.functional.gelu(xn, approximate=mode)
                yn.backward(dy.npu())
                y = yn.detach().cpu().double()
                dx = xn.grad.cpu().double()
                expected_y = torch.nn.functional.gelu(x.double(), approximate=mode)
                expected_dx = torch.ops.aten.gelu_backward.default(
                    dy.double(), x.double(), approximate=mode)
                tolerance = {torch.float32: 1e-5 if mode == "tanh" else 1e-6,
                             torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
                torch.testing.assert_close(y, expected_y, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(dx, expected_dx, rtol=tolerance, atol=tolerance)
                checked += 1
    x = torch.tensor([2.0], device="npu")
    dy = torch.ones_like(x)
    a = torch.ops.aten.gelu_backward.default(dy, x, approximate="none").cpu()
    b = torch.ops.aten.gelu_backward.default(dy, x, approximate="tanh").cpu()
    assert abs(float(a.item() - b.item())) > 5e-4
    for call in (lambda: torch.nn.functional.gelu(x, approximate="invalid"),
                 lambda: torch.ops.aten.gelu_backward.default(dy, x, approximate="invalid")):
        try:
            call()
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid GELU approximation was silently accepted")
    print(f"CROSS_SOC_GELU_PASS cases={checked}")


class TestGeluBackward(TestCase):
    def generate_single_data(self, min_val, max_val, shape, dtype):
        input1 = np.random.uniform(min_val, max_val, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1):
        input1.requires_grad_(True)
        output = torch.nn.functional.gelu(input1)
        z = output.sum()
        z.backward()
        res = input1.grad
        return res.detach().numpy()

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        input1.requires_grad = True
        output = torch.nn.functional.gelu(input1)
        z = output.sum()
        z.backward()
        res = input1.grad.to("cpu")
        return res.detach().numpy()

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float32_1(self):
        input1 = self.generate_single_data(0, 100, (4, 3, 1, 1), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float32_2(self):
        input1 = self.generate_single_data(0, 100, (15, 3, 1), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float32_3(self):
        input1 = self.generate_single_data(0, 100, (4, 4), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float16(self):
        input1 = self.generate_single_data(0, 100, (5, 10, 100), np.float16)
        cpu_input1 = input1.to(torch.float32)
        cpu_output = self.cpu_op_exec(cpu_input1)
        cpu_output = cpu_output.astype(np.float16)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def get_golden(self, grad_tensor, input_self_tensor, approximate="none"):
        output = torch.ops.aten.gelu_backward(grad_tensor, input_self_tensor, approximate=approximate)
        return output

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float32_none(self):
        shape = [100, 400]
        mode = "none"
        input_self_tensor = torch.rand(shape, dtype=torch.float32).npu()
        grad_tensor = torch.rand(shape, dtype=torch.float32).npu()
        output = torch.ops.aten.gelu_backward(grad_tensor, input_self_tensor, approximate=mode)
        golden = self.get_golden(grad_tensor.cpu(), input_self_tensor.cpu(), mode)
        self.assertRtolEqual(output, golden)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float32_tanh(self):
        shape = [5, 10, 100]
        mode = "none"
        input_self_tensor = torch.rand(shape, dtype=torch.float32).npu()
        grad_tensor = torch.rand(shape, dtype=torch.float32).npu()
        output = torch.ops.aten.gelu_backward(grad_tensor, input_self_tensor, approximate=mode)
        golden = self.get_golden(grad_tensor.cpu(), input_self_tensor.cpu(), mode)
        self.assertRtolEqual(output, golden)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float16_none(self):
        shape = [4, 4]
        mode = "none"
        input_self_tensor = torch.rand(shape, dtype=torch.float16).npu()
        grad_tensor = torch.rand(shape, dtype=torch.float16).npu()
        output = torch.ops.aten.gelu_backward(grad_tensor, input_self_tensor, approximate=mode)
        golden = self.get_golden(grad_tensor.cpu(), input_self_tensor.cpu(), mode)
        self.assertRtolEqual(output, golden)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_gelu_backward_float16_tanh(self):
        shape = [4, 3, 1, 1]
        mode = "tanh"
        input_self_tensor = torch.rand(shape, dtype=torch.float16).npu()
        grad_tensor = torch.rand(shape, dtype=torch.float16).npu()
        output = torch.ops.aten.gelu_backward(grad_tensor, input_self_tensor, approximate=mode)
        golden = self.get_golden(grad_tensor.cpu(), input_self_tensor.cpu(), mode)
        self.assertRtolEqual(output, golden)


    def test_lazy_fusion_approximate_modes(self):
        if not torch.npu.get_device_name(0).startswith(("Ascend910B", "Ascend910_93", "Ascend950")):
            self.skipTest("No-graph fusion is not supported on this device")
        code = ("import runpy; module = runpy.run_path(" + repr(os.path.abspath(__file__))
                + "); module['_check_lazy_fusion_modes']()")
        # Process isolation is required: the fusion switch and compatible mode
        # are cached when their operators first run.
        for fusion, compatible in (("False", "1"), ("True", "0"), ("True", "1")):
            with self.subTest(fusion=fusion, compatible=compatible):
                env = os.environ.copy()
                env["TORCH_NPU_LAZY_FUSION"] = (
                    "True enable_ops_only=gelu,gelu_backward" if fusion == "True" else "False")
                env["TORCH_NPU_USE_COMPATIBLE_IMPL"] = compatible
                result = subprocess.run([sys.executable, "-c", code], cwd="/tmp", env=env,
                                        capture_output=True, text=True, timeout=600)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CROSS_SOC_GELU_PASS cases=12", result.stdout)


if __name__ == "__main__":
    np.random.seed(20)
    torch_npu.npu.use_compatible_impl(True)
    run_tests()
