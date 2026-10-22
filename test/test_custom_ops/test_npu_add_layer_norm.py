import unittest
from contextlib import contextmanager
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices
torch.npu.set_compile_mode(jit_compile=False)


@contextmanager
def jit_compile_mode(enabled):
    previous = not torch.npu.is_jit_compile_false()
    try:
        torch.npu.set_compile_mode(jit_compile=enabled)
        yield
    finally:
        torch.npu.set_compile_mode(jit_compile=previous)


class AddLayerNormOutputParams:
    def __init__(self, y, mean, rstd, x):
        self.y = y
        self.mean = mean
        self.rstd = rstd
        self.x = x


class TestNPUAddLayerNorm(TestCase):
    def supported_op_exec(self, x1, x2, gamma, beta):
        if x1.dtype is not torch.float and x2.dtype is torch.float:
            x1_tensor = x1.to(torch.float)
        else:
            x1_tensor = x1
        if x1.dtype is torch.float and x2.dtype is not torch.float:
            x2_tensor = x2.to(torch.float)
        else:
            x2_tensor = x2

        epsilon = 1e-5
        x = x1_tensor + x2_tensor
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        try:
            rstd = 1 / np.sqrt(variance + epsilon)
        except ZeroDivisionError as err:
            raise err
        y = ((x - mean) * rstd) * gamma + beta
        return AddLayerNormOutputParams(y, mean, rstd, x)

    def custom_op_exec(self, x1, x2, gamma, beta):
        y, mean, rstd, x = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, 1e-5, True)
        return AddLayerNormOutputParams(y.cpu().numpy(), mean.cpu().numpy(), rstd.cpu().numpy(), x.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x1 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_x2 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [8192]).astype(np.float32)

        npu_input_x1 = torch.from_numpy(cpu_input_x1).to(device)
        npu_input_x2 = torch.from_numpy(cpu_input_x2).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)

        supported_output = self.supported_op_exec(cpu_input_x1, cpu_input_x2, cpu_input_gamma, cpu_input_beta)
        custom_output = self.custom_op_exec(npu_input_x1, npu_input_x2, npu_input_gamma, npu_input_beta)

        self.assertRtolEqual(supported_output.y, custom_output.y)
        self.assertRtolEqual(supported_output.mean, custom_output.mean)
        self.assertRtolEqual(supported_output.rstd, custom_output.rstd)
        self.assertRtolEqual(supported_output.x, custom_output.x)

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_x1x2_different_dtype(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x1 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float16)
        cpu_input_x2 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [8192]).astype(np.float32)

        npu_input_x1 = torch.from_numpy(cpu_input_x1).to(device)
        npu_input_x2 = torch.from_numpy(cpu_input_x2).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)

        supported_output = self.supported_op_exec(cpu_input_x1, cpu_input_x2, cpu_input_gamma, cpu_input_beta)
        custom_output = self.custom_op_exec(npu_input_x1, npu_input_x2, npu_input_gamma, npu_input_beta)

        self.assertEqual(supported_output.y.dtype, "float32")
        self.assertRtolEqual(supported_output.y, custom_output.y)
        self.assertRtolEqual(supported_output.mean, custom_output.mean)
        self.assertRtolEqual(supported_output.rstd, custom_output.rstd)
        self.assertRtolEqual(supported_output.x, custom_output.x)

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_optional_output(self):
        dtype_pairs = [(dtype, dtype) for dtype in (torch.float32, torch.float16, torch.bfloat16)]
        dtype_pairs += [(torch.float16, torch.float32), (torch.float32, torch.float16)]
        for shape in ((32, 128), (2, 16, 256)):
            for dtype1, dtype2 in dtype_pairs:
                with self.subTest(shape=shape, dtype1=dtype1, dtype2=dtype2):
                    x1 = torch.randn(shape, dtype=dtype1)
                    x2 = torch.randn(shape, dtype=dtype2)
                    output_dtype = dtype1 if dtype1 == dtype2 else torch.float32
                    gamma = torch.randn(shape[-1], dtype=output_dtype)
                    beta = torch.randn_like(gamma)
                    summed = x1.float() + x2.float()
                    expected_y = torch.nn.functional.layer_norm(
                        summed, (shape[-1],), gamma.float(), beta.float(), 1e-5)
                    expected_mean = summed.mean(-1, keepdim=True)
                    expected_rstd = (summed.var(-1, unbiased=False, keepdim=True) + 1e-5).rsqrt()
                    inputs = [t.npu() for t in (x1, x2, gamma, beta)]
                    outputs = []
                    for enabled in (False, True):
                        result = torch_npu.npu_add_layer_norm(*inputs, additional_output=enabled)
                        y, mean, rstd, x = result
                        tolerance = 2e-2 if output_dtype == torch.bfloat16 else 2e-3
                        torch.testing.assert_close(y.cpu().float(), expected_y, atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(mean.cpu(), expected_mean, atol=1e-4, rtol=1e-4)
                        torch.testing.assert_close(rstd.cpu(), expected_rstd, atol=1e-4, rtol=1e-4)
                        self.assertEqual(y.dtype, output_dtype)
                        self.assertEqual(x.dtype, output_dtype)
                        self.assertEqual(x.device, y.device)
                        if enabled:
                            torch.testing.assert_close(x.cpu(), summed.to(output_dtype))
                            self.assertNotEqual(x.data_ptr(), y.data_ptr())
                        else:
                            self.assertEqual(x.shape, (0,))
                            self.assertEqual(x.untyped_storage().nbytes(), 0)
                        outputs.append(result)
                    for disabled, enabled in zip(outputs[0][:3], outputs[1][:3]):
                        torch.testing.assert_close(disabled, enabled, atol=0, rtol=0)
                    # Omitting the flag must have the same allocation behavior as False.
                    self.assertEqual(torch_npu.npu_add_layer_norm(*inputs)[3].numel(), 0)

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_optional_output_noncontiguous(self):
        x1 = torch.randn(128, 32, device='npu').t()
        x2 = torch.randn_like(x1)
        gamma = torch.ones(128, device='npu')
        beta = torch.zeros_like(gamma)
        self.assertFalse(x1.is_contiguous())
        disabled = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta)
        enabled = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, additional_output=True)
        expected = torch.nn.functional.layer_norm(x1.cpu() + x2.cpu(), (128,))
        torch.testing.assert_close(disabled[0].cpu(), expected, atol=1e-4, rtol=1e-4)
        for actual, reference in zip(disabled[:3], enabled[:3]):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        self.assertEqual(disabled[3].untyped_storage().nbytes(), 0)

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_optional_output_backward(self):
        for enabled in (False, True):
            with self.subTest(additional_output=enabled):
                inputs = [torch.randn(16, 128), torch.randn(16, 128),
                          torch.randn(128), torch.randn(128)]
                cpu_inputs = [t.requires_grad_() for t in inputs]
                npu_inputs = [t.detach().npu().requires_grad_() for t in inputs]
                summed = cpu_inputs[0] + cpu_inputs[1]
                reference = torch.nn.functional.layer_norm(
                    summed, (128,), cpu_inputs[2], cpu_inputs[3], 1e-5)
                y, _, _, x = torch_npu.npu_add_layer_norm(*npu_inputs, additional_output=enabled)
                grad_y = torch.randn_like(reference)
                loss = (y * grad_y.npu()).sum()
                reference_loss = (reference * grad_y).sum()
                if enabled:
                    grad_x = torch.randn_like(summed)
                    loss = loss + (x * grad_x.npu()).sum()
                    reference_loss = reference_loss + (summed * grad_x).sum()
                else:
                    loss = loss + x.sum()
                loss.backward()
                reference_loss.backward()
                for actual, expected in zip(npu_inputs, cpu_inputs):
                    torch.testing.assert_close(actual.grad.cpu(), expected.grad, atol=1e-4, rtol=1e-4)

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_optional_output_aclops(self):
        # JIT-enabled dispatch selects aclops even when ACLNN is available.
        with jit_compile_mode(True):
            self.test_add_layer_norm_optional_output()

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_optional_output_noncontiguous_aclops(self):
        with jit_compile_mode(True):
            self.test_add_layer_norm_optional_output_noncontiguous()

    @SupportedDevices(['Ascend910B'])
    def test_add_layer_norm_optional_output_backward_aclops(self):
        with jit_compile_mode(True):
            self.test_add_layer_norm_optional_output_backward()

    def test_add_layer_norm_optional_output_meta(self):
        for dtype1, dtype2 in ((torch.float16, torch.float16),
                              (torch.float16, torch.float32), (torch.float32, torch.float16)):
            for enabled in (False, True):
                with self.subTest(dtype1=dtype1, dtype2=dtype2, additional_output=enabled):
                    output_dtype = dtype1 if dtype1 == dtype2 else torch.float32
                    x1 = torch.empty((2, 16, 128), dtype=dtype1, device='meta')
                    x2 = torch.empty_like(x1, dtype=dtype2)
                    gamma = torch.empty((128,), dtype=output_dtype, device='meta')
                    y, mean, rstd, x = torch_npu.npu_add_layer_norm(
                        x1, x2, gamma, gamma, additional_output=enabled)
                    self.assertEqual(y.shape, x1.shape)
                    self.assertEqual(mean.shape, (2, 16, 1))
                    self.assertEqual(rstd.shape, mean.shape)
                    self.assertEqual(y.dtype, output_dtype)
                    self.assertEqual(x.dtype, output_dtype)
                    self.assertEqual(x.shape, x1.shape if enabled else (0,))
                    self.assertEqual(x.device.type, 'meta')


if __name__ == "__main__":
    run_tests()
