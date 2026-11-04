import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestBatchNormStats(TestCase):
    def cuda_op_exec(self, *args):
        cpu_mean, cpu_invstd = torch.batch_norm_stats(*args)
        return cpu_mean.numpy(), cpu_invstd.numpy()

    def cuda_expect_result(self):
        cpu_output0 = np.array([5.401827, 5.444219, 5.7656665], dtype=np.float32)
        cpu_output1 = np.array([0.37123242, 0.38706362, 0.37435925], dtype=np.float32)
        return cpu_output0, cpu_output1

    def npu_op_exec(self, *args):
        npu_mean, npu_invstd = torch.batch_norm_stats(*args)
        out_mean = npu_mean.cpu().numpy()
        out_invstd = npu_invstd.cpu().numpy()
        return out_mean, out_invstd

    def test_batch_norm_stats(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float16, -1, [2, 3, 12, 12]], 1e-5],
        ]
        for item in shape_format:
            cpu_input1, npu_inputfp16 = create_common_tensor(item[0], 1, 10)
            npu_input1fp32 = npu_inputfp16.float()
            if torch.cuda.is_available():
                cpu_output = self.cuda_op_exec(cpu_input1.cuda(), item[-1])
            else:
                cpu_output = self.cuda_expect_result()
            npu_outputfp16 = self.npu_op_exec(npu_inputfp16, item[-1])
            npu_outputfp32 = self.npu_op_exec(npu_input1fp32, item[-1])

            self.assertRtolEqual(cpu_output[0], npu_outputfp16[0])
            self.assertRtolEqual(cpu_output[1], npu_outputfp16[1], 1e-2)

            self.assertRtolEqual(cpu_output[0], npu_outputfp32[0])
            self.assertRtolEqual(cpu_output[1], npu_outputfp32[1], 1e-2)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_batch_norm_stats_bfloat16(self):
        # Compare statistics of the values actually representable in BF16, using
        # FP32 population variance. CPU batch_norm_stats itself has no CPU kernel.
        generator = torch.Generator().manual_seed(2026)
        shapes = [(7, 3), (2, 5, 11), (3, 7, 9, 13), (2, 3, 4, 5, 7)]
        for shape in shapes:
            for layout in ('contiguous', 'strided'):
                for eps in (1e-5, 1e-3):
                    for constant in (False, True):
                        with self.subTest(shape=shape, layout=layout, eps=eps, constant=constant):
                            host = torch.randn(shape, generator=generator) * 3 + 2
                            if constant:
                                host.fill_(1.25)
                            host = host.to(torch.bfloat16)
                            if layout == 'strided':
                                backing = torch.empty((*shape[:-1], shape[-1] * 2), dtype=host.dtype)
                                backing[..., ::2].copy_(host)
                                host = backing[..., ::2]
                            value = host.npu()
                            if layout == 'strided':
                                backing_npu = torch.empty((*shape[:-1], shape[-1] * 2),
                                                          dtype=host.dtype, device=value.device)
                                backing_npu[..., ::2].copy_(value)
                                value = backing_npu[..., ::2]
                                self.assertFalse(value.is_contiguous())
                            original = value.clone()
                            dimensions = tuple(i for i in range(len(shape)) if i != 1)
                            variance, expected_mean = torch.var_mean(host.float(), dim=dimensions, correction=0)
                            expected_invstd = torch.rsqrt(variance + eps)

                            mean, invstd = torch.batch_norm_stats(value, eps)

                            self.assertEqual(mean.dtype, torch.float32)
                            self.assertEqual(invstd.dtype, torch.float32)
                            self.assertEqual(mean.device, value.device)
                            self.assertEqual(invstd.device, value.device)
                            self.assertEqual(tuple(mean.shape), (shape[1],))
                            self.assertEqual(tuple(invstd.shape), (shape[1],))
                            self.assertRtolEqual(expected_mean.numpy(), mean.cpu().numpy(), 1e-4)
                            self.assertRtolEqual(expected_invstd.numpy(), invstd.cpu().numpy(), 1e-4)
                            self.assertEqual(value, original)


if __name__ == "__main__":
    run_tests()
