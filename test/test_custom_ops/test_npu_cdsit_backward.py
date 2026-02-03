import unittest

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUCdistBackward(TestCase):


    def compare(self, a, b, benchmark):
        diff_abs = torch.abs(a - b)
        max_diff_abs = diff_abs.max()

        if max_diff_abs.item() == 0:
            return True

        rel_error = 0
        abs_error = 0
        for i in range(a.numel()):
            ai = a[i].item()
            bi = b[i].item()
            diff = diff_abs[i].item()

            if ai == 0 and bi == 0:
                diff_rel_item = 0
            elif ai == 0 or bi == 0:
                diff_rel_item = 1
            else:
                diff_rel_item = diff / abs(ai)

            if abs(ai) < 1 and diff > benchmark:
                abs_error += 1
            elif abs(ai) >= 1 and diff_rel_item > benchmark:
                rel_error += 1

            if (rel_error + abs_error) > 10:
                break

        return (rel_error + abs_error) == 0


    def npu_cdist_backward_golden(self, grad, x1, x2, cdist, p):
        return torch.ops.aten._cdist_backward(
            grad.to(torch.float32),
            x1.to(torch.float32),
            x2.to(torch.float32),
            p,
            cdist.to(torch.float32)
        ).to(grad.dtype)


    @unittest.skip("skip until CANN is updated to support aclnnCdistBackward")
    @SupportedDevices(['Ascend910B'])
    def test_npu_cdist_backward(self):
        shape_list = [
            (4, 5, 3),
            (2, 6, 8),
        ]
        p = 1.7633918456382829
        benchmark = 1e-3

        for m, n, d in shape_list:
            grad = torch.randn((m, n), dtype=torch.float32)
            x1 = torch.randn((m, d), dtype=torch.float32)
            x2 = torch.randn((n, d), dtype=torch.float32)
            cdist = torch.rand((m, n), dtype=torch.float32) + 1e-6
            dx1_ref = self.npu_cdist_backward_golden(
                grad, x1, x2, cdist, p
            )
            grad_npu = grad.npu()
            x1_npu = x1.npu()
            x2_npu = x2.npu()
            cdist_npu = cdist.npu()
            dx1_npu = torch_npu.npu_cdist_backward(
                grad_npu, x1_npu, x2_npu, cdist_npu, p
            )
            dx1_ref_flat = dx1_ref.reshape(-1).cpu()
            dx1_npu_flat = dx1_npu.reshape(-1).cpu()
            self.assertTrue(
                self.compare(dx1_ref_flat, dx1_npu_flat, benchmark)
            )


    @unittest.skip("skip until CANN is updated to support aclnnCdistBackward")
    @SupportedDevices(['Ascend910B'])
    def test_npu_cdist_backward_fp16(self):
        shape_list = [
            (4, 5, 3),
            (2, 6, 8),
        ]
        p = 1.7633918456382829
        benchmark = 1e-3

        for m, n, d in shape_list:
            grad = torch.randn((m, n), dtype=torch.float16)
            x1 = torch.randn((m, d), dtype=torch.float16)
            x2 = torch.randn((n, d), dtype=torch.float16)
            cdist = torch.rand((m, n), dtype=torch.float16) + 1e-6
            dx1_ref = self.npu_cdist_backward_golden(
                grad, x1, x2, cdist, p
            )
            grad_npu = grad.npu()
            x1_npu = x1.npu()
            x2_npu = x2.npu()
            cdist_npu = cdist.npu()
            dx1_npu = torch_npu.npu_cdist_backward(
                grad_npu, x1_npu, x2_npu, cdist_npu, p
            )
            dx1_ref_flat = dx1_ref.reshape(-1).cpu()
            dx1_npu_flat = dx1_npu.reshape(-1).cpu()
            self.assertTrue(
                self.compare(dx1_ref_flat, dx1_npu_flat, benchmark)
            )


if __name__ == "__main__":
    run_tests()