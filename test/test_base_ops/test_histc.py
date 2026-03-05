import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHistc(TestCase):
    def test_histc(self):
        case_list = [
            [np.float32, -1, [2]],
            [np.float32, -1, [2]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4, 5]],
            [np.float32, -1, [2, 3, 4, 5]],
        ]
        for item in case_list:
            for bins in (10, 2048):
                cpu_input, npu_input = create_common_tensor(item, -2, 2)
                cpu_output = torch.histc(cpu_input, bins, -2, 2)
                npu_output = torch.histc(npu_input, bins, -2, 2)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_out(self):
        case_list = [
            [np.float32, -1, [2]],
            [np.float32, -1, [2]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4]],
            [np.float32, -1, [2, 3, 4, 5]],
            [np.float32, -1, [2, 3, 4, 5]],
        ]
        for item in case_list:
            for bins in (10, 2048):
                cpu_input, npu_input = create_common_tensor(item, -2, 2)
                cpu_output = torch.from_numpy(np.zeros([bins], dtype=item[0]))
                npu_output = cpu_output.npu()
                torch.histc(cpu_input, bins, -2, 2, out=cpu_output)
                torch.histc(npu_input, bins, -2, 2, out=npu_output)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_check_logic(self):
        device = "npu"

        # negative nbins throws
        with self.assertRaisesRegex(RuntimeError, 'bins must be > 0'):
            torch.histc(torch.tensor([1], dtype=torch.float, device=device), bins=-1)

        # empty tensor
        actual = torch.histc(torch.tensor([], dtype=torch.float, device=device), min=0, max=3)
        expected = torch.zeros(100, dtype=torch.float, device=device)
        self.assertEqual(expected, actual)

        # without nbins
        actual = torch.histc(torch.tensor([2, 5], dtype=torch.float, device=device))
        expected = torch.zeros(100, dtype=torch.float, device=device)
        expected[0] = 1
        expected[99] = 1
        self.assertEqual(expected, actual)

        # tensor with the same element
        actual = torch.histc(torch.ones(5, dtype=torch.float, device=device), bins=5)
        self.assertEqual(torch.tensor([0, 0, 5, 0, 0], dtype=torch.float, device=device), actual)

        # no element falls between [min, max]
        actual = torch.histc(torch.ones(5, dtype=torch.float, device=device), bins=5, min=2, max=3)
        self.assertEqual(torch.tensor([0, 0, 0, 0, 0], dtype=torch.float, device=device), actual)

        # element falls below min + integral bin size
        actual = torch.histc(torch.tensor([2, 4, 2, 2, 5, 4], dtype=torch.float, device=device), bins=5, min=1, max=5)
        self.assertEqual(torch.tensor([0, 3, 0, 2, 1], dtype=torch.float, device=device), actual)

        # non-integral bin size
        actual = torch.histc(torch.tensor([1, 2, 1], dtype=torch.float, device=device), bins=4, min=0, max=3)
        self.assertEqual(torch.tensor([0, 2, 1, 0], dtype=torch.float, device=device), actual)

        # scalar input and 1 bin -- should return a 1-dimensional tensor, not a scalar.
        actual = torch.histc(torch.tensor(0, dtype=torch.float, device=device), bins=1, min=0, max=3)
        self.assertEqual(torch.tensor([1], dtype=torch.float, device=device), actual)

        # tensors with inf; min, max not provided -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([float("-inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([float("-inf"), float("inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([1., 2., float("inf")], dtype=torch.float, device=device))

        # tensors with inf; min, max provided
        self.assertEqual(torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device), bins=1, min=0, max=3), torch.tensor([0], dtype=torch.float, device=device))
        self.assertEqual(torch.histc(torch.tensor([1., 2., float("inf")], dtype=torch.float, device=device), bins=4, max=3), torch.tensor([0, 1, 1, 0], dtype=torch.float, device=device))

        # tensor with nan; min, max not provided -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, r'range of \[nan, nan\] is not finite'):
            torch.histc(torch.tensor([float("nan")], dtype=torch.float, device=device))

        # tensor with nan; min, max provided -- nan is ignored
        self.assertEqual(torch.histc(torch.tensor([1., 2., float("nan")], dtype=torch.float, device=device), bins=4, max=3), torch.tensor([0, 1, 1, 0], dtype=torch.float, device=device))

        # tensors with min > max -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, "max must be larger than min"):
            torch.histc(torch.tensor([1., 2., 3.], dtype=torch.float, device=device), bins=4, min=5, max=1)


if __name__ == "__main__":
    run_tests()
