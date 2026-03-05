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

    def test_histc_bins_invalid(self):
        npu_input = torch.tensor([1.0], dtype=torch.float32).npu()
        msg = "bins must be > 0"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.histc(npu_input, bins=0)
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.histc(npu_input, bins=-1)

    def test_histc_half_not_supported(self):
        npu_input = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float16).npu()
        msg = "HalfTensor is not supported"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.histc(npu_input, bins=4, min=0, max=3)

    def test_histc_min_max_invalid(self):
        npu_input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).npu()
        msg = "max must be larger than min"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.histc(npu_input, bins=4, min=5, max=1)

    def test_histc_min_max_equal_non_empty(self):
        cpu_input = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = torch.histc(cpu_input, bins=4, min=5, max=5)
        npu_output = torch.histc(npu_input, bins=4, min=5, max=5)
        expected = torch.tensor([2, 0, 0, 1], dtype=torch.float32)
        self.assertRtolEqual(expected, cpu_output)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_min_max_equal_empty(self):
        cpu_input = torch.tensor([], dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = torch.histc(cpu_input, bins=5, min=0, max=0)
        npu_output = torch.histc(npu_input, bins=5, min=0, max=0)
        expected = torch.zeros(5, dtype=torch.float32)
        self.assertRtolEqual(expected, cpu_output)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_non_finite_range_error(self):
        msg = r"range of \[.*\] is not finite"
        npu_inf = torch.tensor([float("inf")], dtype=torch.float32).npu()
        npu_nan = torch.tensor([float("nan")], dtype=torch.float32).npu()
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.histc(npu_inf)
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.histc(npu_nan)

    def test_histc_non_finite_values_with_finite_min_max(self):
        inputs = [
            torch.tensor([1.0, 2.0, float("inf")], dtype=torch.float32),
            torch.tensor([1.0, 2.0, float("nan")], dtype=torch.float32),
        ]
        for cpu_input in inputs:
            npu_input = cpu_input.npu()
            cpu_output = torch.histc(cpu_input, bins=4, min=0, max=3)
            npu_output = torch.histc(npu_input, bins=4, min=0, max=3)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_out_check_align(self):
        cpu_input = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_out = torch.zeros(4, dtype=torch.float32)
        npu_out = cpu_out.npu()

        with self.assertRaisesRegex(RuntimeError, "bins must be > 0"):
            torch.histc(npu_input, bins=0, out=npu_out)

        with self.assertRaisesRegex(RuntimeError, "max must be larger than min"):
            torch.histc(npu_input, bins=4, min=5, max=1, out=npu_out)

        torch.histc(cpu_input, bins=4, min=5, max=5, out=cpu_out)
        torch.histc(npu_input, bins=4, min=5, max=5, out=npu_out)
        expected = torch.tensor([2, 0, 0, 1], dtype=torch.float32)
        self.assertRtolEqual(expected, cpu_out)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
