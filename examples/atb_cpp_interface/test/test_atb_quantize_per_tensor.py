import torch
import torch_npu
import atb_cpp_interface
from torch_npu.testing.testcase import TestCase, run_tests


class TestAtbQuantizePerTensor(TestCase):
    def test_quantize_per_tensor(self):
        x_npu = torch.tensor(
            [
                -2.0, -1.0, -0.5, 0.0,
                0.5, 1.0, 1.5, 2.0,
                -1.5, -0.25, 0.25, 0.75,
                1.25, 1.75, 2.25, -2.25,
            ],
            dtype=torch.float16,
        ).view(1, 4, 4).npu()
        scale_npu = torch.full((4, 4), 0.5, dtype=torch.float16).npu()
        zero_point_npu = torch.ones((4, 4), dtype=torch.int8).npu()

        expected = torch.empty_like(x_npu, dtype=torch.int8)
        torch.ops.atb._npu_quantize_per_tensor(x_npu, scale_npu, zero_point_npu, expected)

        actual = atb_cpp_interface.quantize_per_tensor(x_npu, scale_npu, zero_point_npu)

        self.assertRtolEqual(actual, expected)


if __name__ == "__main__":
    run_tests()
