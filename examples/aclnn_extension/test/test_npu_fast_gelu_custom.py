import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

import aclnn_extension


class TestFastGelu(TestCase):

    # compare with npu_fast_gelu
    def supported_op_exec(self, input1):
        output = torch.ops.npu.npu_fast_gelu(input1)
        return output.cpu().detach()

    def custom_op_exec(self, input1):
        output = torch.ops.npu.npu_fast_gelu_custom(input1)
        return output.cpu().detach()

    def test_npu_fast_gelu_custom(self, device="npu"):
        item = [np.float32, 0, [3, 16, 32]]
        _, npu_input = create_common_tensor(item, 0, 100)

        supported_output = self.supported_op_exec(npu_input)
        custom_output = self.custom_op_exec(npu_input)
        self.assertRtolEqual(supported_output, custom_output)

    def test_nested_to_padded_tensor(self, device="npu"):
        nested = torch.nested.nested_tensor(
            [
                torch.tensor([1.0, 2.0]),
                torch.tensor([3.0]),
            ],
            dtype=torch.float32,
            device=device,
        )
        output = torch.nested.to_padded_tensor(nested, -1.0)
        expected = torch.tensor([[1.0, 2.0], [3.0, -1.0]])

        self.assertRtolEqual(output.cpu(), expected)


if __name__ == "__main__":
    run_tests()
