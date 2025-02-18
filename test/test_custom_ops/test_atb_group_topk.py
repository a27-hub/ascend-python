import os
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAtbGroupTopk(TestCase):

    @SupportedDevices(['Ascend910B'])
    def test_atb_group_topk(self, device="npu"):
        input1 = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).npu().half()
        out = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]).npu().half()
        torch.ops.atb.npu_group_topk(input1, group_num=2, k=1, n=1, out=out)
        expected_out = torch.tensor([[0.0000, 0.0000, 0.3000, 0.3999],
                                     [0.0000, 0.0000, 0.7002, 0.7998]]).half()
        self.assertRtolEqual(expected_out, input1)


if __name__ == "__main__":
    run_tests()
