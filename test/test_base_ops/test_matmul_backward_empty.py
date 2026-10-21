import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests

import torch_npu
import torch_npu.testing  # noqa: F401


class TestMatmulBackwardEmpty(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("shapes", [
        ((0,), (0,)),
        ((5, 0), (0,)),
        ((0,), (0, 5)),
        ((5, 0), (0, 7)),
        ((0, 5), (5, 7)),
        ((5, 7), (7, 0)),
        ((5, 5, 0), (0, 5)),
        ((5, 0), (5, 0, 5)),
        ((0, 5, 7), (7, 3)),
        ((3, 5), (0, 5, 7)),
        ((0, 0), (5, 0, 0)),
        ((0, 5, 7), (1, 7, 3)),
        ((1, 3, 5), (0, 5, 7)),
    ])
    @parametrize("mask", [(True, True), (True, False), (False, True), (False, False)])
    def test_empty(self, device, dtype, shapes, mask):
        references = tuple(torch.randn(shape, dtype=dtype, requires_grad=True) for shape in shapes)
        output = references[0] @ references[1]
        grad = torch.randn_like(output)
        expected = torch.autograd.grad(output, references, grad_outputs=grad)
        inputs = tuple(value.detach().to(device) for value in references)
        versions = tuple(value._version for value in inputs)
        actual = torch.ops.aten.matmul_backward.default(grad.to(device), *inputs, mask)
        self.assertEqual(actual, tuple(value.to(device) if enabled else None for value, enabled in zip(expected, mask)))
        self.assertEqual(tuple(value._version for value in inputs), versions)


instantiate_device_type_tests(TestMatmulBackwardEmpty, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
