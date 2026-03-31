import os
import unittest

import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_npu.testing.testcase import TestCase, run_tests


class TestMatmulBackwardMeta(TestCase):
    @staticmethod
    def _randn(shape, dtype, requires_grad=False):
        if len(shape) == 0:
            return torch.randn((), dtype=dtype, device="npu", requires_grad=requires_grad)
        return torch.randn(shape, dtype=dtype, device="npu", requires_grad=requires_grad)

    def _run_fake_matmul_backward(self, grad_shape, mat1_shape, mat2_shape, grad_dtype, mat1_dtype, mat2_dtype):
        with FakeTensorMode():
            grad = self._randn(grad_shape, grad_dtype)
            mat1 = self._randn(mat1_shape, mat1_dtype, requires_grad=True)
            mat2 = self._randn(mat2_shape, mat2_dtype, requires_grad=True)
            return torch.ops.aten.matmul_backward.default(grad, mat1, mat2, [True, True])

    def _assert_fake_backward_matches_real(self, mat1_shape, mat2_shape, dtype):
        mat1 = self._randn(mat1_shape, dtype, requires_grad=True)
        mat2 = self._randn(mat2_shape, dtype, requires_grad=True)
        output = torch.matmul(mat1, mat2)
        grad = torch.randn(output.size(), dtype=output.dtype, device="npu")
        output.backward(grad)

        with FakeTensorMode():
            fake_mat1 = self._randn(mat1_shape, dtype, requires_grad=True)
            fake_mat2 = self._randn(mat2_shape, dtype, requires_grad=True)
            fake_output = torch.matmul(fake_mat1, fake_mat2)
            fake_grad = torch.randn(fake_output.size(), dtype=fake_output.dtype, device="npu")
            fake_output.backward(fake_grad)

        self.assertEqual(mat1.grad.shape, fake_mat1.grad.shape)
        self.assertEqual(mat1.grad.dtype, fake_mat1.grad.dtype)
        self.assertEqual(mat2.grad.shape, fake_mat2.grad.shape)
        self.assertEqual(mat2.grad.dtype, fake_mat2.grad.dtype)

    @unittest.skipIf(
        os.getenv("TORCH_NPU_USE_COMPATIBLE_IMPL") == "1",
        "matmul_backward meta registration is disabled in compatible impl",
    )
    def test_matmul_backward_meta_shapes_and_dtypes(self):
        test_cases = [
            {
                "name": "vector_vector",
                "grad_shape": (),
                "mat1_shape": (8,),
                "mat2_shape": (8,),
                "grad_dtype": torch.float32,
                "mat1_dtype": torch.float16,
                "mat2_dtype": torch.float16,
                "expected_self_shape": (1, 8),
                "expected_other_shape": (8,),
                "expected_self_dtype": torch.float32,
                "expected_other_dtype": torch.float16,
            },
            {
                "name": "vector_matrix",
                "grad_shape": (6,),
                "mat1_shape": (5,),
                "mat2_shape": (5, 6),
                "grad_dtype": torch.float32,
                "mat1_dtype": torch.float16,
                "mat2_dtype": torch.float16,
                "expected_self_shape": (1, 5),
                "expected_other_shape": (5, 6),
                "expected_self_dtype": torch.float32,
                "expected_other_dtype": torch.float16,
            },
            {
                "name": "matrix_vector_squeeze_other_grad",
                "grad_shape": (3,),
                "mat1_shape": (3, 5),
                "mat2_shape": (5,),
                "grad_dtype": torch.float32,
                "mat1_dtype": torch.float32,
                "mat2_dtype": torch.float16,
                "expected_self_shape": (3, 5),
                "expected_other_shape": (5,),
                "expected_self_dtype": torch.float32,
                "expected_other_dtype": torch.float32,
            },
            {
                "name": "matrix_batched_matrix_special_self_grad",
                "grad_shape": (2, 3, 4),
                "mat1_shape": (3, 5),
                "mat2_shape": (2, 5, 4),
                "grad_dtype": torch.float16,
                "mat1_dtype": torch.float16,
                "mat2_dtype": torch.float16,
                "expected_self_shape": (3, 5),
                "expected_other_shape": (2, 5, 4),
                "expected_self_dtype": torch.float16,
                "expected_other_dtype": torch.float16,
            },
            {
                "name": "batched_matrix_matrix_special_other_grad",
                "grad_shape": (2, 3, 6),
                "mat1_shape": (2, 3, 5),
                "mat2_shape": (5, 6),
                "grad_dtype": torch.float16,
                "mat1_dtype": torch.float16,
                "mat2_dtype": torch.float16,
                "expected_self_shape": (2, 3, 5),
                "expected_other_shape": (5, 6),
                "expected_self_dtype": torch.float16,
                "expected_other_dtype": torch.float16,
            },
            {
                "name": "leading_singleton_self_batch",
                "grad_shape": (1, 4, 3, 6),
                "mat1_shape": (1, 4, 3, 5),
                "mat2_shape": (5, 6),
                "grad_dtype": torch.float32,
                "mat1_dtype": torch.float16,
                "mat2_dtype": torch.float16,
                "expected_self_shape": (1, 4, 3, 5),
                "expected_other_shape": (5, 6),
                "expected_self_dtype": torch.float32,
                "expected_other_dtype": torch.float16,
            },
            {
                "name": "leading_singleton_other_batch",
                "grad_shape": (1, 4, 3, 6),
                "mat1_shape": (3, 5),
                "mat2_shape": (1, 4, 5, 6),
                "grad_dtype": torch.float32,
                "mat1_dtype": torch.float32,
                "mat2_dtype": torch.float16,
                "expected_self_shape": (3, 5),
                "expected_other_shape": (1, 4, 5, 6),
                "expected_self_dtype": torch.float32,
                "expected_other_dtype": torch.float32,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                self_grad, other_grad = self._run_fake_matmul_backward(
                    case["grad_shape"],
                    case["mat1_shape"],
                    case["mat2_shape"],
                    case["grad_dtype"],
                    case["mat1_dtype"],
                    case["mat2_dtype"],
                )
                self.assertEqual(self_grad.shape, torch.Size(case["expected_self_shape"]))
                self.assertEqual(other_grad.shape, torch.Size(case["expected_other_shape"]))
                self.assertEqual(self_grad.dtype, case["expected_self_dtype"])
                self.assertEqual(other_grad.dtype, case["expected_other_dtype"])

    @unittest.skipIf(
        os.getenv("TORCH_NPU_USE_COMPATIBLE_IMPL") == "1",
        "matmul_backward meta registration is disabled in compatible impl",
    )
    def test_matmul_backward_fake_autograd_matches_real_grad_meta(self):
        test_cases = [
            ((8,), (8, 4), torch.float16),
            ((4, 8), (2, 8, 6), torch.float16),
            ((1, 4, 3, 5), (5, 6), torch.float32),
        ]

        for mat1_shape, mat2_shape, dtype in test_cases:
            with self.subTest(mat1_shape=mat1_shape, mat2_shape=mat2_shape, dtype=dtype):
                self._assert_fake_backward_matches_real(mat1_shape, mat2_shape, dtype)


if __name__ == "__main__":
    run_tests()
