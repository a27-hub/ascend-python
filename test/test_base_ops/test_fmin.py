import itertools

import torch
import torch_npu  # noqa: F401 - register the NPU backend
from torch_npu.testing.testcase import TestCase, run_tests


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
DTYPES = FLOAT_DTYPES + INT_DTYPES + (torch.bool,)


def make_input(shape, dtype, generator):
    if dtype.is_floating_point:
        value = torch.randn(shape, generator=generator).to(dtype)
        if value.numel():
            value.reshape(-1)[0] = float("nan")
        return value
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=generator).bool()
    low = 0 if dtype == torch.uint8 else -100
    return torch.randint(low, 100, shape, generator=generator).to(dtype)


class TestFmin(TestCase):
    def assert_result(self, actual, expected):
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)

    def test_special_values(self):
        nan, inf = float("nan"), float("inf")
        for dtype in FLOAT_DTYPES:
            with self.subTest(dtype=dtype):
                info = torch.finfo(dtype)
                a = torch.tensor([nan, 1, nan, 3, inf, -inf, info.max, info.min, 0, -0.0], dtype=dtype)
                b = torch.tensor([2, nan, nan, -1, -inf, inf, info.min, info.max, -0.0, 0], dtype=dtype)
                self.assert_result(torch.fmin(a.npu(), b.npu()), torch.fmin(a, b))
                self.assert_result(torch.fmin(b.npu(), a.npu()), torch.fmin(b, a))

    def test_subnormal_values(self):
        for dtype in FLOAT_DTYPES:
            with self.subTest(dtype=dtype):
                info = torch.finfo(dtype)
                small = info.tiny * info.eps
                a = torch.tensor([small, -small, info.tiny, -info.tiny], dtype=dtype)
                b = torch.zeros(4, dtype=dtype)
                self.assert_result(torch.fmin(a.npu(), b.npu()), torch.fmin(a, b))
                self.assert_result(torch.fmin(b.npu(), a.npu()), torch.fmin(b, a))

    def test_dtype_shape_generalization(self):
        # 120 reproducible cases per dtype, including scalar, empty and broadcast.
        generator = torch.Generator().manual_seed(20261105)
        shapes = [((), ()), ((0, 3), (1, 3)), ((3, 1), (1, 5)), ((2, 1, 4), (1, 3, 4))]
        for dtype in DTYPES:
            for index in range(120):
                with self.subTest(dtype=dtype, case=index):
                    if index < len(shapes):
                        shape_a, shape_b = shapes[index]
                    else:
                        shape_a = (index % 11 + 1, index % 7 + 1)
                        shape_b = shape_a if index % 2 else (1, shape_a[1])
                    a = make_input(shape_a, dtype, generator)
                    b = make_input(shape_b, dtype, generator)
                    self.assert_result(torch.fmin(a.npu(), b.npu()), torch.fmin(a, b))

    def test_mixed_dtype_promotion(self):
        generator = torch.Generator().manual_seed(3)
        for dtype_a, dtype_b in itertools.product(DTYPES, repeat=2):
            with self.subTest(self_dtype=dtype_a, other_dtype=dtype_b):
                a = make_input((3, 1), dtype_a, generator)
                b = make_input((1, 5), dtype_b, generator)
                self.assert_result(torch.fmin(a.npu(), b.npu()), torch.fmin(a, b))

    def test_integer_extremes(self):
        for dtype in INT_DTYPES:
            with self.subTest(dtype=dtype):
                info = torch.iinfo(dtype)
                a = torch.tensor([info.min, info.max, 1, 0], dtype=dtype)
                b = a.flip(0)
                self.assert_result(torch.fmin(a.npu(), b.npu()), torch.fmin(a, b))
        a = torch.tensor([2**53 + 1, 2**53 + 3, -(2**53 + 1)], dtype=torch.int64)
        b = torch.tensor([2**53 + 2, 2**53 + 2, -(2**53 + 2)], dtype=torch.int64)
        self.assert_result(torch.fmin(a.npu(), b.npu()), torch.fmin(a, b))

    def test_non_contiguous_inputs(self):
        for view in (lambda t: t.t(), lambda t: t[::2]):
            a, b = torch.randn(8, 6), torch.randn(8, 6)
            npu_a, npu_b = view(a.npu()), view(b.npu())
            self.assertFalse(npu_a.is_contiguous())
            self.assert_result(torch.fmin(npu_a, npu_b), torch.fmin(view(a), view(b)))
        base = torch.randn(2, 4, 5, 3)
        a = base.permute(0, 3, 1, 2)
        npu_a = base.npu().permute(0, 3, 1, 2)
        self.assertTrue(npu_a.is_contiguous(memory_format=torch.channels_last))
        self.assertFalse(npu_a.is_contiguous())
        self.assert_result(torch.fmin(npu_a, npu_a), torch.fmin(a, a))

    def test_api_variants_and_cpu_scalar(self):
        a = torch.tensor([1, float("nan"), -2], dtype=torch.float16)
        npu_a = a.npu()
        scalar = torch.tensor(0.5, dtype=torch.float64)
        self.assert_result(npu_a.fmin(scalar), torch.fmin(a, scalar))
        self.assert_result(torch.fmin(scalar, npu_a), torch.fmin(scalar, a))
        self.assert_result(torch.ops.aten.fmin.default(npu_a, npu_a), torch.fmin(a, a))

    def test_out_variants(self):
        a = torch.tensor([[float("nan")], [2.0]])
        b = torch.tensor([[1.0, float("nan"), -3.0]])
        expected = torch.fmin(a, b)
        npu_a, npu_b = a.npu(), b.npu()
        for shape, dtype in [((2, 3), torch.float32), ((0,), torch.float32), ((2, 3), torch.float64)]:
            with self.subTest(shape=shape, dtype=dtype):
                out = torch.empty(shape, dtype=dtype, device="npu")
                returned = torch.fmin(npu_a, npu_b, out=out)
                self.assertIs(returned, out)
                self.assert_result(out, expected.to(dtype))
        out = torch.empty(2, 6, device="npu")[:, ::2]
        self.assertFalse(out.is_contiguous())
        self.assert_result(torch.fmin(npu_a, npu_b, out=out), expected)
        out = torch.empty(1, device="npu")
        self.assertWarns(lambda: torch.fmin(npu_a, npu_b, out=out))
        self.assert_result(out, expected)

    def test_out_aliases(self):
        a = torch.tensor([float("nan"), 2.0, -1.0])
        b = torch.tensor([1.0, float("nan"), 3.0])
        expected = torch.fmin(a, b)
        for index in (0, 1):
            inputs = [a.npu(), b.npu()]
            out = inputs[index]
            self.assertIs(torch.fmin(*inputs, out=out), out)
            self.assert_result(out, expected)

    def test_out_casts(self):
        a, b = torch.tensor([1, 3]), torch.tensor([2, -1])
        for dtype in (torch.int32, torch.float32, torch.complex64):
            out = torch.empty(2, dtype=dtype, device="npu")
            expected = torch.empty(2, dtype=dtype)
            torch.fmin(a, b, out=expected)
            self.assert_result(torch.fmin(a.npu(), b.npu(), out=out), expected)
        for input_dtype, out_dtype in [(torch.int32, torch.bool), (torch.float32, torch.int32)]:
            with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                torch.fmin(
                    a.to(input_dtype).npu(), b.to(input_dtype).npu(), out=torch.empty(2, dtype=out_dtype, device="npu")
                )

    def test_invalid_inputs(self):
        for dtype in (torch.complex64, torch.complex128):
            # Construct on CPU so an unsupported complex fill cannot mask the fmin error.
            a = torch.ones(2, dtype=dtype).npu()
            with self.assertRaisesRegex(TypeError, "fmin not implemented for complex"):
                torch.fmin(a, a)
            with self.assertRaisesRegex(TypeError, "fmin not implemented for complex"):
                torch.fmin(a, a, out=torch.empty_like(a))
        a = torch.ones(3, device="npu")
        with self.assertRaises(RuntimeError):
            torch.fmin(a, torch.ones(3))
        with self.assertRaises(RuntimeError):
            torch.fmin(a, a, out=torch.empty(3))
        with self.assertRaises(RuntimeError):
            torch.fmin(a, torch.ones(4, device="npu"))

    def test_output_overlap(self):
        base = torch.arange(5.0, device="npu")
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            torch.fmin(base[:4], torch.ones(4, device="npu"), out=base[1:])
        out = torch.empty(1, device="npu").expand(4)
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            torch.fmin(torch.ones(4, device="npu"), torch.ones(4, device="npu"), out=out)

    def test_backward(self):
        a = torch.tensor([[float("nan")], [2.0], [1.0]], requires_grad=True)
        b = torch.tensor([[1.0, float("nan"), 2.0]], requires_grad=True)
        npu_a = a.detach().npu().requires_grad_()
        npu_b = b.detach().npu().requires_grad_()
        torch.fmin(a, b).sum().backward()
        torch.fmin(npu_a, npu_b).sum().backward()
        self.assert_result(npu_a.grad, a.grad)
        self.assert_result(npu_b.grad, b.grad)
        with self.assertRaisesRegex(RuntimeError, "automatic differentiation"):
            torch.fmin(npu_a, npu_b, out=torch.empty(3, 3, device="npu"))

    def test_float64(self):
        a = torch.tensor([1 + 2**-41, 1 + 2**-40, -(1 + 2**-40), float("nan"), -float("inf")], dtype=torch.float64)
        b = torch.tensor([1 + 2**-40, 1 + 2**-41, -(1 + 2**-41), 2.0, float("nan")], dtype=torch.float64)
        self.assert_result(torch.fmin(a.npu(), b.npu()), torch.fmin(a, b))


if __name__ == "__main__":
    run_tests()
