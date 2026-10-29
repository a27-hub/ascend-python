import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


def bits16(out):
    # bitwise compare: pre/post fix differ by 1 ulp, rtol cannot distinguish
    return out.cpu().contiguous().view(torch.uint16).flatten().tolist()


def fp64_expected(scalar_value, tensor_value, out_dtype):
    # expected result computed from fp64 math instead of hardcoded bits
    return (torch.tensor(scalar_value, dtype=torch.float64)
            * torch.tensor(tensor_value, dtype=torch.float64)).to(out_dtype)


class TestMulPrecision(TestCase):
    def test_mul_half_scalar_opmath_precision(self):
        # These scalars expose early Half rounding, including scalar underflow
        # and overflow. Compare exactly: a tolerance can hide a one-ULP error.
        cpu = torch.tensor([0.125, 0.33325, 0.7, 1.0, 1.5, -1.5, 10.0], dtype=torch.float16)
        for scalar in [0.1, 0.1234567, 1.0 / 3.0, 1e-8, 65536.0]:
            with self.subTest(scalar=scalar):
                expected = cpu * scalar
                actual = cpu.npu() * scalar
                torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_mul_half_cpu_scalar_and_out_precision(self):
        # The Tensor overload and out= must preserve the same scalar precision
        # as mul.Scalar when the right operand is a CPU zero-dimensional tensor.
        cpu = torch.tensor([1.5, -1.5, 0.7], dtype=torch.float16)
        scalar = torch.tensor(0.1, dtype=torch.float32)
        expected = cpu * scalar
        actual = torch.mul(cpu.npu(), scalar)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        out = torch.empty_like(cpu, device="npu")
        returned = torch.mul(cpu.npu(), scalar, out=out)
        self.assertIs(returned, out)
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

    def test_mul_cpu0dim_fp32_scalar_bf16(self):
        # issue #5392: CPU 0-dim fp32 scalar times bf16 tensor on NPU
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        out = a * b
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, -3.7969, torch.bfloat16)))

    def test_mul_cpu0dim_fp64_scalar_bf16_opmath(self):
        # fp64 scalar also takes the float opmath path, aligned with native
        a = torch.tensor(0.22401839494705, dtype=torch.float64)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        out = a * b
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, -3.7969, torch.bfloat16)))

    def test_mul_operand_order_consistency(self):
        # a*b, b*a and mul_ must be bitwise identical
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        self.assertEqual(bits16(a * b), bits16(b * a))
        inplace = b.clone()
        inplace.mul_(a)
        self.assertEqual(bits16(a * b), bits16(inplace))

    def test_mul_out_variant(self):
        # out= variant goes through mul_out, keep scalar at opmath precision
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        out = torch.empty(1, dtype=torch.bfloat16, device="npu")
        torch.mul(a, b, out=out)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, -3.7969, torch.bfloat16)))

    def test_mul_cpu0dim_fp32_scalar_fp16(self):
        # fp16 guard: fp32 path product and fp16 pre-rounded product fall on different
        # sides of the fp16 rounding midpoint (0x3353 vs 0x3352)
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([1.021484375], dtype=torch.float16).npu()
        out = a * b
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, 1.021484375, torch.float16)))

    def test_mul_fp32_tensor_regression(self):
        # fp32 output does not hit the fix branch, must stay unchanged
        a = torch.tensor(0.22401839494705)
        b_cpu = torch.tensor([-3.7969], dtype=torch.float32)
        b_npu = torch.tensor([-3.7969], dtype=torch.float32).npu()
        self.assertEqual((a * b_npu).cpu(), a * b_cpu)


if __name__ == "__main__":
    run_tests()
