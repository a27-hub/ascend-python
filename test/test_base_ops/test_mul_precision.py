import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestMulPrecision(TestCase):
    # Discriminating pair for the 1-ulp scalar rounding issue #5392:
    # the fp32 product 0.224018394947052 * 3.796875 = 0.850569... is above the
    # bf16 rounding midpoint 0.849609375, so it must round to 0x3F5A
    # (0.8515625). Pre-rounding the scalar to bf16 gives 0.2236328125, whose
    # product 0.849105... falls below the midpoint and rounds to 0x3F59
    # (0.84765625). The scalar must keep opmath (float) precision until the
    # final cast to the output dtype.
    SCALAR = 0.224018394947052
    TENSOR_VALUE = 3.796875
    EXPECTED_BITS = 0x3F5A

    def npu_out_bits(self, out):
        return out.cpu().contiguous().view(torch.uint16).flatten().tolist()

    def make_bf16_tensor(self):
        return torch.full((2, 3), self.TENSOR_VALUE, dtype=torch.bfloat16, device="npu")

    def test_mul_cpu0dim_fp32_scalar_bf16(self):
        scalar = torch.tensor(self.SCALAR)
        tensor = self.make_bf16_tensor()
        out = scalar * tensor
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(self.npu_out_bits(out), [self.EXPECTED_BITS] * 6)

    def test_mul_fp64_scalar_opmath(self):
        scalar = torch.tensor(self.SCALAR, dtype=torch.float64)
        tensor = self.make_bf16_tensor()
        out = scalar * tensor
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(self.npu_out_bits(out), [self.EXPECTED_BITS] * 6)

    def test_mul_operand_order_consistency(self):
        scalar = torch.tensor(self.SCALAR)
        tensor = self.make_bf16_tensor()
        out_ab = self.npu_out_bits(scalar * tensor)
        out_ba = self.npu_out_bits(tensor * scalar)
        self.assertEqual(out_ab, out_ba)
        inplace = tensor.clone()
        inplace.mul_(scalar)
        self.assertEqual(out_ab, self.npu_out_bits(inplace))

    def test_mul_out_variant(self):
        scalar = torch.tensor(self.SCALAR)
        tensor = self.make_bf16_tensor()
        out = torch.empty((2, 3), dtype=torch.bfloat16, device="npu")
        torch.mul(scalar, tensor, out=out)
        self.assertEqual(self.npu_out_bits(out), [self.EXPECTED_BITS] * 6)

    def test_mul_fp32_tensor_regression(self):
        scalar = torch.tensor(self.SCALAR)
        cpu_tensor = torch.full((2, 3), self.TENSOR_VALUE, dtype=torch.float32)
        npu_tensor = torch.full((2, 3), self.TENSOR_VALUE, dtype=torch.float32, device="npu")
        expected = scalar * cpu_tensor
        out = scalar * npu_tensor
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.cpu(), expected)


if __name__ == "__main__":
    run_tests()
