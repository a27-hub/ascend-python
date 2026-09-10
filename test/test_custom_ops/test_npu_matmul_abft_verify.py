# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion, SupportedDevices


class TestNpuMatmulAbftVerify(TestCase):
    # V-ABFT GEMM fault-tolerance verification op: a[M,K] b[K,N] c[M,N]
    # (precomputed A×B, always fp32) w[N] (row-checksum encoding vector).
    # Output comp_row[ceil(M/8)*ceil(N/256)] uint8 bitstream, one bit per
    # row-segment verdict: 1 = correct, 0 = error detected.
    # Golden-reference strategy (CANN-side test convention G4): a correct C
    # means all row segments pass, so comp_row is all 255; corrupting rows of
    # C makes the corresponding bytes (8 rows per byte) non-255. The all-255 /
    # all-0 byte assertions are independent of the bit order.

    def generate_abft_inputs(self, m, n, k, dtype):
        a = (torch.randn(m, k, dtype=torch.float32) * 0.1).to(dtype)
        b = (torch.randn(k, n, dtype=torch.float32) * 0.1).to(dtype)
        w = torch.ones(n, dtype=dtype)
        # c comes from a float64 golden-reference matmul, cast to the fp32
        # required by the op contract.
        c = torch.matmul(a.to(torch.float64), b.to(torch.float64)).to(torch.float32)
        return a, b, c, w

    def cpu_op_exec(self, m, n):
        """Reference: a correct C means all row segments pass, so comp_row is all 255."""
        return np.full((m + 7) // 8 * ((n + 255) // 256), 255, dtype=np.uint8)

    def npu_op_exec(self, a, b, c, w, e_max=None):
        a, b, c, w = a.npu(), b.npu(), c.npu(), w.npu()
        if e_max is None:
            out = torch_npu._npu_matmul_abft_verify(a, b, c, w)
        else:
            out = torch_npu._npu_matmul_abft_verify(a, b, c, w, e_max=e_max)
        return out.cpu().numpy()

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_basic_fp16(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(256, 256, 256, torch.float16)
        expected = self.cpu_op_exec(256, 256)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_basic_bf16(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(256, 256, 256, torch.bfloat16)
        expected = self.cpu_op_exec(256, 256)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_basic_fp32(self, device="npu"):
        torch.manual_seed(0)
        a, b, c, w = self.generate_abft_inputs(256, 256, 256, torch.float32)
        expected = self.cpu_op_exec(256, 256)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_e_max_default(self, device="npu"):
        torch.manual_seed(0)
        # e_max omitted → the default value 0.001 is used
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_e_max_explicit_bf16(self, device="npu"):
        torch.manual_seed(0)
        # BF16 recommended value 0.001
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.bfloat16)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w, e_max=0.001)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_e_max_explicit_fp32(self, device="npu"):
        torch.manual_seed(0)
        # FP32 recommended value 0.00002
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w, e_max=0.00002)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_ceil_boundary_shape(self, device="npu"):
        torch.manual_seed(0)
        # M=33 is not a multiple of 8 and N=100 not of 256 → output [ceil(33/8)*ceil(100/256)] = [5*1] = [5].
        # With splitN=1, byte i covers rows 8i..8i+7: rows 0..31 (bytes 0..3) are full bytes, asserted all 255;
        # byte 4 has 7 padding bits (undefined value), so only its valid bits are asserted non-zero (>0).
        m, n, k = 33, 100, 67
        a, b, c, w = self.generate_abft_inputs(m, n, k, torch.float16)
        expected = self.cpu_op_exec(m, n)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertRtolEqual(expected[:4], actual[:4])
        self.assertTrue(actual[4] > 0)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_large_scale(self, device="npu"):
        torch.manual_seed(0)
        # Recommended scale for matrix ops: M, N, K >= 1024
        a, b, c, w = self.generate_abft_inputs(1024, 1024, 1024, torch.bfloat16)
        expected = self.cpu_op_exec(1024, 1024)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_error_detection(self, device="npu"):
        torch.manual_seed(0)
        # Detection case: corrupt the first 8 rows of C → rows 0..7 fail verification → byte 0 is all-zero bits, remaining bytes all 255
        m, n, k = 128, 128, 128
        a, b, c, w = self.generate_abft_inputs(m, n, k, torch.float16)
        c[0:8, :] += 50.0
        expected = self.cpu_op_exec(m, n)
        expected[0] = 0
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_output_shape_dtype(self, device="npu"):
        torch.manual_seed(0)
        # M=37, N=300 → output [ceil(37/8)*ceil(300/256)] = [5*2] = [10], dtype uint8.
        # With splitN=2 the bit order inside a byte is undefined: full bytes (rows 0..31, 4 bytes per column segment)
        # are asserted all 255; the two trailing padding bytes are only counted (non-255 bytes <= splitN=2).
        m, n, k = 37, 300, 129
        a, b, c, w = self.generate_abft_inputs(m, n, k, torch.float16)
        expected = self.cpu_op_exec(m, n)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, torch.Size([10]))
        self.assertEqual(actual.dtype, np.uint8)
        self.assertGreaterEqual(np.count_nonzero(actual == 255), actual.size - 2)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_a_invalid_dtype(self, device="npu"):
        # a/b/checksum_weight all int64 (c stays float32) → dtype outside the supported set
        a = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        b = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.randint(0, 10, (128,), dtype=torch.int64).npu()
        with self.assertRaisesRegex(RuntimeError, "only support float16/bfloat16/float32"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_b_invalid_dtype(self, device="npu"):
        # b is int64 while a is float16 → dtype mismatch
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_c_invalid_dtype(self, device="npu"):
        # c is int64 → c only supports float32
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "c only supports float32"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_checksum_weight_invalid_dtype(self, device="npu"):
        # checksum_weight is int64 while a is float16 → dtype mismatch
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.randint(0, 10, (128,), dtype=torch.int64).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_a_invalid_dim(self, device="npu"):
        # a is 3D → must be 2D
        a = torch.randn(2, 128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "a must be a 2D tensor"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_k_mismatch(self, device="npu"):
        # K dim of a (64) does not equal the K dim of b (128)
        a = torch.randn(128, 64, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must be equal to the K dim of b"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_c_shape_mismatch(self, device="npu"):
        # c shape (128, 127) ≠ [M, N]
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 127, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "c must have shape"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_checksum_weight_shape(self, device="npu"):
        # checksum_weight shape (127,) ≠ [N]
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(127, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "checksum_weight must have shape"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_dtype_linkage(self, device="npu"):
        # a is float16, b is bfloat16 → dtype linkage violation
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.bfloat16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_e_max_negative(self, device="npu"):
        # e_max is negative
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "must be non-negative"):
            torch_npu._npu_matmul_abft_verify(a.npu(), b.npu(), c.npu(), w.npu(), e_max=-0.001)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_non_contiguous(self, device="npu"):
        # non-contiguous a (sliced from (128, 256) to (128, 128) on the NPU; slicing on CPU then .npu() would not work — the H2D copy makes it contiguous again)
        a_base = torch.randn(128, 256, dtype=torch.float16).npu()
        a = a_base[:, :128]
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "only support contiguous tensors"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)


if __name__ == "__main__":
    run_tests()
