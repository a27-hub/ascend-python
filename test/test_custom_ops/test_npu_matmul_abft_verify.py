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
    # V-ABFT GEMM 容错校验算子: a[M,K] b[K,N] c[M,N](预计算 A×B, 恒 fp32) w[N](行校验和编码向量)
    # 输出 comp_row[ceil(M/8)*ceil(N/256)] uint8 位流, 每 bit 一行一段检错结果, 1 正确 0 错误.
    # 金标准策略(CANN 侧测试 G4 约定): 正确 C → 全部行段通过 → comp_row 全 255;
    #   破坏 C 的若干行 → 对应字节(每字节 8 行)非 255. 全 255/全 0 字节断言与 bit 序无关.

    def generate_abft_inputs(self, m, n, k, dtype):
        a = (torch.randn(m, k, dtype=torch.float32) * 0.1).to(dtype)
        b = (torch.randn(k, n, dtype=torch.float32) * 0.1).to(dtype)
        w = torch.ones(n, dtype=dtype)
        # c 由 float64 金标准矩阵乘得到, 转成算子要求的 float32
        c = torch.matmul(a.to(torch.float64), b.to(torch.float64)).to(torch.float32)
        return a, b, c, w

    def cpu_op_exec(self, m, n):
        """参考实现: 正确 C 时全部行段校验通过 → comp_row 全 255."""
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
        # 不传 e_max → 使用默认值 0.001
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_e_max_explicit_bf16(self, device="npu"):
        torch.manual_seed(0)
        # BF16 推荐值 0.001
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.bfloat16)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w, e_max=0.001)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_e_max_explicit_fp32(self, device="npu"):
        torch.manual_seed(0)
        # FP32 推荐值 0.00002
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        expected = self.cpu_op_exec(128, 128)
        actual = self.npu_op_exec(a, b, c, w, e_max=0.00002)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_ceil_boundary_shape(self, device="npu"):
        torch.manual_seed(0)
        # M=33 非 8 倍数, N=100 非 256 倍数 → 输出 [ceil(33/8)*ceil(100/256)] = [5*1] = [5]
        # splitN=1 时 byte i ↔ 行 8i..8i+7: 行 0..31 (字节 0..3) 为完整字节, 值断言全 255;
        # 字节 4 含 7 个 padding bit(取值未定义), 仅断言其有效位非全 0 (>0).
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
        # Matrix ops 推荐规模 M,N,K ≥ 1024
        a, b, c, w = self.generate_abft_inputs(1024, 1024, 1024, torch.bfloat16)
        expected = self.cpu_op_exec(1024, 1024)
        actual = self.npu_op_exec(a, b, c, w)
        self.assertEqual(actual.shape, expected.shape)
        self.assertRtolEqual(expected, actual)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_error_detection(self, device="npu"):
        torch.manual_seed(0)
        # 检错场景: 破坏 C 的前 8 行 → 行 0..7 校验失败 → 字节 0 全 bit 为 0, 其余字节全 255
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
        # M=37, N=300 → 输出 [ceil(37/8)*ceil(300/256)] = [5*2] = [10], dtype uint8
        # splitN=2 时字节内 bit 序未定义: 完整字节(行 0..31, 每列分块 4 字节)值断言全 255,
        # 两个含 padding 的末字节仅计数断言 (非 255 字节数 ≤ splitN=2).
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
        # a/b/checksum_weight 均为 int64 (c 保持 float32) → dtype 不在支持范围
        a = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        b = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.randint(0, 10, (128,), dtype=torch.int64).npu()
        with self.assertRaisesRegex(RuntimeError, "only support float16/bfloat16/float32"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_b_invalid_dtype(self, device="npu"):
        # b 为 int64, a 为 float16 → dtype 不一致
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_c_invalid_dtype(self, device="npu"):
        # c 为 int64 → c 仅支持 float32
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randint(0, 10, (128, 128), dtype=torch.int64).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "c only supports float32"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_checksum_weight_invalid_dtype(self, device="npu"):
        # checksum_weight 为 int64, a 为 float16 → dtype 不一致
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.randint(0, 10, (128,), dtype=torch.int64).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_a_invalid_dim(self, device="npu"):
        # a 为 3D → 必须 2 维
        a = torch.randn(2, 128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "a must be a 2D tensor"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_k_mismatch(self, device="npu"):
        # a 的 K(64) 与 b 的 K(128) 不相等
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
        # a float16, b bfloat16 → dtype 联动违规
        a = torch.randn(128, 128, dtype=torch.float16).npu()
        b = torch.randn(128, 128, dtype=torch.bfloat16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "must have the same dtype"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_e_max_negative(self, device="npu"):
        # e_max 为负数
        a, b, c, w = self.generate_abft_inputs(128, 128, 128, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "must be non-negative"):
            torch_npu._npu_matmul_abft_verify(a.npu(), b.npu(), c.npu(), w.npu(), e_max=-0.001)

    @SkipIfNotGteCANNVersion("9.2.0")
    @SupportedDevices(['Ascend910B'])
    def test_neg_non_contiguous(self, device="npu"):
        # a 非连续 (NPU 上由 (128, 256) 切片得到 (128, 128); 不能先切片再 .npu() — H2D 拷贝会重新连续化)
        a_base = torch.randn(128, 256, dtype=torch.float16).npu()
        a = a_base[:, :128]
        b = torch.randn(128, 128, dtype=torch.float16).npu()
        c = torch.randn(128, 128, dtype=torch.float32).npu()
        w = torch.ones(128, dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, "only support contiguous tensors"):
            torch_npu._npu_matmul_abft_verify(a, b, c, w)


if __name__ == "__main__":
    run_tests()
