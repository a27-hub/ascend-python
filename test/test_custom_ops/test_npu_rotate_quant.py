import unittest
import math
import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuRotateQuant(TestCase):
    """
    torch_npu.npu_rotate_quant 接口测试用例。

    Atlas A2（Ascend910B）平台看护场景：
        覆盖 int8（torch.int8）和 int4（torch.quint4x2）两种量化输出类型。
        输入 x 为 2 维 bf16 张量，shape 为 [512, 1024]；rotation 为 [1024, 1024] 的归一化矩阵。
        标杆实现通过 conv_rot 完成分块旋转（torch.matmul），再调用 npu_dynamic_quant 完成量化，
        与被测 npu_rotate_quant 算子的输出进行精度对比。int8 用例采用 assertEqual（容差1）比较
        量化结果、assertRtolEqual 比较 scale；int4 用例采用 compare_tensor_nibbles 逐半字节比较
        量化结果（差值绝对值 ≤ 1）、assertRtolEqual 比较 scale。

    Atlas A5（Ascend950）平台看护场景：
        覆盖 mxfp4（torch_npu.float4_e2m1fn_x2）、fp8_e5m2（torch.float8_e5m2）、
        fp8_e4m3fn（torch.float8_e4m3fn）三种 MX 类型量化输出。
        输入 x 为 bf16 张量，rotation 为 [k, k] 的归一化矩阵，k 分别取 128、64、32，
        覆盖 A5 平台支持的全部 rotation 尺寸；其中 fp8_e5m2 用例将 x reshape 为 3 维 [4, 64, 512]，
        体现 A5 平台支持 1~7 维输入的约束。标杆实现通过 block_rotation（使用 out_dtype=torch.float
        的 mm/bmm）完成旋转、npu_dynamic_mx_quant 完成量化，与被测算子输出进行 bit-exact 对比
        （assertEqual），同时看护量化结果和 scale。
    """

    def compare_tensor_nibbles(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """
        比较两个形状相同的张量，每个元素是 int32，由 8 个 int4 拼成。
        要求对应位置的每个半字节（视为有符号 int4）差值绝对值 ≤ 1。
        """
        if a.shape != b.shape:
            return False

        # 转换为无符号 32 位整数（用 int64 防止溢出）
        a_uint = a.to(torch.int64) & 0xFFFFFFFF
        b_uint = b.to(torch.int64) & 0xFFFFFFFF

        # 准备所有移位偏移（从高位到低位：28, 24, 20, ..., 0）
        shifts = torch.tensor([28 - 4 * i for i in range(8)], dtype=torch.int64, device=a.device)

        # 提取所有半字节：a_nibbles shape = (8, *a.shape)，每个值为 0~15
        a_nibbles = (a_uint.unsqueeze(0) >> shifts.view(-1, *([1] * a.dim()))) & 0xF
        b_nibbles = (b_uint.unsqueeze(0) >> shifts.view(-1, *([1] * b.dim()))) & 0xF

        # 将无符号半字节转换为有符号 int4（范围 -8 ~ 7）
        a_int4 = torch.where(a_nibbles >= 8, a_nibbles - 16, a_nibbles)
        b_int4 = torch.where(b_nibbles >= 8, b_nibbles - 16, b_nibbles)

        # 逐半字节比较差值绝对值
        diff = torch.abs(a_int4 - b_int4)
        mask = diff > 1

        if mask.any():
            # 获取所有错误位置的索引
            indices = torch.nonzero(mask, as_tuple=True)
            nibble_indices = indices[0]          # 半字节位置 (0~7)
            tensor_indices = indices[1:]         # 每个维度的坐标

            print("Found mismatches (|diff| > 1) with int4 interpretation:")
            for i in range(len(nibble_indices)):
                nibble_pos = nibble_indices[i].item()
                tensor_pos = tuple(idx[i].item() for idx in tensor_indices)
                a_val = a_int4[(nibble_pos,) + tensor_pos].item()
                b_val = b_int4[(nibble_pos,) + tensor_pos].item()
                diff_val = abs(a_val - b_val)
                print(f"  - nibble index {nibble_pos}, tensor index {tensor_pos}: a={a_val}, b={b_val}, diff={diff_val}")
            return False
        else:
            return True

    def conv_rot(self,  input_matrix, rot_matrix):
        stride = rot_matrix.shape[0]
        h, w = input_matrix.shape
        num_blocks = w // stride
        input_matrix = input_matrix.view(h, num_blocks, stride).reshape(-1, stride)
        out = torch.matmul(input_matrix, rot_matrix)
        mat_rot = out.reshape(h, w)
        return mat_rot

    def rotate_quant(self, x, rot_matrix, dst_dtype):
        x_rot = self.conv_rot(x, rot_matrix)
        xdtype = x.dtype
        torch_npu.npu.synchronize()
        x_rot_int8, x_rot_scale = torch_npu.npu_dynamic_quant(x_rot.to(xdtype), dst_type=dst_dtype)
        return x_rot_int8, x_rot_scale

    def gen_input_data(self, M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        # 归一化旋转矩阵
        rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-6)
        return x, rotation

    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_int8(self, device="npu"):
        # 生成数据
        M = 512
        N = 1024
        K = 1024
        dst_dtype = torch.int8  # int8
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.cpu(), 1)
        self.assertRtolEqual(output1, output1_npu.cpu())

    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_int4(self, device="npu"):
        # 生成数据
        M = 512
        N = 1024
        K = 1024
        dst_dtype = torch.quint4x2  # int4
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.compare_tensor_nibbles(output0.cpu(), output0_npu.cpu())
        self.assertRtolEqual(output1, output1_npu.cpu())

    def block_rotation(self, x, rotation, block_size, axis):
        blockNum = x.shape[axis] // block_size
        origin_shape = x.shape

        if (len(rotation.shape) == 2):
            x_r = torch.mm(x.reshape(-1, block_size), rotation, out_dtype=torch.float)
        else:
            x = x.reshape(-1, blockNum, block_size).permute(1, 0, 2)
            x_r = torch.bmm(x, rotation, out_dtype=torch.float).permute(1, 0, 2)
        return x_r.reshape(origin_shape).to(torch.bfloat16)

    def npu_rotate_quant_golden(self, x, rotation, alpha, axis, round_mode, scale_alg, dst_type_max, dst_dtype):
        # 此标杆需在torch_npu >= 2.8环境运行（因为out_dtype入参需要2.8+才支持）
        blockSize = rotation.shape[-1]
        x_r = self.block_rotation(x, rotation, blockSize, axis)
        npu_dst_type = {
            23: torch.float8_e5m2,
            24: torch.float8_e4m3fn,
            296: torch_npu.float4_e2m1fn_x2,
        }.get(dst_dtype, torch.float8_e5m2)

        if alpha is not None and alpha.numel() > 0 and alpha[0] > 0 and alpha[0] < 1:
            x_c = x_r.view(-1, 32)
            groupMaxVal = x_c.abs().amax(dim=-1, keepdim=True)
            limit = alpha[0] * groupMaxVal
            x_c = torch.clamp(x_c, min=-limit, max=limit)
            x_r = x_c.view(x_r.shape)

        out, mxscale = torch_npu.npu_dynamic_mx_quant(
            x_r, axis=axis, round_mode=round_mode, dst_type=npu_dst_type,
            block_size=32, scale_alg=scale_alg, dst_type_max=dst_type_max
        )
        return out.to(torch.float), mxscale.to(torch.float)

    @SupportedDevices(['Ascend950'])
    def test_npu_rotate_quant_fp4(self, device="npu"):
        # 生成数据
        M = 512
        N = 1024
        K = 128
        dst_dtype = torch_npu.float4_e2m1fn_x2  # fp4，枚举值296
        x, rotation = self.gen_input_data(M, N, K)
        # 标杆：使用 out_dtype=torch.float 的 mm + npu_dynamic_mx_quant
        output0, output1 = self.npu_rotate_quant_golden(
            x.npu(), rotation.npu(), alpha=None, axis=-1,
            round_mode="rint", scale_alg=0, dst_type_max=0.0,
            dst_dtype=int(dst_dtype) if isinstance(dst_dtype, torch.dtype) else dst_dtype
        )
        # 被测算子
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.to(torch.float))
        self.assertEqual(output1, output1_npu.view(torch.uint8).to(torch.float))

    @SupportedDevices(['Ascend950'])
    def test_npu_rotate_quant_fp8_e5m2(self, device="npu"):
        # 生成数据：3维输入，体现A5平台支持1~7维
        M = 256
        N = 512
        K = 64
        dst_dtype = torch.float8_e5m2  # fp8_e5m2，枚举值23
        x, rotation = self.gen_input_data(M, N, K)
        x = x.view(4, 64, N)  # reshape为3维，最后一维512可被K=64整除
        # 标杆：使用 out_dtype=torch.float 的 mm + npu_dynamic_mx_quant
        output0, output1 = self.npu_rotate_quant_golden(
            x.npu(), rotation.npu(), alpha=None, axis=-1,
            round_mode="rint", scale_alg=0, dst_type_max=0.0,
            dst_dtype=int(dst_dtype) if isinstance(dst_dtype, torch.dtype) else dst_dtype
        )
        # 被测算子
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.to(torch.float))
        self.assertEqual(output1, output1_npu.view(torch.uint8).to(torch.float))

    @SupportedDevices(['Ascend950'])
    def test_npu_rotate_quant_fp8_e4m3fn(self, device="npu"):
        # 生成数据：K=32，最小支持的rotation尺寸
        M = 128
        N = 256
        K = 32
        dst_dtype = torch.float8_e4m3fn  # fp8_e4m3fn，枚举值24
        x, rotation = self.gen_input_data(M, N, K)
        # 标杆：使用 out_dtype=torch.float 的 mm + npu_dynamic_mx_quant
        output0, output1 = self.npu_rotate_quant_golden(
            x.npu(), rotation.npu(), alpha=None, axis=-1,
            round_mode="rint", scale_alg=0, dst_type_max=0.0,
            dst_dtype=int(dst_dtype) if isinstance(dst_dtype, torch.dtype) else dst_dtype
        )
        # 被测算子
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.to(torch.float))
        self.assertEqual(output1, output1_npu.view(torch.uint8).to(torch.float))

if __name__ == "__main__":
    run_tests()
