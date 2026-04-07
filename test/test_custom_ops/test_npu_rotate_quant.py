import unittest
import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuRotateQuant(TestCase):
    def rotate_quant(self, x: torch.Tensor, rotation: torch.Tensor, alpha: float, dst_dtype: int):
        """
        执行旋转和量化的操作。

        参数:
            x (torch.Tensor): 输入张量，形状为 (m, n)。
            rotation (torch.Tensor): 旋转矩阵，形状为 (k, k)，k为4的幂次方。
            alpha (float): 旋转角度缩放因子。
            dst_dtype (int): 目标数据类型，2表示int8，29表示int4。

        返回:
            quantOutput (torch.Tensor): 量化后的输出张量，形状为 (m, n)。
            quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (m,)。
        """
        # 执行旋转变换
        m, n = x.shape
        k = rotation.shape[0]

        # 只对前k列进行旋转变换
        x_rot = x[:, :k] @ rotation.T * (1.0 + alpha)

        # 拼接未变换的部分（如果n > k）
        if n > k:
            x_rot = torch.cat([x_rot, x[:, k:]], dim=1)

        # 对输出进行量化
        abs_max = torch.max(torch.abs(x_rot), -1).values  # 找到最大绝对值以计算缩放因子
        if dst_dtype == 2:  # int8
            quantScaleOutput = 127.0 / (abs_max + 1e-6)  # 计算量化缩放因子
            quantOutput = torch.round(x_rot * quantScaleOutput.reshape(m, 1)).to(torch.int8)  # 量化为 int8
        elif dst_dtype == 29:  # int4
            quantScaleOutput = 7.0 / (abs_max + 1e-6)  # 计算量化缩放因子
            quantOutput = torch.round(x_rot * quantScaleOutput.reshape(m, 1)).to(torch.int8)  # 量化为 int4（用int8存储）
        else:
            raise ValueError(f"Unsupported dst_dtype: {dst_dtype}")

        quantScaleOutput = 1.0 / quantScaleOutput  # 反向量化缩放因子以便后续反量化

        return quantOutput, quantScaleOutput

    def gen_input_data(self, M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        # 归一化旋转矩阵
        rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-6)
        return x, rotation

    @unittest.skip("skip case")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_int8(self, device="npu"):
        # 生成数据
        M = 512
        N = 1024
        K = 1024
        dst_dtype = 2  # int8
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x, rotation, alpha=0.0, dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), alpha=0.0, dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.cpu(), 1)
        self.assertRtolEqual(output1, output1_npu.cpu())

    @unittest.skip("skip case")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_int4(self, device="npu"):
        # 生成数据
        M = 512
        N = 1024
        K = 1024
        dst_dtype = 29  # int4
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x, rotation, alpha=0.0, dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), alpha=0.0, dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.cpu(), 1)
        self.assertRtolEqual(output1, output1_npu.cpu())

    @unittest.skip("skip case")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rotate_quant_with_alpha(self, device="npu"):
        # 生成数据，测试alpha参数
        M = 256
        N = 512
        K = 256
        dst_dtype = 2  # int8
        alpha = 0.5
        x, rotation = self.gen_input_data(M, N, K)
        output0, output1 = self.rotate_quant(x, rotation, alpha=alpha, dst_dtype=dst_dtype)
        output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), alpha=alpha, dst_dtype=dst_dtype)
        self.assertEqual(output0, output0_npu.cpu(), 1)
        self.assertRtolEqual(output1, output1_npu.cpu())

if __name__ == "__main__":
    run_tests()
