import unittest
import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuGroupedMatmulSwigluQuant(TestCase):
    def GMM_Swiglu_quant(self, x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, m: int):
        """
        执行量化的 GMM（通用矩阵乘法）操作，并使用 SwiGLU 激活函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (m, n)。
            weight (torch.Tensor): 权重张量，形状为 (n, k)。
            perChannelScale (torch.Tensor): 每个通道的缩放因子，形状为 (k,)。
            perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (m,)。
            m (int): token 的数量（x 的行数）。

        返回:
            quantOutput (torch.Tensor): 量化后的输出张量，形状为 (m, k // 2)。
            quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (m,)。
        """
        # 使用 int32 精度执行矩阵乘法
        c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
        c_temp1 = c_temp1.to(torch.float32)  # 转换回 float32 以便进行缩放

        # 应用每个通道和每个 token 的缩放
        c_temp2 = torch.mul(c_temp1, perChannelScale)
        c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))

        # 将结果分成两部分以应用 SwiGLU 激活函数
        c_temp4, gate = c_temp3.chunk(2, dim=-1)
        c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
        c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘

        # 对输出进行量化
        abs_max = torch.max(torch.abs(c_temp6), -1).values  # 找到最大绝对值以计算缩放因子
        quantScaleOutput = 127 / abs_max  # 计算量化缩放因子
        quantOutput = torch.round(c_temp6 * quantScaleOutput.reshape(m, 1)).to(torch.int8)  # 量化为 int8
        quantScaleOutput = 1 / quantScaleOutput  # 反向量化缩放因子以便后续反量化

        return quantOutput, quantScaleOutput


    def process_groups(self, x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, groupList: torch.Tensor):
        """
        按组处理输入数据，并调用 GMM_Swiglu_quant 函数进行量化计算。

        参数:
            x (torch.Tensor): 输入张量，形状为 (M, N)。
            weight (torch.Tensor): 权重张量列表，每个元素的形状为 (n, k)。
            perChannelScale (torch.Tensor): 每个通道的缩放因子列表，每个元素的形状为 (k,)。
            perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (M,)。
            groupList (list): 定义每个组的 token 数量的列表。

        返回:
            quantOutput (torch.Tensor): 量化后的输出张量，形状为 (M, N // 2)。
            quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (M,)。
        """
        M, N = x.shape[0], weight.shape[2]  # 获取输入张量的形状
        quantOutput = torch.zeros(M, N // 2).to(torch.int8)  # 初始化量化输出张量
        quantScaleOutput = torch.zeros(M).to(torch.float32)  # 初始化量化缩放因子张量

        start_idx = 0  # 起始索引
        preV = 0  # 前一个组的 token 数量
        groupList = groupList.tolist()
        # 遍历 groupList，按组处理数据
        for i, v in enumerate(groupList):
            currV = v
            tempV = currV - preV  # 计算当前组的 token 数量
            preV = currV  # 更新前一个组的 token 数量
            if (tempV > 0):
            # 调用 GMM_Swiglu_quant 处理当前组
                quantOutput[start_idx:start_idx + tempV], quantScaleOutput[start_idx:start_idx + tempV] = \
                    self.GMM_Swiglu_quant(x[start_idx:start_idx + tempV], 
                                    weight[i], 
                                    perChannelScale[i], 
                                    perTokenScale[start_idx:start_idx + tempV], 
                                    tempV)

            start_idx += tempV  # 更新起始索引以处理下一组
        return quantOutput, quantScaleOutput

    def generate_non_decreasing_sequence(self, length, upper_limit):
        """
        生成一个随机非减的一维 Tensor，且最后一个值小于上限。

        参数:
            length (int): 序列的长度。
            upper_limit (int): 最后一个值的上限。

        返回:
            torch.Tensor: 生成的一维 Tensor。
        """
        # 生成随机递增序列
        random_increments = torch.randint(0, 128, (length,))  # 随机增量，范围 0~9
        sequence = torch.cumsum(random_increments, dim=0)  # 累加生成非减序列

        # 确保最后一个值小于上限
        if sequence[-1] >= upper_limit:
            scale_factor = upper_limit / sequence[-1]  # 计算缩放因子
            sequence = (sequence * scale_factor).to(torch.int64)  # 缩放并转换为整数

        return sequence

    def gen_input_data(self, E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = self.generate_non_decreasing_sequence(E, M)
        # pylint:disable=too-many-return-values
        return x, weight, weightScale, xScale, groupList

    @unittest.skip("skip case")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant(self, device="npu"):
        # 生成数据
        E = 16
        M = 512
        K = 7168
        N = 4096
        torch.npu.config.allow_internal_format = True
        x, weight, weightScale, xScale, groupList = self.gen_input_data(E, M, K, N)
        output0, output1 = self.process_groups(x, weight, weightScale, xScale, groupList)
        # 注：有效数据截至到groupList[-1] 即output0[:groupList[-1],:],output0[:groupList[-1]]
        output0_valid = output0[:groupList[-1], :]
        output1_valid = output1[:groupList[-1]]
        weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
        output0_npu, output1_npu, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(x.npu(), weight_npu, groupList.npu(), weightScale.npu(), xScale.npu())
        output0_npu_valid = output0_npu[:groupList[-1], :]
        output1_npu_valid = output1_npu[:groupList[-1]]
        self.assertEqual(output0_valid, output0_npu_valid.cpu(), 1)
        self.assertRtolEqual(output1_valid, output1_npu_valid.cpu())

    @unittest.skip("skip case")
    @SupportedDevices(['Ascend910B'])
    def test_npu_grouped_matmul_swiglu_quant_A8W4(self):
        def x_int8_to_x_int4(x: torch.Tensor):
            m, k = x.shape
            x_high_4bit = torch.floor(x.to(torch.float16) // 16).to(torch.int8)
            x_low_4bit = (torch.bitwise_and(x.view(torch.int16), 0x0f0f).view(torch.int8) - 8)
            x_int4 = torch.empty((2 * m, k), dtype=torch.int8)
            x_int4[::2, :] = x_high_4bit
            x_int4[1::2, :] = x_low_4bit
            return x_int4

        def custom_mm(x: torch.Tensor, weight: torch.Tensor, weightScale: torch.Tensor, m: int):
            """
            执行量化的 GMM（通用矩阵乘法）操作

            参数:
                x (torch.Tensor): 输入张量，形状为 (m, k)。
                weight (torch.Tensor): 权重张量，形状为 (k, n)。
                weightScale (torch.Tensor): 每个通道的缩放因子，
                 - perGroup 场景: 形状为 (k_group_num, n)。注：当k_group_num==1时为perChannel场景
                 - perChannel 场景: 形状为 (n)。
                perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (m,)。
                m (int): token 的数量（x 的行数）。

            返回:
                MMOut(fp16): MatMul + perGroup或perChannel反量化结果
            """
            # 使用 int32 精度执行矩阵乘法
            k, n = weight.shape
            mm_out = torch.zeros((m, n), dtype=torch.float16)
            # perGroup 场景
            if len(weightScale.shape) == 2 and weightScale.shape[0] != 1:
                k_group = weightScale.shape[0]
                per_group_ele = k // k_group
                x_grouped = x.view(-1, k_group, per_group_ele).transpose(0, 1)
                weight_grouped = weight.view(k_group, per_group_ele, n)

                c_temp = np.bmm(x_grouped.to(torch.int32), weight_grouped.to(torch.int32)).to(torch.float16)
                for k_idx in range(k_group):
                    mm_out += (c_temp[k_idx] * weightScale[k_idx].view(1, -1).to(torch.float16)).to(torch.float16)
            # perChannel 场景
            elif len(weightScale.shape) == 1 or (len(weightScale.shape) == 2 and weightScale.shape[0] == 1):
                c_temp = torch.matmul(x.to(torch.int32), weight.to(torch.int32)).to(torch.float32)
                mm_out = c_temp * weightScale.view(1, -1).to(torch.float16)
            return mm_out.to(torch.float32)

        def gmmSwiglu_quant_golden_A8W4(x: torch.Tensor, weight: torch.Tensor, weightScale: torch.Tensor,
                                        perTokenScale: torch.Tensor, bias: torch.Tensor,
                                        groupList: torch.Tensor):
            """
            按组处理输入数据，并调用 GMM_Swiglu_quant 函数进行量化计算。

            参数:
                x (torch.Tensor): 输入张量，形状为 (M, K) INT8。
                weight (torch.Tensor): 权重张量列表，每个元素的形状为 (E, K, N)数据类型 INT8 但数据范围为INT4, 实际代表INT4。
                weightScale (torch.Tensor): 每个通道的缩放因子，
                 - perGroup 场景: 形状为 (E, k_group_num, N)。
                 - perChannel 场景: 形状为 (E, N)。
                perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (M,)。
                groupList (list): 定义每个组的 token 数量的列表。

            返回:
                quantOutput (torch.Tensor): 量化后的输出张量，形状为 (M, N // 2)。
                quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (M,)。
            """
            M, N = x.shape[0], weight.shape[2]  # 获取输入张量的形状
            quantOutput = torch.zeros(M, N // 2).to(torch.int8)  # 初始化量化输出张量
            quantScaleOutput = torch.zeros(M).to(torch.float32)  # 初始化量化缩放因子张量
            # 前处理 X_INT8 -> X_INT4
            x_int4 = x_int8_to_x_int4(x)
            start_idx = 0  # 起始索引
            preV = 0  # 前一个组的 token 数量
            groupList = groupList.tolist()
            # 遍历 groupList，按组处理数据
            for i, v in enumerate(groupList):
                currV = v
                tempV = int((currV - preV) * 2)  # 计算当前组的 token 数量 “ * 2 ”是因为 1行Int8--> 2行Int4
                preV = currV  # 更新前一个组的 token 数量
                if (tempV > 0):
                    # 调用 MM_Swiglu_quant 处理当前组

                    mm_out = custom_mm(x_int4[int(start_idx): int(start_idx + tempV)],
                                      weight[i],
                                      weightScale[i],
                                      tempV)
                    mm_num_concat = ((mm_out[::2] * 16 + mm_out[1::2]) + bias[i].view(1, -1))

                    per_token_quant = mm_num_concat * perTokenScale[start_idx // 2: (start_idx + tempV) // 2].view(-1, 1)
                    # 将结果分成两部分以应用 SwiGLU 激活函数
                    swiglu, gate = per_token_quant.chunk(2, dim=-1)

                    temp = swiglu * torch.sigmoid(swiglu)  # SwiGLU 激活
                    temp = temp * gate  # 与门控值进行逐元素相乘
                    # 对输出进行量化
                    max_value = torch.max(torch.abs(temp), dim=-1).values  # 找到最大绝对值以计算缩放因子
                    quantScaleOutput_temp = 127 / max_value  # 计算量化缩放因子
                    quantOutput[start_idx // 2: (start_idx + tempV) // 2] = torch.round(
                        temp * quantScaleOutput_temp.reshape(tempV // 2, 1)).to(torch.int8)  # 量化为 int8
                    quantScaleOutput[
                    start_idx // 2: (start_idx + tempV) // 2] = 1 / quantScaleOutput_temp  # 反向量化缩放因子以便后续反量化
                start_idx += tempV  # 更新起始索引以处理下一组
            return quantOutput, quantScaleOutput

        # 生成数据
        E = 16
        M = 512
        K = 7168
        N = 4096
        torch.npu.config.allow_internal_format = True
        x = torch.randint(-5, 5, (M, K), dtype=torch.int8).npu()
        weight_ori = torch.randint(-5, 5, (E, K, N), dtype=torch.int8)
        weight_nz = torch_npu.npu_format_cast(weight_ori.npu().to(torch.float32), 29)
        pack_weight = torch_npu.npu_quantize(weight_nz, torch.tensor([1.], device='npu'), None, torch.quint4x2, -1,
                                             False)

        weightScale = torch.randn(E, 1, N)
        scale_np = weightScale.cpu().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        pertoken_scale = torch.randn(M).to(torch.float32).npu()
        groupList = self.generate_non_decreasing_sequence(E, M).npu()
        bias = torch.zeros((E, N), dtype=torch.float32, device="npu").uniform_(-5, 5)

        output_golden, output_scale_golden = gmmSwiglu_quant_golden_A8W4(x.cpu(), weight_ori, weightScale,
                                                                         pertoken_scale.cpu(), bias.cpu(),
                                                                         groupList.cpu())

        output, output_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
            x=x,
            weight=pack_weight,
            bias=bias,
            group_list=groupList,
            weight_scale=scale_uint64_tensor,
            x_scale=pertoken_scale)
        self.assertEqual(output_golden, output.cpu(), 1)
        self.assertRtolEqual(output_scale_golden, output_scale.cpu(), 0.005)

if __name__ == "__main__":
    run_tests()
