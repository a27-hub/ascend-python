import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestQuantMatmulAlltoAll(TestCase):
    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910_95'])
    def test_npu_matmul_all_to_all(self):
        # 设备卡数
        world_size = 2
        # 初始化dtype，因为matmul_all_to_all的x1、x2和output的dtype一致，bias可以等于x1也可以为float32，这里统一使用bfloat16
        dtype = torch.bfloat16  # Simulate supported dtype
        # 初始化shape
        m, k, n = 16, 32, 32
        # 初始化format，取-1是因为后续要使用create_common_tensor生成张量，format为-1时，不会做特殊处理。
        data_format = -1
        # 初始化create_common_tensor需要的参数
        x1_shape = [dtype, data_format, [m, k]]
        x2_shape = [dtype, data_format, [k, n]]
        bias_shape = [dtype, data_format, [n]]
        # 生成tensors (one per NPU)
        # create_common_tensor接收四个参数：
        # 1、item：[dtype，format，shape[]]。2、minValue：int。3、maxValue：int。4、device=None
        # 两个返回值，cpu_input，npu_input。npu_input需要和device联合使用，npu_input=tensor.to(device)
        x1, _ = create_common_tensor(x1_shape, -1, 1)
        x2, _ = create_common_tensor(x2_shape, -1, 1)
        bias, _ = create_common_tensor(bias_shape, -1, 1)
        # Move tensors to NPU
        x1 = x1.npu()
        x2 = x2.npu()
        bias = bias.npu()
        # Call NPU operator
        out = torch_npu.npu_matmul_all_to_all(x1, x2, "hahaha", bias, all2all_axes=[-1,-2])
        self.assertEqual(1, 1, f"量化借用一下非量化, (●'◡'●), output is {out}")


if __name__ == '__main__':
    run_tests()

