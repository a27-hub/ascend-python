# import os
# import unittest
#
# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch_npu
#
# from torch_npu.testing.testcase import TestCase, run_tests
# from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
# from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
#
#
# class TestMatmulAlltoAll(TestCase):
#     @classmethod
#     def _init_dist_hccl(cls, rank, world_size):
#         os.environ['MASTER_ADDR'] = '127.0.0.1'
#         os.environ['MASTER_PORT'] = '50000'
#         os.environ['HCCL_WHITELIST_DISABLE'] = '1'
#         torch_npu.npu.set_device(rank)
#         dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
#         return dist
#
#
#     @classmethod
#     def _test_npu_matmul_all_to_all(cls, rank, input_list):
#         # Unpack inputs
#         x1, x2, world_size, init_pg, c2p = input_list
#
#         # Initialize process group
#         pg = init_pg(rank, world_size)
#         group = pg.distributed_c10d._get_default_group()
#
#         # Get HCCL communication name
#         if torch.__version__ > '2.0.1':
#             hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
#         else:
#             hcom_name = group.get_hccl_comm_name(rank)
#
#         # Move tensors to NPU
#         x1 = x1.npu()
#         x2 = x2.npu()
#
#         # Call NPU operator
#         out = torch_npu.npu_matmul_all_to_all(x1, x2, hcom_name, bias=None, all2all_axes=[-1,-2])
#
#         # Return result
#         c2p.put((rank, out.cpu()))
#         pg.barrier()
#
#     def _test_multiprocess(self, f, init_pg, input_list): # f function, init_pg function, input_list list[]
#         # Input: expt_out_list, x1_list, x2_list, bias_list, ..., world_size
#         expt_out_list, x1_list, x2_list, bias_list, world_size = input_list
#
#         ctx = mp.get_context('spawn')
#         c2p = ctx.Queue(world_size)
#         ps = []
#
#         for i in range(world_size):
#             p = ctx.Process(
#                 target=f,
#                 # args for line30
#                 args=(i, [x1_list[i], x2_list[i], bias_list[i], world_size, init_pg, c2p])
#             )
#             p.start()
#             ps.append(p)
#
#         # Collect results
#         for _ in range(world_size):
#             rank, out_put = c2p.get()
#             print(f"rank {rank} Receive tensor was {out_put}")
#
#         # Wait for all processes to finish
#         for p in ps:
#             p.join()
#
#     @skipIfUnsupportMultiNPU(2)
#     @SupportedDevices(['Ascend910_95'])
#     def test_npu_quant_reduce_scatter(self):
#         world_size = 2
#         x_dtype = np.int8
#         scales_dtype = np.float32
#         data_format = -1
#         # T-G量化
#         x_shape = [x_dtype, data_format, [2048, 5120]]
#         scales_shape = [scales_dtype, data_format, [2048, 40]]
#         x_list = []
#         scales_list = []
#         for _ in range(world_size):
#             x, _ = create_common_tensor(x_shape, -1, 1)
#             x_list.append(x)
#             scales, _ = create_common_tensor(scales_shape, 1, 100)
#             scales_list.append(scales)
#         expt_out_list = self._construct_excepted_result(x_list, scales_list, world_size)
#         self._test_multiprocess(TestQuantReduceScatter._test_npu_quant_reduce_scatter,
#                                 TestQuantReduceScatter._init_dist_hccl,
#                                 [expt_out_list, x_list, scales_list, world_size])
#
#     @skipIfUnsupportMultiNPU(2)
#     @SupportedDevices(['Ascend910_95'])
#     def test_npu_matmul_all_to_all(self):
#         # 设备卡数
#         world_size = 2
#         # 初始化dtype，因为matmul_all_to_all的x1、x2和output的dtype一致，bias可以等于x1也可以为float32，这里统一使用bfloat16
#         dtype = torch.bfloat16  # Simulate supported dtype
#         # 初始化shape
#         m, k, n = 16, 32, 32
#         # 初始化format，取-1是因为后续要使用create_common_tensor生成张量，format为-1时，不会做特殊处理。
#         data_format = -1
#         # 初始化create_common_tensor需要的参数
#         x1_shape = [dtype, data_format, [m, k]]
#         x2_shape = [dtype, data_format, [k, n]]
#         bias_shape = [dtype, data_format, [n]]
#         # 生成tensors (one per NPU)
#         x1_list = []
#         x2_list = []
#         bias_list = []
#         for _ in range(world_size):
#             # create_common_tensor接收四个参数：
#             # 1、item：[dtype，format，shape[]]。2、minValue：int。3、maxValue：int。4、device=None
#             # 两个返回值，cpu_input，npu_input。npu_input需要和device联合使用，npu_input=tensor.to(device)
#             x1, _ = create_common_tensor(x1_shape, -1, 1)
#             x1_list.append(x1)
#             x2, _ = create_common_tensor(x2_shape, -1, 1)
#             x2_list.append(x2)
#             bias, _ = create_common_tensor(bias_shape, -1, 1)
#             bias_list.append(bias)
#         # x1_scale_list = []
#         # x2_scale_list = []
#         # output_scale_list = []
#         # common_scale_list = []
#         # x1_offset_list = []
#         # x2_offset_list = []
#         # output_offset_list = []
#
#         for _ in range(world_size):
#             # x1: [m, k]
#             x1, _ = create_common_tensor([dtype, -1, [m, k]], -1, 1)
#             x1_list.append(x1)
#
#             # x2: [k, n]
#             x2, _ = create_common_tensor([dtype, -1, [k, n]], -1, 1)
#             x2_list.append(x2)
#
#             # bias: [m, n]
#             bias, _ = create_common_tensor([dtype, -1, [m, n]], 0, 1)
#             bias_list.append(bias)
#
#             # 暂未支持
#             # # Quantization-related tensors (optional, set to None)
#             # x1_scale, _ = create_common_tensor([torch.float32, -1, [m, k]], 0.1, 1.0)
#             # x1_scale_list.append(x1_scale)
#
#             # x2_scale, _ = create_common_tensor([torch.float32, -1, [k, n]], 0.1, 1.0)
#             # x2_scale_list.append(x2_scale)
#
#             # output_scale, _ = create_common_tensor([torch.float32, -1, [m, n]], 0.1, 1.0)
#             # output_scale_list.append(output_scale)
#
#             # common_scale, _ = create_common_tensor([torch.float32, -1, [1]], 0.1, 1.0)
#             # common_scale_list.append(common_scale)
#
#             # x1_offset, _ = create_common_tensor([torch.int32, -1, [m, k]], 0, 0)
#             # x1_offset_list.append(x1_offset)
#
#             # x2_offset, _ = create_common_tensor([torch.int32, -1, [k, n]], 0, 0)
#             # x2_offset_list.append(x2_offset)
#
#             # output_offset, _ = create_common_tensor([torch.int32, -1, [m, n]], 0, 0)
#             # output_offset_list.append(output_offset)
#
#         # Construct expected output
#         expt_out_list = self._construct_expected_result(x1_list, x2_list, bias_list, world_size)
#
#         # Run multi-process test
#         self._test_multiprocess(
#             TestMatmulAlltoAll._test_npu_matmul_all_to_all,
#             TestMatmulAlltoAll._init_dist_hccl,
#             [expt_out_list, x1_list, x2_list, bias_list, world_size]
#         )
#
#
# if __name__ == '__main__':
#     run_tests()


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


if __name__ == '__main__':
    run_tests()

