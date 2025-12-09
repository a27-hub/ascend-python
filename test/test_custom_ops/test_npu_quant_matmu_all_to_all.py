import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestQuantMatmulAlltoAll(TestCase):
    
    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    
    @classmethod
    def _test_npu_quant_matmul_all_to_all(cls, rank, input_list):
        # Unpack inputs
        x1, x2, world_size, init_pg, c2p = input_list

        # Initialize process group
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()

        # Get HCCL communication name
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        # Move tensors to NPU
        x1 = x1.npu()
        x2 = x2.npu()

        # Call NPU operator
        out = torch_npu.npu_matmul_all_to_all(x1, x2, hcom_name, bias=None)

        # Return result
        c2p.put((rank, out.cpu()))
        pg.barrier()
        
    def _test_multiprocess(self, f, init_pg, input_list): # f function, init_pg function, input_list list[]
        # Input: expt_out_list, x1_list, x2_list, bias_list, ..., world_size
        expt_out_list, x1_list, x2_list, bias_list, world_size = input_list

        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                # args for line30
                args=(i, [x1_list[i], x2_list[i], bias_list[i], world_size, init_pg, c2p])
            )
            p.start()
            ps.append(p)

        # Collect results
        for _ in range(world_size):
            rank, out_put = c2p.get()
            self.assertEqual(out_put, expt_out_list[rank],
                                f"rank {rank} Expected receive tensor {expt_out_list[rank]} but got {out_put}")

        # Wait for all processes to finish
        for p in ps:
            p.join()

    # golden
    def _construct_expected_result(self, x1_list, x2_list, bias_list, world_size):
        # Compute manually on CPU: x1 @ x2 + bias
        out = None
        out_list = []
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            bias = bias_list[i] if bias_list[i] is not None else 0

            # Perform matmul + bias
            matmul_result = torch.matmul(x1, x2)
            result = matmul_result + bias
            
            # Quantitative scenarios to be supplemented

            if out is None:
                out = result
            else:
                out = out + result  # Accumulate results from all processes (simulates final all2all output)
            out_list.append(out.to(torch.bfloat16))
        return out_list  # Simulate default output dtype

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910_95'])
    def test_npu_quant_matmul_all_to_all(self):
        world_size = 2
        dtype = torch.bfloat16  # Simulate supported dtype
        m, k, n = 128, 256, 512

        # Generate tensors (one per NPU)
        x1_list = []
        x2_list = []
        bias_list = []
        # x1_scale_list = []
        # x2_scale_list = []
        # output_scale_list = []
        # common_scale_list = []
        # x1_offset_list = []
        # x2_offset_list = []
        # output_offset_list = []

        for _ in range(world_size):
            # x1: [m, k]
            x1, _ = create_common_tensor([dtype, -1, [m, k]], -1, 1)
            x1_list.append(x1)

            # x2: [k, n]
            x2, _ = create_common_tensor([dtype, -1, [k, n]], -1, 1)
            x2_list.append(x2)

            # bias: [m, n]
            bias, _ = create_common_tensor([dtype, -1, [m, n]], 0, 1)
            bias_list.append(bias)

            # 暂未支持
            # # Quantization-related tensors (optional, set to None)
            # x1_scale, _ = create_common_tensor([torch.float32, -1, [m, k]], 0.1, 1.0)
            # x1_scale_list.append(x1_scale)

            # x2_scale, _ = create_common_tensor([torch.float32, -1, [k, n]], 0.1, 1.0)
            # x2_scale_list.append(x2_scale)

            # output_scale, _ = create_common_tensor([torch.float32, -1, [m, n]], 0.1, 1.0)
            # output_scale_list.append(output_scale)

            # common_scale, _ = create_common_tensor([torch.float32, -1, [1]], 0.1, 1.0)
            # common_scale_list.append(common_scale)

            # x1_offset, _ = create_common_tensor([torch.int32, -1, [m, k]], 0, 0)
            # x1_offset_list.append(x1_offset)

            # x2_offset, _ = create_common_tensor([torch.int32, -1, [k, n]], 0, 0)
            # x2_offset_list.append(x2_offset)

            # output_offset, _ = create_common_tensor([torch.int32, -1, [m, n]], 0, 0)
            # output_offset_list.append(output_offset)

        # Construct expected output
        expt_out_list = self._construct_expected_result(x1_list, x2_list, bias_list, world_size)

        # Run multi-process test
        self._test_multiprocess(
            TestQuantMatmulAlltoAll._test_npu_quant_matmul_all_to_all,
            TestQuantMatmulAlltoAll._init_dist_hccl,
            [expt_out_list, x1_list, x2_list, bias_list, world_size]
        )


if __name__ == '__main__':
    run_tests()
