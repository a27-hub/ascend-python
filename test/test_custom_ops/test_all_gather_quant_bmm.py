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


class TestAllGatherQuantBatchMatMul(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_all_gather_quant_bmm(cls, rank, input_list):
        x1, x2, scale, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        scale = scale.npu()
        scale = torch_npu.npu_trans_quant_param(scale)
        out = torch_npu.npu_all_gather_quant_bmm(x1,
                                                 x2,
                                                 scale,
                                                 hcom_name,
                                                 world_size,
                                                 bias=None,
                                                 pertoken_scale=None,
                                                 output_dtype=torch.float16,
                                                 gather_axis=0,
                                                 gather_output=False,
                                                 comm_turn=0)

        c2p.put((rank, out[0].to(torch.float32).cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x1, x2, scale, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1, x2, scale, world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list[rank], output))

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, scale_list, world_size):
        gather_out = torch.cat(x1_list)
        out_list = []
        out_dype = gather_out.dtype
        for i in range(world_size):
            output = torch.matmul(gather_out.to(torch.int32), x2_list[i].to(torch.int32))
            output = torch.mul(output, scale_list[i].to(torch.float32)).to(torch.float32)
            out_list.append(output)
        return out_list

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B'])
    def test_npu_all_gather_quant_bmm(self):
        world_size = 2
        dtype = np.int8
        scale_dtype = np.float32
        data_format = -1
        x1_shape = [dtype, data_format, [1, 256, 8192]]
        x2_shape = [dtype, data_format, [8192, 4096]]
        scale_shape = [scale_dtype, data_format, [4096]]
        x1_list = []
        x2_list = []
        scale_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            scale, _ = create_common_tensor(scale_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
            scale_list.append(scale)

        expt_out_list = self._construct_excepted_result(x1_list, x2_list, scale_list, world_size)
        self._test_multiprocess(TestAllGatherQuantBatchMatMul._test_npu_all_gather_quant_bmm,
                                TestAllGatherQuantBatchMatMul._init_dist_hccl, [expt_out_list, x1, x2, scale, world_size])


if __name__ == '__main__':
    run_tests()
