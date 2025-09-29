import os
import unittest
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

class TestMoeDistributeBarrier(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size, ep_world_size, tp_world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        ep_ranks_list = []
        tp_ranks_list = []
        for i in range(tp_world_size):
            ep_ranks_list.append(list(range(i, world_size, tp_world_size)))
        for i in range(ep_world_size):
            tp_ranks_list.append(list(range(i * tp_world_size, (i + 1) * tp_world_size)))
        for i in range(tp_world_size):
            ep_group = dist.new_group(backend='hccl', ranks=ep_ranks_list[i])
            if rank in ep_ranks_list[i]:
                ep_group_tmp = ep_group
        for i in range(ep_world_size):
            tp_group = dist.new_group(backend='hccl', ranks=tp_ranks_list[i])
            if rank in tp_ranks_list[i]:
                tp_group_tmp = tp_group
        return dist, ep_group_tmp, tp_group_tmp

    @classmethod
    def gen_elastic_info(is_elastic, world_size, shared_expert_rank_num, shared_broken_card_num,
                         moe_broken_card_num, local_moe, ep_world_size, moe_expert_num):
        if not is_elastic: return None
        elastic_info = torch.zeros(4 + 2 * ep_world_size, dtype = torch.int32)
        elastic_info[0] = is_elastic
        elastic_info[1] = world_size - shared_broken_card_num -moe_broken_card_num
        elastic_info[2] = shared_expert_rank_num
        elastic_info[3] = moe_expert_num - local_moe*(moe_broken_card_num + shared_broken_card_num)
        table1 = [-1] * ep_world_size
        table2 = [-1] * ep_world_size

        if is_elastic:
            shared_ranks = list(range(shared_expert_rank_num))
            non_shared_ranks = list(range(shared_expert_rank_num, world_size))
            elastic_rank = random.sample(shared_ranks, elastic_info[2]) + random.sample(non_shared_ranks, elastic_info[1] - elastic_info[2])
            elastic_info.sort()
        for loacl_rank_id, ep_rank_id in enumerate(elastic_info):
            if ep_rank_id < ep_world_size:
                table1[ep_rank_id] = loacl_rank_id
                table2[loacl_rank_id] = ep_rank_id
        for i in range(ep_world_size):
            elastic_info[4 + i] = table1[i]
        for i in range(ep_world_size):
            elastic_info[4 + ep_world_size + i] = table2[i]
        assert elastic_info.shape[0] == 4 + 2 * ep_world_size

        if is_elastic:
            table1 = elastic_info[4 : 4 + ep_world_size]
            table2 = elastic_info[4 + ep_world_size: 4 + 2 * ep_world_size]
        return elastic_info
    
    @classmethod
    def _test_npu_moe_distribute_barrier(cls, rank, x_ref, time_out, elastic_info,
                                         ep_world_size, tp_world_size, init_pg, c2p, p2c ):
        
        pg, ep_group, tp_group = init_pg(rank, ep_world_size * tp_world_size, ep_world_size, tp_world_size)
        ep_hcomm_name = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        tp_hcomm_name = tp_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        if rank in elastic_info[4 + ep_world_size: 4 + 2 * ep_world_size]:
            out = torch_npu.npu_moe_distribute_barrier(
                                                    x_ref = x_ref,
                                                    time_out = time_out,
                                                    elastic_info = elastic_info,
                                                    group_ep=ep_hcomm_name,
                                                    world_size=ep_world_size,
                                                    )
        else:
            out = None
        c2p.put((rank, out[0].cpu()))
        p2c.get()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x_ref, time_out, elastic_info, \
            ep_world_size, tp_world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ep_world_size * tp_world_size)
        p2c = ctx.Queue(ep_world_size * tp_world_size)
        ps = []

        for i in range(ep_world_size * tp_world_size):
            p = ctx.Process(
                target=f,
                args=(i, x_ref, time_out, elastic_info, ep_world_size, tp_world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)

        for _ in range(ep_world_size * tp_world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list[rank], output))

        for _ in range(ep_world_size * tp_world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x_ref, elastic_info, ep_world_size): 
        out_list = []
        for i in elastic_info[4 + ep_world_size: 4 + 2 * ep_world_size]:
            out_list.append(x_ref)
        else:
            out_list.append(None)
        return out_list
    
    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910_'])
    def test_npu_moe_distribute_combine_add_rms_norm(self):
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 7
        shared_broken_card_num_2 = 2
        shared_broken_card_num_1 = 1
        shared_expert_rank_num_2 = 2
        moe_broken_card_num_2 = 2
        moe_broken_card_num_1 = 1
        local_moe_4 = 4
        local_moe_2 = 2
        moe_expert_num_4 = local_moe_4 * (ep_world_size * tp_world_size - shared_expert_rank_num_2)
        moe_expert_num_2 = local_moe_2 * (ep_world_size * tp_world_size - shared_expert_rank_num_2)
        is_elastic = 1
        x_ref = torch.ones(1, dtype =np.int32)
        time_out = torch.tensor([100000], dtype = torch.int32).npu()
        elastic_info_x1 = TestMoeDistributeBarrier.gen_elastic_info(is_elastic, ep_world_size * tp_world_size,
                                                                    shared_broken_card_num_2,
                                                                    shared_expert_rank_num_2,
                                                                    moe_broken_card_num_2,
                                                                    local_moe_4,
                                                                    ep_world_size,
                                                                    moe_expert_num_4)
        
        elastic_info_x2 = TestMoeDistributeBarrier.gen_elastic_info(is_elastic, ep_world_size * tp_world_size,
                                                                    shared_broken_card_num_1,
                                                                    shared_expert_rank_num_2,
                                                                    moe_broken_card_num_2,
                                                                    local_moe_4,
                                                                    ep_world_size,
                                                                    moe_expert_num_4)
        
        elastic_info_x3 = TestMoeDistributeBarrier.gen_elastic_info(is_elastic, ep_world_size * tp_world_size,
                                                                    shared_broken_card_num_2,
                                                                    shared_expert_rank_num_2,
                                                                    moe_broken_card_num_1,
                                                                    local_moe_4,
                                                                    ep_world_size,
                                                                    moe_expert_num_4)
        
        elastic_info_x4 = TestMoeDistributeBarrier.gen_elastic_info(is_elastic, ep_world_size * tp_world_size,
                                                                    shared_broken_card_num_2,
                                                                    shared_expert_rank_num_2,
                                                                    moe_broken_card_num_2,
                                                                    local_moe_2,
                                                                    ep_world_size,
                                                                    moe_expert_num_4)
        
        for _ in range(ep_world_size):
            expt_out_list_1 = self._construct_excepted_result(x_ref, elastic_info_x1, ep_world_size)
            expt_out_list_2 = self._construct_excepted_result(x_ref, elastic_info_x1, ep_world_size)
            expt_out_list_3 = self._construct_excepted_result(x_ref, elastic_info_x1, ep_world_size)
            expt_out_list_4 = self._construct_excepted_result(x_ref, elastic_info_x1, ep_world_size)

        self._test_multiprocess(TestMoeDistributeBarrier._test_npu_moe_distribute_barrier,
                TestMoeDistributeBarrier._init_dist_hccl, [expt_out_list_1, x_ref, time_out, elastic_info_x1,
                                                            ep_world_size, tp_world_size])
        self._test_multiprocess(TestMoeDistributeBarrier._test_npu_moe_distribute_barrier,
                TestMoeDistributeBarrier._init_dist_hccl, [expt_out_list_2, x_ref, time_out, elastic_info_x2,
                                                            ep_world_size, tp_world_size])
        self._test_multiprocess(TestMoeDistributeBarrier._test_npu_moe_distribute_barrier,
                TestMoeDistributeBarrier._init_dist_hccl, [expt_out_list_3, x_ref, time_out, elastic_info_x3,
                                                            ep_world_size, tp_world_size])
        self._test_multiprocess(TestMoeDistributeBarrier._test_npu_moe_distribute_barrier,
                TestMoeDistributeBarrier._init_dist_hccl, [expt_out_list_4, x_ref, time_out, elastic_info_x4,
                                                            ep_world_size, tp_world_size])

if __name__ == '__main__':
    run_tests()