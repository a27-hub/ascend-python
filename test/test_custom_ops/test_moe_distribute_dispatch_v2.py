import os
import unittest
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestMoeDistributeDispatch(TestCase):
    """
    torch_npu.npu_moe_distribute_dispatch_v2 测试用例集

    用例覆盖场景说明：
    | 用例编号 | 场景描述 | 数据类型 | 共享专家 | TP | 量化模式 | K值 | batch size |
    |---------|---------|---------|---------|-----|---------|-----|-----------|
    | 1-4     | 非量化场景，TP=2 | fp16/bf16 | 有/无 | 2 | 0 | 4 | 8 |
    | 5-8     | 非量化场景，TP=1 | fp16/bf16 | 有/无 | 1 | 0 | 4 | 8 |
    | 9-12    | 动态量化场景，TP=2 | fp16/bf16 | 有/无 | 2 | 2 | 4 | 8 |
    | 13-14   | 动态量化场景，TP=1 | fp16/bf16 | 无 | 1 | 2 | 4 | 8 |
    | 15-16   | 多共享专家卡场景 | fp16/bf16 | 2个 | 2 | 0 | 4 | 8 |
    | 17-18   | 不同K值场景(K=2) | fp16/bf16 | 无 | 2 | 0 | 2 | 8 |
    | 19-20   | 大batch size场景 | fp16/bf16 | 无 | 2 | 0 | 4 | 16 |

    约束说明：
    - x数据类型支持：bfloat16、float16
    - expert_ids数据类型：int32
    - ep_world_size取值范围：[2, 768]
    - moe_expert_num取值范围：[1, 1024]
    - quant_mode：0表示非量化，2表示动态量化
    - tp_world_size：0和1表示无TP域通信，2表示有TP域通信
    """

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
    def _test_npu_moe_distribute_dispatch_v2(cls, rank, input_list):
        expt_token_list, x1_list, x2_list, topk1_list, topk2_list, ep_world_size, tp_world_size, globalBS, \
            sharedExpertRankNum, moeExpertNum, h, quant_mode, init_pg, c2p, p2c = input_list
        tp_world_size_2 = 2
        if rank % tp_world_size_2 == 0:
            x = x1_list[rank // tp_world_size_2]
            topk = topk1_list[rank // tp_world_size_2]
        else:
            x = x2_list[rank // tp_world_size_2]
            topk = topk2_list[rank // tp_world_size_2]
        pg, ep_group, tp_group = init_pg(rank, ep_world_size * tp_world_size_2, ep_world_size, tp_world_size_2)
        ep_hcomm_name = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        tp_hcomm_name = tp_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x = x.npu()
        topk = topk.npu()
        out, _, _, _, _, _, _ = torch_npu.npu_moe_distribute_dispatch_v2(x=x,
                                                                         expert_ids=topk,
                                                                         group_ep=ep_hcomm_name,
                                                                         ep_world_size=ep_world_size,
                                                                         ep_rank_id=int(rank // tp_world_size_2),
                                                                         moe_expert_num=moeExpertNum,
                                                                         scales=None,
                                                                         group_tp=tp_hcomm_name,
                                                                         tp_world_size=tp_world_size,
                                                                         tp_rank_id=int(
                                                                             rank % tp_world_size) if tp_world_size != 1 else 0,
                                                                         expert_shard_type=0,
                                                                         shared_expert_num=int(sharedExpertRankNum > 0),
                                                                         shared_expert_rank_num=sharedExpertRankNum,
                                                                         quant_mode=quant_mode,
                                                                         global_bs=globalBS)

        if tp_world_size == 1:
            _ = torch_npu._npu_distribute_barrier(
                x_ref=x,
                group=ep_hcomm_name,
                world_size=ep_world_size)

        if rank // tp_world_size_2 < sharedExpertRankNum:
            A = int(globalBS // sharedExpertRankNum)
        else:
            local = int(moeExpertNum // (ep_world_size - sharedExpertRankNum))
            A = int(globalBS * local)
        out = (out.reshape(tp_world_size * A, h))[:int(expt_token_list[rank]), :]
        c2p.put((rank, out.cpu()))
        p2c.get()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, expt_token_list, x1_list, x2_list, topk1_list, topk2_list, ep_world_size, tp_world_size, globalBS, \
            sharedExpertRankNum, moeExpertNum, h, quant_mode = input_list
        ctx = mp.get_context('spawn')
        tp_world_size_2 = 2
        c2p = ctx.Queue(ep_world_size * tp_world_size_2)
        p2c = ctx.Queue(ep_world_size * tp_world_size_2)
        ps = []

        for i in range(ep_world_size * tp_world_size_2):
            p = ctx.Process(
                target=f,
                args=(i, [expt_token_list, x1_list, x2_list, topk1_list, topk2_list, ep_world_size, tp_world_size,
                          globalBS, sharedExpertRankNum, moeExpertNum, h, quant_mode, init_pg, c2p, p2c]))
            p.start()
            ps.append(p)

        for _ in range(ep_world_size * tp_world_size_2):
            rank, output = c2p.get()
            self.assertEqual(output, expt_out_list[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expt_out_list[rank], output))

        for _ in range(ep_world_size * tp_world_size_2):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, topk1_list, topk2_list, bs, h, k, globalBS,
                                   sharedExpertRankNum, moeExpertNum, ep_world_size, tp_world_size):
        col_idx = torch.arange(0, globalBS * k, dtype=torch.int32)
        row_idx = col_idx.view(k, -1).permute(1, 0)
        row_idx = row_idx.reshape([globalBS, k]).contiguous()

        x1 = torch.cat(x1_list, dim=0).view(-1, h)
        x2 = torch.cat(x2_list, dim=0).view(-1, h)
        topk1 = torch.cat(topk1_list, dim=0).view(-1, k)
        topk2 = torch.cat(topk2_list, dim=0).view(-1, k)

        expandX1, expand_row1, expand_expert1 = torch_npu.npu_moe_init_routing(x1.npu(), row_idx=row_idx.npu(),
                                                                               expert_idx=topk1.npu(),
                                                                               active_num=globalBS)
        expandX2, expand_row2, expand_expert2 = torch_npu.npu_moe_init_routing(x2.npu(), row_idx=row_idx.npu(),
                                                                               expert_idx=topk2.npu(),
                                                                               active_num=globalBS)

        expandX1 = expandX1.cpu()
        expandX2 = expandX2.cpu()
        expand_expert1 = expand_expert1.cpu()
        expand_expert2 = expand_expert2.cpu()
        shared_list = []
        shared_tokens = []
        for i in range(sharedExpertRankNum):
            tmp_list = []
            shared_tokens.append(bs * (int(moeExpertNum / sharedExpertRankNum) + 1))
            tmp_list.append(x1[(bs * i):(bs * (i + 1)), :])
            for j in range(int(moeExpertNum / sharedExpertRankNum)):
                tmp_list.append(
                    x1[(bs * (i + (j + 1) * sharedExpertRankNum)):(bs * (i + (j + 1) * sharedExpertRankNum + 1)), :])
            tmp_list = torch.cat(tmp_list, dim=0).to(torch.float16)
            shared_list.append(tmp_list)
        if sharedExpertRankNum != 0:
            shared_x1 = torch.cat(shared_list, dim=0)
        token1 = torch.cat((torch.tensor(shared_tokens), torch.bincount(expand_expert1, minlength=moeExpertNum)))
        token2 = torch.cat((torch.tensor(shared_tokens), torch.bincount(expand_expert2, minlength=moeExpertNum)))
        shared_list = []
        for i in range(sharedExpertRankNum):
            tmp_list = []
            tmp_list.append(x2[(bs * i):(bs * (i + 1)), :])
            for j in range(int(moeExpertNum / sharedExpertRankNum)):
                tmp_list.append(
                    x2[(bs * (i + (j + 1) * sharedExpertRankNum)):(bs * (i + (j + 1) * sharedExpertRankNum + 1)), :])
            tmp_list = torch.cat(tmp_list, dim=0).to(torch.float16)
            shared_list.append(tmp_list)
        if sharedExpertRankNum != 0:
            shared_x2 = torch.cat(shared_list, dim=0)
            golden_expandX1 = torch.cat((shared_x1, expandX1)).view(-1, h)
            golden_expandX2 = torch.cat((shared_x2, expandX2)).view(-1, h)
        else:
            golden_expandX1 = expandX1.view(-1, h)
            golden_expandX2 = expandX2.view(-1, h)

        sums1 = 0
        sums2 = 0
        out_list = []
        token_list = []
        for i in range(ep_world_size):
            start1 = sums1
            end1 = sums1 + int(token1[i])
            sums1 = end1
            start2 = sums2
            end2 = sums2 + int(token2[i])
            sums2 = end2
            if tp_world_size == 2:
                out_list.append(torch.cat((golden_expandX1[start1:end1, :], golden_expandX2[start2:end2, :])))
                out_list.append(torch.cat((golden_expandX2[start2:end2, :], golden_expandX1[start1:end1, :])))
                token_list.append(int(token1[i]) + int(token2[i]))
                token_list.append(int(token1[i]) + int(token2[i]))
            else:
                out_list.append(golden_expandX1[start1:end1, :])
                out_list.append(golden_expandX2[start2:end2, :])
                token_list.append(int(token1[i]))
                token_list.append(int(token2[i]))

        return out_list, token_list

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_no_shared_tp2(self):
        """
        用例1: 非量化场景 - float16, 无共享专家, TP=2
        测试float16数据类型，无共享专家，有TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_no_shared_tp2(self):
        """
        用例2: 非量化场景 - bfloat16, 无共享专家, TP=2
        测试bfloat16数据类型，无共享专家，有TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_shared_tp2(self):
        """
        用例3: 非量化场景 - float16, 有共享专家, TP=2
        测试float16数据类型，有共享专家，有TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 1
        moe_expert_num = 7
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_shared_tp2(self):
        """
        用例4: 非量化场景 - bfloat16, 有共享专家, TP=2
        测试bfloat16数据类型，有共享专家，有TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 1
        moe_expert_num = 7
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_no_shared_tp1(self):
        """
        用例5: 非量化场景 - float16, 无共享专家, TP=1
        测试float16数据类型，无共享专家，无TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_no_shared_tp1(self):
        """
        用例6: 非量化场景 - bfloat16, 无共享专家, TP=1
        测试bfloat16数据类型，无共享专家，无TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_shared_tp1(self):
        """
        用例7: 非量化场景 - float16, 有共享专家, TP=1
        测试float16数据类型，有共享专家，无TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 1
        moe_expert_num = 7
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_shared_tp1(self):
        """
        用例8: 非量化场景 - bfloat16, 有共享专家, TP=1
        测试bfloat16数据类型，有共享专家，无TP通信域的场景
        """
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 1
        moe_expert_num = 7
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_quant_tp2(self):
        """
        用例9: 动态量化场景 - float16, 无共享专家, TP=2
        测试动态量化模式(quant_mode=2)，float16数据类型，有TP通信域
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 2

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_quant_tp2(self):
        """
        用例10: 动态量化场景 - bfloat16, 无共享专家, TP=2
        测试动态量化模式(quant_mode=2)，bfloat16数据类型，有TP通信域
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 2

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_quant_shared_tp2(self):
        """
        用例11: 动态量化场景 - float16, 有共享专家, TP=2
        测试动态量化模式，有共享专家，有TP通信域
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 1
        moe_expert_num = 7
        global_bs = bs * ep_world_size
        quant_mode = 2

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_quant_shared_tp2(self):
        """
        用例12: 动态量化场景 - bfloat16, 有共享专家, TP=2
        测试动态量化模式，bfloat16数据类型，有共享专家，有TP通信域
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 1
        moe_expert_num = 7
        global_bs = bs * ep_world_size
        quant_mode = 2

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_quant_tp1(self):
        """
        用例13: 动态量化场景 - float16, 无共享专家, TP=1
        测试动态量化模式，无TP通信域
        """
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 2

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_quant_tp1(self):
        """
        用例14: 动态量化场景 - bfloat16, 无共享专家, TP=1
        测试动态量化模式，bfloat16数据类型，无TP通信域
        """
        ep_world_size = 8
        tp_world_size = 1
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 2

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_shared2_tp2(self):
        """
        用例15: 非量化场景 - float16, 2个共享专家卡, TP=2
        测试多个共享专家卡的场景
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 2
        moe_expert_num = 6
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_shared2_tp2(self):
        """
        用例16: 非量化场景 - bfloat16, 2个共享专家卡, TP=2
        测试多个共享专家卡的场景，bfloat16数据类型
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 4
        shared_expert_rank_num = 2
        moe_expert_num = 6
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_k2_tp2(self):
        """
        用例17: 非量化场景 - float16, K=2, TP=2
        测试不同的topK值(K=2)
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 2
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_k2_tp2(self):
        """
        用例18: 非量化场景 - bfloat16, K=2, TP=2
        测试不同的topK值(K=2)，bfloat16数据类型
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 8
        h = 7168
        k = 2
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_fp16_large_bs_tp2(self):
        """
        用例19: 非量化场景 - float16, 大batch size, TP=2
        测试较大的batch size场景
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 16
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.float16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])

    @skipIfUnsupportMultiNPU(16)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_moe_distribute_dispatch_v2_bf16_large_bs_tp2(self):
        """
        用例20: 非量化场景 - bfloat16, 大batch size, TP=2
        测试较大的batch size场景，bfloat16数据类型
        """
        ep_world_size = 8
        tp_world_size = 2
        bs = 16
        h = 7168
        k = 4
        shared_expert_rank_num = 0
        moe_expert_num = 8
        global_bs = bs * ep_world_size
        quant_mode = 0

        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = [topk.clone() for _ in range(ep_world_size)]
        topk2_list = [topk.clone() for _ in range(ep_world_size)]

        x1_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]
        x2_list = [torch.randn(bs, h, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(ep_world_size)]

        expt_out_list, expt_token_list = self._construct_excepted_result(
            x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
            global_bs, shared_expert_rank_num, moe_expert_num, ep_world_size, tp_world_size)

        self._test_multiprocess(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2,
                                TestMoeDistributeDispatch._init_dist_hccl,
                                [expt_out_list, expt_token_list, x1_list, x2_list, topk1_list,
                                 topk2_list, ep_world_size, tp_world_size, global_bs, shared_expert_rank_num,
                                 moe_expert_num, h, quant_mode])


if __name__ == '__main__':
    run_tests()