   @classmethod
    def _test_npu_moe_distribute_dispatch_v2_fp8(cls, rank, input_list):
        expt_token_list, x1_list, x2_list, topk1_list, topk2_list, ep_world_size, globalBS, \
            sharedExpertRankNum, moeExpertNum, h, init_pg, c2p, p2c = input_list
        if rank % 2 == 0:
            x = x1_list[rank // 2]
            topk = topk1_list[rank // 2]
        else:
            x = x2_list[rank // 2]
            topk = topk2_list[rank // 2]
        pg, ep_group, _ = init_pg(rank, ep_world_size * 2, ep_world_size, 2)
        ep_hcomm_name = ep_group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x = x.npu()
        topk = topk.npu()
        scales = torch.randn(4, 7168, dtype=torch.float32, device="npu")
        out, _, _, _, _, _, _ = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=topk,
            elastic_info=None,
            group_ep=ep_hcomm_name,
            ep_world_size=ep_world_size,
            ep_rank_id=int(rank // 2),
            moe_expert_num=moeExpertNum,
            scales=scales,
            group_tp="",
            tp_world_size=0,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=int(sharedExpertRankNum > 0),
            shared_expert_rank_num=sharedExpertRankNum,
            quant_mode=1,
            global_bs=globalBS,
            y_dtype=torch.float8_e4m3fn)

        _ = torch_npu._npu_distribute_barrier(
            x_ref=x,
            group=ep_hcomm_name,
            world_size=ep_world_size)

        if rank // 2 < sharedExpertRankNum:
            A = int(globalBS // sharedExpertRankNum)
        else:
            local = int(moeExpertNum // (ep_world_size - sharedExpertRankNum))
            A = int(globalBS * local)
        out = (out.reshape(A, h))[:int(expt_token_list[rank]), :]
        c2p.put((rank, out.cpu()))
        p2c.get()

    def _test_multiprocess_float8(self, f, init_pg, input_list):
        expt_out_list, expt_token_list, x1_list, x2_list, topk1_list, topk2_list, ep_world_size, globalBS,\
            sharedExpertRankNum, moeExpertNum, h = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ep_world_size * 2)
        p2c = ctx.Queue(ep_world_size * 2)
        ps = []

        for i in range(ep_world_size * 2):
            p = ctx.Process(
                target=f,
                args=(i, [expt_token_list, x1_list, x2_list, topk1_list, topk2_list, ep_world_size,
                          globalBS, sharedExpertRankNum, moeExpertNum, h, init_pg, c2p, p2c]))
            p.start()
            ps.append(p)

        for _ in range(ep_world_size * 2):
            rank, output = c2p.get()
            self.assertRtolEqual(output, expt_out_list[rank].float(), 0.01, 0.01)

        for _ in range(ep_world_size * 2):
            p2c.put(0)

        for p in ps:
            p.join()

    # @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950'])
    def test_npu_moe_distribute_dispatch_v2_fp8(self):
        ep_world_size = 4
        bs = 8
        h = 7168
        k = 2
        shared_expert_rank_num_1 = 1
        moe_expert_num_3 = 3
        shared_expert_rank_num_0 = 0
        moe_expert_num_4 = 4
        global_bs = bs * ep_world_size
        dtype = np.float16
        data_format = -1
        topk = torch.tile(torch.arange(k), (bs,)).int().view(-1, k)
        topk1_list = []
        topk2_list = []
        x1_shape = [dtype, data_format, [bs, h]]
        x2_shape = [dtype, data_format, [bs, h]]
        x1_list = []
        x2_list = []
        for _ in range(ep_world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
            topk1_list.append(topk)
            topk2_list.append(topk)
        expt_out_list_1, expt_token_list_1 = self._construct_excepted_result(x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
                                                            global_bs, shared_expert_rank_num_1, moe_expert_num_3, ep_world_size, 1)
        expt_out_list_2, expt_token_list_2 = self._construct_excepted_result(x1_list, x2_list, topk1_list, topk2_list, bs, h, k,
                                                            global_bs, shared_expert_rank_num_0, moe_expert_num_4, ep_world_size, 1)

        self._test_multiprocess_float8(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2_fp8,
                TestMoeDistributeDispatch._init_dist_hccl, [expt_out_list_1, expt_token_list_1, x1_list, x2_list, topk1_list,
                topk2_list, ep_world_size, global_bs, shared_expert_rank_num_1, moe_expert_num_3, h])
        self._test_multiprocess_float8(TestMoeDistributeDispatch._test_npu_moe_distribute_dispatch_v2_fp8,
                TestMoeDistributeDispatch._init_dist_hccl, [expt_out_list_2, expt_token_list_2, x1_list, x2_list, topk1_list,
                topk2_list, ep_world_size, global_bs, shared_expert_rank_num_0, moe_expert_num_4, h])