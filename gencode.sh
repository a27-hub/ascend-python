    @classmethod
    def _test_npu_all_gather_quant_mm_float8_e5m2(cls, rank, input_list):
        x1_list, x2_list, ydtype, world_size, init_pg, c2p = input_list
        x1 = x1_list[rank]
        x2 = x2_list[rank]
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, gather_out = torch_npu.npu_all_gather_quant_mm(x1,
                                                            x2,
                                                            hcom_name,
                                                            world_size,
                                                            bias=None,
                                                            x1_scale=None,
                                                            x2_scale=None,
                                                            quant_scale=None,
                                                            gather_index=0,
                                                            gather_output=True,
                                                            comm_turn=0,
                                                            y_dtype=ydtype,
                                                            x1_dtype=torch.float8_e5m2,
                                                            x2_dtype=torch.float8_e5m2)

        c2p.put((rank, out.cpu(), gather_out.cpu()))
        pg.barrier()

    @SupportedDevices(['Ascend950'])
    def test_npu_all_gather_quant_mm_float8_e5m2(self):
        world_size = 2
        dtype = np.float16
        ydtype = 5
        data_format = -1
        x1_shape = [dtype, data_format, [16, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e5m2))
            x2_list.append(x2.to(torch.float8_e5m2))
        expt_out_list, expt_gather = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float16)
        self._test_multiprocess(TestAllGatherQuantMm._test_npu_all_gather_quant_mm_float8_e5m2,
                                TestAllGatherQuantMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, ydtype, world_size])

    @classmethod
    def _test_npu_all_gather_quant_mm_float8_e4m3fn(cls, rank, input_list):
        x1_list, x2_list, ydtype, world_size, init_pg, c2p = input_list
        x1 = x1_list[rank]
        x2 = x2_list[rank]
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, gather_out = torch_npu.npu_all_gather_quant_mm(x1,
                                                            x2,
                                                            hcom_name,
                                                            world_size,
                                                            bias=None,
                                                            x1_scale=None,
                                                            x2_scale=None,
                                                            quant_scale=None,
                                                            gather_index=0,
                                                            gather_output=True,
                                                            comm_turn=0,
                                                            y_dtype=ydtype,
                                                            x1_dtype=torch.float8_e4m3fn,
                                                            x2_dtype=torch.float8_e4m3fn)

        c2p.put((rank, out.cpu(), gather_out.cpu()))
        pg.barrier()

    @SupportedDevices(['Ascend950'])
    def test_npu_all_gather_quant_mm_float8_e4m3fn(self):
        world_size = 2
        dtype = np.float16
        ydtype = 5
        data_format = -1
        x1_shape = [dtype, data_format, [16, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e4m3fn))
            x2_list.append(x2.to(torch.float8_e4m3fn))
        expt_out_list, expt_gather = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float16)
        self._test_multiprocess(TestAllGatherQuantMm._test_npu_all_gather_quant_mm_float8_e4m3fn,
                                TestAllGatherQuantMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, ydtype, world_size])

    @classmethod
    def _test_npu_all_gather_quant_mm_hifloat8(cls, rank, input_list):
        x1_list, x2_list, ydtype, world_size, init_pg, c2p = input_list
        x1 = x1_list[rank]
        x2 = x2_list[rank]
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, gather_out = torch_npu.npu_all_gather_quant_mm(x1,
                                                            x2,
                                                            hcom_name,
                                                            world_size,
                                                            bias=None,
                                                            x1_scale=None,
                                                            x2_scale=None,
                                                            quant_scale=None,
                                                            gather_index=0,
                                                            gather_output=True,
                                                            comm_turn=0,
                                                            y_dtype=ydtype,
                                                            x1_dtype=torch_npu.hifloat8,
                                                            x2_dtype=torch_npu.hifloat8)

        c2p.put((rank, out.cpu(), gather_out.cpu()))
        pg.barrier()

    @SupportedDevices(['Ascend950'])
    def test_npu_all_gather_quant_mm_hifloat8(self):
        world_size = 2
        dtype = np.float16
        ydtype = 5
        data_format = -1
        x1_shape = [dtype, data_format, [16, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.int8))
            x2_list.append(x2.to(torch.int8))
        expt_out_list, expt_gather = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float16)
        self._test_multiprocess(TestAllGatherQuantMm._test_npu_all_gather_quant_mm_hifloat8,
                                TestAllGatherQuantMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, ydtype, world_size])

    @classmethod
    def _test_npu_all_gather_quant_mm_float8_mixed_e5m2_e4m3fn(cls, rank, input_list):
        x1_list, x2_list, ydtype, world_size, init_pg, c2p = input_list
        x1 = x1_list[rank]
        x2 = x2_list[rank]
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, gather_out = torch_npu.npu_all_gather_quant_mm(x1,
                                                            x2,
                                                            hcom_name,
                                                            world_size,
                                                            bias=None,
                                                            x1_scale=None,
                                                            x2_scale=None,
                                                            quant_scale=None,
                                                            gather_index=0,
                                                            gather_output=True,
                                                            comm_turn=0,
                                                            y_dtype=ydtype,
                                                            x1_dtype=torch.float8_e5m2,
                                                            x2_dtype=torch.float8_e4m3fn)

        c2p.put((rank, out.cpu(), gather_out.cpu()))
        pg.barrier()

    @SupportedDevices(['Ascend950'])
    def test_npu_all_gather_quant_mm_float8_mixed_e5m2_e4m3fn(self):
        world_size = 2
        dtype = np.float16
        ydtype = 15
        data_format = -1
        x1_shape = [dtype, data_format, [16, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e5m2))
            x2_list.append(x2.to(torch.float8_e4m3fn))
        expt_out_list, expt_gather = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.bfloat16)
        self._test_multiprocess(TestAllGatherQuantMm._test_npu_all_gather_quant_mm_float8_mixed_e5m2_e4m3fn,
                                TestAllGatherQuantMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, ydtype, world_size])

    @classmethod
    def _test_npu_all_gather_quant_mm_float8_mixed_e4m3fn_e5m2(cls, rank, input_list):
        x1_list, x2_list, ydtype, world_size, init_pg, c2p = input_list
        x1 = x1_list[rank]
        x2 = x2_list[rank]
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, gather_out = torch_npu.npu_all_gather_quant_mm(x1,
                                                            x2,
                                                            hcom_name,
                                                            world_size,
                                                            bias=None,
                                                            x1_scale=None,
                                                            x2_scale=None,
                                                            quant_scale=None,
                                                            gather_index=0,
                                                            gather_output=True,
                                                            comm_turn=0,
                                                            y_dtype=ydtype,
                                                            x1_dtype=torch.float8_e4m3fn,
                                                            x2_dtype=torch.float8_e5m2)

        c2p.put((rank, out.cpu(), gather_out.cpu()))
        pg.barrier()

    @SupportedDevices(['Ascend950'])
    def test_npu_all_gather_quant_mm_float8_mixed_e4m3fn_e5m2(self):
        world_size = 2
        dtype = np.float16
        ydtype = 6
        data_format = -1
        x1_shape = [dtype, data_format, [16, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e4m3fn))
            x2_list.append(x2.to(torch.float8_e5m2))
        expt_out_list, expt_gather = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float32)
        self._test_multiprocess(TestAllGatherQuantMm._test_npu_all_gather_quant_mm_float8_mixed_e4m3fn_e5m2,
                                TestAllGatherQuantMm._init_dist_hccl, [expt_out_list, expt_gather, x1_list, x2_list, ydtype, world_size])

    def _construct_excepted_result_float8(self, x1_list, x2_list, world_size, out_dtype):
        gather_out = torch.cat([x.float() for x in x1_list])
        out_list = []
        for i in range(world_size):
            out_list.append(torch.matmul(gather_out.npu(), x2_list[i].float().npu()).to(out_dtype).cpu())
        return out_list, gather_out