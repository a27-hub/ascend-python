    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_mm_all_reduce_base_float8_e5m2(self):
        world_size = 8
        dtype = np.float16
        dtype_quant = np.int8
        data_format = -1
        x1_shape = [dtype, data_format, [1, 256]]
        x2_shape = [dtype_quant, data_format, [256, 256]]
        scale_shape = [dtype, data_format, [256]]
        offset_shape = [dtype, data_format, [256]]
        x1_list = []
        x2_list = []
        scale, _ = create_common_tensor(scale_shape, 0.0010, 0.0110)
        offset, _ = create_common_tensor(offset_shape, -1, 1)
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e5m2))
            x2_list.append(x2.to(torch.float8_e5m2))
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, scale, offset, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base_float8_e5m2,
                                TestMmAllReduceBase._init_dist_hccl,
                                [expt_out_list, x1_list, x2_list, scale, offset, world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_float8_e5m2(cls, rank, input_list):
        x1, x2, scale, offset, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        scale = scale.npu()
        offset = offset.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', antiquant_scale=scale,
                                               antiquant_offset=offset, bias=None, comm_turn=0,
                                               x1_dtype=torch.float8_e5m2, x2_dtype=torch.float8_e5m2)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_mm_all_reduce_base_float8_e4m3fn(self):
        world_size = 8
        dtype = np.float16
        dtype_quant = np.int8
        data_format = -1
        x1_shape = [dtype, data_format, [1, 256]]
        x2_shape = [dtype_quant, data_format, [256, 256]]
        scale_shape = [dtype, data_format, [256]]
        offset_shape = [dtype, data_format, [256]]
        x1_list = []
        x2_list = []
        scale, _ = create_common_tensor(scale_shape, 0.0010, 0.0110)
        offset, _ = create_common_tensor(offset_shape, -1, 1)
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e4m3fn))
            x2_list.append(x2.to(torch.float8_e4m3fn))
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, scale, offset, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base_float8_e4m3fn,
                                TestMmAllReduceBase._init_dist_hccl,
                                [expt_out_list, x1_list, x2_list, scale, offset, world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_float8_e4m3fn(cls, rank, input_list):
        x1, x2, scale, offset, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        scale = scale.npu()
        offset = offset.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', antiquant_scale=scale,
                                               antiquant_offset=offset, bias=None, comm_turn=0,
                                               x1_dtype=torch.float8_e4m3fn, x2_dtype=torch.float8_e4m3fn)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_mm_all_reduce_base_hifloat8(self):
        world_size = 8
        dtype = np.float16
        dtype_quant = np.int8
        data_format = -1
        x1_shape = [dtype, data_format, [1, 256]]
        x2_shape = [dtype_quant, data_format, [256, 256]]
        scale_shape = [dtype, data_format, [256]]
        offset_shape = [dtype, data_format, [256]]
        x1_list = []
        x2_list = []
        scale, _ = create_common_tensor(scale_shape, 0.0010, 0.0110)
        offset, _ = create_common_tensor(offset_shape, -1, 1)
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.int8))
            x2_list.append(x2.to(torch.int8))
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, scale, offset, world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base_hifloat8,
                                TestMmAllReduceBase._init_dist_hccl,
                                [expt_out_list, x1_list, x2_list, scale, offset, world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_hifloat8(cls, rank, input_list):
        x1, x2, scale, offset, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        scale = scale.npu()
        offset = offset.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', antiquant_scale=scale,
                                               antiquant_offset=offset, bias=None, comm_turn=0,
                                               x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_mm_all_reduce_base_bfloat16(self):
        world_size = 8
        dtype = np.float32
        dtype_quant = np.int8
        data_format = -1
        x1_shape = [dtype, data_format, [1, 256]]
        x2_shape = [dtype_quant, data_format, [256, 256]]
        scale_shape = [dtype, data_format, [256]]
        offset_shape = [dtype, data_format, [256]]
        x1_list = []
        x2_list = []
        scale, _ = create_common_tensor(scale_shape, 0.0010, 0.0110)
        offset, _ = create_common_tensor(offset_shape, -1, 1)
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.bfloat16))
            x2_list.append(x2)
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, scale.to(torch.bfloat16), offset.to(torch.bfloat16), world_size)
        self._test_multiprocess(TestMmAllReduceBase._test_npu_mm_all_reduce_base_bfloat16,
                                TestMmAllReduceBase._init_dist_hccl,
                                [expt_out_list, x1_list, x2_list, scale.to(torch.bfloat16), offset.to(torch.bfloat16), world_size])

    @classmethod
    def _test_npu_mm_all_reduce_base_bfloat16(cls, rank, input_list):
        x1, x2, scale, offset, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcom_name = group.get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        scale = scale.npu()
        offset = offset.npu()
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', antiquant_scale=scale,
                                               antiquant_offset=offset, bias=None, comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()