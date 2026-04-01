    @classmethod
    def _test_npu_quant_mm_reduce_scatter_float8_e5m2(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                    x2,
                                                    hcom_name,
                                                    world_size,
                                                    reduce_op='sum',
                                                    bias=None,
                                                    comm_turn=0,
                                                    y_dtype=5,
                                                    x1_dtype=torch.float8_e5m2,
                                                    x2_dtype=torch.float8_e5m2)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950'])
    def test_npu_quant_mm_reduce_scatter_float8_e5m2(self):
        world_size = 8
        dtype = np.float16
        data_format = -1
        x1_shape = [dtype, data_format, [128, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e5m2))
            x2_list.append(x2.to(torch.float8_e5m2))
        expt_out_list = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float16)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_float8_e5m2,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @classmethod
    def _test_npu_quant_mm_reduce_scatter_float8_e4m3fn(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                    x2,
                                                    hcom_name,
                                                    world_size,
                                                    reduce_op='sum',
                                                    bias=None,
                                                    comm_turn=0,
                                                    y_dtype=5,
                                                    x1_dtype=torch.float8_e4m3fn,
                                                    x2_dtype=torch.float8_e4m3fn)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950'])
    def test_npu_quant_mm_reduce_scatter_float8_e4m3fn(self):
        world_size = 8
        dtype = np.float16
        data_format = -1
        x1_shape = [dtype, data_format, [128, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e4m3fn))
            x2_list.append(x2.to(torch.float8_e4m3fn))
        expt_out_list = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float16)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_float8_e4m3fn,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @classmethod
    def _test_npu_quant_mm_reduce_scatter_hifloat8(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                    x2,
                                                    hcom_name,
                                                    world_size,
                                                    reduce_op='sum',
                                                    bias=None,
                                                    comm_turn=0,
                                                    y_dtype=5,
                                                    x1_dtype=torch_npu.hifloat8,
                                                    x2_dtype=torch_npu.hifloat8)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950'])
    def test_npu_quant_mm_reduce_scatter_hifloat8(self):
        world_size = 8
        dtype = np.float16
        data_format = -1
        x1_shape = [dtype, data_format, [128, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.int8))
            x2_list.append(x2.to(torch.int8))
        expt_out_list = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float16)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_hifloat8,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @classmethod
    def _test_npu_quant_mm_reduce_scatter_bfloat16(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                    x2,
                                                    hcom_name,
                                                    world_size,
                                                    reduce_op='sum',
                                                    bias=None,
                                                    comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950'])
    def test_npu_quant_mm_reduce_scatter_bfloat16(self):
        world_size = 8
        dtype = np.float32
        data_format = -1
        x1_shape = [dtype, data_format, [128, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.bfloat16))
            x2_list.append(x2.to(torch.bfloat16))
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_bfloat16,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @classmethod
    def _test_npu_quant_mm_reduce_scatter_float8_mixed_e5m2_e4m3fn(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                    x2,
                                                    hcom_name,
                                                    world_size,
                                                    reduce_op='sum',
                                                    bias=None,
                                                    comm_turn=0,
                                                    y_dtype=15,
                                                    x1_dtype=torch.float8_e5m2,
                                                    x2_dtype=torch.float8_e4m3fn)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950'])
    def test_npu_quant_mm_reduce_scatter_float8_mixed_e5m2_e4m3fn(self):
        world_size = 8
        dtype = np.float16
        data_format = -1
        x1_shape = [dtype, data_format, [128, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e5m2))
            x2_list.append(x2.to(torch.float8_e4m3fn))
        expt_out_list = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.bfloat16)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_float8_mixed_e5m2_e4m3fn,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @classmethod
    def _test_npu_quant_mm_reduce_scatter_float8_mixed_e4m3fn_e5m2(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                    x2,
                                                    hcom_name,
                                                    world_size,
                                                    reduce_op='sum',
                                                    bias=None,
                                                    comm_turn=0,
                                                    y_dtype=6,
                                                    x1_dtype=torch.float8_e4m3fn,
                                                    x2_dtype=torch.float8_e5m2)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950'])
    def test_npu_quant_mm_reduce_scatter_float8_mixed_e4m3fn_e5m2(self):
        world_size = 8
        dtype = np.float16
        data_format = -1
        x1_shape = [dtype, data_format, [128, 256]]
        x2_shape = [dtype, data_format, [256, 256]]
        x1_list = []
        x2_list = []
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1.to(torch.float8_e4m3fn))
            x2_list.append(x2.to(torch.float8_e5m2))
        expt_out_list = self._construct_excepted_result_float8(x1_list, x2_list, world_size, torch.float32)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_float8_mixed_e4m3fn_e5m2,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    def _construct_excepted_result_float8(self, x1_list, x2_list, world_size, out_dtype):
        out = None
        for i in range(world_size):
            x1 = x1_list[i].float()
            x2 = x2_list[i].float()
            out_single = torch.matmul(x1, x2)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        out = out.to(out_dtype)
        index = x1_list[0].shape[0] // world_size
        return [out[index * i:index * (i + 1), :] for i in range(world_size)]