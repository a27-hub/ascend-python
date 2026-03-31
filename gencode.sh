    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_mm_all_reduce_base_float8_e5m2(cls, rank, input_list):
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
        deqscale = torch.randn(1, dtype=torch.float32).npu()
        dequant_scale = torch_npu.npu_trans_quant_param(deqscale, None)
        out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', antiquant_scale=scale,
                                               antiquant_offset=offset, bias=None, comm_turn=0, dequant_scale=dequant_scale,
                                               x1_dtype=torch.float8_e5m2, x2_dtype=torch.float8_e5m2, y_dtype=torch.float16)

        c2p.put((rank, out.cpu()))
        pg.barrier()