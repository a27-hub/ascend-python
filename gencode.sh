    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_hifloat8_static(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float32) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        torch.npu.synchronize()

        dst_type = torch_npu.hifloat8
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'round'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), input_offset_tensor.cpu(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_hifloat8_hybrid_round(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float32) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        torch.npu.synchronize()

        dst_type = torch_npu.hifloat8
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'hybrid'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), input_offset_tensor.cpu(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e5m2_dynamic(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float32) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e5m2
        quant_mode = 'dynamic'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=None,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), None, approximate, quant_mode=quant_mode)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)
        if out_scale is not None:
            self.assertRtolEqual(out_scale.type(torch.float32), golden_out_scale.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e4m3fn_dynamic(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float32) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e4m3fn
        quant_mode = 'dynamic'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=None,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), None, approximate, quant_mode=quant_mode)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)
        if out_scale is not None:
            self.assertRtolEqual(out_scale.type(torch.float32), golden_out_scale.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e5m2_approximate_none(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float32) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e5m2
        quant_mode = 'static'
        approximate = "none"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), input_offset_tensor.cpu(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e4m3fn_approximate_none(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float32) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e4m3fn
        quant_mode = 'static'
        approximate = "none"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), input_offset_tensor.cpu(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_hifloat8_approximate_none(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float32) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float32).npu()
        torch.npu.synchronize()

        dst_type = torch_npu.hifloat8
        quant_mode = 'static'
        approximate = "none"
        round_mode = 'round'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu(), input_scale_tensor.cpu(), input_offset_tensor.cpu(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e5m2_float16_input(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float16) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e5m2
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu().float(), input_scale_tensor.cpu().float(), input_offset_tensor.cpu().float(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e4m3fn_float16_input(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float16) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e4m3fn
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu().float(), input_scale_tensor.cpu().float(), input_offset_tensor.cpu().float(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_hifloat8_float16_input(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.float16) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.float16).npu()
        torch.npu.synchronize()

        dst_type = torch_npu.hifloat8
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'round'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu().float(), input_scale_tensor.cpu().float(), input_offset_tensor.cpu().float(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e5m2_bfloat16_input(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.bfloat16) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.bfloat16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.bfloat16).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e5m2
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu().float(), input_scale_tensor.cpu().float(), input_offset_tensor.cpu().float(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_float8_e4m3fn_bfloat16_input(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.bfloat16) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.bfloat16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.bfloat16).npu()
        torch.npu.synchronize()

        dst_type = torch.float8_e4m3fn
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'rint'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu().float(), input_scale_tensor.cpu().float(), input_offset_tensor.cpu().float(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)

    @SupportedDevices(['Ascend950P'])
    def test_npu_gelu_quant_hifloat8_bfloat16_input(self):
        shape = [100, 400]

        input_self_tensor = (torch.rand(shape, dtype=torch.bfloat16) * 256).npu()
        input_scale_tensor = torch.rand(shape[-1:], dtype=torch.bfloat16).npu()
        input_offset_tensor = torch.rand(shape[-1:], dtype=torch.bfloat16).npu()
        torch.npu.synchronize()

        dst_type = torch_npu.hifloat8
        quant_mode = 'static'
        approximate = "tanh"
        round_mode = 'round'
        output, out_scale = torch.ops.npu.npu_gelu_quant(input_self_tensor,
                                                         input_scale=input_scale_tensor,
                                                         input_offset=input_offset_tensor,
                                                         approximate=approximate,
                                                         quant_mode=quant_mode,
                                                         dst_type=dst_type,
                                                         round_mode=round_mode)
        torch.npu.synchronize()
        golden, golden_out_scale = self.get_golden(input_self_tensor.cpu().float(), input_scale_tensor.cpu().float(), input_offset_tensor.cpu().float(), approximate)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32), prec=1.01)