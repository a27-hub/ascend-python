    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e5m2_bfloat16_input(self):
        x = self.generate_data(-2, 2, (4, 128), np.float32)
        x = x.to(torch.bfloat16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e5m2)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e4m3fn_bfloat16_input(self):
        x = self.generate_data(-2, 2, (4, 128), np.float32)
        x = x.to(torch.bfloat16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e4m3fn)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_hifloat8_bfloat16_input(self):
        x = self.generate_data(-2, 2, (4, 128), np.float32)
        x = x.to(torch.bfloat16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch_npu.hifloat8)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e5m2_block_size_128_128(self):
        x = self.generate_data(-2, 2, (128, 256), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e5m2, row_block_size=128, col_block_size=128)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e4m3fn_block_size_128_128(self):
        x = self.generate_data(-2, 2, (128, 256), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e4m3fn, row_block_size=128, col_block_size=128)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_hifloat8_block_size_128_128(self):
        x = self.generate_data(-2, 2, (128, 256), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch_npu.hifloat8, row_block_size=128, col_block_size=128)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e5m2_block_size_256_256(self):
        x = self.generate_data(-2, 2, (256, 512), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e5m2, row_block_size=256, col_block_size=256)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e4m3fn_block_size_256_256(self):
        x = self.generate_data(-2, 2, (256, 512), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e4m3fn, row_block_size=256, col_block_size=256)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_hifloat8_block_size_256_256(self):
        x = self.generate_data(-2, 2, (256, 512), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch_npu.hifloat8, row_block_size=256, col_block_size=256)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e5m2_block_size_512_192(self):
        x = self.generate_data(-2, 2, (512, 384), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e5m2, row_block_size=512, col_block_size=192)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e4m3fn_block_size_512_192(self):
        x = self.generate_data(-2, 2, (512, 384), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e4m3fn, row_block_size=512, col_block_size=192)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_hifloat8_block_size_512_192(self):
        x = self.generate_data(-2, 2, (512, 384), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch_npu.hifloat8, row_block_size=512, col_block_size=192)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e5m2_block_size_1_64(self):
        x = self.generate_data(-2, 2, (4, 128), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e5m2, row_block_size=1, col_block_size=64)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_float8_e4m3fn_block_size_1_64(self):
        x = self.generate_data(-2, 2, (4, 128), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch.float8_e4m3fn, row_block_size=1, col_block_size=64)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_dynamic_block_quant_hifloat8_block_size_1_64(self):
        x = self.generate_data(-2, 2, (4, 128), np.float16)
        cpu_output = self.cpu_op_exec(x)
        npu_output = self.npu_op_exec(x, torch_npu.hifloat8, row_block_size=1, col_block_size=64)
        self.assertEqual(cpu_output[0].shape, npu_output[0].shape)
        self.assertEqual(cpu_output[1].shape, npu_output[1].shape)
