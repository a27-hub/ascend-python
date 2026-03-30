    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_hifloat8_e8m0fnu_scale(self):
        torch.manual_seed(0)
        M = 16
        K = 256
        N = 32
        g = 1
        group_list = torch.tensor([K, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = torch.randint(1, 2, size=(int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(1, 2, size=(int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0, 1)

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale, group_list, 
                                                     x1_scale=x1_scale, group_list_type=0, 
                                                     group_sizes=None, x1_dtype=torch_npu.hifloat8,
                                                     x2_dtype=torch_npu.hifloat8,
                                                     x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                                     x2_scale_dtype=torch_npu.float8_e8m0fnu)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e5m2_float32_scale(self):
        torch.manual_seed(0)
        M = 16
        K = 256
        N = 32
        g = 1
        group_list = torch.tensor([K, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e5m2).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e5m2).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = np.random.uniform(low=0.5, high=2, size=(g, N)).astype(np.float32)
        x1_scale = np.random.uniform(low=0.5, high=2, size=(g,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale_clone, group_list, 
                                                     x1_scale=x1_scale_clone, group_list_type=0, 
                                                     group_sizes=None, x1_dtype=torch.float8_e5m2,
                                                     x2_dtype=torch.float8_e5m2)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e4m3fn_float32_scale(self):
        torch.manual_seed(0)
        M = 16
        K = 256
        N = 32
        g = 1
        group_list = torch.tensor([K, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e4m3fn).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e4m3fn).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = np.random.uniform(low=0.5, high=2, size=(g, N)).astype(np.float32)
        x1_scale = np.random.uniform(low=0.5, high=2, size=(g,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale_clone, group_list, 
                                                     x1_scale=x1_scale_clone, group_list_type=0, 
                                                     group_sizes=None, x1_dtype=torch.float8_e4m3fn,
                                                     x2_dtype=torch.float8_e4m3fn)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_hifloat8_multi_group(self):
        torch.manual_seed(0)
        M = 16
        K = 512
        N = 32
        g = 2
        group_list = torch.tensor([K // 2, K], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()
        x2_scale = np.random.uniform(low=0.5, high=2, size=(g, N)).astype(np.float32)
        x1_scale = np.random.uniform(low=0.5, high=2, size=(g,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale_clone, group_list, 
                                                     x1_scale=x1_scale_clone, group_list_type=0, 
                                                     group_sizes=None, x1_dtype=torch_npu.hifloat8,
                                                     x2_dtype=torch_npu.hifloat8)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e5m2_multi_group(self):
        torch.manual_seed(0)
        M = 16
        K = 512
        N = 32
        g = 2
        group_list = torch.tensor([K // 2, K], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e5m2).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e5m2).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = torch.randint(1, 2, size=(int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(1, 2, size=(int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0, 1)

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale, group_list, 
                                                     x1_scale=x1_scale, group_list_type=0, 
                                                     group_sizes=None, x1_dtype=torch.float8_e5m2,
                                                     x2_dtype=torch.float8_e5m2,
                                                     x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                                     x2_scale_dtype=torch_npu.float8_e8m0fnu)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e4m3fn_multi_group(self):
        torch.manual_seed(0)
        M = 16
        K = 512
        N = 32
        g = 2
        group_list = torch.tensor([K // 2, K], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e4m3fn).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e4m3fn).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = torch.randint(1, 2, size=(int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(1, 2, size=(int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0, 1)

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale, group_list, 
                                                     x1_scale=x1_scale, group_list_type=0, 
                                                     group_sizes=None, x1_dtype=torch.float8_e4m3fn,
                                                     x2_dtype=torch.float8_e4m3fn,
                                                     x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                                     x2_scale_dtype=torch_npu.float8_e8m0fnu)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_hifloat8_group_list_type1(self):
        torch.manual_seed(0)
        M = 16
        K = 256
        N = 32
        g = 1
        group_list = torch.tensor([K, ], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()
        x2_scale = np.random.uniform(low=0.5, high=2, size=(g, N)).astype(np.float32)
        x1_scale = np.random.uniform(low=0.5, high=2, size=(g,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale_clone, group_list, 
                                                     x1_scale=x1_scale_clone, group_list_type=1, 
                                                     group_sizes=None, x1_dtype=torch_npu.hifloat8,
                                                     x2_dtype=torch_npu.hifloat8)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e5m2_group_list_type1(self):
        torch.manual_seed(0)
        M = 16
        K = 256
        N = 32
        g = 1
        group_list = torch.tensor([K, ], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e5m2).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e5m2).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = torch.randint(1, 2, size=(int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(1, 2, size=(int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0, 1)

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale, group_list, 
                                                     x1_scale=x1_scale, group_list_type=1, 
                                                     group_sizes=None, x1_dtype=torch.float8_e5m2,
                                                     x2_dtype=torch.float8_e5m2,
                                                     x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                                     x2_scale_dtype=torch_npu.float8_e8m0fnu)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e4m3fn_group_list_type1(self):
        torch.manual_seed(0)
        M = 16
        K = 256
        N = 32
        g = 1
        group_list = torch.tensor([K, ], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e4m3fn).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e4m3fn).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = torch.randint(1, 2, size=(int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(1, 2, size=(int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0, 1)

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale, group_list, 
                                                     x1_scale=x1_scale, group_list_type=1, 
                                                     group_sizes=None, x1_dtype=torch.float8_e4m3fn,
                                                     x2_dtype=torch.float8_e4m3fn,
                                                     x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                                     x2_scale_dtype=torch_npu.float8_e8m0fnu)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_hifloat8_multi_group_type1(self):
        torch.manual_seed(0)
        M = 16
        K = 512
        N = 32
        g = 2
        group_list = torch.tensor([K // 2, K // 2], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()
        x2_scale = np.random.uniform(low=0.5, high=2, size=(g, N)).astype(np.float32)
        x1_scale = np.random.uniform(low=0.5, high=2, size=(g,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale_clone, group_list, 
                                                     x1_scale=x1_scale_clone, group_list_type=1, 
                                                     group_sizes=None, x1_dtype=torch_npu.hifloat8,
                                                     x2_dtype=torch_npu.hifloat8)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e5m2_multi_group_type1(self):
        torch.manual_seed(0)
        M = 16
        K = 512
        N = 32
        g = 2
        group_list = torch.tensor([K // 2, K // 2], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e5m2).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e5m2).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = torch.randint(1, 2, size=(int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(1, 2, size=(int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0, 1)

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale, group_list, 
                                                     x1_scale=x1_scale, group_list_type=1, 
                                                     group_sizes=None, x1_dtype=torch.float8_e5m2,
                                                     x2_dtype=torch.float8_e5m2,
                                                     x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                                     x2_scale_dtype=torch_npu.float8_e8m0fnu)
        self.assertEqual(custom_output.shape, y.shape)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e4m3fn_multi_group_type1(self):
        torch.manual_seed(0)
        M = 16
        K = 512
        N = 32
        g = 2
        group_list = torch.tensor([K // 2, K // 2], dtype=torch.int64).npu()

        x1 = torch.randint(2, 3, size=(K, M), dtype=torch.int8).view(torch.float8_e4m3fn).npu().transpose(0, 1)
        x2 = torch.randint(2, 3, size=(K, N), dtype=torch.int8).view(torch.float8_e4m3fn).npu()

        y = torch.randint(2, 3, size=(g, M, N), dtype=torch.float32)
        y_clone = y.clone().npu()

        x2_scale = torch.randint(1, 2, size=(int(K/64 + g), N, 2), dtype=torch.int8).npu()
        x1_scale = torch.randint(1, 2, size=(int(K/64 + g), M, 2), dtype=torch.int8).npu().transpose(0, 1)

        custom_output = torch_npu.npu_add_quant_gmm_(y_clone, x1, x2, x2_scale, group_list, 
                                                     x1_scale=x1_scale, group_list_type=1, 
                                                     group_sizes=None, x1_dtype=torch.float8_e4m3fn,
                                                     x2_dtype=torch.float8_e4m3fn,
                                                     x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                                     x2_scale_dtype=torch_npu.float8_e8m0fnu)
        self.assertEqual(custom_output.shape, y.shape)
