	self: Tensor类型，需要进行数据类型转换的源数据张量，数据格式支持ND，支持非连续的Tensor，shape支持0-8维。
−	Atlas 推理系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64。
−	Atlas 训练系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64、bfloat16。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64、bfloat16。
−	PR/Ascend 950DT：数据类型支持float16、float32、double、int8、uint8、int16、int32、int64、bool、complex32、complex64、bfloat16、hifloat8、float8_e5m2、float8_e4m3fn、float4_e1m2fn_x2、float4_e2m1fn_x2。
	dtype: int类型，目标数据类型对应的枚举值，取值范围为out支持的数据类型。
	input_dtype: int类型，可选输入，源数据类型对应的枚举值，默认值为None。当原生torch不支持输入张量的数据类型时，通过该参数表明输入张量的数据类型，取值范围为self支持的数据类型。
输出说明
	out: Tensor类型，完成数据类型转换后的输出张量，数据类型为参数dtype对应的类型，shape与输入张量self相同，数据格式支持ND，且数据格式与self一致。
−	Atlas 推理系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64。
−	Atlas 训练系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64、bfloat16。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float16、float32、int8、uint8、int16、int32、int64、bool、complex64、bfloat16。
−	PR/Ascend 950DT：数据类型支持float16、float32、double、int8、uint8、int16、int32、int64、bool、complex32、complex64、bfloat16、hifloat8、float8_e5m2、float8_e4m3fn、float4_e1m2fn_x2、float4_e2m1fn_x2、int4。
约束说明
	该接口支持推理、训练场景下使用。
	该接口支持TorchAir图模式。
	该接口当前hifloat8、float4_e1m2fn_x2、float4_e2m1fn_x2、int4数据类型实际由uint8承接，且float4_e1m2fn_x2、float4_e2m1fn_x2、int4在使用时需成对拼接成uint8，不支持单独使用。
	当使用uint8承载float4_e1m2fn_x2、float4_e2m1fn_x2作为输入转成非float4数据类型时，输出的尾轴size将变为输入尾轴的2倍。
	当非float4、int4数据类型转成使用uint8承载的float4_e1m2fn_x2、float4_e2m1fn_x2、int4时，输入的尾轴只支持偶数，且输出的尾轴size将变为输入尾轴的1/2。
    @SupportedDevices(['Ascend950P'])
    def test_dtype_cast_shape_format_hifloat8(self):
        cpu_input = torch.randn(1, 1, 3, 3, dtype=torch.float32)
        npu_input = cpu_input.npu()
        npu_output = torch_npu.npu_dtype_cast(npu_input, torch_npu.hifloat8)
        self.assertEqual(npu_output.shape, cpu_input.shape)

    @SupportedDevices(['Ascend950P'])
    def test_dtype_cast_shape_format_float8_e5m2(self):
        cpu_input = torch.randn(1, 1, 3, 3, dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = cpu_input.to(torch.float8_e5m2)
        npu_output = torch_npu.npu_dtype_cast(npu_input, torch.float8_e5m2)
        self.assertEqual(npu_output.shape, cpu_output.shape)
        self.assertEqual(npu_output.dtype, torch.float8_e5m2)


    @SupportedDevices(['Ascend950P'])
    def test_dtype_cast_shape_format_float8_e4m3fn(self):
        cpu_input = torch.randn(1, 1, 3, 3, dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = cpu_input.to(torch.float8_e4m3fn)
        npu_output = torch_npu.npu_dtype_cast(npu_input, torch.float8_e4m3fn)
        self.assertEqual(npu_output.shape, cpu_output.shape)
        self.assertEqual(npu_output.dtype, torch.float8_e4m3fn)

