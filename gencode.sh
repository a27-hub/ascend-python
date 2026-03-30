	in_channels：int类型，必选参数，Conv2d的输入的channel值，满足in_channels%groups=0。
	out_channels：int类型，必选参数，Conv2d的输出的channel值，满足out_channels%groups=0。
	kernel_size：int或者tuple类型，必选参数，卷积核大小，支持单个数字输入或者2维tuple输入。当前仅支持kernel_size范围为[1, 255]。
	output_dtype：torch.dtype类型或int类型，必选参数，输出张量的数据类型。
−	Atlas 推理系列产品：数据类型支持float16。
−	PR/Ascend 950DT：当quant_conv2d_input数据类型为int8（且input_dtype=None）时，output_dtype仅支持float16。当quant_conv2d_input数据类型为float8_e4m3fn（且input_dtype=None）时，output_dtype支持float16、bfloat16、float32。当input_dtype为hifloat8时，output_dtype支持float16、bfloat16、float32、hifloat8（int类型）。
	stride：int或者tuple类型，可选参数，默认值为1，Conv2d的stride，滑动窗口的步长，支持单个数字输入或者2维tuple输入。当前仅支持stride范围为[1, 63]。
	padding：int或者tuple类型，可选参数，默认值为0，Conv2d的padding，应用于输入4条边的填充量，支持单个数字输入或者2维tuple输入。当前仅支持padding范围为[0, 255]。
	dilation：int或者tuple类型，可选参数，默认值为1，Conv2d的dilation，卷积核点与点之间的距离，支持单个数字输入或者2维tuple输入。当前仅支持dilation范围为[1, 255]。
	groups：int类型，可选参数，默认值为1，Conv2d的group。
−	Atlas 推理系列产品：仅支持取1。
−	PR/Ascend 950DT：支持取[1, 65535]。
	bias：bool类型，可选参数，默认值为True，代表是否需要bias计算参数。如果设置成False，则bias不会加入QuantConv2d计算。
	offset：bool类型，可选参数，默认值为False，当前仅支持False。
	offset_x：int类型，可选参数，默认值为0，补的pad中填入的实际值。
示例：如果padding为[1,1]，offset_x为1，fmap变化如图所示：
 
	round_mode：str类型，可选参数，默认值为"rint"。
−	PR/Ascend 950DT：当output_dtype为hifloat8时，仅支持配置“round”，其他场景不做要求，建议不配置。
	input_dtype：int类型，可选参数，默认值为None。
−	PR/Ascend 950DT：需要使能hifloat8类型计算时配置为hifloat8，其他场景配置为None。input_dtype和weight_dtype必须相同。
	weight_dtype：int类型，可选参数，默认值为None。。
−	Atlas 推理系列产品：暂不支持，使用默认值即可。
−	PR/Ascend 950DT：需要使能hifloat8类型计算时配置为hifloat8，其他场景配置为None。weight_dtype和input_dtype必须相同。
	device：可选参数，默认值为None。预留参数，暂未使用。
	dtype：可选参数，默认值为None。预留参数，暂未使用。
输入说明
quant_conv2d_input（计算输入）：Tensor类型，数据格式支持NCHW，shape为4维，其中C轴大小是weight的C轴大小的groups倍。
	Atlas 推理系列产品：数据类型支持int8。
	PR/Ascend 950DT：数据类型支持int8、uint8（仅当input_dtype为hifloat8时支持）、float8_e4m3fn。当input_dtype为hifloat8时，quant_conv2d_input数据类型还支持int8，表示在使能hifloat8数据类型计算时，quant_conv2d_input tensor的数据用8bit类型承载。
变量说明
	weight：Tensor类型，数据格式支持NCHW，shape为4维。
−	PR/Ascend 950DT：数据类型支持int8、uint8（仅当weight_dtype为hifloat8时支持）、float8_e4m3fn。当weight_dtype为hifloat8时，weight数据类型还支持int8，表示在使能hifloat8数据类型计算时，weight tensor的数据用8bit类型承载。
	scale：Tensor类型，数据类型支持int64。数据格式支持ND，shape是1维(n,)，其中n与weight的out_channels一致。
	bias：Tensor类型，可选参数。数据格式支持ND，shape支持1维(n,)，n与weight的out_channels一致。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持int32。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持int32。
−	Atlas 推理系列产品：数据类型支持int32。
−	PR/Ascend 950DT：如果quant_conv2d_input数据类型是int8，bias数据类型为int32。如果quant_conv2d_input数据类型是float8_e4m3fn或input_dtype配置为hifloat8时，则bias数据类型是float32。
输出说明
一个Tensor类型的输出，代表QuantConv2d的计算结果，数据类型和output_dtype一致，数据格式支持NCHW。
	Atlas 推理系列产品：不支持W轴为1的场景。
约束说明
	该接口支持推理场景下使用。
	该接口支持静态图模式在线推理。
	PR/Ascend 950DT：支持单算子模式调
