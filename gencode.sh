	self（Tensor）：必选输入，需要gelu函数激活并量化的数据，数据类型支持float16、bfloat16、float32，数据格式支持ND。
−	动态量化场景下（即quant_mode为'dynamic'时），self的维度是2~8维。
−	静态量化场景下（即quant_mode为'static'时），self的维度是1~8维。
	input_scale（Tensor）：可选输入，表示输入的量化尺度（静态场景）或平滑缩放尺度（动态场景），quant_mode为'static'（即静态场景）时为必选输入且不能为None。数据类型支持float16、bfloat16、float32，shape仅支持1维，大小只能是self的尾轴维度大小或1。
	input_offset（Tensor）：可选输入，表示静态量化场景下量化输入的偏置。数据类型支持float16、bfloat16、float32。shape仅支持1维，且需要与input_offset的shape保持一致。当quant_mode为'dynamic'（即动态场景）时，若input_scale为None，则input_offset也需要为None。
	approximate（Str）：可选输入，表示gelu激活函数的模式，支持'none'、'tanh'，分别对应gelu激活函数的erf和tanh模式。默认值为'none'。
	quant_mode（Str）：可选输入，表示量化模式，支持'dynamic'、'static'，分别对应动态量化和静态量化模式。默认值为'dynamic'。
	dst_type（int）：可选输入，指定输出y的数据类型，默认值为None（实际采用torch.int8作为目标量化类），数据类型支持取值int8、hifloat8、float8_e5m2、float8_e4m3fn。
	round_mode（Str）：可选输入，表示最终cast到量化输出的舍入模式。默认值为'rint'。
−	当dst_type为int8、float8_e5m2、float8_e4m3fn时，支持'rint'。
−	当dst_type为hifloat8时，支持'round'、'hybrid'。
返回值说明
	y（Tensor）：self量化后的对应结果，数据类型由参数dst_type指定，支持int8、hifloat8、float8_e5m2、float8_e4m3fn。shape与输入self一致。
	out_scale（Tensor）：表示动态量化的尺度，数据类型支持float32，shape为输入self剔除最后一个维度的结果。如self的shape为[a, b, c]，则out_scale的shape为[a, b]。当quant_mode为'static'时，输出为None。
约束说明
	该接口支持推理场景下使用。
	该接口支持TorchAir图模式。
	dst_type（即目标数据类型）为hifloat8时，必须传入该数据类型支持的round_mode（即'round'或'hybrid'），不支持默认值'rint'。
	dst_type（即目标数据类型）为hifloat8时，输出y用uint8承载（即输出y的数据类型为uint8，但数据排布仍表示hifloat8的数值）。
	input_scale的数据类型需要与self的数据类型保持一致，或者采用比self的数据类型精度更高的数据类型（例如self为bfloat16，input_scale为float32）。
	input_offset的数据类型需要与input_scale的数据类型保持一致
