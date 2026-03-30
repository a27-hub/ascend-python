
	x：Tensor类型，需要量化的数据。只支持2维输入。数据类型支持bfloat16、float16。
	min_scale：float类型的属性，scale的最小值，取值需要不小于0。默认值为0.0。
	round_mode：string类型的属性，指定量化后结果cast到输出y数据类型的模式。默认值"rint"。
−	dst_type为float8_e5m2、float8_e4m3fn、float8_e4m3fn时，本参数仅支持取"rint"。
−	dst_type为hifloat8时，本参数仅支持取"round"。
	dst_type：int类型的属性，指定输出y的数据类型，支持取值float8_e5m2、float8_e4m3fn、hifloat8。默认类型为float8_e5m2。
	row_block_size：int类型的属性，指定一个block的行大小。支持取值1、128、256、512。默认值1。
	col_block_size：int类型的属性，指定一个block的列大小。支持取值64、128、192、256。默认值128。
输出说明
	y：Tensor类型，量化后的输出数据。数据类型由参数dst_type决定。shape和输入x一致。
−	单算子模式/静态图模式：当dst_type为hifloat8时，y输出的tensor数据类型为uint8（实际承载的是hifloat8类型）。
−	动态图模式：当dst_type为hifloat8时，y输出的tensor数据类型为torch.bits8（实际承载的是hifloat8类型）。
	scale：Tensor类型，量化时使用的scale。数据类型为float32。如果输入x的shape为[M, N]，scale的shape为[ceil(M/row_block_size), ceil(N/col_block_size)]。
约束说明
	该接口支持推理、训练场景下使用。
	该接口支持TorchAir图模式。
