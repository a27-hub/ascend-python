	self （Tensor）：必选参数，表示matmul计算中的左矩阵，对应公式中的x1，不支持非连续的tensor，数据格式支持ND, 数据类型支持float16、bfloat16、float8_e4m3fn、float8_e5m2、hifloat8，输入shape支持2维，形如(m, k)、(k, n)轴满足matmul算子入参要求，k轴相等，且k轴取值范围为[256, 65535)。
	x2（Tensor）：必选参数，表示matmul计算中的右矩阵，数据格式支持ND，数据类型支持float16、bfloat16、float8_e4m3fn、float8_e5m2、hifloat8，输入shape维度需要和self保持一致。
	hcom（String）：必选参数，表示通信域handle名，通过get_hccl_comm_name接口获取。
	world_size（int）：必选参数，表示通信域内的rank总数， 支持 2、4、8、16、32、64卡。
	*：代表其之前的变量是位置相关，按照顺序输入，必选；之后的变量是键值对赋值的，位置无关，可选（不输入会使用默认值）。
	bias（Tensor）：可选参数，数据格式支持ND，数据类型支持float16、bfloat16、float32。bias仅支持一维，且维度大小与output的第1维大小相同。情形1不支持bias输入为非0的场景，bias数据类型和输入类型一致，情形2 bias数据类型为float32，情形3 bias输入只能为None。
	x1_scale（Tensor）：可选参数，数据格式支持ND，数据类型支持float32、float8_e8m0
	x2_scale（Tensor）：可选参数，数据格式支持ND，数据类型支持float32、float8_e8m0。
	quant_scale（Tensor）：可选参数，数据格式支持ND，数据类型支持float32。
	block_size（int）：可选参数，表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化，block_size由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为block_size= blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0。当前版本只支持block_size输入为0。
	gather_index（int）：可选参数， 表示gather操作对象，0（默认值）：对input做gather，1：对x2做gather。当前版本仅支持输入0。
	gather_output（bool）：可选参数，表示是否需要gather输出。默认值true。
	comm_turn（int）：可选参数，表示rank间通信切分粒度，0（默认值）：表示框架内置切分方式。当前版本仅支持输入0。
	group_sizes（List[Int]）：可选参数， 表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化，group_sizes为[groupSizeM，groupSizeN，groupSizeK]。groupSizeM、groupSizeN、groupSizeK分别表示一个反量化系数在各个维度对应的数的个数。当x1Scale/x2Scale输入都是2维，且数据类型都为FLOAT32时，[groupSizeM，groupSizeN，groupSizeK]取值组合仅支持[128, 128, 128]; 当x1Scale/x2Scale输入都是3维，且数据类型都为FLOAT8_E8M0时，[groupSizeM, groupSizeN, groupSizeK]取值组合仅支持[1, 1, 32]；其他场景输入，当前版本仅支持输入[0, 0, 0]。
	amax_output（bool）:  可选参数, 代表是否输出MM计算的最大值结果，默认值为False，当前版本只支持False。
	y_dtype （int）：可选参数，代表输出数据类型，取值支持：5表示float16、6表示float32、15表示bfloat16。当输入self数据类型为float16或bfloat16时不需要输入，其他场景需要输入。
	x1_dtype（int）：可选参数，代表输入的左矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
	x2_dtype（int）：可选参数，代表输入的右矩阵的数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
	x1_scale_dtype（int）：可选参数，代表输入的左矩阵的量化系数数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
	x2_scale_dtype（int）：可选参数，代表输入的右矩阵的量化系数数据类型，PyTorch原生不支持的数据类型需通过该参数指定。
返回值说明
	y（Tensor）：allgather和matmul计算后输出的结果，如果self数据类型为float16、bfloat16，输出数据类型和self保持一致；如果self类型为float8_e4m3fn、float8_e5m2、hifloat8，输出数据类型由y_dtype指定，shape为[world_size * M, N], 数据格式支持ND。
	gatherout（Tensor）：allgather的结果，作为matmul计算的左矩阵。数据类型和self保持一致。当is_gather_out为true时，shape为(world_size * M, K), 数据格式支持ND。
	amaxout（Tensor）：当前只支持输出为None，数据格式支持ND。
约束说明
	该接口支持训练场景下使用。
	该接口支持TorchAir图模式。
	self的M轴在情形三必须满足为128的整数倍。
	world_size 取值只能为2、4、8、16、32、64。
	输出output为2维，其维度为(m*world_size, n)，world_size为卡数。
	输出gatherout为2维，其维度为(m*rank_size, k)，rank_size为卡数。
	当x1、x2的数据类型为float16、bfloat16时，output计算输出数据类型和x1、x2保持一致，bias暂不支持输入为非0的场景，且不支持amaxOut的输出。
	当x1、x2的数据类型为float8_e4m3fn、float8_e5m2、hifloat8时，output输出数据类型支持float16、float32、bfloat16。支持bias输入为float32。
	当x1、x2的数据类型为float16、bfloat16、hifloat8时，x1和x2数据类型需要保持一致。
	当x1、x2数据类型为float8_e4m3fn、float8_e5m2时，x1和x2数据类型可以为其中一种。
	当x1、x2数据类型为float16、bfloat16、hifloat8、float8_e4m3fn 、float8_e5m2时，x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景。
	情形一和情形二时，group_sizes取值为[0, 0, 0]; 情形三时，group_sizes取值为[128, 128, 128]，bias必须为空tensor。
