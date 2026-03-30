	self（Tensor）：必选输入，待累加矩阵，数据类型支持float32，tensor支持3维，shape为(g, M, N)。数据格式支持ND。
	x1（Tensor）：必选输入，表示矩阵乘法中的左矩阵，数据类型支持float8_e5m2、float8_e4m3fn、hifloat8，tensor支持2维，shape为(K, M)，其中hifloat8需配置可选参数x1_dtype为对应类型，此时x1本身的dtype不再生效，但仍需保证x1本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	x2（Tensor）：必选输入，表示矩阵乘法中的右矩阵，数据类型支持float8_e5m2、float8_e4m3fn、hifloat8，tensor支持2维，shape为(K, N)，其中hifloat8需配置可选参数x2_dtype为对应类型，此时x2本身的dtype不再生效，但仍需保证x2本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	x2_scale（Tensor）：必选输入，表示矩阵乘法中的右矩阵的缩放因子，数据类型支持float8_e8m0fnu、float32，shape支持2维或3维，其中float8_e8m0fnu需配置可选参数x2_scale_dtype为对应类型，此时x2_scale本身的dtype不再生效，但仍需保证x2_scale本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	group_list（Tensor）：必选输入，代表输入和输出分组轴方向的matmul大小分布, 数据类型支持int64, 1维tensor，shape为(g，), 数据格式支持ND。当group_list_type为0时，group_list必须为非负单调非递减数列；当group_list_type为1时，group_list必须为非负数列。
	x1_scale（Tensor）：可选输入，表示矩阵乘法中的左矩阵的缩放因子，数据类型支持float8_e8m0fnu、float32，shape支持1维、2维或3维，其中float8_e8m0fnu需配置可选参数x1_scale_dtype为对应类型，此时x1_scale本身的dtype不再生效，但仍需保证x1_scale本身的dtype为8bit位的数据类型，以保证shape正确。数据格式支持ND。
	group_list_type（int）：可选输入，代表group_list的表达形式。数据类型支持int32。
0：默认值，group_list中数值为分组轴大小的cumsum结果（累积和）。
1：group_list中数值为分组轴上每组大小。
	group_sizes （List[int]）：可选输入，列表大小为3，分别表示M、N、K三个轴上多少个参数共用一个缩放因子。数据类型支持int32，目前只支持默认值None。
	x1_dtype（int）：可选输入，可用于在x1无法用torch原生数据类型表示时显式指定x1的数据类型。
None：默认值，表示输入x1真实的数据类型与输入x1的dtype相同。
赋予输入x1真实的数据类型，当前仅支持hifloat8。
	x2_dtype（int）：可选输入，可用于在x2无法用torch原生数据类型表示时显式指定x2的数据类型。
None：默认值，表示输入x2真实的数据类型与输入x2的dtype相同。
赋予输入x2真实的数据类型，当前仅支持hifloat8。
	x1_scale_dtype（int）：可选输入，用于在x1_scale无法用torch原生数据类型表示时显式指定x1_scale的数据类型。
None：默认值，表示输入x1_scale真实的数据类型与输入x1_scale的dtype相同。
赋予输入x1_scale真实的数据类型，当前仅支持float8_e8m0fnu。
	x2_scale_dtype（int）：可选输入，用于在x2_scale无法用torch原生数据类型表示时显示指定x2_scale的数据类型。
None：默认值，表示输入x2_scale真实的数据类型与输入x2_scale的dtype相同。
赋予输入x2_scale真实的数据类型，当前仅支持float8_e8m0fnu。
返回值说明
self（Tensor）：groupedMatmul计算完成后与待累加矩阵相加得到的最后结果矩阵，支持数据类型、shape、数据格式均与输入self保持一致。
约束说明
	该接口支持训练场景下使用。
	该接口支持图模式（目前仅支持Pytorch2.1版本）。
	group_list第1维最大支持1024，即最多支持1024个group。
	数据类型约束：
