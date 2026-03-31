	x1：Tensor类型，数据格式支持ND，输入shape支持2维或者3维。
−	PR/Ascend 950DT：数据类型支持int8、float16、bfloat16、float8_e4m3fn、float8_e5m2、hifloat8、float4_e2m1fn_x2。
	x2：Tensor类型。非量化场景，数据类型需要和x1保持一致，输入shape维度第0维和x1的最后一维保持一致。
−	PR/Ascend 950DT：数据类型支持int8、float16、bfloat16、float8_e4m3fn、float8_e5m2、hifloat8、float4_e2m1fn_x2。数据格式支持ND。
−	Atlas 推理系列加速卡产品上数据格式仅支持NZ。
	hcom：String类型，通信域handle名，通过get_hccl_comm_name接口获取。
	*：代表其之前的变量是位置相关，按照顺序输入，必选；之后的变量是键值对赋值的，位置无关，可选（不输入会使用默认值）。
	reduce_op：String类型，reduce操作类型，当前版本仅支持'sum'，默认值：'sum'。
	bias：Tensor类型，可选参数，数据格式支持ND。bias当前仅支持一维，且维度大小与output/x2的最后一维大小相同。
−	PR/Ascend 950DT：数据类型支持int32、float16、bfloat16、float32。
	antiquant_scale：Tensor类型，可选参数，伪量化场景对x2进行去量化的系数，数据类型支持float16、bfloat16，数据格式支持ND。伪量化场景数据类型需要和x1保持一致。
−	pertensor场景：shape为[1]。
−	perchannel场景：shape为[1,n]或者[n]，n为x2最后一维的大小。
−	pergroup场景：shape为[ceil(k, antiquant_group_size), n]。其中k为x2第一维的大小，n为x2最后一维的大小，antiquant_group_size为伪量化场景对输入x2进行反量化计算的groupSize输入。
 
ceil(k, antiquant_group_size)的计算逻辑为：(k+antiquant_group_siz-1)/antiquant_group_size，并对计算结果取整数部分。
	antiquant_offset：Tensor类型，可选参数，伪量化场景对x2进行去量化的系数，数据类型支持float16、bfloat16，数据格式支持ND。数据类型、shape需要和antiquant_scale保持一致。
	x3：Tensor类型，可选参数，matmul计算后的偏移。数据格式支持ND。数据类型、shape需要和输出output保持一致。
−	PR/Ascend 950DT：数据类型支持float16、bfloat16、float32。
	dequant_scale：Tensor类型，可选参数，matmul计算后的去量化系数。数据格式支持ND。
支持的量化场景如下：
−	pertensor场景：shape为[1]。
−	perchannel场景：shape为[n]/[1,n]，n为x2最后一维的大小。
−	mx量化场景：数据类型为float8_e8m0fnu时， 仅支持转置，x2 shape为[n, k]时，x2Scale的shape为[n, ceilDiv(k, 32)]，且必须保证ceilDiv(k, 32)为偶数。
−	perblock场景：shape为[ceilDiv(n, 128), ceilDiv(k, 128)]。
−	PR/Ascend 950DT：支持pertensor、perchannel、mx、perblock场景。数据类型支持int64、uint64、bfloat16、float32、float8_e8m0fnu。
	pertoken_scale：Tensor类型，可选参数，matmul计算后的pertoken去量化系数。若数据类型为float32，当x1为[m,k]时pertoken_scale shape为[m]；当x1为[b, s, k]时pertoken_scale shape为[b*s]。
−	PR/Ascend 950DT：数据类型支持float32、float8_e8m0fnu。若数据类型为float8_e8m0fnu，mx场景：shape为[s, ceilDiv(k, 64), 2]或者[m, ceilDiv(k, 64), 2]，且必须保证ceilDiv(k, 32)为偶数；perblock场景：shape为[ceilDiv(m, 128), ceilDiv(k, 128)]。
	comm_quant_scale_1: Tensor类型，可选参数，alltoall通信前后的量化、去量化系数。支持float16、bfloat16，支持ND格式。x2为[k, n]时shape为[1, n]或[n]，用户需保证每张卡上数据保持一致且正确。
	comm_quant_scale_2: Tensor类型，可选参数，allgather通信前后的量化、去量化系数。支持float16、bfloat16，支持ND格式。x2为[k, n]时shape为[1, n]或[n]，用户需保证每张卡上数据保持一致且正确。
	antiquant_group_size：int类型，表示伪量化pre-group算法模式下，对输入x2进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在k轴方向的大小。当伪量化算法模式不为pre-group时传入0；当伪量化算法模式为pre-group时传入值的范围为[32, min(k-1, INT_MAX)]且值要求是32的倍数，其中k为x2第一维的大小。默认值0，为0则表示非per-group场景。
	comm_turn：int类型，表示rank间通信切分粒度，默认值：0，表示默认的切分方式。当前版本仅支持输入0。
	group_sizes：List[int]类型, 用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。group_sizes为[groupSizeM，groupSizeN，groupSizeK]。groupSizeM，groupSizeN，groupSizeK表示一个反量化系数在各个维度对应的数的个数。
	y_dtype：int类型, 可选参数, 代表输出数据类型 (5表示float16，6表示float32，15表示bfloat16）。
	x1_dtype：int类型，可选参数，代表输入数据类型，当self tensor的数据类型为hifloat8时，需要输入hifloat8，类型为float4_e2m1fn_x2时输入float4_e2m1fn_x2。
	x2_dtype：int类型，可选参数，代表输入数据类型，当x2 tensor的数据类型为hifloat8时，需要输入hifloat8，类型为float4_e2m1fn_x2时输入float4_e2m1fn_x2。
	dequant_scale_dtype：int类型，可选参数，代表输入数据类型，当dequant_scale数据类型为float8_e8m0fnu时，需输入float8_e8m0fnu类型。
	pertoken_scale_dtype：int类型，可选参数，代表输入数据类型，当pertoken_scale数据类型为float8_e8m0fnu时，需输入float8_e8m0fnu类型。
	comm_quant_mode：int类型，可选参数，代表低比特通信的量化模式，取值为0或1，当需要动态量化的低比特通信时，需要输入为1。
输出说明
Tensor类型，数据类型非量化场景以及伪量化场景与x1保持一致，全量化场景输出数据类型为float16或bfloat16。shape第0维度和x1的0维保持一致，若x1为2维，shape第1维度和x2的1维保持一致，若x1为3维，shape第1维度和x1的1维保持一致，shape第2维度和x2的1维保持一致。
约束说明
	增量场景不使能该融合算子，全量场景使能该融合算子。
	该接口支持推理场景下使用。
	该接口支持TorchAir图模式。
	输入x1可为2维或者3维、x2必须是2维，分别为(b, s, k)/(m, k), (k, n)，k轴满足mm算子入参要求，k轴相等。bias当前仅支持一维，且维度大小与output的最后一维大小相同。x3的shape与output的shape相同。
	x1不支持输入转置后的tensor，x2转置后输入，需要满足shape的第一维大小与x1的最后一维相同，满足matmul的计算条件。
	antiquant_group_size中k值的范围与matmul一致，为[1,65535]，INT_MAX大于(k-1)。
	Atlas 推理系列加速卡产品：
−	x1、x2支持为空tensor，即k轴为0。
−	只支持2卡。
−	非量化、伪量化、全量化等场景仅支持图模式，且需要使能weightNZ。如果是输入为ND格式数据，需要调用torch_npu.npu_format_cast进行格式转换（处理方法见对应的接口使用说明），由于格式限制，k、n需要分别32、16对齐。
−	非量化场景支持x2矩阵转置或非转置场景，伪量化和全量化仅支持x2矩阵转置场景，需要调用.transpose()接口构造，可参考调用示例。
	全量化场景：m取值范围均为[1, 2147483647]，x1、x2的最后一维范围为[1, 65535]，即k的取值范围为[1, 65535]、仅当x2(shape=[n, k])为转置时n可以大于65535。
	伪量化场景：m取值范围均为[1, 2147483647]，k、n的取值范围为[1, 65535]。
	一个模型中的通算融合MC2算子，仅支持相同通信域。
	在长序列场景，随着b/s或者m的增大，可能出现内存不足或者计算超时。
	不同场景下数据类型支持情况：
