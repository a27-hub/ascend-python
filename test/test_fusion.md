5.10 torch_npu.npu_fused_infer_attention_score_v2（910D）
（obp：沈维铃/测试-赵静，david：王哲）
首发：C23
产品支持情况
Atlas A2 训练系列产品/Atlas A2 推理系列产品
Atlas A3 训练系列产品/Atlas A3 推理系列产品
PR/Ascend 950DT
功能说明
	接口功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当不涉及system prefix、左padding、kv量化参数合一、per-tensor全量化的场景，推荐使用本接口，否则使用老接口npu_fused_infer_attention_score。
	计算公式：
 
其中Q和KT的乘积代表输入X的注意力，为避免该值变得过大，通常除以d的开根号进行缩放，并对每行进行Softmax归一化，与V相乘后得到一个n*d的矩阵。
函数原型
torch_npu.npu_fused_infer_attention_score_v2(query, key, value, *, query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None) -> (Tensor, Tensor)
参数说明
 
	query、key、value参数的维度含义：B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
	Q_S表示query shape中的S，KV_S表示key和value shape中的S，N表示num_query_heads。
	query（Tensor）：必选参数，表示attention结构的Query输入，对应公式中的Q。不支持非连续的Tensor，数据格式支持ND。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持float16、bfloat16。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float16、bfloat16。
−	PR/Ascend 950DT：数据类型支持float16、bfloat16、int8、hifloat8、float8_e4m3fn。
	key（Tensor）：必选参数，表示attention结构的Key输入，对应公式中的K。不支持非连续的Tensor，数据格式支持ND。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持float16、bfloat16、int8、int4（int32）。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float16、bfloat16、int8、int4（int32）。
−	PR/Ascend 950DT：数据类型支持float16、bfloat16、int8、int4（int32）、hifloat8、float8_e4m3fn、float4_e2m1fn_x2。
	value（Tensor）：必选参数，表示attention结构的Value输入，对应公式中的V。不支持非连续的Tensor，数据格式支持ND。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持float16、bfloat16、int8、int4（int32）。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float16、bfloat16、int8、int4（int32）。
−	PR/Ascend 950DT：数据类型支持float16、bfloat16、int8、int4（int32）、hifloat8、float8_e4m3fn、float4_e2m1fn_x2。
	*：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。
	query_rope（Tensor）：可选参数，表示MLA（Multi-head Latent Attention）结构中query的rope信息，数据类型支持float16、bfloat16，不支持非连续的Tensor，数据格式支持ND。
	key_rope（Tensor）：可选参数，表示MLA（Multi-head Latent Attention）结构中的key的rope信息，数据类型支持float16、bfloat16，不支持非连续的Tensor，数据格式支持ND。
	pse_shift（Tensor）：可选参数，表示attention结构内部的位置编码参数，数据类型支持float16、bfloat16，数据类型与query数据类型需满足类型推导规则。不支持非连续的Tensor，数据格式支持ND。如不使用该功能可传入None。
−	Q_S不为1，当query为bfloat16类型时，要求pse_shift为bfloat16类型；query为其他类型时要求pse_shift为float16类型。输入shape类型需为(B, N, Q_S, KV_S)或(1, N, Q_S, KV_S)。其中Q_S为query的shape中的S，KV_S为key和value的shape中的S。对于pse_shift的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
−	Q_S为1，当pse_shift为float16类型时，要求query为float16类型；当pse_shift为bfloat16类型时，要求query为bfloat16类型。输入shape类型需为(B, N, 1, KV_S)或(1, N, 1, KV_S)。对于pse_shift的KV_S为非32对齐的场景，建议padding到32字节来提高性能，多余部分的填充值不做要求。
	atten_mask（Tensor）：可选参数，对QK结果进行mask，用来指示是否计算Token间的相关性。数据类型支持bool、int8和uint8。不支持非连续的Tensor，数据格式支持ND。如不使用该功能可传入None。
−	Q_S不为1时，建议shape输入(Q_S, KV_S)、(B, Q_S, KV_S)、(1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S)。
−	Q_S为1时，建议shape输入(B, KV_S)、(B, 1, KV_S)、(B, 1, 1, KV_S)。
如果Q_S、KV_S非16或32对齐，可以向上取到对齐的S。综合约束请见约束说明。
	actual_seq_qlen（List[int]）：可选参数，表示不同Batch中query的有效seqlen，数据类型支持int64。如果不指定seqlen可传入None，表示和query的shape的S长度相同。
Atlas A2 训练系列产品/Atlas A2 推理系列产品/Atlas A3 训练系列产品/Atlas A3 推理系列产品有如下限制：
该入参中每个batch的有效seqlen应该不大于query中对应batch的seqlen。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度大于等于Batch时取seqlen的前Batch个数。其他长度不支持。当query的input_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
	actual_seq_kvlen（List[int]）：可选参数，表示不同Batch中key/value的有效seqlenKv，数据类型支持int64。如果不指定None，表示和key/value的shape的S长度相同。不同O_S值有不同的约束，具体参见约束说明。
	block_table（Tensor）：可选参数，表示PageAttention中KV存储使用的block映射表，数据类型支持int32。数据格式支持ND。如不使用该功能可传入None。
	dequant_scale_query（Tensor）：可选参数，表示query的反量化参数，支持per-token叠加per-head和per-block。数据类型支持float32。数据格式支持ND，如不使用该功能可传入None，综合约束请见约束说明。
	dequant_scale_key（Tensor）：可选参数，表示key的反量化因子，用于kv伪量化参数分离和FP8 per-block全量化场景。数据格式支持ND。kv伪量化参数分离场景下，通常支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale和per-token-group。FP8 全量化场景下支持per-block。如不使用该功能可传入None，综合约束请见约束说明。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持float16、bfloat16、float32。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float16、bfloat16、float32。
−	PR/Ascend 950DT：数据类型支持float16、bfloat16、float32、float8_e8m0fnu。
	dequant_offset_key（Tensor）：可选参数，kv伪量化参数分离时表示key的反量化偏移。数据类型支持float16、bfloat16、float32。数据格式支持ND。支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset。如不使用该功能可传入None，综合约束请见约束说明。
	dequant_scale_value（Tensor）：可选参数，表示value的反量化因子，用于kv伪量化参数分离和FP8 per-block全量化场景。数据类型支持float16、bfloat16、float32、float8_e8m0fnu。数据格式支持ND。kv伪量化参数分离场景下，支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale和per-token-group。FP8 全量化场景下支持per-block。如不使用该功能可传入None，综合约束请见约束说明。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品：数据类型支持float16、bfloat16、float32。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float16、bfloat16、float32。
−	PR/Ascend 950DT：数据类型支持float16、bfloat16、float32、float8_e8m0fnu。
	dequant_offset_value（Tensor）：可选参数，kv伪量化参数分离时表示value的反量化偏移。数据类型支持float16、bfloat16、float32。数据格式支持ND。支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset。如不使用该功能可传入None，综合约束请见约束说明。
	dequant_scale_key_rope（Tensor）：可选参数，预留参数，暂未使用，使用默认值即可。
	quant_scale_out（Tensor）：可选参数，表示输出的量化因子。数据类型支持float32、bfloat16。数据格式支持ND。支持per-tensor、per-channel。当输入为bfloat16时，同时支持float32、bfloat16，否则仅支持float32。per-channel格式，当输出layout为BSH时，要求quant_scale_out所有维度的乘积等于H；其他layout要求乘积等于N*D（建议输出layout为BSH时，quant_scale_out shape传入(1, 1, H)或(H,)；输出为BNSD时，建议传入(1, N, 1, D)或(N, D)；输出为BSND时，建议传入(1, 1, N, D)或(N, D)）；输出为TND时，建议传入(1, N, 1, D)或(N, D)。如不使用该功能可传入None，综合约束请见约束说明。
	quant_offset_out（Tensor）：可选参数，表示输出的量化偏移。数据类型支持float32、bfloat16。数据格式支持ND。支持per-tensor、per-channel。若传入quant_offset_out，需保证其类型和shape信息与quant_scale_out一致。如不使用该功能可传入None，综合约束请见约束说明。
	num_query_heads（int）：可选参数，代表query的head个数，数据类型支持int64，在BNSD场景下，需要与shape中的query的N轴shape值相同，否则执行异常。
	num_key_value_heads（int）：可选参数，代表key、value中head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，数据类型支持int64。用户不特意指定时可传入默认值0，表示key/value/query的head个数相等，需要满足num_query_heads整除num_key_value_heads，num_query_heads与num_key_value_heads的比值不能大于64。在BSND、BNSD、BNSD_BSND（仅支持Q_S大于1）场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常。
	softmax_scale（float）：可选参数，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持float32。数据类型与query数据类型需满足数据类型推导规则。用户不特意指定时可传入默认值1.0。
	pre_tokens（int）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联，数据类型支持int64。用户不特意指定时可传入默认值2147483647，Q_S为1时该参数无效。
	next_tokens（int）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持int64。用户不特意指定时可传入默认值2147483647，Q_S为1时该参数无效。
	input_layout（str）：可选参数，用于标识输入query、key、value的数据排布格式，用户不特意指定时可传入默认值"BSH"。
 
注意排布格式带下划线时，下划线左边表示输入query的layout，下划线右边表示输出output的格式，算子内部会进行layout转换。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持BSH、BSND、BNSD、BNSD_BSND（输入为BNSD时，输出格式为BSND，仅支持Q_S大于1）、BSH_NBSD、BSND_NBSD、BNSD_NBSD（输出格式为NBSD时，仅支持Q_S大于1且小于等于16）、TND、TND_NTD、NTD_TND（TND相关场景综合约束请见约束说明）。
−	PR/Ascend 950DT：支持BSH、BSND、BNSD、BNSD_BSND（输入为BNSD时，输出格式为BSND，仅支持Q_S大于1）、BSH_NBSD、BSND_NBSD、BNSD_NBSD（输出格式为NBSD时，仅支持Q_S大于1且小于等于16）、TND（TND相关场景综合约束请见约束说明）。
其中BNSD_BSND含义指当输入为BNSD，输出格式为BSND，仅支持Q_S大于1。
	sparse_mode（int）：可选参数，表示sparse的模式。数据类型支持int64。Q_S为1且不带rope输入时该参数无效。input_layout为TND、TND_NTD、NTD_TND时，综合约束请见约束说明。
−	sparse_mode为0时，代表defaultMask模式，如果atten_mask未传入则不做mask操作，忽略pre_tokens和next_tokens（内部赋值为INT_MAX）；如果传入，则需要传入完整的atten_mask矩阵（S1*S2），表示pre_tokens和next_tokens之间的部分需要计算。
−	sparse_mode为1时，代表allMask，必须传入完整的attenmask矩阵（S1*S2）。
−	sparse_mode为2时，代表leftUpCausal模式的mask，需要传入优化后的atten_mask矩阵（2048*2048）。
−	sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的atten_mask矩阵（2048*2048）。
−	sparse_mode为4时，代表band模式的mask，需要传入优化后的atten_mask矩阵（2048*2048）。
−	sparse_mode为5、6、7、8时，分别代表prefix、global、dilated、block_local，均暂不支持。用户不特意指定时可传入默认值0。综合约束请见约束说明。
	block_size（int）：可选参数，表示PageAttention中KV存储每个block中最大的token个数，默认为0，数据类型支持int64。
	query_quant_mode（int）：可选参数， 表示query的反量化方式。代表模式3：per-token叠加per-head模式。代表模式7：FP8 per-block全量化模式。
	key_quant_mode（int）：可选参数，表示key的反量化方式。用户不特意指定时可传入默认值0，取值除了key_antiquant_mode为0且value_quant_mode为1的场景外，其他场景取值需要与value_antiquant_mode一致。综合约束请见约束说明。
Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：当Q_S≥2时，仅支持传入值为0、1；当Q_S=1时，支持取值0、1、2、3、4、5。
PR/Ascend 950DT：支持取值0、1、2、3、4、5、6、7。
−	key_quant_mode为0时，代表per-channel模式（per-channel包含per-tensor）。
−	key_quant_mode为1时，代表per-token模式。
−	key_quant_mode为2时，代表per-tensor叠加per-head模式。
−	key_quant_mode为3时，代表per-token叠加per-head模式。
−	key_quant_mode为4时，代表per-token叠加使用page attention模式管理scale/offset模式。
−	key_quant_mode为5时，代表per-token叠加per head并使用page attention模式管理scale/offset模式。
−	key_quant_mode为6时，代表per-token-group模式。
−	key_quant_mode为7时，代表FP8 per-block全量化模式。
	value_quant_mode（int）：可选参数，表示value的反量化方式，模式编号与key_quant_mode一致。用户不特意指定时可传入默认值0，取值除了key_quant_mode为0且value_quant_mode为1的场景外，其他场景取值需要与key_aquant_mode一致。综合约束请见约束说明。
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：当Q_S≥2时，仅支持传入值为0、1；当Q_S=1时，支持取值0、1、2、3、4、5。
−	PR/Ascend 950DT：支持取值0、1、2、3、4、5、6、7。
	inner_precise（int）：可选参数，数据类型支持int64，支持4种模式：0、1、2、3。一共两位bit位，第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。当Q_S>1时，sparse_mode为0或1，并传入用户自定义mask的情况下，建议开启行无效；Q_S为1时该参数仅支持取0和1。综合约束请见约束说明。
−	inner_precise为0时，代表开启高精度模式，且不做行无效修正。
−	inner_precise为1时，代表高性能模式，且不做行无效修正。
−	inner_precise为2时，代表开启高精度模式，且做行无效修正。
−	inner_precise为3时，代表高性能模式，且做行无效修正。
 
bfloat16和int8不区分高精度和高性能，行无效修正对float16、bfloat16和int8均生效。当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。
	return_softmax_lse（bool）：可选参数，表示是否输出softmax_lse，支持S轴外切（增加输出）。true表示输出，false表示不输出；用户不特意指定时可传入默认值false。
	query_dtype（int）：可选参数，表示query的数据类型，预留参数，暂未使用，使用默认值即可。
	key_dtype（int）：可选参数，表示key的数据类型。
Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：预留参数，暂未使用，使用默认值即可。
PR/Ascend 950DT：key的数据类型为float4_e1m2fn_x2、float4_e2m1fn_x2、hifloat8时，需要传值。
	value_dtype（int）：可选参数，表示value的数据类型。
Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：预留参数，暂未使用，使用默认值即可。
PR/Ascend 950DT：value的数据类型为float4_e1m2fn_x2、float4_e2m1fn_x2、hifloat8时，需要传值。
	query_rope_dtype（int）：可选参数，表示query_repo的数据类型，预留参数，暂未使用，使用默认值即可。
	key_rope_dtype（int）：可选参数，表示key_rope的数据类型，预留参数，暂未使用，使用默认值即可。
	key_shared_prefix_dtype（int）：可选参数，表示key_shared_prefix的数据类型，预留参数，暂未使用，使用默认值即可。
	value_shared_prefix_dtype（int）：可选参数，表示value_shared_prefix的数据类型，预留参数，暂未使用，使用默认值即可。
	dequant_scale_query_dtype（int）：可选参数，表示dequant_scale_query的数据类型，预留参数，暂未使用，使用默认值即可。
	dequant_scale_key_dtype（int）：可选参数，表示dequant_scale_key的数据类型。
Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：预留参数，暂未使用，使用默认值即可。
PR/Ascend 950DT：dequant_scale_key的数据类型为float8_e8m0fnu时，需要传值。
	dequant_scale_value_dtype（int）：可选参数，表示dequant_scale_value的数据类型。
Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：预留参数，暂未使用，使用默认值即可。
PR/Ascend 950DT：dequant_scale_value的数据类型为float8_e8m0fnu时，需要传值。
	dequant_scale_key_rope_dtype（int）：可选参数，表示dequant_scale_key_rope的数据类型，预留参数，暂未使用，使用默认值即可。
输出说明
	attention_out（Tensor）：公式中的输出，数据类型支持float16、bfloat16、int8。数据格式支持ND。限制：该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。
	softmax_lse（Tensor）：ring attention算法对query乘key的结果先取max得到softmax_max，query乘key的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum，最后对softmax_sum取log，再加上softmax_max得到的结果。数据类型支持float32，softmax_lse_flag为True时，一般情况下，输出shape为(B, N, Q_S, 1)的Tensor，当input_layout为TND/NTD_TND时，输出shape为(T,N,1)的Tensor；softmax_lse_flag为False时，则输出shape为[1]的值为0的Tensor。
约束说明
	该接口支持推理场景下使用。
	该接口支持TorchAir图模式。
	该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
	入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则填充全零的对应shape的输出（填充attention_out）。attention_out为空Tensor时，框架会处理。
	参数key、value中对应tensor的shape需要完全一致；非连续场景下key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。
	int8量化相关入参数量与输入、输出数据格式的综合限制：
入参quant_offset_out和quant_scale_out支持per-tensor或per-channel格式，数据类型支持float32、bfloat16。
	query_rope和key_rope输入时即为MLA场景，参数约束如下：
−	query_rope的数据类型、数据格式与query一致。
−	key_rope的数据类型、数据格式与key一致。
−	query_rope和key_rope要求同时配置或同时不配置，不支持只配置其中一个。
−	当query_rope和key_rope非空时，支持如下特性：
	当query的D等于512时：
□	sparse：Q_S等于1时只支持sparse=0且不传mask，Q_S大于1时只支持sparse=3且传入mask；
□	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品约束如下：
query_rope配置时要求query的s为1-16、n为32、64、128，query_rope的shape中D为64，其余维度与query一致；
key_rope配置时要求key的N为1，D为512，key_rope的shape中D为64，其余维度与key一致；
支持key、value、key_rope的input_layout格式为ND或NZ。当input_layout为NZ时，数据类型为float16或bfloat16时，输入参数key和value的格式为[blockNum, N, D/16, blockSize, 16]；当数据类型为int8时，输入参数key和value的格式为[blockNum, N, D/32, blockSize, 32]；
input_layout形状支持BSH、BSND、BNSD、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND、TND_NTD，当数据格式为NZ时input_layout不支持BNSD、BNSD_NBSD。
该场景下，必须开启PageAttention，此时block_size支持16、128，其中数据格式为NZ时block_size不支持配置16。
不支持开启softmax_lse、tensorlist、pse、伪量化、后量化。
支持全量化场景，即输入query/key/value全为int8，query_rope和key_rope为bfloat16，输出为bfloat16的场景：
①入参dequant_scale_query、dequant_scale_key、dequant_scale_value需要同时存在，且其数据类型仅支持float32。
②不支持传入quant_scale_out、quant_offset_out、dequant_offset_key、dequant_offset_value（即不为None），否则报错并返回。
③query_quant_mode仅支持per-token叠加per-head模式，key_quant_mode和value_quant_mode仅支持per-tensor模式。④支持key、value、key_rope的input_layout格式为NZ。
□	PR/Ascend 950DT约束如下：
queryRope配置时要求query的s为1-16、n为32、64、128，d为512，queryRope的shape中b、n、s与query一致，d为64。
keyRope配置时要求key的n为1，d为512，keyRope的shape中b、n、s与key一致，d为64。
sparse：支持sparse=0且不传mask、sparse=3且传入mask。
key&value支持ND输入。
inputLayout：BSH、BSND、BNSD、BNSD_BSND、TND。
支持actualSeqLengths、actualSeqLengthsKv参数。
	当query的D等于128时：
□	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品约束如下：
inputLayout：TND、NTD_TND。 query_rope配置时要求query_rope的shape中d为64，其余维度与query一致。 key_rope配置时要求key_rope的shape中d为64，其余维度与key一致。 不支持tensorlist、pse、page attention、伪量化、全量化、后量化。 其余约束同TND、NTD_TND场景下的综合限制保持一致。
□	PR/Ascend 950DT约束如下：
query_rope配置时要求query_rope的shape中b、n、s与query一致，d为64。
key_rope配置时要求key_rope的shape中b、n、s与key一致，d为64。
input_layout：BSH、BSND、BNSD、BNSD_BSND、TND。
不支持page attention、prefix、伪量化、全量化、后量化。
当kv为tensorlist时，key_rope的shape中b需要与tensorlist长度保持一致，n、s需要与tensorlist中每个tensor的n、s相等，d为64。
−	TND场景下query、key、value输入的综合限制：
	actual_seq_qlen和actual_seq_kvlen必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值。
	不支持左padding、tensorlist、pse、prefix。
−	TND、TND_NTD、NTD_TND场景下query、key、value输入的综合限制：
	sparse模式仅支持sparse=0且不传mask，或sparse=3且传入mask；
	actual_seq_qlen和actual_seq_kvlen必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值；
	当query的D等于512时：
□	支持TND、TND_NTD;
□	必须开启page attention，此时actual_seq_kvlen长度等于key/value的batch值，代表每个batch的实际长度，值不大于KV_S；
□	支持query每个batch的s为1-16；
□	要求query的n为32/64/128，key、value的n为1；
□	要求query_rope和key_rope不等于空，query_rope和key_rope的d为64；
□	不支持开启softmax_lse、tensorlist、pse、伪量化、全量化、后量化。
	当query的D不等于512时：
□	当query_rope和key_rope为空时：TND场景，要求Q_D、K_D、V_D等于128，或者Q_D、K_D等于192，V_D等于128/192；NTD_TND场景，要求Q_D、K_D等于128/192，V_D等于128。当query_rope和key_rope不为空时，要求Q_D、K_D、V_D等于128；
□	Q_N、K_N、V_N：需要满足K_N、V_N相等，Q_N整除K_N，Q_N与K_N的比值不能大于64；
□	支持TND、NTD_TND；
□	数据类型仅支持BFLOAT16；
□	当sparse=3时，要求每个batch单独的actual_seq_qlen<actual_seq_kvlen；
□	不支持tensorlist、pse、page attention、伪量化、全量化、后量化；
□	不支持图模式配置Tiling调度优化（tiling_schedule_optimize=True）、reduce-overhead执行模式（config.mode="reduce-overhead"）。
	GQA伪量化场景下KV为NZ格式时的参数约束如下：
−	仅支持per-channel模式，query数据类型固定为bfloat16，key&value固定为int8；query&key&value的D仅支持128；query Sequence Length仅支持1-16；
−	input_layout仅支持BSH、BSND、BNSD；
−	key&value仅支持NZ输入，输入格式为[blockNum, N, D/32, blockSize, 32]；
−	dequant_scale_key和dequant_scale_value的shape：当layout为BSH时，必须传入[H]；layout为BNSD时，必须传入[N,1,D]；输出为BSND时，必须传入[N, D]；
−	仅支持KV分离；
−	仅支持高性能模式；
−	当MTP等于0时，支持sparse_mode=0且不传mask；当MTP大于0、小于16时，支持sparse_mode=3且传入优化后的atten_mask矩阵，atten_mask矩阵shape必须传入（2048*2048）；
−	不支持配置query_rope和key_rope；
−	不支持tensorlist、pse、page attention、后量化；
−	num_query_heads与num_key_value_heads支持组合有(10, 1)、(64, 8)、(80, 8)、(128, 16)。
	当Q_S大于1时：
−	query、key、value输入，功能使用限制如下：
	支持B轴小于等于65536，D轴32byte不对齐时仅支持到128。
	支持N轴小于等于256，支持D轴小于等于512；input_layout为BSH或者BSND时，建议N*D小于65535。
	S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致PFA算子执行超时（aicore error类型报错，errorStr为timeout or trap error），此场景下建议做S切分处理（注：这里计算量会受B、S、N、D等的影响，值越大计算量越大），典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
□	B=1，Q_N=20，Q_S=2097152，D=256，KV_N=1，KV_S=2097152。
□	B=1，Q_N=2，Q_S=20971520，D=256，KV_N=2，KV_S=20971520。
□	B=20，Q_N=1，Q_S=2097152，D=256，KV_N=1，KV_S=2097152。
□	B=1，Q_N=10，Q_S=2097152，D=512，KV_N=1，KV_S=2097152。
	query、key、value输入类型包含int8时，D轴需要32对齐；输入类型全为float16、bfloat16时，D轴需16对齐。
	D轴限制：
□	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：query、key、value输入类型包含int8时，D轴需要32对齐；query、key、value或attentionOut类型包含INT4时，D轴需要64对齐；输入类型全为float16、bfloat16时，D轴需16对齐。
□	PR/Ascend 950DT：非量化场景：query，key，value的类型全部为FLOAT16、BFLOAT16，D轴1-512全部支持。伪量化场景：query类型为FLOAT16、BFLOAT16，key、value类型为INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E1M2/FLOAT4_E2M1/INT4（INT32），其中当key、value类型为INT4（INT32）D轴仅支持64对齐（INT32仅支持D 8对齐）
−	actual_seq_qlen：
	PR/Ascend 950DT：该参数应为非负数，在input_layout不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，其长度为1或大于等于query的Batch值，该入参中的值代表每个Batch的实际长度，其值应该不大于Q_S。当input_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：该入参中每个batch的有效Sequence Length应该≤query中对应batch的Sequence Length。seqlen的传入长度为1时，每个Batch使用相同seqlen；传入长度≥Batch时取seqlen的前Batch个数。其他长度不支持。当query的inputLayout为TND/NTD_TND时，综合约束请见约束说明。
−	actual_seq_kvlen：
	PR/Ascend 950DT：该参数传入时应为非负数，在input_layout不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，该入参中每个Batch的有效seqlenKv应该不大于key/value中对应Batch的seqlenKv。当本参数的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当key/value的input_layout为TND时，该入参必须传入，且该入参元素的数量等于Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKv和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当key/value的inputLayout为TND/NTD_TND时，综合约束请见约束说明。
−	参数sparse_mode当前仅支持值为0、1、2、3、4的场景，取其它值时会报错。
	sparse_mode=0时，atten_mask如果为None，则忽略入参pre_tokens、next_tokens（内部赋值为INT_MAX）。
	sparse_mode=2、3、4时，atten_mask的shape需要为(S, S)或(1, S, S)或(1, 1, S, S)，其中S的值需要固定为2048，且需要用户保证传入的atten_mask为下三角，不传入atten_mask或者传入的shape不正确报错。
	sparse_mode=1、2、3的场景忽略入参pre_tokens、next_tokens并按照相关规则赋值。
−	page attention场景：
	page attention的使能必要条件是block_table存在且有效，同时key、value是按照block_table中的索引在一片连续内存中排布，支持key、value数据类型为float16、bfloat16。在该场景下key、value的input_layout参数无效。block_table中填充的是blockid，当前不会对blockid的合法性进行校验，需用户自行保证。
	block_size是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，block_size最小为128，最大为512，且要求是128的倍数。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
	page attention场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且KV_N*D超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小KV_N）或调整kv cache排布格式为（blocknum, KV_N, blocksize, D）解决。当query的input_layout为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）、（blocknum, KV_N, blocksize, D）和（blocknum, KV_N, D/16, blocksize, 16）三种格式，当query的input_layout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）和（blocknum, KV_N, D/16, blocksize, 16）两种格式。blocknum不能小于根据actual_seq_lengths_kv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
	page attention伪量化场景
□	PR/Ascend 950DT：支持query为float16/bfloat16，支持key、value dtype为int8/hifloat8/float8_e4m3fn/float8_e5m2/float4_e1m2fn_x2/float4_e2m1fn_x2/int4（int32）。当kv cache为五维时，kv cache排布为（blocknum, KV_N, D/16, blocksize, 16）；同时，当key、value dtype为int32时，kv cache排布为（blocknum, KV_N, D/2, blocksize, 2）。
	page attention不支持tensorlist场景。
	page attention场景下，必须传入actual_seq_kvlen。
	page attention场景下，block_table必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actual_seq_lengths_kv对应的block数量）。
	page attetion场景下，支持两种格式和float32/bfloat16，不支持输入query为int8的场景。
	page attention使能场景下，以下场景输入需满足KV_S>=maxBlockNumPerSeq*blockSize：
□	传入atten_mask时，如mask shape为（B, 1, Q_S, KV_S）。
□	传入pse_shift时，如pse_shiftshape为（B, N, Q_S, KV_S）。
□	使能伪量化per-token模式：输入参数antiquantScale和antiquantOffset的shape均为(2, B, S)。
□	使能per-token叠加per-head模式：两个参数的shape均为(B, N, S)，数据类型固定为FLOAT32，当key、value数据类型为INT8、INT4(INT32)时支持。
−	入参quant_scale_out和quant_offset_out支持per-tensor、per-channel量化，支持float32、bfloat16类型。若传入quant_offset_out，需保证其类型和shape信息与quant_scale_out一致。当输入为bfloat16时，同时支持float32和bfloat16，否则仅支持float32。per-channel场景下，当输出layout为BSH时，要求quant_scale_out所有维度的乘积等于H；其他layout要求乘积等于N*D。当输出layout为BSH时，quant_scale_out shape建议传入(1, 1, H)或(H,)；当输出layout为BNSD时，建议传入(1, N, 1, D)或(N, D)；当输出为BSND时，建议传入(1, 1, N, D)或(N, D)。
−	输出为int8，quant_scale_out和quant_offset_out为per-channel时，暂不支持Ring Attention或者D非32Byte对齐的场景。
−	输出为int8时，暂不支持sparse为band且preTokens/nextTokens为负数。
−	pse_shift功能使用限制如下：
	支持query数据类型为float16、bfloat16、int8场景下使用该功能。
	query、key、value数据类型为float16且pse_shift存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
	Q_S需大于等于query的S长度，KV_S需大于等于key的S长度。
	PR/Ascend 950DT：非量化场景无对齐限制。
−	输出为int8，入参quant_offset_out传入非None和非空tensor值，并且sparse_mode、pre_tokens和next_tokens满足以下条件，矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截：
	sparse_mode=0，atten_mask如果非None，每个batch actual_seq_qlen-actual_seq_kvlen-pre_tokens>0或next_tokens<0时，满足拦截条件。
	sparse_mode=1或2，不会出现满足拦截条件的情况。
	sparse_mode=3，每个batch actual_seq_kvlen-actual_seq_qlen<0，满足拦截条件。
	sparse_mode=4，pre_tokens<0或每个batch next_tokens+actual_seq_kvlen-actual_seq_qlen<0时，满足拦截条件。
−	kv伪量化参数分离：
	当伪量化参数和KV分离量化参数同时传入时，以KV分离量化参数为准。
	除了key_quant_mode为0并且value_quant_mode为1的场景外，key_quant_mode和value_quant_mode取值需要保持一致。
	dequant_scale_key和dequant_scale_value要么都为空，要么都不为空；dequant_offset_key和dequant_offset_value要么都为空，要么都不为空。
	dequant_scale_key和dequant_scale_value都不为空时，其shape需要保持一致；dequant_offset_key和dequant_offset_value都不为空时，其shape需要保持一致。
	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
□	仅支持per-token和per-channel模式，per-token模式下要求两个参数的shape均为(B, S)，数据类型固定为float32；per-channel模式下要求两个参数的shape为（N，D），(N, D)，(H)，数据类型固定为bfloat16。
□	dequant_scale_key与dequant_scale_value非空场景，要求query的s小于等于16；要求query的dtype为bfloat16，key、value的dtype为int8，输出的dtype为bfloat16；不支持tensorlist、page attention。
	PR/Ascend 950DT：
□	支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token使用page attention模式管理scale/offset、per-token叠加per-head并使用page attention模式管理scale/offset、key支持per-channel叠加value支持per-token和per-token-group九种模式。
□	dequant_scale_key和dequant_scale_value都不为空时，除了key_quant_mode为0并且value_quant_mode为1的场景外，其shape需要保持一致；dequant_offset_key和dequant_offset_value都不为空时，除了key_quant_mode为0并且value_quant_mode为1的场景外，其shape需要保持一致。
	管理scale/offset的量化模式如下：
 
注意scale、offset具体指key_antiquant_scale、value_antiquant_scale、key_antiquant_offset、value_antiquant_offset参数。
量化模式	该场景下scale和offset条件	该场景下key和value条件	支持的产品
per-channel模式	两个参数shape支持(N, 1, D)，(N, D)，(H)，(1, N, 1, D)，(1, N, D)，(1, H)数据类型和query数据类型相同。		Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：仅支持key、value数据类型为int8。
	PR/Ascend 950DT：当key、value数据类型为int4（int32）、int8、hifloat8、float8_e5m2、float8_e4m3fn时支持。其中，当为hifloat8、float8_e5m2、float8_e4m3fn时，不支持带antiquantOffset。		Atlas A2 训练系列产品/Atlas A2 推理系列产品
	Atlas A3 训练系列产品/Atlas A3 推理系列产品
	PR/Ascend 950DT
per-token模式	两个参数的shape均为(B, S)，数据类型固定为float32。		Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：仅支持key、value数据类型为int8。
	PR/Ascend 950DT：当key、value数据类型为int4（int32）、int8时支持。	
per-tensor模式	两个参数的shape均为(1,)，数据类型和query数据类型相同。	当key、value数据类型为int4（int32）或int8时支持。	PR/Ascend 950DT
per-tensor叠加per-head模式	两个参数的shape均为(N,)，数据类型和query数据类型相同。	当key、value数据类型为int4（int32）或int8时支持。	
per-token叠加per-head模式	两个参数的shape均为(B, N, S)，数据类型固定为float32。	key、value数据类型为int4（int32）或int8时支持。	
per-token叠加使用page attention模式	两个参数的shape均为(blocknum, blocksize)，数据类型固定为float32。	key、value数据类型为int8时支持。	
per-token叠加per head并使用page attention模式	两个参数的shape均为(blocknum, N, blocksize)，数据类型固定为float32。	key、value数据类型为int8时支持。	
key支持per-channel叠加value支持per-token模式	对于key支持per-channel，两个参数的shape可支持(N, 1, D)、(N, D)、(H)，(1, N, 1, D)、(1, N, D)、(1, H)且参数数据类型和query数据类型相同。	当key、value数据类型为int4（int32）或int8时支持。	
	对于value支持per-token，两个参数的shape均为(B, S)并且数据类型固定为float32。		
per-token-group模式	antiquant_scale的shape为(1, B, N, S, D/32)，数据类型固定为float8_e8m0fnu，不支持带antiquant_offset。	当key、value数据类型为float4_e1m2fn_x2或float4_e2m1fn_x2时支持。	

−	qkv FP8 per-block全量化：
	query、key和value的数据类型支持float8_e5m2和float8_e4m3fn。
	query_quant_mode、key_quant_mode和value_quant_mode均为7。
	dequant_scale_query、dequant_scale_key和dequant_scale_value的数据类型固定为float32。
	dequant_scale_query的shape为(B，Q_N，ceil(Q_S，128)，1)，dequant_scale_key和dequant_scale_value的shape为(B，K_N，ceil(K_S，256)，1)。
	输出数据类型支持float16和bfloat16。
	不支持TND格式输入。
	D轴支持1-128。
	不支持叠加任何高阶特性。
	当Q_S等于1时：
−	query、key、value输入，功能使用限制如下：
	支持B轴小于等于65536，支持N轴小于等于256，支持S轴小于等于262144，支持D轴小于等于512。
	query、key、value输入类型均为int8的场景暂不支持。
	在int4（int32）伪量化场景下，PyTorch入图调用仅支持KV int4拼接成int32输入（建议通过dynamicQuant生成int4格式的数据，因为dynamicQuant就是一个int32包括8个int4）。
	在int4（int32）伪量化场景下，若KV int4拼接成int32输入，那么KV的N、D或者H是实际值的八分之一（prefix同理）。并且，int4伪量化仅支持D 64对齐（int32支持D 8对齐）。
	key、value在特定数据数据类型下存在对于D轴的限制
PR/Ascend 950DT：key、value输入类型为float4_e1m2fn_x2/float4_e2m1fn_x2/int4（int32）时，query的D轴以及key、value的D轴需要64对齐（INT32仅支持key、value的D 8对齐）。
−	actual_seq_qlen：
	PR/Ascend 950DT：该参数在未输入rope参数时不生效。输入rope参数时生效，传入时应为非负数，在input_layout不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，其长度为1或大于等于query的Batch值，该入参中的值代表每个Batch的实际长度，其值应该不大于Q_S。当input_layout为TND时，该入参必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和，因此后一个元素的值必须大于等于前一个元素的值，且不能出现负值。
	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：当query的input_layout不为TND时，Q_S为1时该参数无效。当query的input_layout为TND/TND_NTD时，综合约束请见约束说明。
−	actual_seq_kvlen：
	PR/Ascend 950DT：该参数应为非负数，在input_layout不同时，其含义与拦截条件不同：一般情况下，该入参为可选入参，该入参中每个Batch的有效Sequence Length应该不大于key/value中对应Batch的seqlenKv。当本参数的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当input_layout为TND/TND_NTD时，该入参必须传入，在非PA场景下，第b个值表示前b个Batch的S轴累加长度，其值应递增（大于等于前一个值）排列，且该入参元素的数量代表总Batch数，在PA场景下，其长度等于key/value的Batch值，代表每个Batch的实际长度，值不大于KV_S。
	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：该入参中每个batch的有效Sequence Length应该不大于key/value中对应batch的Sequence Length。seqlenKv的传入长度为1时，每个Batch使用相同seqlenKv；传入长度大于等于Batch时取seqlenKv的前Batch个数。其他长度不支持。当key/value的inputLayout为TND/TND_NTD时，综合约束请见约束说明。
−	page attention场景：
	使能必要条件是block_table存在且有效，同时key、value是按照block_table中的索引在一片连续内存中排布，在该场景下key、value的input_layout参数无效。
	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
□	支持key、value数据类型为float16、bfloat16、int8。
□	不支持Q为bfloat16、float16、key、value为int4（int32）的场景。
	PR/Ascend 950DT：支持key、value数据类型为float16、bfloat16、int8、int4（int32）、float8_e5m2、float8_e4m3fn、hifloat8、float4_e1m2fn_x2、float4_e2m1fn_x2。
	该场景下，block_size是用户自定义的参数，该参数的取值会影响page attention的性能。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：key、value输入类型为float16/bfloat16时需要16对齐；key、value 输入类型为int8时需要32对齐，推荐使用128。
PR/Ascend 950DT：key、value输入类型为float16/bfloat16时需要16对齐；key、value 输入类型为int8/float8_e5m2/float8_e4m3fn/hifloat8时需要32对齐；key、value输入类型为float4_e1m2fn_x2/float4_e2m1fn_x2/int4（int32）时需要64对齐。
	参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围。
	page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actual_seq_kvlen对应的block数量）。
	page attention场景下，当query的input_layout为BNSD、TND时，kv cache排布支持（blocknum, blocksize, H）、（blocknum, KV_N, blocksize, D）和（blocknum, KV_N, D/16, blocksize, 16）三种格式，当query的input_layout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）和（blocknum, KV_N, D/16, blocksize, 16）两种格式。blocknum不能小于根据actual_seq_kvlen和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
	PR/Ascend 950DT：page attention伪量化场景下，当kv cache为五维时，kv cache排布为（blocknum, KV_N, D/16, blocksize, 16）；同时，当key、value dtype为int32时，kv cache排布为（blocknum, KV_N, D/2, blocksize, 2）。
	page attention场景下，kv cache排布为（blocknum, KV_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV_N, blocksize, D）格式。
	page attention使能场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且numKvHeads*headDim超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小numKvHeads）或调整kvCache排布格式为（blocknum, numKvHeads, blocksize, D）解决。
	page attention场景的参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围。
−	kv伪量化参数分离：
	除了key_quant_mode为0并且value_quant_mode为1的场景外，key_quant_mode和value_quant_mode取值需要保持一致。
	dequant_scale_key和dequant_scale_value要么都为空，要么都不为空；dequant_offset_key和dequant_offset_value要么都为空，要么都不为空。
	dequant_scale_key和dequant_scale_value都不为空时，除了key_quant_mode为0并且value_quant_mode为1的场景外，其shape需要保持一致；dequant_offset_key和dequant_offset_value都不为空时，除了key_quant_mode为0并且value_quant_mode为1的场景外，其shape需要保持一致。
	int4（int32）伪量化场景不支持后量化。
	管理scale/offset的量化模式如下：
 
注意scale、offset两个参数指key_antiquant_scale、value_antiquant_scale、key_antiquant_offset、value_antiquant_offset。
量化模式	该场景下scale和offset条件	该场景下key和value条件	支持的产品
per-channel模式	两个参数shape支持(1, N, 1, D)，(1, N, D)，(1, H)，数据类型和query数据类型相同。		Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：当key、value数据类型为int4（int32）或int8时支持。
	PR/Ascend 950DT：当key、value数据类型为int4（int32）、int8、hifloat8、float8_e5m2、float8_e4m3fn时支持。其中，当为hifloat8、float8_e5m2、float8_e4m3fn时，不支持带antiquantOffset。		Atlas A2 训练系列产品/Atlas A2 推理系列产品
	Atlas A3 训练系列产品/Atlas A3 推理系列产品
	PR/Ascend 950DT
per-tensor模式	两个参数的shape均为(1)，数据类型和query数据类型相同。		Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：当key、value数据类型为int8时支持。
	PR/Ascend 950DT：当key、value数据类型为int4（int32）或int8时支持。	
per-token模式	两个参数的shape均为(1, B, S)，数据类型固定为float32。	key、value数据类型为int4（int32）或int8时支持。	
per-tensor叠加per-head模式	两个参数的shape均为(N)，数据类型和query数据类型相同。		Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：当key、value数据类型为int8时支持。
	PR/Ascend 950DT：当key、value数据类型为int4（int32）或int8时支持。	
per-token叠加per-head模式	两个参数的shape均为(B, N, S)，数据类型固定为float32。	key、value数据类型为int4（int32）或int8时支持。	
per-token叠加使用page attention模式	两个参数的shape均为(blocknum, blocksize)，数据类型固定为float32。	key、value数据类型为int8时支持。	
per-token叠加per head并使用page attention模式	两个参数的shape均为(blocknum, N, blocksize)，数据类型固定为float32。	key、value数据类型为int8时支持。	
key支持per-channel叠加value支持per-token模式	对于key支持per-channel，两个参数的shape可支持(1, N, 1, D)、(1, N, D)、(1, H)，且参数数据类型和query数据类型相同。		Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：当key、value数据类型为int4（int32）或int8时支持；当key和value的数据类型为int8时，仅支持query和输出的dtype为float16。
	PR/Ascend 950DT：当key、value数据类型为int4（int32）或int8时支持。	
	对于value支持per-token，两个参数的shape均为(1, B, S)并且数据类型固定为float32。		
per-token-group模式	antiquant_scale的shape为(1, B, N, S, D/32)，数据类型固定为float8_e8m0fnu，不支持带antiquant_offset。	当key、value数据类型为float4_e1m2fn_x2或float4_e2m1fn_x2时支持。	PR/Ascend 950DT

−	pse_shift功能使用限制如下：
pse_shift数据类型需与query数据类型保持一致。
支持的PyTorch版本
PyTorch 2.1
PyTorch 2.3
PyTorch 2.4
调用示例
	单算子模式调用
import torch 
import torch_npu 
import math 
# 生成随机数据, 并发送到npu 
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu() 
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu() 
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu() 
softmax_scale = 1/math.sqrt(128.0) 
actseqlen = [164] 
actseqlenkv = [1024] 
 
# 调用FIA算子 
out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, actual_seq_qlen = actseqlen, actual_seq_kvlen = actseqlenkv, 
num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535) 
 
# 执行上述代码的输出out类似如下 
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140], 
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117], 
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062], 
        .. 
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039], 
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164], 
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]], 
        device='npu:0', dtype=torch.float16)
	图模式调用
−	Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
import torch 
import torch_npu 
import math 
import torchair as tng 
 
from torchair.configs.compiler_config import CompilerConfig 
import torch._dynamo 
TORCHDYNAMO_VERBOSE=1 
TORCH_LOGS="+dynamo" 
 
# 支持入图的打印宏 
import logging 
from torchair.core.utils import logger 
logger.setLevel(logging.DEBUG) 
config = CompilerConfig() 
config.debug.graph_dump.type = "pbtxt" 
npu_backend = tng.get_npu_backend(compiler_config=config) 
from torch.library import Library, impl 
 
# 数据生成 
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu() 
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu() 
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu() 
softmax_scale = 1/math.sqrt(128.0) 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
    def forward(self): 
        return torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535) 
def MetaInfershape(): 
    with torch.no_grad(): 
        model = Model() 
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True) 
        graph_output = model() 
    single_op = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535) 
    print("single op output with mask:", single_op[0], single_op[0].shape) 
    print("graph output with mask:", graph_output[0], graph_output[0].shape) 
if __name__ == "__main__": 
    MetaInfershape() 
 
# 执行上述代码的输出类似如下 
single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140], 
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117], 
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062], 
        ..., 
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039], 
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164], 
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]], 
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128]) 
 
graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140], 
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117], 
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062], 
        ..., 
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039], 
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164], 
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]], 
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
−	float4_e1m2fn_x2伪量化图模式调用（仅PR/Ascend 950DT支持）
import torch 
import torch_npu 
import math 
import torchair as tng 
 
from torchair.configs.compiler_config import CompilerConfig 
import torch._dynamo 
TORCHDYNAMO_VERBOSE=1 
TORCH_LOGS="+dynamo" 
 
# 支持入图的打印宏 
import logging 
from torchair.core.utils import logger 
logger.setLevel(logging.DEBUG) 
config = CompilerConfig() 
config.debug.graph_dump.type = "pbtxt" 
npu_backend = tng.get_npu_backend(compiler_config=config) 
from torch.library import Library, impl 
 
# 数据生成 
# fp4伪量化含有pytorch不支持的数据类型，需要伪装成uint8输入 
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu() 
k = torch.randint(0, 256, (1, 8, 1024, 64), dtype=torch.uint8, device='npu') 
v = torch.randint(0, 256, (1, 8, 1024, 64), dtype=torch.uint8, device='npu') 
key_antiquant_scale = torch.randint(0, 256, (1, 1, 8, 1024, 4), dtype=torch.uint8, device='npu') 
value_antiquant_scale = torch.randint(0, 256, (1, 1, 8, 1024, 4), dtype=torch.uint8, device='npu') 
softmax_scale = 1/math.sqrt(128.0) 
quant_mode = 6 
key_dtype = torch_npu.float4_e1m2fn_x2 
value_dtype = torch_npu.float4_e1m2fn_x2 
key_antiquant_scale_dtype = torch_npu.float8_e8m0fnu 
value_antiquant_scale_dtype = torch_npu.float8_e8m0fnu 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
    def forward(self): 
        return torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, dequant_scale_key = key_antiquant_scale, dequant_scale_value = value_antiquant_scale, key_dtype=key_dtype, value_dtype=value_dtype, dequant_scale_key_dtype=key_antiquant_scale_dtype,  
dequant_scale_value_dtype=value_antiquant_scale_dtype, key_quant_mode = quant_mode, value_quant_mode = quant_mode) 
def MetaInfershape(): 
    with torch.no_grad(): 
        model = Model() 
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True) 
        graph_output = model() 
    single_op = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535, dequant_scale_key = key_antiquant_scale, dequant_scale_value = value_antiquant_scale, key_dtype=key_dtype, value_dtype=value_dtype, dequant_scale_key_dtype=key_antiquant_scale_dtype, 
dequant_scale_value_dtype=value_antiquant_scale_dtype, key_quant_mode = quant_mode, value_quant_mode = quant_mode) 
    print("single op output with mask:", single_op[0], single_op[0].shape) 
    print("graph output with mask:", graph_output[0], graph_output[0].shape) 
if __name__ == "__main__": 
    MetaInfershape() 
 
# 执行上述代码的输出类似如下 
single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140], 
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117], 
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062], 
        ..., 
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039], 
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164], 
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]], 
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128]) 
 
graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140], 
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117], 
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062], 
        ..., 
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039], 
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164], 
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]], 
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
