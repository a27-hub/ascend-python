	x (Tensor)：必选参数，表示计算使用的token数据，需根据expert_ids来发送给其他卡。要求为2维张量，shape为(BS, H)，表示有BS个token，数据类型支持bfloat16、float16，数据格式为ND，支持非连续的Tensor。
	expert_ids (Tensor)：必选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为(BS, K)，数据类型支持int32，数据格式为ND，支持非连续的Tensor。对应torch_npu.npu_moe_distribute_combine_v2的expert_ids输入，张量里value取值范围为[0, moe_expert_num)，且同一行中的K个value不能重复。
	group_ep (str)：必选参数，EP通信域名称，专家并行的通信域。字符串长度范围为[1,128)。取值不能和group_tp相同。
	ep_world_size（int）：必选参数，EP通信域size。
	comm_alg设置为"fullmesh"时，ep_world_size取值范围为16、32、64、128、256。
	comm_alg设置为"hierarchy"时，ep_world_size取值范围为16、32、64。
−	PR/Ascend 950DT：取值支持[2, 768]。
	ep_rank_id (int)：必选参数，EP通信域本卡ID，取值范围[0, ep_world_size)，同一个EP通信域中各卡的ep_rank_id不重复。
	moe_expert_num (int)：必选参数，MoE专家数量，取值范围[1, 1024]，并且满足以下条件：moe_expert_num%(ep_world_size - shared_expert_rank_num)=0。
	*：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
	scales (Tensor)：可选参数，表示每个专家的权重。
−	PR/Ascend 950DT：数据类型支持float32。
	当quant_mode取值为0时，传入None。
	当quant_mode取值为1时，必须传入有效数据，其shape为 (1, )、(H, )、(shared_expert_num+ moe_expert_num, H)，(moe_expert_num, H）。
	当quant_mode取值为2或3时，可传入None或者有效数据，传入有效数据时，其shape为 (shared_expert_num+ moe_expert_num, H)，(moe_expert_num, H)。
	当quant_mode取值为4时，传默认值即可。
	x_active_mask (Tensor)：可选参数，表示token是否参与通信。
−	PR/Ascend 950DT：取值范围如下所示。
	comm_alg设置为"mte"时，要求为一个1维或2维张量。当输入为1维时，shape为(BS, ); 当输入为2维时，shape为(BS, K)。数据类型支持bool，数据格式要求为ND，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入；当输入为2维时，参数为true表示当前token对应的expert_ids参与通信，若当前token对应的K个bool值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。
	comm_alg设置为"ccu"时，要求为一个1维张量。shape为(BS, );。数据类型支持bool，数据格式要求为ND，支持非连续的Tensor。参数为true表示对应的token参与通信，true必须排到false之前。
	expert_scales (Tensor)：可选参数，表示每个token的topK个专家权重。
−	PR/Ascend 950DT：暂不支持该参数，使用默认值即可。
	elastic_info (Tensor)：预留参数，当前版本不支持，传默认值None即可。
	group_tp (string)：可选参数，TP通信域名称，数据并行的通信域。若有TP域通信需要传参，若无TP域通信，使用默认值""即可。
−	PR/Ascend 950DT：不支持TP域通信，使用默认输入。
	tp_world_size (int)：可选参数，TP通信域size。有TP域通信才需要传参。
−	PR/Ascend 950DT：不支持TP域通信，使用默认值0即可。
	tp_rank_id (int)：可选参数，TP通信域本卡ID。有TP域通信才需要传参。
−	PR/Ascend 950DT：不支持TP域通信，使用默认值0即可。
	expert_shard_type (int)：可选参数，表示共享专家卡排布类型。
−	PR/Ascend 950DT：当前仅支持0，表示共享专家卡排在MoE专家卡前面。
	shared_expert_num (int)：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。
−	PR/Ascend 950DT：取值范围[0, 4]，0表示无共享专家，默认值为1。
	shared_expert_rank_num (int)：可选参数，表示共享专家卡数量。
−	PR/Ascend 950DT：取值范围[0, ep_world_size)。取0表示无共享专家，不取0需满足shared_expert_rank_num%shared_expert_num=0。
	quant_mode (int)：可选参数，表示量化模式。
−	PR/Ascend 950DT：0表示非量化（默认），1表示静态量化，2表示动态pertoken量化，3表示pergroup量化，4表示mxfp8量化。当quant_mode为2、3、4，dynamic_scales不为None；当quant_mode为0或1，dynamic_scales为None。
	global_bs (int)：可选参数，表示EP域全局的batch size大小。当每个rank的BS不同时，支持传入max_bs*ep_world_size，其中max_bs表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS*ep_world_size。
	expert_token_nums_type (int)：可选参数，表示输出expert_token_nums的值类型，取值范围[0, 1]，0表示每个专家收到token数量的前缀和，1表示每个专家收到的token数量（默认）。
	comm_alg (string)：可选参数，表示通信亲和内存布局算法。
	"": 配置HCCL_INTRA_PCIE_ENABLE=1和HCCL_INTRA_ROCE_ENABLE=0时，调用"hierarchy"算法，否则调用"fullmesh"算法。不推荐使用该方式。
	"fullmesh": token数据直接通过RDMA方式发往topk个目标专家所在的卡。
	"hierarchy": token数据经过跨机、机内两次发送，仅不同server同号卡之间使用RDMA通信，server内使用HCCS通信。
−	Atlas A3 训练系列产品/Atlas A3 推理系列产品：当前版本仅支持""，"fullmesh_v1"，"fullmesh_v2"三种输入方式。
	""：默认值，不使能fullmesh_v2模板；
	"fullmesh_v1"：不使能fullmesh_v2模板；
	"fullmesh_v2"：使能fullmesh_v2模板，其中commAlg仅在tp_world_size取值为1时生效，且不支持在各卡Bs不一致、输入xActiveMask和特殊专家场景下使能。
−	PR/Ascend 950DT：当前支持"mte"和"ccu"输入，默认为"mte"。
	"mte": 基于UB-MEM进行跨卡通信。
	"ccu": 基于CCU进行跨卡通信。
	zero_expert_num (int)：可选参数，表示零专家的数量。
−	PR/Ascend 950DT：当前版本不支持，传0即可。
	copy_expert_num (int)：可选参数，表示拷贝专家的数量。
−	PR/Ascend 950DT：当前版本不支持，传0即可。
	const_expert_num (int)：可选参数，表示常量专家的数量。
−	PR/Ascend 950DT：当前版本不支持，传0即可。
	y_dtype: int类型，可选参数，表示输出的数据类型。
−	PR/Ascend 950DT ：当quant_mode不为0时应当指定，不指定默认为int8类型。
	x_dtype：int类型，可选参数，表示输入的tensor x的数据类型，当其为PyTorch原生不支持的数据类型需通过该参数指定。
	scales_dtype: int类型，可选参数，代表输入的tensor scales的数据类型，当其为PyTorch原生不支持的数据类型需通过该参数指定。
返回值说明
	expand_x (Tensor)：表示本卡收到的token数据，要求为2维张量，shape为(max(tp_world_size, 1) *A, H)，A表示在EP通信域可能收到的最大token数。量化时类型为int8，非量化时与x数据类型保持一致。数据格式为ND，支持非连续的Tensor。
	PR/Ascend 950DT：数据类型支持float16、bfloat16、int8、float8_e5m2、float8_e4m3fn、hifloat8。
	dynamic_scales (Tensor)：表示计算得到的动态量化参数。当quant_mode不为0或1时才有该输出，数据格式支持ND，支持非连续的Tensor。
	PR/Ascend 950DT：数据类型支持float32、float8_e8m0fnu。。quant_mode取值为2时，其shape为 (A, )，quant_mode取值为3时，其shape为 (A, ceil(H/128))；quant_mode取值为4时，其shape为 (A, ceil(H/32))且ceil(H/32) % 2 == 0。
	assist_info_for_combine (Tensor)：表示给同一专家发送的token个数，要求是一个1维张量，shape为(A * 128, )。数据类型支持int32，数据格式为ND，支持非连续的Tensor。对应torch_npu.npu_moe_distribute_combine_v2的assist_info_for_combine输入。
	expert_token_nums (Tensor)：本卡每个专家实际收到的token数量，要求为1维张量，shape为(local_expert_num,)，数据类型int64，数据格式支持ND，支持非连续的Tensor。
	ep_recv_counts (Tensor)：表示EP通信域各卡收到的token数（token数以前缀和的形式表示），要求为1维张量，数据类型int32，数据格式支持ND，支持非连续的Tensor。对应torch_npu.npu_moe_distribute_combine_v2的ep_send_counts输入。
−	PR/Ascend 950DT：要求shape为(ep_world_size*max(tp_world_size, 1)*local_expert_num, )。
	tp_recv_counts (Tensor)：表示TP通信域各卡收到的token数量。对应torch_npu.npu_moe_distribute_combine_v2的tp_send_counts输入。
−	PR/Ascend 950DT：不支持TP通信域，暂无该输出。
	expand_scales (Tensor)：表示expert_scales与x一起进行alltoallv之后的输出。
−	PR/Ascend 950DT：暂不支持该输出，返回None。
约束说明
	该接口支持推理场景下使用。
	该接口支持单算子模式和静态图模式调用。
Atlas A3 训练系列产品/Atlas A3 推理系列产品：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
	本文公式中的“/”表示整除。
	该接口必须与npu_moe_distribute_dispatch_v2和npu_moe_distribute_combine_v2配套使用。
	在不同产品型号、不同通信算法或不同版本中，npu_moe_distribute_dispatch_v2的Tensor输出assist_info_for_combine、ep_recv_counts、tp_recv_counts、expand_scales中的元素值可能不同，使用时直接将上述Tensor传给npu_moe_distribute_combine_v2对应参数即可，模型其他业务逻辑不应对其存在依赖。
	调用接口过程中使用的group_ep、ep_world_size、moe_expert_num、group_tp、tp_world_size、expert_shard_type、shared_expert_num、shared_expert_rank_num、global_bs参数取值所有卡需保持一致，group_ep、ep_world_size、group_tp、tp_world_size、expert_shard_type、global_bs网络中不同层中也需保持一致，且和torch_npu.npu_moe_distribute_combine_v2对应参数也保持一致。
	moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num < MAX_INT32。
	参数说明里shape格式说明：
−	A：表示本卡接收的最大token数量，取值范围如下
	对于共享专家，要满足A=BS*shared_expert_num/shared_expert_rank_num。
	对于MoE专家，当global_bs为0时，要满足A>=BS*ep_world_size*min(local_expert_num, K)；当global_bs不为0时，要满足A>=global_bs* min(local_expert_num, K)。
−	H：表示hidden size隐藏层大小。
	Atlas A2 训练系列产品：H的取值范围如下所示。
□	comm_alg设置为"fullmesh"时，H的取值范围(0, 7168]，且保证是32的整数倍。
□	comm_alg设置为"hierarchy"且驱动版本不低于25.0.RC1.1时，H的取值范围(0, 10 * 1024]，且保证是32的整数倍。
	Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值为[1024, 8192]。
	PR/Ascend 950DT：取值范围[1024, 8192]。
−	BS：表示batch sequence size，即本卡最终输出的token数量。
	PR/Ascend 950DT：取值范围为0<BS≤512。
−	K：表示选取topK个专家，取值范围为0<K≤16，同时满足0 < K ≤ moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num。
−	server_num：表示服务器的节点数，取值只支持2、4、8。
−	local_expert_num：表示本卡专家数量。
	对于共享专家卡，local_expert_num为1。
	对于MoE专家卡，local_expert_num=moe_expert_num/(ep_world_size-shared_expert_rank_num)，当local_expert_num大于1时，不支持TP域通信。