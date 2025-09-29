# torch\_npu.npu\_moe\_distribute\_barrier<a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>  | √   |

## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   API功能：

    完成通信域内的全卡同步，`x_ref`仅用于构建Tensor依赖，接口内不对`x_ref`做任何操作
     - 支持动态缩容场景，在创建通信域后，出现故障卡，完成通信域内除故障卡外的全卡同步，算子可正常执行（无需重新编译），通过传入`elastic_info`参数使能该特性（可选）；
     - 支持超时检测场景，通过传入大于0的`time_out`参数使能本特性。当算子内部全卡同步等待时间超过给定`time_out`时，认为该卡存在超时异常（可选）。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
npu_distribute_barrier(Tensor x_ref, str group, int world_size, *, Tensor? time_out=None, Tensor? elastic_info=None) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

-   **x\_ref** (`Tensor`)：必选参数，Device侧的aclTensor，无业务语义，仅用于输入Tensor依赖，接口内不做任何操作。数据类型支持BFLOAT16, FLOAT16、FLOAT32、BOOL、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64。 数据格式要求为ND，支持非连续的Tensor。
-   **group** (`str`)：必选参数，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)。
-   **world\_size** (`int`)：必选参数，通信域size，取值范围[2, 384]。
-   **time\_out** (`int`)：可选参数，表示全卡同步的超时设定时间，当算子内部同步等待时间超过给定`time_out`时，认为该卡存在超时异常。
-   **elastic\_info** (`Tensor`)：可选参数，表示EP通信域的动态缩容信息。当某些通信卡因异常而从通信域中剔除，实际参与通信的卡数可从本参数中获取。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：预留参数，当前版本不支持，传空指针即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：可选择传入有效数据或填空指针，传入空指针时表示不使能动态缩容功能；当传入有效数据时，要求是一个1D的Tensor，shape为(4 + 2\*ep\_world\_size,)，数据类型支持int32；数据格式要求为ND，支持非连续的Tensor。Tensor中的前四个数字分别表示(是否缩容，缩容后实际rank数，缩容后共享专家使用的rank数，缩容后moe专家的个数)，后2 * epWorldSize表示2个rank映射表，缩容后本卡中因部分rank异常而从EP通信域中剔除，第一个Table的映射关系为Table1[epRankId]=localEpRankId或-1，localEpRankId表示新EP通信域中的rank Index，-1表示epRankId这张卡从通信域中被剔除，第二个Table映射关系为Table2[localEpRankId] = epRankId。

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>
`Tensor`
表示barrier构建Tensor依赖成功，接口内不做任何操作。数据类型支持BFLOAT16, FLOAT16、FLOAT32、BOOL、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64。 数据格式要求为ND，支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持静态图模式（不低于PyTorch 2.1.0版本）。
-   动态缩容后的部署信息通过`elastic_info`参数传递给算子，无需修改其他参数，缩容参数仅在`tp_world_size`取值为1时生效。动态缩容后，MOE专家卡上的本卡部署MOE专家数需与缩容前保持一致。<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

-   通信域使用约束：

    -   一个模型中的`npu_moe_distribute_barrier`需要使用单独通信域，且该通信域中不允许有其他算子。

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   单算子模式调用

    ```python
    import os，socket, torch, csv, time, torch_npu, random, torchair
    from typing import List
    import numpy as np
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch.multiprocessing as mp
    from torchair.configs.compiler_config import CompilerConfig
    from math import ceil
    from enum import Enum
    import traceback
    server_num = 1
    server_index = 0 if server_num > 1 else 0
    master_ip_address = '71.20.45.123'
    world_size = 16
    ep_world_size = world_size
    rank_per_dev = 16
    bs = 16  # token数量
    h = 7168  # 每个token的长度
    k = 2
    shared_broken_card_num = 2
    mor_broken_card_num = 2
    shared_expert_rank_num = 2
    local_moe = 4
    moe_expert_num = local_moe * (world_size - shared_expert_rank_num)
    ep_ranks_list = [list(range(0, ep_world_size))]
    
    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
    def init_hccl_comm(rank):
        dist.init_process_group(backend = "hccl", rank = rank, world_size = world_size, init_method = f'tcp://{master_ip}::50001')
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend = "hccl", rank = ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        return ep_hcomm_info, ep_group


    def warm_up_barrier(ep_hcomm_info):
        x_ref = torch.ones(1, dtype = torch.int32)
        warmup_barriers = {
            'x_ref' :x_ref.npu(),
            'group' :ep_hcomm_info,
            'world_size' :world_size,
        }
        out_res = torch_npu.npu_distribute_barrier(**warmup_barriers)
        return out_res

    def run_barrier_process(rank, queue):
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        x_ref = torch.ones(1 ,dtype = torch.int32)
        time_out = torch.tensor([1000000], dtype = torch.int32).npu()
        elastic_info = torch.tensor([1 , 12, 0, 1, -1, 2, 3, 4, -1 ,5, -1, 6, 7, -1, 8, 9, 10, 11, 
                                    0, 1, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15, -1, -1, -1, -1], dtype = torch.int32).npu()
        available_ranks = [0, 1, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15]
        group = ep_hcomm_info
        out = warm_up_barrier(ep_hcomm_info)

        if rank in available_ranks:
            out = torch_npu.npu_distribute_barrier(
                x_ref = x_ref,
                group = group,
                world_size = world_size,
                time_out = time_out,
                elastic_info = elastic_info)

        
        print(f'rank {rank} barrier npu finished! \n')


    if __name__ == "__main__":
        p_list = []
        queue = Manager().Queue()
        mp.set_start_method("spawn", force = True)
        for rank in range(rank_per_dev):
            p = Process(target=run_barrier_process, args=(rank, queue, ))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        out_ref = [None] * rank_per_dev
    ```

-   图模式调用

    ```python
    import os, socket, torch, csv, time, torch_npu, random, torchair
    from typing import List
    import numpy as np
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch.multiprocessing as mp
    from torchair.configs.compiler_config import CompilerConfig
    from math import ceil
    from enum import Enum
    import traceback
    server_num = 1
    server_index = 0 if server_num > 1 else 0
    master_ip = '71.20.45.123' if server_num > 1 else '127.0.0.1'
    is_elastic = 1
    world_size = 16
    ep_world_size = world_size
    rank_per_dev = 16
    bs = 16  # token数量
    h = 7168  # 每个token的长度
    k = 2
    shared_broken_card_num = 2
    mor_broken_card_num = 2
    shared_expert_rank_num = 2
    local_moe = 4
    moe_expert_num = local_moe * (world_size - shared_expert_rank_num)
    ep_ranks_list = [list(range(0, ep_world_size))]
    
    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
    def inithccl_coom(rank):
        dist.init_process_group(backend = "hccl", rank = rank, world_size = world_size, init_method = f'tcp://{master_ip}:50001')
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend = "hccl", ranks = ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        return ep_hcomm_info =, ep_group

    def compile_model(model):
        compiler_config = None
        npu_backend = torchair.get_npu_backend(compiler_config = compiler_comfig)

        if graph_type == 1:
            compiled_model = =torch.compile(model, backend = npu_backend, dynamic = False)
        elif graph_type == 2:
            compiled_model = =torch.compile(model, backend = npu_backend, dynamic = True)
        return compiled_model

    def warm_up_barrier(ep_hcomm_info):
        x_ref = torch.ones(1, dtype=torch.int32)
        warmup_barriers = (
            'x_ref' :x_ref.npu(),
            'group' :ep_hcomm_info,
            'world_size' :world_size,
        )
        out_res = torch_npu.npu_distribute_barrier(**warmup_barriers)
        return out_res
    class DISTRIBUTE_BARRIER_GRAPH_Model(torch.nn.Module):
        def __init__(self)
            self().__init__()
        def forward(self, x_ref, group, world_size, time_out, elastic_info):
            x_ref = torch_npu.npu_moe_distribute_barrier(
                x_ref=x_ref,
                group=ep_hcomm_info,
                world_size=world_size,
                time_out=time_out,
                elastic_info=elastic_info)

    def run_barrier_process(rank, queue):
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        x_ref = torch.ones(1 ,dtype = torch.int32)
        time_out = torch.tensor([1000000], dtype = torch.int32).npu()
        elastic_info = torch.tensor([1 , 12, 0, 1, -1, 2, 3, 4, -1 ,5, -1, 6, 7, -1, 8, 9, 10, 11, 
                                    0, 1, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15, -1, -1, -1, -1], dtype=torch.int32).npu()
        available_ranks = [0, 1, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15]
        group = ep_hcomm_info
        out = warm_up_barrier(ep_hcomm_info)
        barrier_model = compile_model(DISTRIBUTE_BARRIER_GRAPH_Model())
        if rank in available_ranks:
            out = barrier_model(
                x_ref=x_ref,
                group=ep_hcomm_info,
                world_size=world_size,
                time_out=time_out,
                elastic_info=elastic_info)

        
        print(f'rank {rank} barrier npu finished! \n')


    if __name__ == "__main__":
        p_list = []
        queue = Manager().Queue()
        mp.set_start_method("spawn", force = True)
        for rank in range(rank_per_dev):
            p = Process(target=run_barrier_process, args=(rank, queue, ))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        out_ref = [None] * rank_per_dev
    ```
