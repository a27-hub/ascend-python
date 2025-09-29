# torch\_npu.npu\_alltoallv\_gmm_v2<a name="ZH-CN_TOPIC_0000002350725076"></a>

## 产品支持情况<a name="zh-cn_topic_0000002282815538_section1369303644412"></a>

<a name="zh-cn_topic_0000002282815538_table38301303189"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000002282815538_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002282815538_p1883113061818"><a name="zh-cn_topic_0000002282815538_p1883113061818"></a><a name="zh-cn_topic_0000002282815538_p1883113061818"></a><span id="zh-cn_topic_0000002282815538_ph24751558184613"><a name="zh-cn_topic_0000002282815538_ph24751558184613"></a><a name="zh-cn_topic_0000002282815538_ph24751558184613"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002282815538_p783113012187"><a name="zh-cn_topic_0000002282815538_p783113012187"></a><a name="zh-cn_topic_0000002282815538_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002282815538_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002282815538_p2098311377352"><a name="zh-cn_topic_0000002282815538_p2098311377352"></a><a name="zh-cn_topic_0000002282815538_p2098311377352"></a><span id="zh-cn_topic_0000002282815538_ph1719614396352"><a name="zh-cn_topic_0000002282815538_ph1719614396352"></a><a name="zh-cn_topic_0000002282815538_ph1719614396352"></a><term id="zh-cn_topic_0000002282815538_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002282815538_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002282815538_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></term>
</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002282815538_p7948163910184"><a name="zh-cn_topic_0000002282815538_p7948163910184"></a><a name="zh-cn_topic_0000002282815538_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002282815538_row220181016241">
<td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 ">
<p><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 ">
<p>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002282815538_row220181016242">
<td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1">
<p><term>昇腾910_95 AI处理器</term></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2">
<p>√</p>
</td>
</tr>
</tbody>
</table>


## 功能说明<a name="zh-cn_topic_0000002282815538_section14441124184110"></a>

- API功能：MoE（Mixture of Experts，混合专家模型）网络中，完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。兼容npu\_alltoallv\_gmm接口功能，在其基础上增加aiv模式。

  当通信模式为aiv时，完成路由专家的量化GMM和非量化GMM：

- 非量化：

  ![sp251010_141328](D:\hxy\api\alltoallvgmm\figures\sp251010_141328.png)

  

-   量化：

    ![image-20251010141538862](C:\Users\h00947502\AppData\Roaming\Typora\typora-user-images\image-20251010141538862.png)

    global_token_per_expert为通信矩阵，仅在aiv模式下输出
    
    当通信模式为aicpu或ccu时：
    
-   路由专家计算公式：

    ![](figures/zh-cn_formulaimage_0000002357766385.png)

    -   ata\_out是gmm\_x进行AlltoAllv通信的输出结果，后续用于Permute计算。
    -   permute\_out是ata\_out进行Permute计算的输出结果，作为路由专家进行GroupedMatMul计算的左矩阵。
    -   gmm\_weight指路由专家进行GroupedMatMul计算的右矩阵。
    -   gmm\_y指路由专家进行GroupedMatMul计算的输出。

-   共享专家计算公式：

    ![](figures/zh-cn_formulaimage_0000002323547858.png)

    -   mm\_x指共享专家MatMul计算的左矩阵。
    -   mm\_weight指共享专家MatMul计算的右矩阵。
    -   mm\_y指共享专家MatMul计算的输出。

## 函数原型<a name="zh-cn_topic_0000002282815538_section45077510411"></a>

```
torch_npu.npu_alltoallv_gmm_v2(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False, gmm_x_scale=None, gmm_weight_scale=None, comm_mode="aicpu", y_dtype=0, max_output_token_num=-1) -> (Tensor, Tensor, Tensor， Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002282815538_section112637109429"></a>

-   **gmm\_x**（`Tensor`）：必选参数，AlltoAllv通信与Permute操作后结果作为GroupedMatMul计算的左矩阵。数据类型支持`float16`、`bfloat16`，支持2维，shape为$(BSK, H1)$，数据格式支持ND。
-   **gmm\_weight**（`Tensor`）：必选参数，GroupedMatMul计算的右矩阵。数据类型与`gmm_x`保持一致，支持3维，shape为$(e, H1, N1)$，数据格式支持ND。当comm_mode为“aiv”时，支持ND、NZ数据格式。
-   **hcom**（`str`）：必选参数，专家并行的通信域名，字符串长度要求\(0, 128\)。
-   **ep\_world\_size**（`int`）：必选参数，EP通信域size，Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件，当comm_mode为“aiv”时，取值支持2、4、8。Atlas A3 训练系列产品/Atlas A3 推理系列产品，当comm_mode为“aicpu”时，取值支持8、16、32、64。当comm_mode为“aicpu”时，取值支持2、4、8、16。昇腾910_95 AI处理器，当comm_mode为“ccu”时，取值支持4、8、16、32、64。
-   **send\_counts**（`List[int]`）：可选参数，表示发送给其他卡的token数，数据类型支持int，取值大小为e\*`ep_world_size`，最大为256。**当comm_mode为“aiv”时，不支持该参数，传空链表**。
-   **recv\_counts**（`List[int]`）：可选参数，表示接收其他卡的token数，数据类型支持int，取值大小为e\*`ep_world_size`，最大为256。**当comm_mode为“aiv”时，不支持该参数，传空链表**。
-   **send\_counts\_tensor**（`Tensor`）：可选参数，数据类型支持int，shape为$(e*ep\_world\_size,)$，数据格式支持ND。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件数据类型支持int。Atlas A3 训练系列产品/Atlas A3 推理系列产品当comm_mode为“aiv”时，数据类型支持int；**当comm_mode为“aicpu”时，暂不支持该输入，传nullptr**。昇腾910_95 AI处理器**暂不支持该输入，传nullptr**。
-   **recv\_counts\_tensor**（`Tensor`）：可选参数，数据类型支持int，shape为$(e*ep\_world\_size,)$，数据格式支持ND。**当前版本暂不支持**，使用默认值即可。
-   **mm\_x**（`Tensor`）：可选参数，共享专家MatMul计算中的左矩阵。当需要融合共享专家矩阵计算时，该参数必选，数据类型支持`float16`、`bfloat16`，支持2维，shape为$(BS, H2)$。
-   **mm\_weight**（`Tensor`）：可选参数，共享专家MatMul计算中的右矩阵。当需要融合共享专家矩阵计算时，该参数必选，数据类型与`mm_x`保持一致，支持2维，shape为$(H2, N2)$。
-   **trans\_gmm\_weight**（`bool`）：可选参数，GroupedMatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。
-   **trans\_mm\_weight**（`bool`）：可选参数，共享专家MatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。
-   **permute\_out\_flag**（`bool`）：可选参数，Permute结果是否需要输出，true表明需要输出，false表明不需要输出。
-   **gmm_x_scale**（`Tensor`）：可选参数，mm左矩阵反量化参数。数据类型支持`float`，shape为$(BSK, )$，数据格式支持ND。**当前版本仅在comm_mode为”aiv”时支持**。
-   **gmm_weight_scale**（`Tensor`）：可选参数，mm右矩阵反量化参数。数据类型支持`int64`，shape为$(epWorldSize, N1)$，数据格式支持ND。**当前版本仅在commMode为”aiv”时支持**。
-   **comm_mode**（`str`）：可选参数，Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件支持取值“aiv”，Atlas A3 训练系列产品/Atlas A3 推理系列产品支持取值“aiv”和“aicpu”，昇腾910_95 AI处理器支持取值“ccu”，默认值为“aicpu”。不受环境变量的影响
-   **y_dtype**（`int`）：可选参数，支持取值0和1，分别代表量化场景下，gmm_y的输出类型，0为float16，1为bfloat16。
-   **max_output_token_num**（`int`）：可选参数，int类型，表示gmm_y第一维的shape，**仅在comm_mode为"aiv"时支持。**

## 返回值说明<a name="zh-cn_topic_0000002282815538_section22231435517"></a>

-   **gmm\_y**（`Tensor`）：计算输出，表示最终的计算结果，支持2维，shape为$(A, N1)$。aicpu模式下数据类型与输入gmmX保持一致，aiv量化模式下取决于y_dtype。
-   **mm\_y**（`Tensor`）：计算输出，共享专家MatMul的输出，数据类型与`mm_x`保持一致，支持2维，shape为$(BS, N2)$。仅当传入`mm_x`与`mm_weight`才输出。
-   **permute\_out**（`Tensor`）：计算输出，Permute之后的输出，数据类型与`gmm_x`保持一致。
-   **global_token_per_expert**（`Tensor`）：计算输出，通信矩阵，数据类型支持int32，**仅当comm_mode为"aiv"时才输出。**

## 约束说明<a name="zh-cn_topic_0000002282815538_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   单卡通信量范围：\[2MB, 100MB\]。
-   输入参数Tensor中shape使用的变量说明：
    -   BSK：本卡发送的token数（BS\*K=BSK），是send\_counts参数累加之和，当comm_mode为"aicpu"或"ccu"时，取值范围(0,52428800)。当comm_mode为"aiv"时，取值范围(0, 300000)。
    -   H1：表示路由专家hidden size隐藏层大小，当comm_mode为"aicpu"或"ccu"时，取值范围(0, 65536)，当comm_mode为"aiv"时，取值范围(0, 16384]。

    -   H2：表示共享专家hidden size隐藏层大小，取值范围\(0, 12288\]。
    -   e：表示单卡上专家个数，当comm_mode为aicpu时，e<=32，e\*ep\_world\_size最大支持256。当commMode为aiv时：e<=16。

    -   N1：表示路由专家的head\_num，当comm_mode为aiv时，取值范围(0, 12000)。当comm_mode为aicpu或ccu时，取值范围(0, 65536)。
    -   N2：表示共享专家的head\_num，取值范围\(0, 65536\)。

    -   BS：表示batch sequence size。
    -   K：表示选取topK个专家，K的范围\[2, 8\]。

    -   A：本卡收到的token数，当comm_mode为"aicpu”时，A是recvCounts参数累加之和。当comm_mode为"aiv"时，A的实际大小可以小于本卡收到的token个数，小于部分会被截断。
    -   EP通信域内所有卡上的A累加和等于所有卡上的BSK累加和。

## 调用示例<a name="zh-cn_topic_0000002282815538_section14459801435"></a>

- 单算子aicpu/ccu模式调用

  ```python
  import torch
  import torch_npu
  import torch.distributed as dist
  import torch.multiprocessing as mp
  
  def run_npu_alltoallv_gmm_v2(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
      torch_npu.npu.set_device(rank)
      init_method = 'tcp://' + master_ip + ':' + master_port
      dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
      from torch.distributed.distributed_c10d import _get_default_group
      default_pg = _get_default_group()
      if torch.__version__ > '2.0.1':
          hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
      else:
          hcom_info = default_pg.get_hccl_comm_name(rank)
  
      input = torch.randn(gmm_x, dtype=dtype).npu()
      weight = torch.randn(gmm_w, dtype=dtype).npu()
  
      print(torch_npu.npu_alltoallv_gmm_v2(gmm_x =input,
                                              gmm_weight = weight,
                                              hcom= hcom_info,
                                              ep_world_size = ep_world_size,
                                              send_counts = list(send_counts),
                                              recv_counts = list(recv_counts),
                                              send_counts_tensor = None, #send_counts_tensor,
                                              recv_counts_tensor = None, #recv_counts_tensor,
                                              mm_x = None,
                                              mm_weight = None,
                                              trans_gmm_weight = False,
                                              trans_mm_weight  = False,
                                              permute_out_flag  = False,
                                           	gmm_x_scale = None,
                                           	gmm_weight_scale = None,
                                       		comm_mode = "aicpu",
                                          	y_dtype = 0,
                                          	max_output_token_num = -1))
  
  if __name__ == "__main__":
      epWorkSize = 8
      e = 4
      master_ip = '127.0.0.1'
      master_port = '50001'
      BS = 512
      K = 8
      gmm_x_shape = [BS*K, 2048]
      gmm_weight_shape = [e, 2048, 2048]
      send_counts = [128] * (e * epWorkSize)
      recv_counts = [128] * (e * epWorkSize)
      dtype = torch.float16
      mp.spawn(run_npu_alltoallv_gmm_v2, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
  ```

-   单算子aiv模式调用

    ```python
    import os
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    
    def run_npu_alltoallv_gmm_v2(rank, ep_world_size, master_ip, master_port, gmm_x_shape, gmm_w_shape, e, dtype, max_output_token_num):
        torch_npu.npu.set_device(rank)
        init_method = f'tcp://{master_ip}:{master_port}'
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
            
        gmm_x = torch.randn(gmm_x_shape, dtype=dtype).npu()
        gmm_weight = torch.randn(gmm_w_shape, dtype=dtype).npu()
        
        send_counts_tensor = torch.full(
            (e * ep_world_size,),  
            128,                  
            dtype=torch.int32
        ).npu()
        
        recv_counts_tensor = None
        gmm_x_scale = torch.tensor(1.0, dtype=torch.float32).npu()
        gmm_weight_scale = torch.tensor(1.0, dtype=torch.float32).npu()
        print(f"[Rank {rank}] Running AIV mode npu_alltoallv_gmm_v2")
    
        y_out = torch_npu.npu_alltoallv_gmm_v2(
            gmm_x=gmm_x,
            gmm_weight=gmm_weight,
            hcom=hcom_info,
            ep_world_size=ep_world_size,
            send_counts_tensor=send_counts_tensor,   
            recv_counts_tensor=recv_counts_tensor,   
            gmm_x_scale=gmm_x_scale,                 
            gmm_weight_scale=gmm_weight_scale,      
            comm_mode="aiv",                    
            y_dtype=0,                               
            max_output_token_num=max_output_token_num,                 
        )
    
        print(f"[Rank {rank}] npu_alltoallv_gmm_v2 output:")
        for i, out in enumerate(y_out):
            print(f"  Output[{i}]: {None if out is None else (out.shape, out.dtype)}")
    
    
    if __name__ == "__main__":
        ep_world_size = 8
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50011'
    
        BS = 512
        K = 8
        gmm_x_shape = [BS * K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        dtype = torch.int8
        max_output_token_num = BS * K
    
        mp.spawn(
            run_npu_alltoallv_gmm_v2,
            args=(ep_world_size, master_ip, master_port, gmm_x_shape, gmm_weight_shape, e, dtype, max_output_token_num),
            nprocs=ep_world_size,
            join=True
        )
    
    ```
    
-   图模式调用（当前版本仅aicpu支持）

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchair
    
    class ALLTOALLV_GMM_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,gmm_x, gmm_weight,
                    hcom, ep_world_size,
                    send_counts, recv_counts, send_counts_tensor, recv_counts_tensor,
                    mm_x, mm_weight,
                    trans_gmm_weight, trans_mm_weight, permute_out_flag):
            return torch_npu.npu_alltoallv_gmm_v2(gmm_x =gmm_x,
                                                gmm_weight = gmm_weight,
                                                hcom= hcom,
                                                ep_world_size = ep_world_size,
                                                send_counts = list(send_counts),
                                                recv_counts = list(recv_counts),
                                                send_counts_tensor = None, #send_counts_tensor,
                                                recv_counts_tensor = None, #recv_counts_tensor,
                                                mm_x = mm_x,
                                                mm_weight = mm_weight,
                                                trans_gmm_weight = trans_gmm_weight,
                                                trans_mm_weight  = trans_mm_weight,
                                                permute_out_flag  = permute_out_flag)
    
    def run_npu_alltoallv_gmm_v2(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
    
        input = torch.randn(gmm_x, dtype=dtype).npu()
        weight = torch.randn(gmm_w, dtype=dtype).npu()
    
        model = ALLTOALLV_GMM_GRAPH_Model()
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        # 静态图：dynamic=False；动态图：dynamic=True
        model = torch.compile(ALLTOALLV_GMM_GRAPH_Model(), backend=npu_backend, dynamic=False)
        print(model(gmm_x=input,
                        gmm_weight=weight,
                        send_counts_tensor=None,
                        recv_counts_tensor=None,
                        mm_x=None,
                        mm_weight=None,
                        hcom=hcom_info,
                        ep_world_size=ep_world_size,
                        send_counts=send_counts,
                        recv_counts=recv_counts,
                        trans_gmm_weight=False,
                        trans_mm_weight=False,
                        permute_out_flag=True))
    
    if __name__ == "__main__":
        epWorkSize = 8
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50001'
        BS = 512
        K = 8
        gmm_x_shape = [BS*K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        send_counts = [128] * (e * epWorkSize)
        recv_counts = [128] * (e * epWorkSize)
        dtype = torch.float16
        mp.spawn(run_npu_alltoallv_gmm_v2, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

