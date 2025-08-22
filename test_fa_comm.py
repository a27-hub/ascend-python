import torch
import torch_npu
import os
import torch.distributed as dist

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend='hccl', init_method='env://')
    torch.npu.set_device(dist.get_rank())

def create_npu_streams():
    """创建两个NPU流"""
    attention_stream = torch.npu.Stream()
    comm_stream = torch.npu.Stream()
    return attention_stream, comm_stream

def main():
    # 初始化分布式
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 创建两个NPU流
    attention_stream, comm_stream = create_npu_streams()
    atten_prev = None
    i = 0
    with torch.npu.stream(attention_stream):
        while True:
            LAYER_INDEX = 46
            SPARSE_MODE=3
            q = torch.load(f"q_{LAYER_INDEX}_15.pt").cpu().npu()
            k = torch.load(f"k_{LAYER_INDEX}_15.pt").cpu().npu()
            v = torch.load(f"v_{LAYER_INDEX}_15.pt").cpu().npu()
            attn_mask_npu = torch.load(f"attn_mask_npu_{LAYER_INDEX}_15.pt").cpu().npu()
            cu_seqlens_q = torch.load(f"cu_seqlens_q_{LAYER_INDEX}_15.pt").cpu()
            cu_seqlens_k = torch.load(f"cu_seqlens_k_{LAYER_INDEX}_15.pt").cpu()
            # print(q)
            atten_out_new = torch_npu.npu_fusion_attention(q,k,v,4,
                                        pse=None,
                                        padding_mask=None,
                                        atten_mask=attn_mask_npu,
                                        scale=0.08838834764831845,
                                        keep_prob=1.0,
                                        input_layout="TND",
                                        actual_seq_qlen=tuple(cu_seqlens_q[1:].numpy().tolist()),
                                        actual_seq_kvlen=tuple(cu_seqlens_k[1:].numpy().tolist()),
                                        sparse_mode=SPARSE_MODE)[0]
            if i == 0:
                print("SKIP!")
                atten_prev = atten_out_new
            else:
                if not torch.equal(atten_prev, atten_out_new):
                    print("FAILED!!!")
                    break
                else:
                    print("SUCCESS")
            i += 1
    
    # 在comm_stream上执行集合通信 
    with torch.npu.stream(comm_stream):
        while True:
            input = torch.zeros_like(q).npu()
            output_chunk = torch.chunk(input, world_size, dim=0)[rank]
            reduce_scatter_input = [input] * world_size
            reduce_scatter_output = torch.empty_like(input)
            dist.reduce_scatter(reduce_scatter_output, reduce_scatter_input)
            # 使用all_gather进行结果广播
            all_gather_output = [torch.empty_like(reduce_scatter_output) for _ in range(world_size)]
            dist.all_gather(all_gather_output, reduce_scatter_output)
    
if __name__ == "__main__":
    main()